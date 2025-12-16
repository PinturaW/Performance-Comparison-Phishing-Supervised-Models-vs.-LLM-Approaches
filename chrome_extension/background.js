// background.js - Service Worker
const API_URL = 'http://localhost:5001/api/predict/simple';
const BYPASS_DURATION = 10 * 60 * 1000; // 10 minutes

// In-memory bypass flags: {hostname: timestamp}
const bypassFlags = {};

// Settings cache
let settings = {
  aiProtection: true,
  blockBeforeLoad: true,
  manualBlocklist: true,
  reputationGuard: true,
  whoisCheck: true,
  showNotifications: true
};

// Statistics
let stats = {
  totalChecked: 0,
  phishingBlocked: 0
};

// Load settings on startup
chrome.storage.sync.get(['settings', 'stats'], (result) => {
  if (result.settings) {
    settings = result.settings;
  }
  if (result.stats) {
    stats = result.stats;
  }
});

// Check if bypass is still valid
function isBypassed(hostname) {
  if (bypassFlags[hostname]) {
    const elapsed = Date.now() - bypassFlags[hostname];
    if (elapsed < BYPASS_DURATION) {
      return true;
    } else {
      // Expired, remove it
      delete bypassFlags[hostname];
      return false;
    }
  }
  return false;
}

// Check URL with API
async function checkUrlWithAPI(url) {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: url })
    });

    if (!response.ok) {
      console.error('API error:', response.status);
      return { prediction: 0, error: true };
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('API call failed:', error);
    return { prediction: 0, error: true };
  }
}

// Get hostname from URL
function getHostname(url) {
  try {
    return new URL(url).hostname;
  } catch {
    return null;
  }
}

// Message listener
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  
  // Check URL
  if (request.action === 'checkUrl') {
    // Check if AI protection is enabled
    if (!settings.aiProtection) {
      sendResponse({ prediction: 0, disabled: true });
      return true;
    }

    const hostname = getHostname(request.url);
    
    // Check bypass first
    if (hostname && isBypassed(hostname)) {
      sendResponse({ prediction: 0, bypassed: true });
      return true;
    }

    // Check manual blocklist if enabled
    if (settings.manualBlocklist) {
      chrome.storage.sync.get(['blockedList'], (result) => {
        const blockedList = result.blockedList || [];
        if (blockedList.includes(hostname)) {
          sendResponse({ prediction: 1, manualBlock: true });
          
          // Update stats
          stats.phishingBlocked++;
          chrome.storage.sync.set({ stats: stats });
          
          // Show notification if enabled
          if (settings.showNotifications) {
            chrome.notifications.create({
              type: 'basic',
              iconUrl: 'icon48.png',
              title: '🛡️ Phishing Blocked',
              message: `Manual blocklist: ${hostname}`
            });
          }
          return;
        }
        
        // Continue to API check
        checkWithAPI();
      });
      return true;
    }

    // Check with API
    function checkWithAPI() {
      stats.totalChecked++;
      chrome.storage.sync.set({ stats: stats });

      checkUrlWithAPI(request.url).then(async (result) => {
        // If phishing detected, add to blocked list
        if (result.prediction === 1 && hostname) {
          stats.phishingBlocked++;
          chrome.storage.sync.set({ stats: stats });

          try {
            const storage = await chrome.storage.sync.get(['blockedList']);
            const blockedList = storage.blockedList || [];
            if (!blockedList.includes(hostname)) {
              blockedList.push(hostname);
              await chrome.storage.sync.set({ blockedList: blockedList });
            }
          } catch (error) {
            console.error('Storage error:', error);
          }

          // Show notification if enabled
          if (settings.showNotifications) {
            chrome.notifications.create({
              type: 'basic',
              iconUrl: 'icon48.png',
              title: '⚠️ Phishing Detected',
              message: `AI blocked: ${hostname}`
            });
          }
        }
        
        sendResponse(result);
      });
    }

    checkWithAPI();
    return true; // Async response
  }

  // Proceed to site (set bypass and redirect)
  if (request.action === 'proceedToSite') {
    const { url, hostname } = request;
    
    // Set bypass flag
    bypassFlags[hostname] = Date.now();
    console.log(`Bypass set for ${hostname}, valid until ${new Date(Date.now() + BYPASS_DURATION)}`);
    
    // Redirect using chrome.tabs.update
    chrome.tabs.update(sender.tab.id, { url: url }, () => {
      sendResponse({ success: true });
    });
    
    return true; // Async response
  }

  // Add to manual block list
  if (request.action === 'addToBlockList') {
    chrome.storage.sync.get(['blockedList'], (result) => {
      const blockedList = result.blockedList || [];
      const hostname = request.hostname;
      
      if (!blockedList.includes(hostname)) {
        blockedList.push(hostname);
        chrome.storage.sync.set({ blockedList: blockedList }, () => {
          // Show notification instead of relying on popup alert
          if (settings.showNotifications) {
            chrome.notifications.create({
              type: 'basic',
              iconUrl: 'icon48.png',
              title: '🚫 Site Blocked',
              message: `${hostname} added to blocklist`
            });
          }
          sendResponse({ success: true });
        });
      } else {
        sendResponse({ success: false, message: 'Already in block list' });
      }
    });
    
    return true; // Async response
  }

  // Reset block list
  if (request.action === 'resetBlockList') {
    chrome.storage.sync.set({ blockedList: [] }, () => {
      sendResponse({ success: true });
    });
    
    return true; // Async response
  }

  // Settings updated
  if (request.action === 'settingsUpdated') {
    settings = request.settings;
    console.log('Settings updated:', settings);
    sendResponse({ success: true });
    return true;
  }

  return false;
});