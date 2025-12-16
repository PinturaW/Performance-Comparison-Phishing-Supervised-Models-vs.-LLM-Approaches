// popup.js - Popup logic with API integration

const API_URL = 'http://localhost:5001/api/predict/simple';

// Get hostname from URL
function getHostname(url) {
  try {
    const parsed = new URL(url);
    return parsed.hostname.replace('www.', '');
  } catch {
    return null;
  }
}

// Get registrable domain (simple version)
function getRegistrableDomain(hostname) {
  if (!hostname) return null;
  const parts = hostname.split('.');
  if (parts.length >= 2) {
    return parts.slice(-2).join('.');
  }
  return hostname;
}

// Clean URL input (remove http/https)
function cleanUrl(url) {
  return url.replace(/^https?:\/\//, '').replace(/^www\./, '').trim();
}

// Check URL with API
async function checkUrlWithAPI(url) {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: url })
    });
    
    if (!response.ok) {
      console.error('API error:', response.status);
      return null;
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('API call failed:', error);
    return null;
  }
}

// Block current website
document.getElementById('blockCurrentBtn').addEventListener('click', async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab || !tab.url) {
      alert('❌ Cannot detect current website');
      return;
    }

    const hostname = getHostname(tab.url);
    
    if (!hostname) {
      alert('❌ Invalid URL');
      return;
    }

    // Skip chrome:// and extension pages
    if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
      alert('❌ Cannot block browser pages');
      return;
    }

    // Skip localhost
    if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname.startsWith('192.168.')) {
      alert('❌ Cannot block localhost');
      return;
    }

    // Show loading
    const btn = document.getElementById('blockCurrentBtn');
    const originalText = btn.textContent;
    btn.textContent = 'Checking...';
    btn.disabled = true;

    // Check with API first
    const result = await checkUrlWithAPI(tab.url);
    
    btn.textContent = originalText;
    btn.disabled = false;

    if (!result) {
      alert('⚠️ API connection failed. Added to local blocklist anyway.');
    } else if (result.prediction === 1) {
      alert(`⚠️ WARNING: This site is already detected as PHISHING!\n\nURL: ${hostname}\n\nIt will be blocked automatically.`);
    } else {
      alert(`ℹ️ This site appears legitimate according to AI.\n\nURL: ${hostname}\n\nBut you can still add it to your manual blocklist.`);
    }

    // Add to block list regardless
    const registrable = getRegistrableDomain(hostname);
    chrome.runtime.sendMessage(
      { action: 'addToBlockList', hostname: registrable },
      (response) => {
        if (response && response.success) {
          // Close popup after adding
          window.close();
        } else if (response && response.message === 'Already in block list') {
          // Already blocked - just close
          window.close();
        } else {
          alert(`⚠️ Failed to add to blocklist`);
        }
      }
    );
  } catch (error) {
    console.error('Error:', error);
    alert('❌ Error blocking website');
  }
});

// Reset block list
document.getElementById('resetBtn').addEventListener('click', () => {
  const confirmed = confirm(
    '⚠️ Reset Block List?\n\n' +
    'This will remove ALL manually blocked websites.\n\n' +
    'Note: AI-detected phishing sites will still be blocked.\n\n' +
    'Are you sure?'
  );
  
  if (confirmed) {
    chrome.runtime.sendMessage(
      { action: 'resetBlockList' },
      (response) => {
        if (response && response.success) {
          alert('✅ Manual block list has been reset\n\nAI protection is still active.');
        } else {
          alert('❌ Failed to reset');
        }
      }
    );
  }
});

// Add to block list manually
document.getElementById('addToBlockListBtn').addEventListener('click', async () => {
  const input = document.getElementById('urlInput');
  let url = input.value.trim();
  
  if (!url) {
    alert('❌ Please enter a URL');
    return;
  }

  // Clean URL (remove http/https, www)
  url = cleanUrl(url);
  
  // Validate format
  if (!url.includes('.')) {
    alert('❌ Invalid URL format\n\nExample: example.com');
    return;
  }

  // Show loading
  const btn = document.getElementById('addToBlockListBtn');
  const originalText = btn.textContent;
  btn.textContent = 'Checking...';
  btn.disabled = true;

  // Check with API first
  const fullUrl = url.startsWith('http') ? url : 'https://' + url;
  const result = await checkUrlWithAPI(fullUrl);
  
  btn.textContent = originalText;
  btn.disabled = false;

  if (result) {
    if (result.prediction === 1) {
      alert(`⚠️ PHISHING DETECTED!\n\nURL: ${url}\n\nThis site is already flagged by AI as phishing.\nIt will be blocked automatically.`);
    } else {
      alert(`ℹ️ AI Analysis: Legitimate\n\nURL: ${url}\n\nThe AI model thinks this site is safe.\nAre you sure you want to block it?`);
    }
  } else {
    alert('⚠️ Cannot connect to API\n\nAdding to local blocklist anyway.');
  }

  // Add to block list
  chrome.runtime.sendMessage(
    { action: 'addToBlockList', hostname: url },
    (response) => {
      if (response && response.success) {
        input.value = '';
        // Show success and close after short delay
        btn.textContent = '✅ Added!';
        btn.style.backgroundColor = '#28a745';
        setTimeout(() => {
          window.close();
        }, 1000);
      } else if (response && response.message === 'Already in block list') {
        alert(`⚠️ ${url} is already in block list`);
      } else {
        alert(`⚠️ Failed to add to blocklist`);
      }
    }
  );
});

// Display current status on popup open
(async function init() {
  try {
    // Get current tab info
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url) {
      const hostname = getHostname(tab.url);
      if (hostname && !tab.url.startsWith('chrome://') && !tab.url.startsWith('chrome-extension://')) {
        // Check if site is in blocklist
        chrome.storage.sync.get(['blockedList'], (result) => {
          const blockedList = result.blockedList || [];
          const registrable = getRegistrableDomain(hostname);
          if (blockedList.includes(registrable) || blockedList.includes(hostname)) {
            document.getElementById('blockCurrentBtn').textContent = '🚫 Already Blocked';
            document.getElementById('blockCurrentBtn').style.backgroundColor = '#6c757d';
          }
        });
      }
    }
  } catch (error) {
    console.error('Init error:', error);
  }
})();

// Settings button
document.getElementById('settingsBtn').addEventListener('click', () => {
  chrome.tabs.create({ url: chrome.runtime.getURL('settings.html') });
});