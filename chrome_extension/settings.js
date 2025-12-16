// settings.js - Settings page logic

const API_URL = 'http://localhost:5001/health';

// Default settings
const DEFAULT_SETTINGS = {
  aiProtection: true,
  blockBeforeLoad: true,
  manualBlocklist: true,
  reputationGuard: true,
  whoisCheck: true,
  showNotifications: true
};

// Load settings from storage
async function loadSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(['settings'], (result) => {
      resolve(result.settings || DEFAULT_SETTINGS);
    });
  });
}

// Save settings to storage
async function saveSettings(settings) {
  return new Promise((resolve) => {
    chrome.storage.sync.set({ settings: settings }, () => {
      resolve();
    });
  });
}

// Check API status
async function checkAPIStatus() {
  const statusDiv = document.getElementById('apiStatus');
  const statusText = document.getElementById('apiStatusText');
  
  try {
    const response = await fetch(API_URL, { 
      method: 'GET',
      signal: AbortSignal.timeout(3000)
    });
    
    if (response.ok) {
      statusDiv.className = 'api-status online';
      statusText.textContent = '✓ API Connected (http://localhost:5001)';
    } else {
      throw new Error('API error');
    }
  } catch (error) {
    statusDiv.className = 'api-status offline';
    statusText.textContent = '✗ API Offline - Protection features limited';
    
    // Disable AI-dependent features if API is offline
    document.getElementById('aiProtection').disabled = true;
    document.getElementById('reputationGuard').disabled = true;
  }
}

// Load statistics
async function loadStatistics() {
  chrome.storage.sync.get(['stats', 'blockedList'], (result) => {
    const stats = result.stats || { totalChecked: 0, phishingBlocked: 0 };
    const blockedList = result.blockedList || [];
    
    document.getElementById('totalChecked').textContent = stats.totalChecked || 0;
    document.getElementById('phishingBlocked').textContent = stats.phishingBlocked || 0;
    document.getElementById('manualCount').textContent = blockedList.length;
  });
}

// Initialize page
(async function init() {
  // Check API status
  checkAPIStatus();
  
  // Load settings
  const settings = await loadSettings();
  
  // Apply settings to checkboxes
  document.getElementById('aiProtection').checked = settings.aiProtection;
  document.getElementById('blockBeforeLoad').checked = settings.blockBeforeLoad;
  document.getElementById('manualBlocklist').checked = settings.manualBlocklist;
  document.getElementById('reputationGuard').checked = settings.reputationGuard;
  document.getElementById('whoisCheck').checked = settings.whoisCheck;
  document.getElementById('showNotifications').checked = settings.showNotifications;
  
  // Load statistics
  loadStatistics();
  
  // Enable/disable dependent settings
  updateDependentSettings();
})();

// Update dependent settings
function updateDependentSettings() {
  const aiProtection = document.getElementById('aiProtection').checked;
  
  // If AI protection is off, disable dependent features
  document.getElementById('blockBeforeLoad').disabled = !aiProtection;
  document.getElementById('reputationGuard').disabled = !aiProtection;
}

// Listen for AI protection toggle
document.getElementById('aiProtection').addEventListener('change', updateDependentSettings);

// Save button
document.getElementById('saveBtn').addEventListener('click', async () => {
  const settings = {
    aiProtection: document.getElementById('aiProtection').checked,
    blockBeforeLoad: document.getElementById('blockBeforeLoad').checked,
    manualBlocklist: document.getElementById('manualBlocklist').checked,
    reputationGuard: document.getElementById('reputationGuard').checked,
    whoisCheck: document.getElementById('whoisCheck').checked,
    showNotifications: document.getElementById('showNotifications').checked
  };
  
  await saveSettings(settings);
  
  // Notify background script
  chrome.runtime.sendMessage({ action: 'settingsUpdated', settings: settings });
  
  // Show feedback
  const btn = document.getElementById('saveBtn');
  const originalText = btn.textContent;
  btn.textContent = '✓ Saved!';
  btn.style.backgroundColor = '#218838';
  
  setTimeout(() => {
    btn.textContent = originalText;
    btn.style.backgroundColor = '#28a745';
  }, 2000);
});

// Reset button
document.getElementById('resetBtn').addEventListener('click', async () => {
  const confirmed = confirm(
    '⚠️ Reset to Default Settings?\n\n' +
    'This will restore all settings to their default values.\n\n' +
    'Are you sure?'
  );
  
  if (confirmed) {
    await saveSettings(DEFAULT_SETTINGS);
    
    // Reload page to apply defaults
    window.location.reload();
  }
});

// Back link
document.getElementById('backLink').addEventListener('click', (e) => {
  e.preventDefault();
  window.close();
});