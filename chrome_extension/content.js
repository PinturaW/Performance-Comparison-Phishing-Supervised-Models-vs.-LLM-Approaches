// content.js - Runs at document_start
(function() {
    const currentUrl = window.location.href;
    
    // Skip checking for warning page itself
    if (currentUrl.includes('warning.html')) {
      return;
    }
  
    // Skip non-http(s) URLs
    if (!currentUrl.startsWith('http://') && !currentUrl.startsWith('https://')) {
      return;
    }
  
    // Skip localhost and local IPs
    const hostname = window.location.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname.startsWith('192.168.')) {
      return;
    }
  
    // Check URL with background script
    chrome.runtime.sendMessage(
      { action: 'checkUrl', url: currentUrl },
      (response) => {
        if (chrome.runtime.lastError) {
          console.error('Extension error:', chrome.runtime.lastError);
          return;
        }
  
        if (!response) {
          console.error('No response from background script');
          return;
        }
  
        // If AI protection is disabled
        if (response.disabled) {
          console.log('AI protection disabled');
          return;
        }
  
        // If bypassed, allow page to load
        if (response.bypassed) {
          console.log('Site bypassed, allowing access');
          return;
        }
  
        // If manual block
        if (response.manualBlock) {
          console.log('Site in manual blocklist');
          const warningUrl = chrome.runtime.getURL('warning.html') +
            `?blocked=${encodeURIComponent(hostname)}&redirect=${encodeURIComponent(currentUrl)}&manual=1`;
          window.location.href = warningUrl;
          return;
        }
  
        // If phishing detected, redirect to warning page
        if (response.prediction === 1) {
          const warningUrl = chrome.runtime.getURL('warning.html') +
            `?blocked=${encodeURIComponent(hostname)}&redirect=${encodeURIComponent(currentUrl)}`;
          
          window.location.href = warningUrl;
        }
      }
    );
  })();