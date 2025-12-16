// warning.js - Warning page logic

// Parse query parameters
function getQueryParams() {
    const params = new URLSearchParams(window.location.search);
    return {
      blocked: params.get('blocked'),
      redirect: params.get('redirect')
    };
  }
  
  // Initialize page
  const params = getQueryParams();
  const blockedHostname = params.blocked || 'unknown';
  const redirectUrl = params.redirect || '';
  
  // Display blocked hostname
  document.getElementById('blockedHostname').textContent = blockedHostname;
  
  // Handle "Proceed anyway" click
  document.getElementById('proceedBtn').addEventListener('click', function(e) {
    e.preventDefault();
    
    // Show confirmation dialog
    const confirmed = confirm(
      '⚠️ WARNING: Proceeding may be unsafe!\n\n' +
      'This site has been identified as a potential phishing site.\n\n' +
      'Continuing may expose you to:\n' +
      '• Identity theft\n' +
      '• Stolen credentials\n' +
      '• Malware infection\n\n' +
      'Do you want to continue anyway?'
    );
    
    if (confirmed && redirectUrl) {
      // Send message to background to set bypass and redirect
      chrome.runtime.sendMessage(
        {
          action: 'proceedToSite',
          url: redirectUrl,
          hostname: blockedHostname
        },
        (response) => {
          if (chrome.runtime.lastError) {
            console.error('Error:', chrome.runtime.lastError);
            alert('Failed to proceed. Please try again.');
          } else if (response && response.success) {
            console.log('Proceeding to site with bypass...');
          }
        }
      );
    }
  });