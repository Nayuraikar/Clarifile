const CLIENT_ID = '901002840369-9m5hr6ds4aqs5o2f2laii829e9erhokp.apps.googleusercontent.com';
const SCOPES = [
  'https://www.googleapis.com/auth/drive.file',
  'https://www.googleapis.com/auth/drive.metadata.readonly'
].join(' ');

async function launchAuthAndGetCode() {
  const redirectUri = chrome.identity.getRedirectURL();
  const authUrl = 'https://accounts.google.com/o/oauth2/v2/auth' +
    '?response_type=code' +
    `&client_id=${encodeURIComponent(CLIENT_ID)}` +
    `&scope=${encodeURIComponent(SCOPES)}` +
    `&redirect_uri=${encodeURIComponent(redirectUri)}` +
    '&access_type=offline&prompt=consent';

  return new Promise((resolve, reject) => {
    chrome.identity.launchWebAuthFlow({ url: authUrl, interactive: true }, (redirectedTo) => {
      if (chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
      try {
        const urlObj = new URL(redirectedTo);
        const code = urlObj.searchParams.get('code');
        if (!code) return reject(new Error('No code in redirect URI'));
        resolve(code);
      } catch (e) {
        reject(e);
      }
    });
  });
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === 'oauth') {
    launchAuthAndGetCode()
      .then(code => {
        // send code to your gateway
        fetch('http://127.0.0.1:4000/drive/exchange_code', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code })
        })
        .then(resp => resp.json())
        .then(data => sendResponse({ ok: true, data }))
        .catch(err => sendResponse({ ok: false, error: err.message }));
      })
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true; // keeps the message channel open for async response
  }
});
