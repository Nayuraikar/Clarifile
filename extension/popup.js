const out = document.getElementById('out');
const statusContainer = document.getElementById('status-container');

function showStatus(message, type = 'info') {
  statusContainer.innerHTML = `
    <div class="status ${type}">
      <svg class="icon" viewBox="0 0 24 24">
        ${type === 'success' ? 
          '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>' :
          type === 'error' ? 
          '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm3.5 6L12 10.5 8.5 8 7 9.5 10.5 12 7 14.5 8.5 16 12 13.5 15.5 16 17 14.5 13.5 12 17 9.5 15.5 8z"/>' :
          '<path d="M13,9H11V7H13M13,17H11V11H13M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z"/>'
        }
      </svg>
      ${message}
    </div>
  `;
  
  // Auto-hide after 5 seconds for success/error messages
  if (type !== 'info') {
    setTimeout(() => {
      statusContainer.innerHTML = '';
    }, 5000);
  }
}

function setButtonState(buttonId, disabled, text) {
  const button = document.getElementById(buttonId);
  button.disabled = disabled;
  const iconSvg = button.querySelector('svg');
  const textContent = button.childNodes[button.childNodes.length - 1];
  textContent.textContent = text;
}

async function getAuthTokenInteractive() {
  return new Promise((resolve, reject) => {
    try {
      chrome.identity.getAuthToken({ interactive: true }, (token) => {
        if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
        if (!token) return reject(new Error('No token returned'));
        resolve(token);
      });
    } catch (e) {
      reject(e);
    }
  });
}

function getClientIdFromManifest() {
  const manifest = chrome.runtime.getManifest();
  const cid = manifest && manifest.oauth2 && manifest.oauth2.client_id;
  return cid || '';
}

function getRedirectUri() {
  const extId = chrome.runtime.id;
  return `https://${extId}.chromiumapp.org/`;
}

async function getTokenViaWebAuthFlow(scopes) {
  const clientId = getClientIdFromManifest();
  if (!clientId) throw new Error('OAuth client_id missing in manifest');
  const redirectUri = getRedirectUri();
  const scopeParam = encodeURIComponent(scopes.join(' '));
  const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${encodeURIComponent(clientId)}&response_type=token&redirect_uri=${encodeURIComponent(redirectUri)}&scope=${scopeParam}&prompt=consent&access_type=online&include_granted_scopes=true`;
  const redirect = await new Promise((resolve, reject) => {
    chrome.identity.launchWebAuthFlow({ url: authUrl, interactive: true }, (responseUrl) => {
      if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
      if (!responseUrl) return reject(new Error('No response URL'));
      resolve(responseUrl);
    });
  });
  // responseUrl will be like: https://<extid>.chromiumapp.org/#access_token=...&token_type=Bearer&expires_in=...
  const hash = redirect.split('#')[1] || '';
  const params = new URLSearchParams(hash);
  const token = params.get('access_token');
  if (!token) throw new Error('No access_token in auth response');
  return token;
}

async function listDriveFiles(token) {
  const q = encodeURIComponent("trashed = false and mimeType != 'application/vnd.google-apps.folder'");
  const fields = encodeURIComponent('nextPageToken, files(id,name,mimeType,parents,size)');
  const base = `https://www.googleapis.com/drive/v3/files?pageSize=1000&q=${q}&fields=${fields}&spaces=drive`;
  let files = [];
  let pageToken = '';
  do {
    const url = pageToken ? `${base}&pageToken=${pageToken}` : base;
    const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    if (!res.ok) throw new Error('Drive list failed: ' + res.status);
    const data = await res.json();
    files = files.concat(data.files || []);
    pageToken = data.nextPageToken || '';
  } while (pageToken);
  return files;
}

document.getElementById('authorize').addEventListener('click', async () => {
  setButtonState('authorize', true, ' Authorizing...');
  showStatus('Connecting to Google Drive...', 'info');
  out.textContent = '';
  
  try {
    // Try native token first; if it fails with bad client id, fallback to web auth flow
    try {
      const token = await getAuthTokenInteractive();
      showStatus('Successfully authorized with Google Drive!', 'success');
      out.textContent = 'Authorization successful. You can now organize your Drive files.';
      setButtonState('authorize', false, ' Authorize Google Drive');
      return;
    } catch (e1) {
      const msg = (e1 && e1.message) ? e1.message : String(e1);
      if (!/bad client id/i.test(msg)) throw e1;
      
      showStatus('Using alternative authorization method...', 'info');
      // Fallback to WebAuthFlow using the same client_id and extension redirect URI
      const scopes = [
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.metadata.readonly'
      ];
      const token = await getTokenViaWebAuthFlow(scopes);
      showStatus('Successfully authorized with Google Drive!', 'success');
      out.textContent = 'Authorization successful via WebAuthFlow. You can now organize your Drive files.';
    }
  } catch (e) {
    const errorMsg = e && e.message ? e.message : String(e);
    showStatus('Authorization failed: ' + errorMsg, 'error');
    out.textContent = 'Authorization failed. Please try again or check your internet connection.';
  } finally {
    setButtonState('authorize', false, ' Authorize Google Drive');
  }
});

document.getElementById('organize').addEventListener('click', async () => {
  setButtonState('organize', true, ' Organizing...');
  showStatus('Scanning your Drive files...', 'info');
  out.textContent = '';
  
  try {
    let token;
    try {
      token = await getAuthTokenInteractive();
    } catch (e1) {
      const msg = (e1 && e1.message) ? e1.message : String(e1);
      if (!/bad client id/i.test(msg)) throw e1;
      
      showStatus('🔐 Getting authorization...', 'info');
      token = await getTokenViaWebAuthFlow([
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.metadata.readonly'
      ]);
    }
    
    showStatus('📂 Fetching Drive files...', 'info');
    const files = await listDriveFiles(token);
    
    if (files.length === 0) {
      showStatus('No files found in your Drive', 'info');
      out.textContent = 'No files found in your Google Drive to organize.';
      return;
    }
    
    showStatus(`🤖 Analyzing ${files.length} files with AI...`, 'info');
    const res = await fetch('http://127.0.0.1:4000/drive/organize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files, move: false, auth_token: token })
    });
    
    if (!res.ok) {
      throw new Error(`Server error: ${res.status} ${res.statusText}`);
    }
    
    const data = await res.json();
    
    if (data.error) {
      throw new Error(data.error);
    }
    
    showStatus('Saving organization proposals...', 'info');
    try {
      await fetch('http://127.0.0.1:4000/drive/proposals_ingest', {
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify(data)
      });
    } catch (ingestError) {
      console.warn('Failed to ingest proposals:', ingestError);
    }
    
    showStatus('Organization complete! Check your Clarifile dashboard.', 'success');
    
    // Format the output nicely
    const organized = data.organized || [];
    const summary = ` Successfully analyzed ${files.length} files!\n\n Organization Summary:\n${organized.map(item => `• ${item.name} → ${item.category}`).join('\n')}\n\nOpen your Clarifile dashboard to review and approve the suggestions.`;
    
    out.textContent = summary;
    
  } catch (e) {
    const errorMsg = e && e.message ? e.message : String(e);
    showStatus(' Organization failed: ' + errorMsg, 'error');
    
    if (errorMsg.includes('127.0.0.1:4000')) {
      out.textContent = ` Cannot connect to Clarifile server.\n\nPlease make sure:\n1. Clarifile is running on your computer\n2. The server is accessible at http://127.0.0.1:4000\n3. Check your firewall settings`;
    } else {
      out.textContent = `Error: ${errorMsg}\n\nPlease try again or contact support if the issue persists.`;
    }

  } finally {
    setButtonState('organize', false, ' Organize Drive Files');
  }
});


