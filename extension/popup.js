const out = document.getElementById('out');
function showStatus(message, type = 'info') {
  const container = document.getElementById('status-container');
  container.innerHTML = '';

  const statusDiv = document.createElement('div');
  statusDiv.className = `status ${type}`;

  let icon = '';
  let loadingSpinner = '';

  if (type === 'success') {
    icon = '✅';
  } else if (type === 'error') {
    icon = '❌';
  } else if (type === 'info') {
    icon = 'ℹ️';
    loadingSpinner = '<span class="loading-dots"><span class="loading-dot"></span><span class="loading-dot"></span><span class="loading-dot"></span></span>';
  }

  statusDiv.innerHTML = `
    <span>${icon}</span>
    <span>${message}</span>
    ${loadingSpinner}
  `;

  container.appendChild(statusDiv);

  // Auto-hide after 5 seconds for success/error messages
  if (type !== 'info') {
    setTimeout(() => {
      container.innerHTML = '';
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

      showStatus('Getting authorization...', 'info');
      token = await getTokenViaWebAuthFlow([
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.metadata.readonly'
      ]);
    }

    showStatus('Fetching Drive files...', 'info');
    const files = await listDriveFiles(token);

    if (files.length === 0) {
      showStatus('No files found in your Drive', 'info');
      out.textContent = 'No files found in your Google Drive to organize.';
      return;
    }

    showStatus(`Starting AI analysis of ${files.length} files...`, 'info');
    out.textContent = `🔍 Analyzing ${files.length} files with AI...\n\n`;

    // Show initial progress with loading animation
    let progressText = `📊 Progress: 0/${files.length} files analyzed\n⏳ Processing...`;
    out.textContent += progressText;

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

    // Update progress as files are processed
    const organized = data.organized_files || [];
    let completedFiles = 0;

    // Simulate progress updates during analysis with enhanced visual feedback
    const progressInterval = setInterval(() => {
      if (completedFiles < organized.length) {
        completedFiles++;
        const progress = Math.round((completedFiles / files.length) * 100);
        const currentFile = organized[completedFiles - 1];
        const fileName = currentFile ? currentFile.name : 'Unknown file';

        progressText = `📊 Progress: ${completedFiles}/${files.length} files analyzed (${progress}%)\n`;
        progressText += `✅ Last processed: ${fileName}\n`;
        progressText += `📁 Category: ${currentFile ? currentFile.proposed_category : 'Unknown'}\n`;
        progressText += `⏳ ${completedFiles < files.length ? 'Processing next file...' : 'Finalizing...'}`;

        out.textContent = `🔍 Analyzing ${files.length} files with AI...\n\n${progressText}`;
      }
    }, 800); // Update every 800ms

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

    // Clear the progress interval and show final result
    clearInterval(progressInterval);

    showStatus('Organization complete! Check your Clarifile dashboard.', 'success');

    // Format the final output nicely with enhanced formatting
    let summary = `✅ Successfully analyzed ${files.length} files!\n\n📋 Organization Summary:\n`;

    if (organized.length > 0) {
        summary += organized.map(item => {
            const fileSummary = item.summary ? `\n   📝 Summary: ${item.summary.substring(0, 100)}${item.summary.length > 100 ? '...' : ''}` : '';
            return `• ${item.name} → ${item.proposed_category}${fileSummary}`;
        }).join('\n');
    } else {
        summary += 'No files were organized.';
    }

    summary += `\n\n🎉 Open your Clarifile dashboard to review and approve the suggestions.`;

    out.textContent = summary;

  } catch (e) {
    const errorMsg = e && e.message ? e.message : String(e);
    showStatus(' Organization failed: ' + errorMsg, 'error');

    if (errorMsg.includes('127.0.0.1:4000')) {
      out.textContent = `❌ Cannot connect to Clarifile server.\n\nPlease make sure:\n1. Clarifile is running on your computer\n2. The server is accessible at http://127.0.0.1:4000\n3. Check your firewall settings`;
    } else {
      out.textContent = `❌ Error: ${errorMsg}\n\nPlease try again or contact support if the issue persists.`;
    }

  } finally {
    setButtonState('organize', false, ' Organize Drive Files');
  }
});


