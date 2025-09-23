const out = document.getElementById('out');

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
  out.textContent = 'Authorizing...';
  try {
    // Try native token first; if it fails with bad client id, fallback to web auth flow
    try {
      const token = await getAuthTokenInteractive();
      out.textContent = 'Authorized. Token acquired.';
      return;
    } catch (e1) {
      const msg = (e1 && e1.message) ? e1.message : String(e1);
      if (!/bad client id/i.test(msg)) throw e1;
      // Fallback to WebAuthFlow using the same client_id and extension redirect URI
      const scopes = [
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.metadata.readonly'
      ];
      const token = await getTokenViaWebAuthFlow(scopes);
      out.textContent = 'Authorized via WebAuthFlow. Token acquired.';
    }
  } catch (e) {
    out.textContent = 'Authorization failed: ' + (e && e.message ? e.message : String(e));
  }
});

document.getElementById('organize').addEventListener('click', async () => {
  out.textContent = 'Listing Drive files...';
  try {
    let token;
    try {
      token = await getAuthTokenInteractive();
    } catch (e1) {
      const msg = (e1 && e1.message) ? e1.message : String(e1);
      if (!/bad client id/i.test(msg)) throw e1;
      token = await getTokenViaWebAuthFlow([
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.metadata.readonly'
      ]);
    }
    const files = await listDriveFiles(token);
    const res = await fetch('http://127.0.0.1:4000/drive/organize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files, move: false, auth_token: token })
    });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
    try {
      await fetch('http://127.0.0.1:4000/drive/proposals_ingest', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data)
      });
    } catch (_) {}
  } catch (e) {
    out.textContent = (e && e.message ? e.message : String(e));
  }
});


