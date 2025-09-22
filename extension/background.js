chrome.runtime.onInstalled.addListener(() => {
  console.log('Clarifile Organizer installed');
});

async function getAuthTokenInteractive() {
  return new Promise((resolve, reject) => {
    chrome.identity.getAuthToken({ interactive: true }, (token) => {
      if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
      resolve(token);
    });
  });
}

async function listSelectedDriveFiles() {
  const token = await getAuthTokenInteractive();
  // Pull from My Drive and Shared drives, exclude trashed, expand pageSize
  const q = encodeURIComponent("trashed = false");
  const fields = encodeURIComponent("nextPageToken, files(id,name,mimeType,parents,size)");
  const url = `https://www.googleapis.com/drive/v3/files?pageSize=1000&q=${q}&fields=${fields}&spaces=drive`;
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` }
  });
  const data = await res.json();
  let files = data.files || [];
  // Drain pagination if needed
  let pageToken = data.nextPageToken;
  while (pageToken) {
    const res2 = await fetch(`${url}&pageToken=${pageToken}`, { headers: { Authorization: `Bearer ${token}` } });
    const data2 = await res2.json();
    files = files.concat(data2.files || []);
    pageToken = data2.nextPageToken;
  }
  return { token, files };
}

async function organizeViaGateway(token, files) {
  const resp = await fetch('http://127.0.0.1:4000/drive/organize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ files, move: true, auth_token: token })
  });
  return resp.json();
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    if (message.type === 'LIST_FILES') {
      try {
        const { token, files } = await listSelectedDriveFiles();
        sendResponse({ ok: true, token, files });
      } catch (e) {
        sendResponse({ ok: false, error: e.message || String(e) });
      }
    }
    if (message.type === 'ORGANIZE_FILES') {
      try {
        const data = await organizeViaGateway(message.token, message.files);
        // Push to gateway cache as fallback
        try {
          await fetch('http://127.0.0.1:4000/drive/proposals_ingest', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data)
          });
        } catch(_) {}
        sendResponse({ ok: true, data });
      } catch (e) {
        sendResponse({ ok: false, error: e.message || String(e) });
      }
    }
  })();
  return true;
});

