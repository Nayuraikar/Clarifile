const out = document.getElementById('out');
document.getElementById('organize').addEventListener('click', () => {
  chrome.runtime.sendMessage({ type: 'LIST_FILES' }, async (resp) => {
    if (!resp || !resp.ok) {
      out.textContent = 'Auth or list files failed.';
      return;
    }
    try {
      const res = await fetch('http://127.0.0.1:4000/drive/organize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files: resp.files, move: true, auth_token: resp.token })
      });
      const data = await res.json();
      out.textContent = JSON.stringify(data, null, 2);
      // Fallback push to gateway cache
      try {
        await fetch('http://127.0.0.1:4000/drive/proposals_ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
      } catch(e) {}
    } catch (e) {
      out.textContent = 'Gateway not reachable';
    }
  });
});


