(function injectButton() {
  function ensureButton() {
    if (document.getElementById('clarifile-organize-btn')) return;

    // Create floating button
    const btn = document.createElement('button');
    btn.id = 'clarifile-organize-btn';
    btn.textContent = 'Organize with Clarifile';
    btn.style.position = 'fixed';
    btn.style.right = '16px';
    btn.style.bottom = '16px';
    btn.style.padding = '10px 14px';
    btn.style.zIndex = 99999;
    btn.style.background = '#1a73e8';
    btn.style.color = '#fff';
    btn.style.border = 'none';
    btn.style.borderRadius = '6px';
    btn.style.cursor = 'pointer';

    // On click → fetch Drive files via background, then send to backend
    btn.onclick = async () => {
      chrome.runtime.sendMessage({ type: 'LIST_FILES' }, (resp) => {
        if (!resp || !resp.ok) {
          alert('❌ Failed to fetch Drive files');
          return;
        }
        const { files, token } = resp;
        chrome.runtime.sendMessage({ type: 'ORGANIZE_FILES', token, files }, (r) => {
          if (!r || !r.ok) {
            alert('⚠️ Failed to organize via gateway');
            return;
          }
          const data = r.data || {};
          alert('Clarifile proposed categories for ' + (data.organized_files || []).length + ' files. Open http://127.0.0.1:4000 to view.');
        });
      });
    };

    document.body.appendChild(btn);
  }

  // Inject button once + re-inject if DOM changes
  ensureButton();
  const obs = new MutationObserver(ensureButton);
  obs.observe(document.documentElement, { childList: true, subtree: true });
})();

// ✅ Listen for messages (from popup/background)
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  console.log("📩 Message received in content script:", msg);
  sendResponse({ received: true });
});
