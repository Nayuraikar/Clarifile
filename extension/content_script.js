(function injectButton() {
  function ensureButton() {
    if (document.getElementById('clarifile-organize-btn')) return;

    

    // On click → fetch Drive files via background, then send to backend
    btn.onclick = async () => {
      chrome.runtime.sendMessage({ type: 'LIST_FILES' }, (resp) => {
        if (!resp || !resp.ok) {
          alert(' Failed to fetch Drive files');
          return;
        }
        const { files, token } = resp;
        chrome.runtime.sendMessage({ type: 'ORGANIZE_FILES', token, files }, (r) => {
          if (!r || !r.ok) {
            alert(' Failed to organize via gateway');
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

//  Listen for messages (from popup/background)
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  console.log(" Message received in content script:", msg);
  sendResponse({ received: true });
});