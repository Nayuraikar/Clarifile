// gateway/index.js
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const path = require('path');
const app = express();

app.use(express.json());
app.use(cors());

// ====== SERVICE ENDPOINTS ======
const PARSER = 'http://127.0.0.1:8000';
const EMBED = 'http://127.0.0.1:8002';
const INDEXER = 'http://127.0.0.1:8003';
const DEDUP = 'http://127.0.0.1:8004';

let DRIVE_PROPOSALS = [];
let DRIVE_TOKEN = null;

// ====== GOOGLE OAUTH CONFIG ======
const CLIENT_ID = '1085724686418-fgvn7on50g4dmq2la1q84dflt66at0mo.apps.googleusercontent.com'; 
const CLIENT_SECRET = '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';                         
const REDIRECT_URI = 'https://aanmbhbgdehmchcjbiblopfkdcpejpkh.chromiumapp.org/';      

// ====== UTILITY ENDPOINTS ======
app.post('/scan', async (req, res) => {
  try { const r = await axios.post(`${PARSER}/scan_folder`); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.post('/embed', async (req, res) => {
  try { const r = await axios.post(`${EMBED}/embed_pending`); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.post('/reindex', async (req, res) => {
  try { const r = await axios.post(`${INDEXER}/reindex`); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.get('/knn', async (req, res) => {
  try { const params = { params: req.query }; const r = await axios.get(`${INDEXER}/knn`, params); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.get('/categories', async (req, res) => {
  try { const r = await axios.get(`${PARSER}/categories`); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.get('/proposals', async (req, res) => {
  try { const r = await axios.get(`${PARSER}/list_proposals`); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.get('/list_proposals', async (req, res) => {
  try { const r = await axios.get(`${PARSER}/list_proposals`); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.post('/approve', async (req, res) => {
  try { const r = await axios.post(`${PARSER}/approve`, req.body); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.get('/duplicates', async (req, res) => {
  try { const params = { params: req.query }; const r = await axios.get(`${DEDUP}/duplicates`, params); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.post('/resolve_duplicate', async (req, res) => {
  try { const r = await axios.post(`${DEDUP}/resolve_duplicate`, req.body); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.post('/keep', async (req, res) => {
  try { const r = await axios.post(`${DEDUP}/keep`, req.body); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// ====== DRIVE ENDPOINTS ======

// Exchange auth code from extension for access token
app.post('/drive/exchange_code', async (req, res) => {
  const { code } = req.body;
  if (!code) return res.status(400).json({ error: 'Missing code' });

  try {
    const tokenResp = await axios.post('https://oauth2.googleapis.com/token', null, {
      params: {
        code,
        client_id: CLIENT_ID,
        client_secret: CLIENT_SECRET,
        redirect_uri: REDIRECT_URI,
        grant_type: 'authorization_code'
      },
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });

    DRIVE_TOKEN = tokenResp.data.access_token;
    res.json(tokenResp.data); // returns access_token, refresh_token, expires_in, etc.
  } catch (e) {
    console.error(e.response?.data || e.message);
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

// Organize files in Drive using parser
app.post('/drive/organize', async (req, res) => {
  try {
    console.log('[drive/organize] received', Array.isArray(req.body?.files) ? req.body.files.length : 0, 'files');
    DRIVE_TOKEN = req.body?.auth_token || DRIVE_TOKEN;
    const r = await axios.post(`${PARSER}/organize_drive_files`, req.body);
    const data = r.data || {};
    DRIVE_PROPOSALS = data.organized_files || [];
    console.log('[drive/organize] stored proposals:', DRIVE_PROPOSALS.length);
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.toString() });
  }
});

app.get('/drive/proposals', async (req, res) => {
  try { res.json(DRIVE_PROPOSALS); } catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.post('/drive/proposals_ingest', async (req, res) => {
  try {
    const files = Array.isArray(req.body?.organized_files) ? req.body.organized_files : (Array.isArray(req.body) ? req.body : []);
    DRIVE_PROPOSALS = files;
    res.json({ ok: true, stored: DRIVE_PROPOSALS.length });
  } catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.get('/drive/health', (req, res) => {
  res.json({ proposals: DRIVE_PROPOSALS.length, hasToken: !!DRIVE_TOKEN });
});

app.get('/categories', async (req, res) => {
  try {
    const r = await axios.get(`${PARSER}/categories`);
    const categories = r.data || [];
    const localCounts = {};
    const driveCounts = {};
    for (const f of await axios.get(`${PARSER}/list_proposals`)) {
      const k = f.proposed_category || 'Other';
      localCounts[k] = (localCounts[k] || 0) + 1;
    }
    for (const f of DRIVE_PROPOSALS) {
      const k = f.proposed_category || 'Other';
      driveCounts[k] = (driveCounts[k] || 0) + 1;
    }
    const out = categories.filter(name => localCounts[name] > 0 || driveCounts[name] > 0).map(name => ({ name, local_file_count: localCounts[name] || 0, drive_file_count: driveCounts[name] || 0 }));
    res.json(out);
  } catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.post('/drive/approve', async (req, res) => {
  try {
    const file = req.body?.file;
    const override_category = req.body?.category || null;
    if (!file) return res.status(400).json({ error: 'missing file' });
    if (!DRIVE_TOKEN) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });

    const payload = { files: [file], move: true, auth_token: DRIVE_TOKEN, override_category };
    const r = await axios.post(`${PARSER}/organize_drive_files`, payload);
    const moved = r.data?.organized_files?.[0] || {};
    const reallyMoved = !!r.data?.move_performed;
    if (reallyMoved) {
      DRIVE_PROPOSALS = DRIVE_PROPOSALS.filter(p => p.id !== file.id);
      res.json({ ok: true, moved });
    } else {
      res.status(500).json({ error: 'move failed', moved });
    }
  } catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Analyze files in Drive
app.post('/drive/analyze', async (req, res) => {
  try {
    if (!DRIVE_TOKEN && !req.body?.auth_token) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    const body = { ...req.body, auth_token: req.body?.auth_token || DRIVE_TOKEN };
    const r = await axios.post(`${PARSER}/drive_analyze`, body);
    res.json(r.data);
  } catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Summarize files in Drive
app.post('/drive/summarize', async (req, res) => {
  try {
    if (!DRIVE_TOKEN && !req.body?.auth_token) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    const body = { ...req.body, auth_token: req.body?.auth_token || DRIVE_TOKEN };
    const r = await axios.post(`${PARSER}/drive_summarize`, body);
    res.json(r.data);
  } catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Find similar files in Drive
app.post('/drive/find_similar', async (req, res) => {
  try {
    if (!DRIVE_TOKEN && !req.body?.auth_token) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    const body = { ...req.body, auth_token: req.body?.auth_token || DRIVE_TOKEN };
    const r = await axios.post(`${PARSER}/drive_find_similar`, body);
    res.json(r.data);
  } catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Extract insights from files in Drive
app.post('/drive/extract_insights', async (req, res) => {
  try {
    if (!DRIVE_TOKEN && !req.body?.auth_token) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    const body = { ...req.body, auth_token: req.body?.auth_token || DRIVE_TOKEN };
    const r = await axios.post(`${PARSER}/drive_extract_insights`, body);
    res.json(r.data);
  } catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Get summary of a file
app.get('/file_summary', async (req, res) => {
  try { const r = await axios.get(`${PARSER}/file_summary`, { params: req.query }); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Ask about a local (scanned) file
app.get('/ask', async (req, res) => {
  try { const r = await axios.get(`${PARSER}/ask`, { params: req.query }); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// ====== SERVE UI ======
// Prefer built React app if available (ui/dist), else fall back to legacy ui/
const builtUi = path.join(__dirname, '../ui/dist');
const legacyUi = path.join(__dirname, '../ui');
try {
  app.use(express.static(builtUi));
  app.get('*', (req, res, next) => {
    res.sendFile(path.join(builtUi, 'index.html'), (err) => {
      if (err) next();
    });
  });
} catch (_) {
  app.use(express.static(legacyUi));
}

// ====== START SERVER ======
app.listen(4000, () => console.log('Gateway running on http://127.0.0.1:4000'));
