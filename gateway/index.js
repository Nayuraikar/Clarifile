// gateway/index.js
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const app = express();
app.use(express.json());
app.use(cors());

const PARSER = 'http://127.0.0.1:8000';
const EMBED = 'http://127.0.0.1:8002';
const INDEXER = 'http://127.0.0.1:8003';
const DEDUP = 'http://127.0.0.1:8004';
let DRIVE_PROPOSALS = [];
let DRIVE_TOKEN = null;

app.post('/scan', async (req, res) => {
  try {
    const r = await axios.post(`${PARSER}/scan_folder`);
    res.json(r.data);
  } catch (e) { res.status(500).json({error: e.toString()}); }
});

app.post('/embed', async (req, res) => {
  try {
    const r = await axios.post(`${EMBED}/embed_pending`);
    res.json(r.data);
  } catch (e) { res.status(500).json({error: e.toString()}); }
});

app.post('/reindex', async (req, res) => {
  try {
    const r = await axios.post(`${INDEXER}/reindex`);
    res.json(r.data);
  } catch (e) { res.status(500).json({error: e.toString()}); }
});

app.get('/knn', async (req, res) => {
  try {
    const params = { params: req.query };
    const r = await axios.get(`${INDEXER}/knn`, params);
    res.json(r.data);
  } catch (e) { res.status(500).json({error: e.toString()}); }
});

// Drive integration: forward organize requests to parser and store proposals for UI
app.post('/drive/organize', async (req, res) => {
  try {
    console.log('[drive/organize] received', Array.isArray(req.body?.files) ? req.body.files.length : 0, 'files');
    DRIVE_TOKEN = req.body?.auth_token || null;
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
  try {
    res.json(DRIVE_PROPOSALS);
  } catch (e) {
    res.status(500).json({ error: e.toString() });
  }
});

// Ingest proposals directly (e.g., from parser) as a fallback channel
app.post('/drive/proposals_ingest', async (req, res) => {
  try {
    const files = Array.isArray(req.body?.organized_files) ? req.body.organized_files : (Array.isArray(req.body) ? req.body : []);
    DRIVE_PROPOSALS = files;
    res.json({ ok: true, stored: DRIVE_PROPOSALS.length });
  } catch (e) {
    res.status(500).json({ error: e.toString() });
  }
});

app.get('/drive/health', (req, res) => {
  res.json({ proposals: DRIVE_PROPOSALS.length, hasToken: !!DRIVE_TOKEN });
});

// Drive categories from current proposals (not DB-based)
app.get('/drive/categories', (req, res) => {
  const counts = {};
  for (const f of DRIVE_PROPOSALS) {
    const k = (f.proposed_category || 'Other');
    counts[k] = (counts[k] || 0) + 1;
  }
  const out = Object.keys(counts).map(name => ({ name, file_count: counts[name] }))
    .sort((a,b) => b.file_count - a.file_count);
  res.json(out);
});

// Approve a Drive proposal: move file in Drive using stored token
app.post('/drive/approve', async (req, res) => {
  try {
    const file = req.body?.file;
    const override_category = req.body?.category || null;
    if (!file) return res.status(400).json({ error: 'missing file' });
    if (!DRIVE_TOKEN) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    const payload = { files: [file], move: true, auth_token: DRIVE_TOKEN, override_category };
    const r = await axios.post(`${PARSER}/organize_drive_files`, payload);
    const moved = r.data?.organized_files?.[0] || {};
    // Remove from in-memory proposals if present
    DRIVE_PROPOSALS = DRIVE_PROPOSALS.filter(p => p.id !== file.id);
    res.json({ ok: true, moved });
  } catch (e) {
    res.status(500).json({ error: e.toString() });
  }
});

app.get('/proposals', async (req, res) => {
  try {
    const r = await axios.get(`${PARSER}/list_proposals`);
    res.json(r.data);
  } catch (e) { res.status(500).json({error: e.toString()}); }
});

app.get('/list_proposals', async (req, res) => {
  try {
    const r = await axios.get(`${PARSER}/list_proposals`);
    res.json(r.data);
  } catch (e) { res.status(500).json({error: e.toString()}); }
});

app.post('/approve', async (req, res) => {
  try {
    const r = await axios.post(`${PARSER}/approve`, req.body);
    res.json(r.data);
  } catch (e) { res.status(500).json({error: e.toString()}); }
});

// Dedup endpoints
app.get('/duplicates', async (req, res) => {
  try {
    const params = { params: req.query };
    const r = await axios.get(`${DEDUP}/duplicates`, params);
    res.json(r.data);
  } catch (e) { res.status(500).json({error: e.toString()}); }
});

app.post('/resolve_duplicate', async (req, res) => {
  try {
    const r = await axios.post(`${DEDUP}/resolve_duplicate`, req.body);
    res.json(r.data);
  } catch (e) { res.status(500).json({error: e.toString()}); }
});
app.get('/categories', async (req, res) => {
  try { const r = await axios.get(`${PARSER}/categories`); res.json(r.data); }
  catch (e) { res.status(500).json({error: e.toString()}); }
});

// Proxy Drive analyze (summary, QA) to parser and pass stored token
app.post('/drive/analyze', async (req, res) => {
  try {
    if (!DRIVE_TOKEN && !req.body?.auth_token) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    const body = { ...req.body, auth_token: req.body?.auth_token || DRIVE_TOKEN };
    const r = await axios.post(`${PARSER}/drive_analyze`, body);
    res.json(r.data);
  } catch (e) {
    res.status(500).json({ error: e.toString() });
  }
});

app.get('/file_summary', async (req, res) => {
  try { const r = await axios.get(`${PARSER}/file_summary`, { params: req.query }); res.json(r.data); }
  catch (e) { res.status(500).json({error: e.toString()}); }
});

const path = require('path');
app.use(express.static(path.join(__dirname, '../ui')));

app.listen(4000, () => console.log('Gateway running on http://127.0.0.1:4000'));
