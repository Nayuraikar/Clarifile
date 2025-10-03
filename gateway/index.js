// gateway/index.js
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const fileUpload = require('express-fileupload');
const app = express();
const path = require('path');

app.use(express.json());
app.use(cors());
app.use(fileUpload());

// ====== SERVICE ENDPOINTS ======
const PARSER = 'http://127.0.0.1:8000';
const EMBED = 'http://127.0.0.1:8002';
const INDEXER = 'http://127.0.0.1:8003';
const DEDUP = 'http://127.0.0.1:8004';
// Duplicates are handled by the dedicated dedup service

let DRIVE_PROPOSALS = [];
let DRIVE_TOKEN = null;

// ====== GOOGLE OAUTH CONFIG ======
const CLIENT_ID = '36164233493xxxxxxx'; 
const CLIENT_SECRET = 'GOCSPX-xxxxxxxx';                         
const REDIRECT_URI = 'https://jxxxxxxxxg/';

// ====== UTILITY ENDPOINTS ======
app.post('/scan', async (req, res) => {
  try { const r = await axios.post(`${PARSER}/scan_folder`); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Create (or find) a Drive folder by name. Optionally accepts parentId.
app.post('/drive/create_folder', async (req, res) => {
  try {
    if (!DRIVE_TOKEN) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    const name = (req.body?.name || '').trim();
    const parentId = (req.body?.parentId || '').trim();
    if (!name) return res.status(400).json({ error: 'missing folder name' });

    // First, try to find an existing folder with this name (optionally under parent)
    const qParts = [
      "mimeType='application/vnd.google-apps.folder'",
      'trashed=false',
      `name='${name.replace(/'/g, "\\'")}'`
    ];
    if (parentId) qParts.push(`'${parentId}' in parents`);
    const q = qParts.join(' and ');
    const findResp = await axios.get('https://www.googleapis.com/drive/v3/files', {
      params: { q, fields: 'files(id,name,parents)' },
      headers: { Authorization: `Bearer ${DRIVE_TOKEN}` }
    });
    const existing = (findResp.data?.files || [])[0];
    if (existing) return res.json({ ok: true, id: existing.id, name: existing.name, existed: true });

    // Create folder
    const metadata = { name, mimeType: 'application/vnd.google-apps.folder' };
    if (parentId) metadata.parents = [parentId];
    const createResp = await axios.post('https://www.googleapis.com/drive/v3/files', metadata, {
      headers: { Authorization: `Bearer ${DRIVE_TOKEN}` }
    });
    const f = createResp.data || {};
    res.json({ ok: true, id: f.id, name: f.name, existed: false });
  } catch (e) {
    const msg = e.response?.data || e.message || String(e);
    res.status(500).json({ error: msg });
  }
});

// List contents of a Drive folder (non-trashed files). Accepts folderId, optional limit
app.get('/drive/folder_contents', async (req, res) => {
  try {
    if (!DRIVE_TOKEN) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    const { folderId, limit } = req.query || {};
    if (!folderId) return res.status(400).json({ error: 'missing folderId' });
    const MAX_FILES = Math.max(1, Math.min(Number(limit) || 500, 2000));

    const files = [];
    let pageToken = undefined;
    let fetched = 0;
    const baseParams = {
      fields: 'nextPageToken, files(id,name,size,mimeType,md5Checksum,trashed)'
    };
    const q = `'${folderId}' in parents and trashed=false`;
    while (fetched < MAX_FILES) {
      const params = { ...baseParams, q, pageToken, pageSize: 200 };
      const r = await axios.get('https://www.googleapis.com/drive/v3/files', {
        params,
        headers: { Authorization: `Bearer ${DRIVE_TOKEN}` }
      });
      const batch = r.data?.files || [];
      for (const f of batch) {
        files.push({ id: f.id, name: f.name, size: Number(f.size || 0), mimeType: f.mimeType });
        fetched += 1;
        if (fetched >= MAX_FILES) break;
      }
      pageToken = r.data?.nextPageToken;
      if (!pageToken || fetched >= MAX_FILES) break;
    }

    res.json({ folderId, files });
  } catch (e) {
    res.status(500).json({ error: e.toString() });
  }
});

app.post('/embed', async (req, res) => {
  try { const r = await axios.post(`${EMBED}/embed_pending`); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Assistant generate passthrough to parser
app.post('/api/assistant_generate', async (req, res) => {
  try {
    const r = await axios.post(`${PARSER}/assistant_generate`, req.body, { timeout: 120000 });
    res.json(r.data);
  } catch (e) {
    res.status(500).json({ error: e.toString() });
  }
});

// ====== ENHANCED CATEGORIZATION ENDPOINTS ======

// Categorize content using enhanced system
app.post('/categorize_content', async (req, res) => {
  try { const r = await axios.post(`${EMBED}/categorize_content`, req.body); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Process a single file with enhanced categorization
app.post('/process_file', async (req, res) => {
  try { const r = await axios.post(`${EMBED}/process_file`, req.body); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Batch categorize multiple files
app.post('/batch_categorize', async (req, res) => {
  try { const r = await axios.post(`${EMBED}/batch_categorize`, req.body); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Load a saved categorization model
app.post('/load_categorization_model', async (req, res) => {
  try { const r = await axios.post(`${EMBED}/load_categorization_model`, req.body); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Get model information
app.get('/model_info', async (req, res) => {
  try { const r = await axios.get(`${EMBED}/model_info`); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// List available categorization models
app.get('/list_models', async (req, res) => {
  try { const r = await axios.get(`${EMBED}/list_models`); res.json(r.data); }
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

// Enhanced categories with Drive folder information
app.get('/enhanced_categories', async (req, res) => {
  try {
    const auth_token = DRIVE_TOKEN || req.query.auth_token;
    const url = auth_token ? `${PARSER}/enhanced_categories?auth_token=${auth_token}` : `${PARSER}/enhanced_categories`;
    const r = await axios.get(url);
    res.json(r.data);
  } catch (e) { 
    res.status(500).json({ error: e.toString() }); 
  }
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
  try { const r = await axios.get(`${DEDUP}/duplicates`); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

// Refresh duplicates: run a scan first (to sync DB with drive changes), then fetch duplicates
app.post('/duplicates/refresh', async (req, res) => {
  try {
    // Optionally rescan local files so DB is up to date
    try { await axios.post(`${PARSER}/scan_folder`); } catch (_) { /* non-fatal */ }
    const r = await axios.get(`${DEDUP}/duplicates`);
    res.json(r.data);
  } catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.post('/resolve_duplicate', async (req, res) => {
  try { const r = await axios.post(`${DEDUP}/resolve_duplicate`, req.body); res.json(r.data); }
  catch (e) { res.status(500).json({ error: e.toString() }); }
});

app.post('/keep', async (req, res) => {
  try { const r = await axios.post(`${PARSER}/keep`, req.body); res.json(r.data); }
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

// Keep a Drive file: no Drive mutation, only remove from current proposals cache
app.post('/drive/keep', async (req, res) => {
  try {
    const id = req.body?.id;
    if (!id) return res.status(400).json({ error: 'missing id' });
    // Remove from proposals cache so it won't show in duplicates derived from proposals
    const before = DRIVE_PROPOSALS.length;
    DRIVE_PROPOSALS = (DRIVE_PROPOSALS || []).filter(f => f.id !== id);
    const after = DRIVE_PROPOSALS.length;
    res.json({ ok: true, removedFromCache: before - after, id });
  } catch (e) {
    res.status(500).json({ error: e.toString() });
  }
});

// Organize files in Drive using parser
app.post('/drive/organize', async (req, res) => {
  try {
    console.log('[drive/organize] received', Array.isArray(req.body?.files) ? req.body.files.length : 0, 'files');
    DRIVE_TOKEN = req.body?.auth_token || DRIVE_TOKEN;
    
    // Clear old proposals from database to start fresh
    try {
      console.log('[drive/organize] Clearing old proposals from database...');
      await axios.delete(`${PARSER}/proposals`);
      console.log('[drive/organize] Old proposals cleared successfully');
    } catch (clearError) {
      console.log('[drive/organize] Failed to clear old proposals:', clearError.message);
    }
    
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
    // Get updated categories from database
    let dbUpdates = {};
    try {
      const r = await axios.get(`${PARSER}/proposals`);
      const dbProposals = r.data || [];
      
      // Create a map of file_id -> updated_category from database
      for (const proposal of dbProposals) {
        dbUpdates[proposal.id] = {
          proposed_category: proposal.proposed_category,
          approved: proposal.approved
        };
      }
    } catch (e) {
      console.log('Failed to fetch database updates:', e.message);
    }
    
    // Apply database updates to the original DRIVE_PROPOSALS (from organize)
    const updatedProposals = (DRIVE_PROPOSALS || []).map(file => {
      if (dbUpdates[file.id]) {
        // File has been analyzed - use updated category from database
        return {
          ...file,
          proposed_category: dbUpdates[file.id].proposed_category,
          approved: dbUpdates[file.id].approved
        };
      }
      // File hasn't been analyzed - keep original category and ensure not approved
      return {
        ...file,
        approved: false  // Ensure non-analyzed files are not marked as approved
      };
    });
    
    res.json(updatedProposals);
  } catch (e) { 
    console.log('Error in drive/proposals:', e.message);
    res.json(DRIVE_PROPOSALS || []); 
  }
});

// Drive categories (mirror categories with Drive-specific counts)
app.get('/drive/categories', async (req, res) => {
  try {
    if (!DRIVE_TOKEN) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });

    // 1) Get updated proposals (same logic as /drive/proposals endpoint)
    let dbUpdates = {};
    try {
      const r = await axios.get(`${PARSER}/proposals`);
      const dbProposals = r.data || [];
      
      // Create a map of file_id -> updated_category from database
      for (const proposal of dbProposals) {
        dbUpdates[proposal.id] = {
          proposed_category: proposal.proposed_category,
          approved: proposal.approved
        };
      }
    } catch (e) {
      console.log('Failed to fetch database updates for categories:', e.message);
    }
    
    // Apply database updates to the original DRIVE_PROPOSALS
    const updatedProposals = (DRIVE_PROPOSALS || []).map(file => {
      if (dbUpdates[file.id]) {
        return {
          ...file,
          proposed_category: dbUpdates[file.id].proposed_category,
          approved: dbUpdates[file.id].approved
        };
      }
      return {
        ...file,
        approved: false  // Ensure non-analyzed files are not marked as approved
      };
    });

    const proposalCounts = {};
    for (const f of updatedProposals) {
      const k = f.proposed_category || 'Other';
      proposalCounts[k] = (proposalCounts[k] || 0) + 1;
    }

    // 2) List folders from Drive
    const folders = [];
    let pageToken = undefined;
    do {
      const r = await axios.get('https://www.googleapis.com/drive/v3/files', {
        params: {
          q: "mimeType='application/vnd.google-apps.folder' and trashed=false",
          fields: 'nextPageToken, files(id,name)'
        },
        headers: { Authorization: `Bearer ${DRIVE_TOKEN}` }
      });
      folders.push(...(r.data?.files || []));
      pageToken = r.data?.nextPageToken;
    } while (pageToken);

    // 3) Map folders to category counts by name
    const out = folders.map(f => ({
      name: f.name,
      folder_id: f.id,
      drive_file_count: proposalCounts[f.name] || 0
    }));

    // 4) Include proposed categories without a matching folder
    for (const name of Object.keys(proposalCounts)) {
      if (!out.some(x => x.name === name)) {
        out.push({ name, folder_id: null, drive_file_count: proposalCounts[name], missing_folder: true });
      }
    }

    res.json(out);
  } catch (e) { res.status(500).json({ error: e.toString() }); }
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

// Drive duplicates based on file metadata (md5Checksum preferred; fallback name+size)
app.get('/drive/duplicates', async (req, res) => {
  try {
    if (!DRIVE_TOKEN) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });

    const { folderId, limit } = req.query || {};
    const MAX_FILES = Math.max(1, Math.min(Number(limit) || 300, 2000));

    // Helper: group files by content hash or fallback key
    const groupFiles = (files) => {
      const groups = {};
      for (const m of files) {
        if (!m || m.trashed) continue;
        let key;
        if (m.mimeType === 'application/vnd.google-apps.folder') {
          // Group folders strictly by name (optionally this could include parent id for stricter grouping)
          key = `folder:${m.name || ''}`;
        } else if (m.md5Checksum) {
          key = `md5:${m.md5Checksum}`;
        } else if (m.size) {
          key = `ns:${m.name || ''}:${m.size}`;
        } else {
          // Some Google-native files may not have size; fall back to name-only
          key = `n:${m.name || ''}`;
        }
        if (!groups[key]) groups[key] = [];
        groups[key].push({ id: m.id, name: m.name, size: Number(m.size || 0) });
      }
      const out = [];
      let gi = 1;
      for (const files of Object.values(groups)) {
        if (files.length >= 2) out.push({ group_id: `group_${gi++}`, file_count: files.length, files });
      }
      return out;
    };

    // Case 1: Use current proposals (fast path)
    const proposalIds = (DRIVE_PROPOSALS || []).map(f => f.id).filter(Boolean);
    if (proposalIds.length) {
      const metas = [];
      for (const id of proposalIds) {
        try {
          const r = await axios.get(`https://www.googleapis.com/drive/v3/files/${encodeURIComponent(id)}`, {
            params: { fields: 'id,name,size,md5Checksum,trashed,mimeType' },
            headers: { Authorization: `Bearer ${DRIVE_TOKEN}` }
          });
          metas.push(r.data || {});
        } catch (_) { /* skip */ }
      }
      const dupGroups = groupFiles(metas);
      return res.json({ summary: { duplicate_groups_found: dupGroups.length, total_files_processed: metas.length }, duplicates: dupGroups });
    }

    // Case 2: No proposals – list files from Drive (optionally folder)
    const files = [];
    let pageToken = undefined;
    let fetched = 0;
    const baseParams = {
      fields: 'nextPageToken, files(id,name,size,md5Checksum,trashed,mimeType)',
      pageSize: 200,
      orderBy: 'modifiedTime desc'
    };
    const qParts = ["trashed=false", "mimeType!='application/vnd.google-apps.folder'"];
    if (folderId) qParts.push(`'${folderId}' in parents`);
    const q = qParts.join(' and ');

    while (fetched < MAX_FILES) {
      const params = { ...baseParams, q, pageToken };
      const r = await axios.get('https://www.googleapis.com/drive/v3/files', {
        params,
        headers: { Authorization: `Bearer ${DRIVE_TOKEN}` }
      });
      const batch = (r.data?.files || []);
      for (const f of batch) {
        files.push(f);
        fetched += 1;
        if (fetched >= MAX_FILES) break;
      }
      pageToken = r.data?.nextPageToken;
      if (!pageToken || fetched >= MAX_FILES) break;
    }

    const dupGroups = groupFiles(files);
    res.json({ summary: { duplicate_groups_found: dupGroups.length, total_files_processed: files.length }, duplicates: dupGroups });
  } catch (e) {
    res.status(500).json({ error: e.toString() });
  }
});

// Delete a Drive file by ID
app.post('/drive/delete', async (req, res) => {
  try {
    const id = req.body?.id;
    if (!id) return res.status(400).json({ error: 'missing id' });
    if (!DRIVE_TOKEN) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });

    await axios.delete(`https://www.googleapis.com/drive/v3/files/${encodeURIComponent(id)}`, {
      headers: { Authorization: `Bearer ${DRIVE_TOKEN}` }
    });

    // Also drop from current proposals cache if present
    DRIVE_PROPOSALS = (DRIVE_PROPOSALS || []).filter(f => f.id !== id);
    res.json({ ok: true, id });
  } catch (e) {
    const msg = e.response?.data || e.message || String(e);
    res.status(500).json({ error: msg });
  }
});

app.get('/categories', async (req, res) => {
  try {
    const r = await axios.get(`${PARSER}/categories`).catch(() => ({ data: [] }));
    const raw = Array.isArray(r.data) ? r.data : [];
    // Normalize to array of names (strings)
    let categories = [];
    if (raw.length && typeof raw[0] === 'string') {
      categories = raw;
    } else if (raw.length && typeof raw[0] === 'object' && raw[0] !== null) {
      categories = raw.map(x => x.name).filter(Boolean);
    }
    const localCounts = {};
    const driveCounts = {};
    try {
      const listResp = await axios.get(`${PARSER}/list_proposals`);
      const list = Array.isArray(listResp.data) ? listResp.data : [];
      for (const f of list) {
        const k = f.proposed_category || 'Other';
        localCounts[k] = (localCounts[k] || 0) + 1;
      }
    } catch (_) { /* ignore */ }
    for (const f of DRIVE_PROPOSALS) {
      const k = f.proposed_category || 'Other';
      driveCounts[k] = (driveCounts[k] || 0) + 1;
    }
    // If parser returned nothing, derive categories from Drive proposals
    if (!categories.length) {
      const set = new Set();
      for (const f of DRIVE_PROPOSALS) set.add(f.proposed_category || 'Other');
      categories = Array.from(set);
    }

    const out = categories
      .filter(name => (localCounts[name] || 0) > 0 || (driveCounts[name] || 0) > 0)
      .map(name => ({ name, local_file_count: localCounts[name] || 0, drive_file_count: driveCounts[name] || 0 }));
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
    
    if (reallyMoved || r.data?.already_in_correct_folder) {
      // Update the file in DRIVE_PROPOSALS to mark it as approved
      const updatedFile = { 
        ...file, 
        approved: true, 
        final_category: override_category || file.proposed_category || file.final_category 
      };
      
      // Update or add the file to DRIVE_PROPOSALS
      const existingIndex = DRIVE_PROPOSALS.findIndex(p => p.id === file.id);
      if (existingIndex >= 0) {
        DRIVE_PROPOSALS[existingIndex] = { ...DRIVE_PROPOSALS[existingIndex], ...updatedFile };
      } else {
        DRIVE_PROPOSALS.push(updatedFile);
      }
      
      res.json({ 
        ok: true, 
        moved, 
        file: updatedFile,
        already_in_correct_folder: r.data?.already_in_correct_folder
      });
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
    
    // Check if this is a document generation request
    if (req.body?.q && r.data) {
      const query = req.body.q.toLowerCase().trim();
      const generationPatterns = {
        "flowchart": ["flowchart", "flow chart", "flowchrt", "flwchart", "diagram", "diagramm", "mermaid", "process flow", "create flowchart", "make flowchart", "generate flowchart", "crete flowchart", "mke flowchart"],
        "short_notes": ["short notes", "bullet points", "summary notes", "brief notes", "quick notes", "create notes", "make notes", "sumary notes", "breif notes", "quik notes", "crete notes", "mke notes", "bullet point", "bulet points"],
        "detailed_notes": ["detailed notes", "revision notes", "study notes", "comprehensive notes", "full notes", "create revision", "make revision", "detaild notes", "revison notes", "studie notes", "crete revision", "mke revision", "revision note", "revison note"],
        "timeline": ["timeline", "chronological", "sequence", "steps", "process timeline", "create timeline", "make timeline", "timline", "chronologicl", "sequance", "step", "proces timeline", "crete timeline", "mke timeline"],
        "key_insights": ["key insights", "insights", "takeaways", "main points", "important points", "create insights", "extract insights", "kye insights", "insites", "takeaway", "main point", "importnt points", "crete insights", "extrac insights"],
        "flashcards": ["flashcards", "flash cards", "quiz", "q&a", "questions and answers", "qna", "create flashcards", "make flashcards", "flashcard", "flash card", "quizz", "q and a", "question and answer", "crete flashcards", "mke flashcards"]
      };
      
      const formatPatterns = {
        "pdf": ["pdf", "downloadable pdf", "download pdf", "as pdf", "in pdf"],
        "docx": ["docx", "word", "doc", "downloadable docx", "download docx", "as docx", "in docx", "as word"],
        "txt": ["txt", "text file", "downloadable txt", "download txt", "downloadable format", "download", "as txt", "in txt"],
        "png": ["png", "image", "img", "picture", "as image", "as png", "downloadable image", "download image"]
      };
      
      let detectedKind = null;
      let detectedFormat = null;
      
      // Detect document type
      for (const [kind, patterns] of Object.entries(generationPatterns)) {
        if (patterns.some(pattern => query.includes(pattern))) {
          detectedKind = kind;
          break;
        }
      }
      
      // Detect format
      for (const [fmt, patterns] of Object.entries(formatPatterns)) {
        if (patterns.some(pattern => query.includes(pattern))) {
          detectedFormat = fmt;
          break;
        }
      }
      
      if (detectedKind) {
        try {
          // Call the assistant generator
          const assistantBody = {
            kind: detectedKind,
            file: req.body.file,
            format: detectedFormat || 'txt',
            auth_token: body.auth_token
          };
          
          const assistantResponse = await axios.post(`${PARSER}/assistant_generate`, assistantBody);
          
          if (assistantResponse.data) {
            // Enhance the response with assistant data
            r.data.assistant = {
              type: detectedFormat ? 'download' : 'display',
              kind: assistantResponse.data.kind,
              filename: assistantResponse.data.filename,
              base64: assistantResponse.data.base64,
              mime: assistantResponse.data.mime,
              content: assistantResponse.data.content
            };
            
            // Update the answer
            if (detectedFormat) {
              r.data.qa = r.data.qa || {};
              r.data.qa.answer = `I've generated your ${detectedKind.replace('_', ' ')} as a downloadable ${detectedFormat.toUpperCase()} file.`;
            } else {
              // Format content for display
              let formattedContent = assistantResponse.data.content;
              if (detectedKind === 'flashcards') {
                formattedContent = assistantResponse.data.content.map((card, i) => 
                  `**Q${i+1}:** ${card[0]}\n**A${i+1}:** ${card[1]}`
                ).join('\n\n');
              } else if (detectedKind === 'flowchart') {
                formattedContent = `\`\`\`mermaid\n${assistantResponse.data.content}\n\`\`\``;
              }
              
              r.data.qa = r.data.qa || {};
              r.data.qa.answer = formattedContent;
            }
          }
        } catch (assistantError) {
          console.error('Assistant generation error:', assistantError);
          r.data.qa = r.data.qa || {};
          r.data.qa.answer = `I encountered an error generating the ${detectedKind.replace('_', ' ')}: ${assistantError.message}`;
        }
      }
    }
    
    // Check if analysis resulted in a new category and auto-update
    if (r.data && req.body?.file?.id) {
      const currentCategory = r.data.category;
      const fileId = req.body.file.id;
      
      // If we have a new category from analysis, automatically update it
      if (currentCategory && currentCategory !== 'Unknown') {
        try {
          console.log(`AUTO-UPDATING category for file ${fileId} to: ${currentCategory}`);
          
          // Call the update category endpoint internally
          await axios.post(`${PARSER}/update_category`, {
            file_id: fileId,
            new_category: currentCategory,
            auth_token: req.body.auth_token || DRIVE_TOKEN
          });
          
          console.log(`Successfully auto-updated category for ${fileId} to ${currentCategory}`);
          
          // Mark in response that category was auto-updated
          r.data.category_auto_updated = true;
          
        } catch (updateError) {
          console.error('Failed to auto-update category:', updateError.response?.data || updateError.message);
          // Don't fail the whole request if category update fails
          r.data.category_update_failed = true;
        }
      }
    }
    
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

// Search files in Drive by content
app.post('/search_files', async (req, res) => {
  try {
    if (!DRIVE_TOKEN && !req.body?.auth_token) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    const query = req.body?.query;
    if (!query) return res.status(400).json({ error: 'missing search query' });
    
    const r = await axios.post(`${PARSER}/search_files?auth_token=${encodeURIComponent(DRIVE_TOKEN)}`, req.body);
    res.json(r.data);
  } catch (e) { 
    console.error('Search error:', e.response?.data || e.message);
    res.status(500).json({ error: e.response?.data?.detail || e.toString() }); 
  }
});

// Multi-file analysis
app.post('/analyze_multi', async (req, res) => {
  try {
    if (!DRIVE_TOKEN && !req.body?.auth_token) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    const body = { ...req.body, auth_token: req.body?.auth_token || DRIVE_TOKEN };
    const r = await axios.post(`${PARSER}/analyze_multi`, body);
    res.json(r.data);
  } catch (e) { 
    console.error('Multi-file analysis error:', e.response?.data || e.message);
    res.status(500).json({ error: e.response?.data?.detail || e.toString() }); 
  }
});

// Visual search files in Drive by image content
app.post('/visual_search', async (req, res) => {
  try {
    if (!DRIVE_TOKEN) return res.status(400).json({ error: 'no drive token available; click Organize in Drive again' });
    
    // Forward the multipart form data to the parser
    const FormData = require('form-data');
    const formData = new FormData();
    
    // Add the image file
    if (req.files && req.files.image) {
      formData.append('image', req.files.image.data, {
        filename: req.files.image.name,
        contentType: req.files.image.mimetype
      });
    } else {
      return res.status(400).json({ error: 'missing image file' });
    }
    
    const r = await axios.post(`${PARSER}/visual_search?auth_token=${encodeURIComponent(DRIVE_TOKEN)}`, formData, {
      headers: {
        ...formData.getHeaders()
      }
    });
    res.json(r.data);
  } catch (e) { 
    console.error('Visual search error:', e.response?.data || e.message);
    res.status(500).json({ error: e.response?.data?.detail || e.toString() }); 
  }
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

// Update category for a file
app.post('/drive/update_category', async (req, res) => {
  try {
    console.log('UPDATE_CATEGORY ENDPOINT CALLED!');
    console.log('Request body:', req.body);
    
    const { fileId, newCategory } = req.body;
    
    if (!fileId || !newCategory) {
      return res.status(400).json({ error: 'fileId and newCategory are required' });
    }
    
    // Forward to parser service to update the category
    const r = await axios.post(`${PARSER}/update_category`, {
      file_id: fileId,
      new_category: newCategory,
      auth_token: req.body.auth_token || DRIVE_TOKEN
    });
    
    console.log('Category update response:', r.data);
    res.json(r.data);
  } catch (e) {
    console.error('Update category error:', e.response?.data || e.message);
    res.status(500).json({ error: e.response?.data?.detail || e.toString() });
  }
});

// ====== SERVE UI ======
// Prefer built React app if available (ui/dist), else fall back to legacy ui/
const builtUi = path.join(__dirname, '../ui/dist');
const legacyUi = path.join(__dirname, '../ui');
try {
  app.use(express.static(builtUi));
  // Also try to serve fallback assets from legacy ui for missing files like index.css
  app.use(express.static(legacyUi));
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
