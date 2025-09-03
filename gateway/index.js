// gateway/index.js
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const app = express();
app.use(express.json());
app.use(cors());

const PARSER = 'http://127.0.0.1:8001';
const EMBED = 'http://127.0.0.1:8002';
const INDEXER = 'http://127.0.0.1:8003';
const DEDUP = 'http://127.0.0.1:8004';

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

app.get('/proposals', async (req, res) => {
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

app.listen(4000, () => console.log('Gateway running on http://127.0.0.1:4000'));
