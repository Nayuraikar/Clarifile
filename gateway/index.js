// gateway/index.js
const express = require('express');
const axios = require('axios');
const app = express();
app.use(express.json());

const PARSER = 'http://localhost:8001';
const EMBED = 'http://localhost:8002';
const INDEXER = 'http://localhost:8003';

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

app.listen(4000, () => console.log('Gateway running on http://localhost:4000'));
