// server.js  (complete – hides articles < 128 chars unless failed=true)
require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const { MongoClient, ObjectId } = require('mongodb');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(bodyParser.json());

const client = new MongoClient(process.env.MONGO_URI);
client.connect().then(() => console.log('Mongo connected'));
const coll = client.db('rssnews').collection('articles');

// ---------- helper: flag short articles ----------
function flagShortArticle(doc) {
  if (doc.article && doc.article.length < 128) doc.too_short = true;
  else delete doc.too_short;                 // remove flag if now long enough
}

// GET /api/articles
app.get('/api/articles', async (req, res) => {
  const page   = Math.max(0, parseInt(req.query.page) || 0);
  const size   = Math.min(100, parseInt(req.query.size) || 20);
  const q      = (req.query.q || '').trim();
  const src    = req.query.source;
  const failed = req.query.failed === 'true';
  const from   = req.query.from;
  const to     = req.query.to;
  const tags   = req.query.tags;                // ← comma-separated list
  const sortField = req.query.sort === 'title' ? 'title' : 'published';
  const sortDir   = sortField === 'title' ? 1 : -1;   // A-Z  vs  newest first

  const match = {};
  if (q) match.$text = { $search: q };
  if (src) match.source = src;
  if (from || to) match.published = {};
  if (from) match.published.$gte = new Date(from);
  if (to)   match.published.$lte = new Date(to);
  if (tags) match.tags = { $in: tags.split(',').map(t => t.trim()) };

  if (failed) {
    // user explicitly wants failures – no extra filtering
  } else {
    match.article = { $exists: true };
    match.too_short = { $ne: true }; // <-- hide short articles by default
  }

  const [rows, total] = await Promise.all([
    coll
      .find(match, { projection: { article: 0 } })
      .sort({ [sortField]: sortDir })
      .skip(page * size)
      .limit(size)
      .toArray(),
    coll.countDocuments(match),
  ]);

  res.json({ rows, total, page, pages: Math.ceil(total / size) });
});

// GET /api/articles/date-range  (global min/max)
app.get('/api/articles/date-range', async (_req, res) => {
  const pipeline = [
    { $group: { _id: null, min: { $min: '$published' }, max: { $max: '$published' } } },
  ];
  const [result] = await coll.aggregate(pipeline).toArray();
  if (!result) return res.json({ min: null, max: null });
  res.json({ min: result.min.toISOString().slice(0, 10), max: result.max.toISOString().slice(0, 10) });
});

// GET /api/articles/daily-counts  (heatmap)
app.get('/api/articles/daily-counts', async (_req, res) => {
  const pipeline = [
    { $group: { _id: { $dateToString: { format: '%Y-%m-%d', date: '$published' } }, count: { $sum: 1 } } },
    { $sort: { _id: 1 } },
  ];
  const raw = await coll.aggregate(pipeline).toArray();
  res.json(raw.map(d => ({ date: d._id, count: d.count })));
});

// GET /api/sources
app.get('/api/sources', async (_req, res) => {
  const data = await coll.distinct('source');
  res.json(data);
});

// GET /api/tags
app.get('/api/tags', async (_req, res) => {
  const data = await coll.distinct('tags');
  res.json(data);
});

// GET /api/article/:id   ← REMOVED PROJECTION – entire doc returned
app.get('/api/article/:id', async (req, res) => {
  const doc = await coll.findOne({ _id: new ObjectId(req.params.id) });
  if (!doc) return res.status(404).send('not found');
  res.json(doc);
});

// POST /api/article/:id  (optional) re-clean after editing raw
app.post('/api/article/:id', async (req, res) => {
  const doc = await coll.findOne({ _id: new ObjectId(req.params.id) });
  if (!doc) return res.status(404).send('not found');
  flagShortArticle(doc);
  await coll.replaceOne({ _id: doc._id }, doc);
  res.json(doc);
});

// POST /api/article/:id/tags  { tags: ['politics','asia'] }
app.post('/api/article/:id/tags', async (req, res) => {
  const { tags } = req.body;
  if (!Array.isArray(tags)) return res.status(400).json({ error: 'tags must be an array' });

  const result = await coll.updateOne(
    { _id: new ObjectId(req.params.id) },
    { $set: { tags } }
  );
  if (result.matchedCount === 0) return res.status(404).send('not found');
  res.json({ ok: true });
});

// POST /api/articles/bulk-tag  { ids: [...], tags: ['x','y'] }
app.post('/api/articles/bulk-tag', async (req, res) => {
  const { ids, tags } = req.body;
  if (!Array.isArray(ids) || !Array.isArray(tags))
    return res.status(400).json({ error: 'ids and tags must be arrays' });

  const objIds = ids.map(id => new ObjectId(id));
  const result = await coll.updateMany(
    { _id: { $in: objIds } },
    { $set: { tags } }
  );
  res.json({ matched: result.matchedCount, modified: result.modifiedCount });
});

// DELETE /api/tags/:tag  (removes tag from ALL articles)
app.delete('/api/tags/:tag', async (req, res) => {
  const { tag } = req.params;
  const result = await coll.updateMany(
    { tags: tag },
    { $pull: { tags: tag } }
  );
  res.json({ matched: result.matchedCount, modified: result.modifiedCount });
});

app.listen(process.env.PORT, () =>
  console.log(`API listening on :${process.env.PORT}`)
);