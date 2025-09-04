// server.js  (complete, with date-range support)
require('dotenv').config();
const express = require('express');
const { MongoClient, ObjectId } = require('mongodb');
const cors = require('cors');

const app = express();
app.use(cors());

const client = new MongoClient(process.env.MONGO_URI);
client.connect().then(() => console.log('Mongo connected'));
const coll = client.db('rssnews').collection('articles');

// GET /api/articles
app.get('/api/articles', async (req, res) => {
  const page   = Math.max(0, parseInt(req.query.page) || 0);
  const size   = Math.min(100, parseInt(req.query.size) || 20);
  const q      = (req.query.q || '').trim();
  const src    = req.query.source;
  const failed = req.query.failed === 'true';

  // --- date range ---
  const from = req.query.from;
  const to   = req.query.to;

  const match = {};
  if (q) match.$text = { $search: q };
  if (src) match.source = src;
  if (failed) match.fetch_error = { $exists: true };
  else        match.article      = { $exists: true };

  // add date filters
  if (from || to) {
    match.published = {};
    if (from) match.published.$gte = new Date(from);
    if (to)   match.published.$lte = new Date(to);
  }

  const [rows, total] = await Promise.all([
    coll
      .find(match, { projection: { article: 0 } })
      .sort({ published: -1 })
      .skip(page * size)
      .limit(size)
      .toArray(),
    coll.countDocuments(match),
  ]);

  res.json({ rows, total, page, pages: Math.ceil(total / size) });
});

// GET /api/sources
app.get('/api/sources', async (_req, res) => {
  const data = await coll.distinct('source');
  res.json(data);
});

// GET /api/article/:id
app.get('/api/article/:id', async (req, res) => {
  const doc = await coll.findOne({ _id: new ObjectId(req.params.id) });
  if (!doc) return res.status(404).send('not found');
  res.json(doc);
});

// GET /api/articles/date-range
app.get('/api/articles/date-range', async (req, res) => {
  const pipeline = [
    { $group: { _id: null, min: { $min: '$published' }, max: { $max: '$published' } } }
  ];
  const [result] = await coll.aggregate(pipeline).toArray();
  if (!result) return res.json({ min: null, max: null });
  res.json({ min: result.min.toISOString().slice(0, 10),
             max: result.max.toISOString().slice(0, 10) });
});

// GET /api/articles/daily-counts
app.get('/api/articles/daily-counts', async (_req, res) => {
  const pipeline = [
    {
      $group: {
        _id: {
          $dateToString: { format: '%Y-%m-%d', date: '$published' }
        },
        count: { $sum: 1 }
      }
    },
    { $sort: { _id: 1 } }
  ];
  const raw = await coll.aggregate(pipeline).toArray();
  const counts = raw.map(d => ({ date: d._id, count: d.count }));
  res.json(counts);
});

app.listen(process.env.PORT, () =>
  console.log(`API listening on :${process.env.PORT}`)
);