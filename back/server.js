require('dotenv').config();
const express = require('express');
const { MongoClient, ObjectId } = require('mongodb');
const cors = require('cors');


//run once
// use rssnews
// db.articles.createIndex({ published: -1 })
// db.articles.createIndex({ title: "text", description: "text", article: "text" })


const app = express();
app.use(cors());

const client = new MongoClient(process.env.MONGO_URI);

client.connect().then(() => console.log('Mongo connected'));
const coll = client.db('rssnews').collection('articles');

// GET /api/articles?page=1&size=20&q=query&source=BBC&failed=false
app.get('/api/articles', async (req, res) => {
  const page  = Math.max(0, parseInt(req.query.page)  || 0);
  const size  = Math.min(100, parseInt(req.query.size) || 20);
  const q     = (req.query.q || '').trim();
  const src   = req.query.source;
  const failed = req.query.failed === 'true';

  const match = {};
  if (q) match.$text = { $search: q };
  if (src) match.source = src;
  if (failed) match.fetch_error = { $exists: true };
  else        match.article      = { $exists: true };

  const [rows, total] = await Promise.all([
    coll
      .find(match, { projection: { article: 0 } }) // skip body in list
      .sort({ published: -1 })
      .skip(page * size)
      .limit(size)
      .toArray(),
    coll.countDocuments(match)
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

app.listen(process.env.PORT, () =>
  console.log(`API listening on :${process.env.PORT}`)
);