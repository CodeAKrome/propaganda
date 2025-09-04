// MKindexes.js  – create required unique indexes
require('dotenv').config();
const { MongoClient } = require('mongodb');

(async () => {
  const client = new MongoClient(process.env.MONGO_URI);
  await client.connect();
  const coll = client.db('rssnews').collection('articles');

  // Performance indexes
  await coll.createIndex({ published: -1 });
  await coll.createIndex(
    { title: 'text', description: 'text', article: 'text' }
  );

  // --- uniqueness constraints ---
  await coll.createIndex({ link: 1 }, { unique: true });
  await coll.createIndex({ title: 1 }, { unique: true });

  console.log('✅ Unique indexes on link and title created');
  await client.close();
})();