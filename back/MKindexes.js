require('dotenv').config();
const { MongoClient } = require('mongodb');

(async () => {
  const client = new MongoClient(process.env.MONGO_URI);
  await client.connect();
  const coll = client.db('rssnews').collection('articles');

  await coll.createIndex({ published: -1 });
  await coll.createIndex(
    { title: "text", description: "text", article: "text" }
  );

  console.log('Indexes created');
  await client.close();
})();
