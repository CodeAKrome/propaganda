mongo rssnews --eval '
  db.articles.createIndex({link: 1},  {unique: true});
  db.articles.createIndex({title: 1}, {unique: true});
'
