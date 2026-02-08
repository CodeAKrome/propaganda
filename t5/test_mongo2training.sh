export MONGODB_URI="mongodb://root:example@localhost:27017/"
python mongo2training.py -o train.json --limit 10000 --verbose
