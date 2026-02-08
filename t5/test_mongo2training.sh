export MONGODB_URI="mongodb://root:example@localhost:27017/"
#python mongo2training.py -o train.json --start-date 2026-01-01
python mongo2training2.py -o train.json --limit 10000 --verbose
