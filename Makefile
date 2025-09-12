load:
	go run main.go config/big.tsv config/kill.tsv
testload:
	go run main.go config/test.tsv config/kill.tsv
back:
	cd back; node server.js
front:
	cd front; npm start
