mgconsole:
	docker run -it memgraph/mgconsole:latest --host host.docker.internal
build:
	go build
load:
	./propaganda config/big.tsv config/kill.tsv
#	go run main.go config/big.tsv config/kill.tsv
testload:
	go run main.go config/test.tsv config/kill.tsv
back:
	cd back; node server.js
front:
	cd front; npm start
