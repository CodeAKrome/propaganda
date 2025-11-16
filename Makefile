CONDA_MP3_ENV = kokoro
DB_ENV = db/.venv
SHELL := /bin/bash
MLX_MODEL = mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit
NUMDAYS = -1
TITLEFILE = output/titles.tsv

.PHONY: build load back front vector query mp3 mgconsole testload thingsthatgo fini ner fner fnervector entity

thingsthatgo: load ner vector query mp3 fini

fload: load ner vector entity fini
fner: ner vector entity query mp3 fini
fnervector: ner vector fini
fquerymp3: entity query mp3 fini
fquery: load ner vector query fini
fmp3: mp3 fini

black:
	black db/*.py
	black ner/*.py
entity:
	cd db && ./mongo2chroma.py title --start-date $(NUMDAYS) | sort -k4,4 > $(TITLEFILE)
	cd db && ./mongo2chroma.py dumpentity --start-date $(NUMDAYS) | egrep '(PERSON|GPE|LOC|EVENT)' > output/impentity.tsv
	cd db && ./mongo2chroma.py dumpentity --start-date $(NUMDAYS)  > output/entity.tsv
	cd db && ./mongo2chroma.py dumpentity --start-date -60  > output/entity_60days.tsv
	cd db && ./mongo2chroma.py dumpentity --start-date -60 | egrep '(PERSON|GPE|LOC|EVENT)' > output/impentity_60days.tsv
# build propaganda go binary
build:
	go build
# read RSS feeds and load data into mongodb
load:
	./propaganda config/big.tsv config/kill.tsv
	source $(DB_ENV)/bin/activate && cd db && ./dedupe.py
#	go run main.go config/big.tsv config/kill.tsv
# start back end before front end
back:
	back/RUNME.sh
# start web interface
front:
	front/RUNME.sh
# read data from mongodb and create vectors in chroma
vector:
	$(DB_ENV)/bin/python db/mongo2chroma.py load --start-date $(NUMDAYS)
# Do NER
ner:
	cd ../ner && ./RUNME.sh $(NUMDAYS)
# runseries of queries to generate db/output/*.md
query:
	find db/output -name "*.md" -delete
	find db/output -name "*.txt" -delete
	find db/output -name "*.vec" -delete
	find db/output -name "*.cypher" -delete
	find db/output -name "*.reporter" -delete
	source $(DB_ENV)/bin/activate && cd db && ./runentitybatch.sh
#	source $(DB_ENV)/bin/activate && cd db && ./runbatch.sh
#	source $(DB_ENV)/bin/activate && cd db && ./run_parallel.sh
# generate mp3 files into mp3/mp3 using files in db/output/*.md
mp3:
	find mp3/mp3 -name "*.mp3" -delete
	cd mp3 && ./mkmkbatch.sh
	source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate $(CONDA_MP3_ENV) && \
	cd mp3 && \
	./batch.sh
#	cd mp3 && conda run -n $(CONDA_MP3_ENV) ./batch.sh
fini:
	afplay ~/Music/df\ picking\ up\ man.wav
	say "The news has been loaded, Doctor!"
mgconsole:
	docker run -it memgraph/mgconsole:latest --host host.docker.internal
testload:
	./propaganda config/test.tsv config/kill.tsv
#	go run main.go config/test.tsv config/kill.tsv
