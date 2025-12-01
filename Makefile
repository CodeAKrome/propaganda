CONDA_MP3_ENV = kokoro
DB_ENV = db/.venv
SHELL := /bin/zsh
#MLX_MODEL = mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit
NUMDAYS = -1
TITLEFILE = output/titles.tsv

.PHONY: build load back front vector query mp3 mgconsole testload thingsthatgo fini ner fner fnervector entity fvector bias mkvec fbias querysmall mkvecsmall smallthingsthatgo

thingsthatgo: load ner vector entity mkvec bias query mp3 fini
smallthingsthatgo: load ner vector entity mkvecsmall bias querysmall mp3 fini
oldthingsthatgo: load ner vector entity query mp3 fini

fvector: load ner vector fini
fload: load ner vector entity fini
fner: ner vector entity query mp3 fini
fnervector: ner vector fini
fquerymp3: query mp3 fini
fquery: load ner vector query fini
fmp3: mp3 fini
fbias: mkvec bias query mp3 fini

black:
	black db/*.py
	black ner/*.py
entity:
	cd db && ./mongo2chroma.py title --start-date $(NUMDAYS) | sort -k4,4 > $(TITLEFILE)
	cd db &&  cat $(TITLEFILE) | grep -v thehindu | grep -v indiaexpress > output/titles_nohindu.tsv
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

# run before bias. Create .vec and .ids files to use for bias and generating articles
mkvec:
	find db/output -name "*.vec" -delete
	find db/output -name "*.ids" -delete
	source $(DB_ENV)/bin/activate && cd db && ./runmkvecbatch.sh

mkvecsmall:
	find db/output -name "*.vec" -delete
	find db/output -name "*.ids" -delete
	source $(DB_ENV)/bin/activate && cd db && ./batchquery.sh './mkvec.sh'

# output/ids.txt to run geminize.py
bias:
	source $(DB_ENV)/bin/activate && cd db && ./runbias.sh

# runseries of queries to generate db/output/*.md
query:
	find db/output -name "*.md" -delete
	find db/output -name "*.txt" -delete
#	find db/output -name "*.vec" -delete
	find db/output -name "*.cypher" -delete
	find db/output -name "*.reporter" -delete
	source $(DB_ENV)/bin/activate && cd db && ./runentitybatch.sh

querysmall:
	find db/output -name "*.md" -delete
	find db/output -name "*.txt" -delete
#	find db/output -name "*.vec" -delete
	find db/output -name "*.cypher" -delete
	find db/output -name "*.reporter" -delete
	source $(DB_ENV)/bin/activate && cd db && ./batchquery.sh './report.sh'


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
