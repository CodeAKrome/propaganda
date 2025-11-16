#!/usr/bin/env python3
"""
FastAPI micro-service that wraps
  - Flair NER (ontonotes-large)
  - Flair sentence-level sentiment
  - Targeted sentiment on each entity mention

NEW: response contains a top-level list of unique entities with
     positive / negative / indifferent counts and enum-based types.
"""

import argparse
import time
from collections import defaultdict
from enum import Enum
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter
from NewsSentiment import TargetSentimentClassifier


# ------------------------------------------------------------------
# Enums & Pydantic models
# ------------------------------------------------------------------
class EntityType(str, Enum):
    """Fixed vocabulary of entity types returned by the NER model."""

    PERSON = "PERSON"
    NORP = "NORP"
    FAC = "FAC"
    ORG = "ORG"
    GPE = "GPE"
    LOC = "LOC"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    DATE = "DATE"
    TIME = "TIME"
    PERCENT = "PERCENT"
    MONEY = "MONEY"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"
    UNKNOWN = "UNKNOWN"


class NerRequest(BaseModel):
    text: str


class Span(BaseModel):
    text: str
    start: int
    end: int
    value: str
    score: str
    sentiment: str
    probability: str


class SentenceOut(BaseModel):
    sentence: str
    tag: str
    score: str
    spans: List[Span]


class EntityAggregate(BaseModel):
    text: str
    type: EntityType
    positive: int
    negative: int
    indifferent: int
    total: int


class NerResponse(BaseModel):
    elapsed_ms: float
    entities: List[EntityAggregate]
    sentences: List[SentenceOut]


# ------------------------------------------------------------------
# Core classifier
# ------------------------------------------------------------------
class FlairSentiment:
    NER_TAGGER = "flair/ner-english-ontonotes-large"

    def __init__(self):
        self.sentiment_tagger = Classifier.load("sentiment")
        self.ner_tagger = Classifier.load(self.NER_TAGGER)
        self.splitter = SegtokSentenceSplitter()
        self.tsc = TargetSentimentClassifier()

    # ---------- existing sentence-level processing ----------
    def process_text(self, text: str) -> List[Dict]:
        sentences = self.splitter.split(text)
        self.sentiment_tagger.predict(sentences)
        self.ner_tagger.predict(sentences)

        output = []
        for sentence in sentences:
            spans = []
            sent = sentence.to_plain_string()
            for span in sentence.get_spans("ner"):
                left = sent[: span.start_position]
                mention = sent[span.start_position : span.end_position]
                right = sent[span.end_position :]

                sentiment = self.tsc.infer_from_text(left, mention, right)
                for label in span.labels:
                    value = "" if label.value == "<unk>" else label.value
                    spans.append(
                        {
                            "text": span.text,
                            "start": span.start_position,
                            "end": span.end_position,
                            "value": value,
                            "score": f"{label.score:.2f}",
                            "sentiment": sentiment[0]["class_label"],
                            "probability": f"{sentiment[0]['class_prob']:.2f}",
                        }
                    )

            output.append(
                {
                    "sentence": sent,
                    "tag": sentence.tag.lower(),
                    "score": f"{sentence.score:.2f}",
                    "spans": spans,
                }
            )
        return output

    # ---------- NEW: aggregation ----------
    @staticmethod
    def _aggregate_entities(sentences: List[Dict]) -> List[EntityAggregate]:
        counter = defaultdict(
            lambda: {"positive": 0, "negative": 0, "indifferent": 0, "type": "UNKNOWN"}
        )

        for sent in sentences:
            for sp in sent["spans"]:
                key = sp["text"]
                counter[key]["type"] = sp["value"] or "UNKNOWN"
                pol = sp["sentiment"].lower()
                if pol == "positive":
                    counter[key]["positive"] += 1
                elif pol == "negative":
                    counter[key]["negative"] += 1
                else:
                    counter[key]["indifferent"] += 1

        return [
            EntityAggregate(
                text=text,
                type=EntityType(vals["type"]),
                positive=vals["positive"],
                negative=vals["negative"],
                indifferent=vals["indifferent"],
                total=vals["positive"] + vals["negative"] + vals["indifferent"],
            )
            for text, vals in counter.items()
        ]


# ------------------------------------------------------------------
# FastAPI wiring
# ------------------------------------------------------------------
app = FastAPI(title="Flair NER + Sentiment micro-service")
classifier = FlairSentiment()


@app.get("/heartbeat")
def heartbeat():
    return {"status": "ok"}


@app.post("/ner", response_model=NerResponse)
def ner(payload: NerRequest):
    if not payload.text:
        raise HTTPException(status_code=400, detail="No text provided")

    tic = time.perf_counter()
    sentences = classifier.process_text(payload.text)
    entities = classifier._aggregate_entities(sentences)
    toc = time.perf_counter()

    return NerResponse(
        elapsed_ms=(toc - tic) * 1000,
        entities=entities,
        sentences=sentences,
    )


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI Flair Sentiment Microservice")
    parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=1337,
        help="Port number to run the service on. Defaults to 1337.",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("flair_fast:app", host="0.0.0.0", port=args.port, reload=False)
