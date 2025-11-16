#!/usr/bin/env python3
"""
Read a news script from stdin, break it into sentences with Flair,
detect named entities with OntoNotes-large-fast, and emit:

<text>…</text>
<background>…</background>
<entities><person>…</person><loc>…</loc>…</entities>

The <background> prompt is built from the highest-weighted entities
found in the sentence (weights defined below).
"""

import sys
from typing import List, Dict, Set

from flair.data import Sentence
from flair.models import SequenceTagger

# ------------------------------------------------------------------
# 1.  Entity-type weights (higher → more important for background)
# ------------------------------------------------------------------
ENTITY_WEIGHTS = {
    "EVENT": 10,
    "PERSON": 9,
    "ORG": 8,
    "GPE": 7,
    "FAC": 6,
    "PRODUCT": 5,
    "NORP": 4,
    "MONEY": 3,
    "CARDINAL": 2,
    "DATE": 1,
    "TIME": 1,
}

# Map OntoNotes tag → simple XML tag used in <entities>
TAG_TO_XML = {
    "PERSON": "person",
    "ORG": "org",
    "GPE": "loc",
    "LOC": "loc",
    "FAC": "fac",
    "EVENT": "event",
    "PRODUCT": "product",
    "NORP": "norp",
    "MONEY": "money",
    "CARDINAL": "cardinal",
    "DATE": "date",
    "TIME": "time",
}

# ------------------------------------------------------------------
# 2.  Load Flair sentence splitter and NER tagger
# ------------------------------------------------------------------
try:
    tagger = SequenceTagger.load("ner-ontonotes-large-fast")
except Exception as e:
    sys.stderr.write(f"Cannot load OntoNotes-large-fast model: {e}\n")
    sys.exit(1)

# ------------------------------------------------------------------
# 3.  Helper functions
# ------------------------------------------------------------------
def build_background_prompt(entities: Dict[str, Set[str]]) -> str:
    """
    Build a short prompt for an image-generation LLM.
    We pick the top-weighted entities (up to 3) and weave them
    into a concise scene description.
    """
    # Flatten list with weights
    weighted = []
    for tag, values in entities.items():
        w = ENTITY_WEIGHTS.get(tag, 0)
        for v in values:
            weighted.append((w, v, tag))

    # Sort by weight desc, then keep top 3
    weighted.sort(reverse=True, key=lambda t: t[0])
    top = [v for _, v, _ in weighted[:3]]

    if not top:
        return "generic news studio background"

    phrase = ", ".join(top)
    return f"news report background scene featuring: {phrase}"

def format_entities(entities: Dict[str, Set[str]]) -> str:
    """Convert dict tag→{values} to <tag>v1,v2</tag>… string."""
    parts = []
    for tag in sorted(entities.keys()):
        xml_tag = TAG_TO_XML.get(tag)
        if not xml_tag or not entities[tag]:
            continue
        values = ", ".join(sorted(entities[tag]))
        parts.append(f"<{xml_tag}>{values}</{xml_tag}>")
    return "".join(parts)

# ------------------------------------------------------------------
# 4.  Main pipeline
# ------------------------------------------------------------------
def main() -> None:
    text = sys.stdin.read().strip()
    if not text:
        return

    # Flair expects Sentence objects; we let it split internally
    flair_sentence = Sentence(text, use_tokenizer=True)
    sentences = flair_sentence.split_by_tokenizer()  # list[Sentence]

    for sent in sentences:
        # Predict NER
        tagger.predict(sent)

        # Collect entities
        entities: Dict[str, Set[str]] = {k: set() for k in TAG_TO_XML.keys()}
        for span in sent.get_spans("ner"):
            tag = span.tag
            if tag in entities:
                entities[tag].add(span.text)

        # Build output sections
        background = build_background_prompt(entities)
        entities_xml = format_entities(entities)

        print(f"<text>{sent.to_original_text()}</text>")
        print(f"<background>{background}</background>")
        print(f"<entities>{entities_xml}</entities>")
        print()  # blank line between sentences

if __name__ == "__main__":
    main()
