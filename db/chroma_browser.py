#!/usr/bin/env python3
"""
chroma_browser.py
Browse ChromaDB collections and MongoDB articles using ncurses
"""

import os
import curses
from datetime import datetime
from typing import List, Dict, Tuple

import pymongo
from bson import ObjectId
import chromadb
from chromadb.config import Settings

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"

CHROMA_PATH = "./chroma"

# ------------------------------------------------------------------
# Database connections
# ------------------------------------------------------------------
mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_coll = mongo_client[MONGO_DB][MONGO_COLL]

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
)


# ------------------------------------------------------------------
# Data functions
# ------------------------------------------------------------------
def get_collections() -> List[str]:
    """Get list of ChromaDB collections"""
    collections = chroma_client.list_collections()
    return [c.name for c in collections]


def get_collection_records(collection_name: str) -> List[Dict]:
    """Get all records from a ChromaDB collection"""
    collection = chroma_client.get_collection(collection_name)
    result = collection.get(
        limit=100000, include=["documents", "metadatas"]
    )  # Get all records

    records = []
    for i in range(len(result["ids"])):
        records.append(
            {
                "id": result["ids"][i],
                "document": result["documents"][i] if result["documents"] else "",
                "metadata": result["metadatas"][i] if result["metadatas"] else {},
            }
        )

    return records


def get_article_from_mongo(article_id: str) -> Dict:
    """Fetch article details from MongoDB (for entity info only)"""
    try:
        doc = mongo_coll.find_one({"_id": ObjectId(article_id)})
        if not doc:
            return None

        return {
            "id": str(doc["_id"]),
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "published": doc.get("published", ""),
            "entities": doc.get("ner", {}).get("entities", []),
        }
    except Exception as e:
        return None


def format_entities(entities: List[Dict]) -> str:
    """Format entities for display"""
    if not entities:
        return "(no entities)"

    by_type = {}
    for entity in entities:
        label = entity.get("label", "")
        text = entity.get("text", "")
        if label not in by_type:
            by_type[label] = []
        if text not in by_type[label]:
            by_type[label].append(text)

    lines = []
    for label in sorted(by_type.keys()):
        entity_list = sorted(by_type[label])
        lines.append(f"{label}: {', '.join(entity_list)}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# UI functions
# ------------------------------------------------------------------
def wrap_text(text: str, width: int) -> List[str]:
    """Wrap text to fit within specified width"""
    if not text:
        return [""]

    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split()
        current_line = []
        current_length = 0

        for word in words:
            word_len = len(word)
            if current_length + word_len + len(current_line) <= width:
                current_line.append(word)
                current_length += word_len
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_len

        if current_line:
            lines.append(" ".join(current_line))

    return lines


def draw_article(stdscr, record: Dict, mongo_data: Dict, scroll_offset: int):
    """Draw article details on screen"""
    height, width = stdscr.getmaxyx()

    try:
        stdscr.clear()

        if record is None:
            stdscr.addstr(2, 2, "Record not found", curses.A_BOLD)
            stdscr.refresh()
            return

        # Prepare content
        lines = []
        lines.append(f"ID: {record['id']}")
        lines.append("")

        # Add metadata if present
        if record.get("metadata"):
            lines.append("Metadata:")
            for key, value in record["metadata"].items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Add MongoDB info if available
        if mongo_data:
            lines.append(f"Title: {mongo_data.get('title', 'N/A')}")
            lines.append("")

            published = mongo_data.get("published", "")
            if isinstance(published, datetime):
                published = published.isoformat()
            lines.append(f"Published: {published}")
            lines.append(f"Source: {mongo_data.get('source', 'N/A')}")
            lines.append("")

            # Entities
            entities_str = format_entities(mongo_data.get("entities", []))
            if entities_str != "(no entities)":
                lines.append("Entities:")
                for entity_line in entities_str.split("\n"):
                    lines.append(f"  {entity_line}")
                lines.append("")

        # Document text from ChromaDB
        lines.append("Document:")
        lines.append("-" * (width - 4))
        document_text = record.get("document", "")
        wrapped = wrap_text(document_text, width - 4)
        lines.extend(wrapped)

        # Draw visible portion
        visible_height = height - 3
        visible_lines = lines[scroll_offset : scroll_offset + visible_height]

        for i, line in enumerate(visible_lines):
            if i + 1 < height - 1:
                try:
                    stdscr.addstr(i + 1, 2, line[: width - 4])
                except curses.error:
                    pass

        # Status bar
        status = f"Scroll: {scroll_offset}/{max(0, len(lines) - visible_height)} | j/k: scroll | ↑/↓: record | ←/→: collection | q: quit"
        try:
            stdscr.addstr(height - 1, 0, status[:width], curses.A_REVERSE)
        except curses.error:
            pass

        stdscr.refresh()

        return len(lines)
    except Exception as e:
        stdscr.addstr(0, 0, f"Display error: {str(e)}")
        stdscr.refresh()
        return 0


def main_loop(stdscr):
    """Main browser loop"""
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()

    # Get collections
    collections = get_collections()
    if not collections:
        stdscr.addstr(0, 0, "No ChromaDB collections found!")
        stdscr.refresh()
        stdscr.getch()
        return

    coll_idx = 0
    record_idx = 0
    scroll_offset = 0
    records = []
    current_record = None
    current_mongo = None
    total_lines = 0

    # Load first collection
    records = get_collection_records(collections[coll_idx])
    if records:
        current_record = records[record_idx]
        current_mongo = get_article_from_mongo(current_record["id"])

    while True:
        # Draw header
        height, width = stdscr.getmaxyx()
        header = f"Collection: {collections[coll_idx]} ({coll_idx + 1}/{len(collections)}) | Record: {record_idx + 1}/{len(records)}"
        try:
            stdscr.addstr(0, 0, header[:width], curses.A_REVERSE | curses.A_BOLD)
        except curses.error:
            pass

        # Draw article
        total_lines = draw_article(stdscr, current_record, current_mongo, scroll_offset)

        # Get input
        key = stdscr.getch()

        if key == ord("q") or key == ord("Q"):
            break

        elif key == curses.KEY_LEFT:
            # Previous collection (wrap)
            coll_idx = (coll_idx - 1) % len(collections)
            records = get_collection_records(collections[coll_idx])
            record_idx = 0
            scroll_offset = 0
            if records:
                current_record = records[record_idx]
                current_mongo = get_article_from_mongo(current_record["id"])

        elif key == curses.KEY_RIGHT:
            # Next collection (wrap)
            coll_idx = (coll_idx + 1) % len(collections)
            records = get_collection_records(collections[coll_idx])
            record_idx = 0
            scroll_offset = 0
            if records:
                current_record = records[record_idx]
                current_mongo = get_article_from_mongo(current_record["id"])

        elif key == curses.KEY_UP:
            # Previous record (wrap)
            if records:
                record_idx = (record_idx - 1) % len(records)
                scroll_offset = 0
                current_record = records[record_idx]
                current_mongo = get_article_from_mongo(current_record["id"])

        elif key == curses.KEY_DOWN:
            # Next record (wrap)
            if records:
                record_idx = (record_idx + 1) % len(records)
                scroll_offset = 0
                current_record = records[record_idx]
                current_mongo = get_article_from_mongo(current_record["id"])

        elif key == ord("j") or key == ord("J"):
            # Scroll down
            max_scroll = max(0, total_lines - (height - 3))
            scroll_offset = min(scroll_offset + 1, max_scroll)

        elif key == ord("k") or key == ord("K"):
            # Scroll up
            scroll_offset = max(0, scroll_offset - 1)

        elif key == curses.KEY_NPAGE:  # Page Down
            scroll_offset = min(
                scroll_offset + (height - 3), max(0, total_lines - (height - 3))
            )

        elif key == curses.KEY_PPAGE:  # Page Up
            scroll_offset = max(0, scroll_offset - (height - 3))


def main():
    """Entry point"""
    try:
        curses.wrapper(main_loop)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
