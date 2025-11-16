#!/usr/bin/env python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from flair.data import Sentence
from flair.nn import Classifier
from collections import Counter
from typing import List, Dict
import uvicorn
import logging
import sys
import time
import signal
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ner_service.log"),
    ],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Flair NER Microservice",
    description="Named Entity Recognition using Flair NLP",
    version="1.0.0",
)

# Configuration
MAX_TEXT_LENGTH = 50000  # Maximum characters to process
MAX_SENTENCE_LENGTH = 10000  # Flair sentence length limit
NER_TAGGER = "ner-ontonotes-fast"


# Statistics tracking
class ServiceStats:
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_entities_found = 0
        self.total_text_processed = 0
        self.total_processing_time = 0.0
        self.entity_type_counts = Counter()

    def add_request(
        self,
        success: bool,
        entities: List[Dict] = None,
        text_length: int = 0,
        processing_time: float = 0.0,
    ):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            if entities:
                self.total_entities_found += len(entities)
                for entity in entities:
                    self.entity_type_counts[entity["label"]] += 1
            self.total_text_processed += text_length
            self.total_processing_time += processing_time
        else:
            self.failed_requests += 1

    def print_report(self):
        uptime = time.time() - self.start_time
        uptime_str = time.strftime("%H:%M:%S", time.gmtime(uptime))

        print("\n" + "=" * 70)
        print("FLAIR NER SERVICE - SHUTDOWN STATISTICS REPORT")
        print("=" * 70)
        print(f"Shutdown Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Uptime:         {uptime_str}")
        print("-" * 70)
        print("REQUEST STATISTICS:")
        print(f"  Total Requests:     {self.total_requests}")
        print(f"  Successful:         {self.successful_requests}")
        print(f"  Failed:             {self.failed_requests}")
        if self.total_requests > 0:
            success_rate = (self.successful_requests / self.total_requests) * 100
            print(f"  Success Rate:       {success_rate:.2f}%")
        print("-" * 70)
        print("PROCESSING STATISTICS:")
        print(f"  Total Entities:     {self.total_entities_found}")
        print(f"  Text Processed:     {self.total_text_processed:,} characters")
        print(f"  Processing Time:    {self.total_processing_time:.2f} seconds")
        if self.successful_requests > 0:
            avg_time = self.total_processing_time / self.successful_requests
            avg_entities = self.total_entities_found / self.successful_requests
            print(f"  Avg Time/Request:   {avg_time:.4f} seconds")
            print(f"  Avg Entities/Req:   {avg_entities:.2f}")
        print("-" * 70)
        if self.entity_type_counts:
            print("ENTITY TYPE DISTRIBUTION:")
            for entity_type, count in self.entity_type_counts.most_common():
                print(f"  {entity_type:20s} {count:6,} entities")
        print("=" * 70)
        print()


# Global statistics instance
stats = ServiceStats()

# Load the NER model (this happens once at startup)
try:
    logger.info(f"Loading Flair NER model: {NER_TAGGER}...")
    tagger = Classifier.load(NER_TAGGER)
    logger.info(f"Flair NER model '{NER_TAGGER}' loaded successfully")
except Exception as e:
    logger.error(f"Failed to load NER model '{NER_TAGGER}': {e}")
    raise


class TextInput(BaseModel):
    text: str

    @validator("text")
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text input cannot be empty")
        return v

    @validator("text")
    def text_length_check(cls, v):
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text too long: {len(v)} characters (max: {MAX_TEXT_LENGTH})"
            )
        return v


class EntityInfo(BaseModel):
    text: str
    label: str


class NERResponse(BaseModel):
    entities: List[EntityInfo]
    entity_counts: Dict[str, int]
    total_entities: int
    processed_length: int
    truncated: bool
    elapsed_time_seconds: float


@app.get("/")
async def root():
    return {
        "message": "Flair NER Microservice",
        "model": NER_TAGGER,
        "endpoints": {
            "/extract": "POST - Extract named entities from text",
            "/health": "GET - Health check",
            "/stats": "GET - Current statistics",
        },
        "config": {
            "max_text_length": MAX_TEXT_LENGTH,
            "max_sentence_length": MAX_SENTENCE_LENGTH,
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": NER_TAGGER,
        "model_loaded": tagger is not None,
        "uptime_seconds": round(time.time() - stats.start_time, 2),
    }


@app.get("/stats")
async def get_stats():
    """Get current service statistics"""
    uptime = time.time() - stats.start_time
    return {
        "uptime_seconds": round(uptime, 2),
        "total_requests": stats.total_requests,
        "successful_requests": stats.successful_requests,
        "failed_requests": stats.failed_requests,
        "total_entities_found": stats.total_entities_found,
        "total_text_processed": stats.total_text_processed,
        "total_processing_time": round(stats.total_processing_time, 4),
        "entity_type_counts": dict(stats.entity_type_counts),
    }


@app.post("/extract", response_model=NERResponse)
async def extract_entities(input_data: TextInput):
    """
    Extract named entities from the provided text.

    Returns:
    - entities: List of all entities found with their labels
    - entity_counts: Count of each unique entity
    - total_entities: Total number of entities found
    - processed_length: Actual length of text processed
    - truncated: Whether text was truncated
    - elapsed_time_seconds: Time taken to process the request
    """
    # Start high-resolution timer
    start_time = time.perf_counter()

    try:
        text = input_data.text.strip()
        original_length = len(text)
        truncated = False

        # Truncate if necessary (safety check even with validator)
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(
                f"Text length {len(text)} exceeds limit, truncating to {MAX_TEXT_LENGTH}"
            )
            text = text[:MAX_TEXT_LENGTH]
            truncated = True

        logger.info(f"Processing text of length: {len(text)}")

        # Split into smaller chunks if text is very long to avoid Flair memory issues
        entities = []
        entity_texts = []

        if len(text) > MAX_SENTENCE_LENGTH:
            # Process in chunks
            chunks = []
            chunk_size = MAX_SENTENCE_LENGTH
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                chunks.append(chunk)

            logger.info(f"Processing {len(chunks)} chunks")

            for idx, chunk in enumerate(chunks):
                try:
                    chunk_start = time.perf_counter()
                    sentence = Sentence(chunk)
                    tagger.predict(sentence)
                    chunk_elapsed = time.perf_counter() - chunk_start
                    logger.info(
                        f"Chunk {idx + 1}/{len(chunks)} processed in {chunk_elapsed:.4f} seconds"
                    )

                    for entity in sentence.get_spans("ner"):
                        entity_info = {
                            "text": entity.text,
                            "label": entity.get_label("ner").value,
                        }
                        entities.append(entity_info)
                        entity_texts.append(entity.text)

                except Exception as e:
                    logger.error(
                        f"Error processing chunk {idx + 1}/{len(chunks)}: {str(e)}"
                    )
                    # Continue processing other chunks
                    continue
        else:
            # Process as single text
            try:
                sentence = Sentence(text)
                tagger.predict(sentence)

                for entity in sentence.get_spans("ner"):
                    entity_info = {
                        "text": entity.text,
                        "label": entity.get_label("ner").value,
                    }
                    entities.append(entity_info)
                    entity_texts.append(entity.text)

            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                stats.add_request(success=False)
                raise HTTPException(
                    status_code=500, detail=f"Error processing text: {str(e)}"
                )

        # Count occurrences
        entity_counts = dict(Counter(entity_texts))

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - start_time

        logger.info(f"Found {len(entities)} entities in {elapsed_time:.4f} seconds")

        # Update statistics
        stats.add_request(
            success=True,
            entities=entities,
            text_length=len(text),
            processing_time=elapsed_time,
        )

        return NERResponse(
            entities=entities,
            entity_counts=entity_counts,
            total_entities=len(entities),
            processed_length=len(text),
            truncated=truncated,
            elapsed_time_seconds=round(elapsed_time, 6),
        )

    except ValueError as e:
        # Validation errors
        elapsed_time = time.perf_counter() - start_time
        logger.warning(f"Validation error: {str(e)} (after {elapsed_time:.4f} seconds)")
        stats.add_request(success=False)
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        # Re-raise HTTP exceptions
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"HTTP exception raised after {elapsed_time:.4f} seconds")
        stats.add_request(success=False)
        raise

    except Exception as e:
        # Catch-all for unexpected errors
        elapsed_time = time.perf_counter() - start_time
        logger.error(
            f"Unexpected error processing text: {str(e)} (after {elapsed_time:.4f} seconds)",
            exc_info=True,
        )
        stats.add_request(success=False)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("\nReceived shutdown signal (Ctrl-C). Shutting down gracefully...")
    stats.print_report()
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting Flair NER Service...")
    logger.info(f"Using NER model: {NER_TAGGER}")
    logger.info("Press Ctrl-C to shutdown and view statistics")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info", access_log=True)
    except KeyboardInterrupt:
        # This won't normally be reached due to signal handler,
        # but kept as a fallback
        logger.info("\nShutdown requested...")
        stats.print_report()
    finally:
        logger.info("Service stopped.")
