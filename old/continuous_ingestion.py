import os
import json
import logging
import hashlib
import click
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, parse_obj_as
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from arango import ArangoClient
from dotenv import dotenv_values
from termcolor import colored
import uuid

# === Colored Logging Setup ===
class ColorFormatter(logging.Formatter):
    """
    Custom formatter to add color to log messages based on severity.
    """
    def format(self, record):
        message = super().format(record)
        if record.levelno == logging.DEBUG:
            return colored(message, 'blue')
        elif record.levelno == logging.INFO:
            return colored(message, 'green')
        elif record.levelno == logging.WARNING:
            return colored(message, 'yellow')
        elif record.levelno == logging.ERROR:
            return colored(message, 'red')
        elif record.levelno == logging.CRITICAL:
            return colored(message, 'magenta')
        return message

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(ColorFormatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# === Constants ===
SIMILARITY_DUPLICATE_THRESHOLD = 0.999  # Cosine similarity threshold to detect duplicates
QDRANT_COLLECTION = "edges"           # Qdrant collection name
VECTOR_SIZE = 384                      # Embedding vector dimension
DISTANCE_METRIC = Distance.COSINE      # Similarity metric

# === Load environment configurations ===
qdrant_cfg = dotenv_values("./qdrant-dev-server/.env")
QDRANT_HOST = qdrant_cfg.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(qdrant_cfg.get("QDRANT_PORT", 6333))

arangodb_cfg = dotenv_values("./arangodb-dev-server/.env")
ARANGO_URL = arangodb_cfg.get("ARANGO_URL", "http://localhost:8529")
ARANGO_USER = arangodb_cfg.get("ARANGO_USERNAME", "root")
ARANGO_PASSWORD = arangodb_cfg.get("ARANGO_ROOT_PASSWORD")
ARANGO_DB = arangodb_cfg.get("ARANGO_DB", "test")

# === Pydantic schema for input validation (JSON uses non-underscored fields) ===
class Fact(BaseModel):
    from_concept: str                      # Maps to ArangoDB '_from'
    to_concept: str                        # Maps to ArangoDB '_to'
    relation: str                          # Placeholder relation label
    sentence: str                          # Natural-language fact or explanation
    weight: float                          # Confidence/truth value (0.0-1.0)

    class Config:
        anystr_strip_whitespace = True
        extra = 'forbid'

# === Initialize clients and model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize or create Qdrant collection with correct parameters
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
try:
    qdrant.get_collection(collection_name=QDRANT_COLLECTION)
except Exception:
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC)
    )

# ArangoDB graph storage
arango = ArangoClient(hosts=ARANGO_URL)
db = arango.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASSWORD)
edges_col = db.collection("relations")

# === Helper functions ===
def is_duplicate(sentence: str, threshold: float = SIMILARITY_DUPLICATE_THRESHOLD) -> bool:
    """
    Returns True if an existing vector in Qdrant has cosine similarity >= threshold.
    """
    vector = model.encode(sentence).tolist()
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        limit=1,
        with_payload=False,
        with_vectors=False
    )
    if hits and hits[0].score >= threshold:
        logger.info(colored(f"Skipped duplicate (sim={hits[0].score:.4f}): {sentence}", 'yellow'))
        return True
    return False


def generate_key(sentence: str) -> str:
    # """
    # Generate a deterministic 12-character hex key via MD5 of the sentence.
    # """
    # return hashlib.md5(sentence.encode('utf-8')).hexdigest()[:12]
    return str(uuid.uuid4())


def ingest_fact(fact: dict):
    """
    Ingests a prepared Arango/Qdrant fact dict with keys '_from','_to','relation','sentence','weight','_key'.
    Performs duplicate check, inserts into ArangoDB, then upserts into Qdrant.
    """
    sentence = fact.get('sentence')
    if not sentence:
        logger.warning(colored("Missing 'sentence'; skipping.", 'yellow'))
        return

    if is_duplicate(sentence):
        return

    # Insert into ArangoDB
    try:
        edges_col.insert(fact, overwrite=False)
    except Exception as e:
        logger.error(colored(f"ArangoDB insert failed: {e}", 'red'))
        return

    # Encode vector and prepare Qdrant point
    vector = model.encode(sentence).tolist()
    point = PointStruct(id=fact['_key'], vector=vector, payload={"arangodb_id": fact['_key']})
    try:
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=[point])
    except Exception as e:
        logger.error(colored(f"Qdrant upsert failed: {e}", 'red'))
        return

    logger.info(colored(f"Inserted fact ID={fact['_key']}: {sentence}", 'green'))

# === CLI entrypoint ===
@click.command()
@click.option('--json_data', required=True, help='Path to JSON file with list of fact objects.')
def main(json_data):
    logger.info(colored(f"Loading JSON from: {json_data}", 'blue'))
    try:
        with open(json_data, 'r') as f:
            raw = json.load(f)
    except Exception as e:
        logger.error(colored(f"Failed to read JSON: {e}", 'red'))
        return

    # Ensure JSON is list of objects
    if not isinstance(raw, list):
        logger.error(colored("Input JSON must be a list of fact objects.", 'red'))
        return

    # Validate against Pydantic schema
    try:
        facts = parse_obj_as(List[Fact], raw)
    except ValidationError as e:
        logger.error(colored(f"Validation errors: {e}", 'red'))
        return

    logger.info(colored(f"Validated {len(facts)} facts, preparing ingestion...", 'green'))

    # Build and ingest each fact
    for fact in facts:
        db_fact = {
            '_from': "concepts/"+fact.from_concept,
            '_to': "concepts/"+fact.to_concept,
            'relation': fact.relation,
            'sentence': fact.sentence,
            'weight': fact.weight,
            '_key':  generate_key(fact.sentence)
        }
        ingest_fact(db_fact)

    logger.info(colored("All facts ingested.", 'green'))

if __name__ == '__main__':
    main()
