import os
import time
import logging
import click
import random
import numpy as np
from dotenv import dotenv_values
from arango import ArangoClient
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from termcolor import colored
from tqdm import tqdm
import hashlib

# === Tunable constants ===
TOP_K = 20                    # Number of nearest neighbors to retrieve
MAX_EDGES_PER_START = 75     # Max edges to collect per initial end node
N_EDGES_PER_PROP = 5        # Number of edges to sample each propagation step
MAX_DEPTH = 50              # Max propagation fronts (depth)
CUTOFF_RELEVANCY = 0.5   
CUTOFF_TRUTH = 0.75 # 
HASH_HEX_CHARS = 24

# === Configure logging with color support ===
class ColorFormatter(logging.Formatter):
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# === Load configs ===
qdrant_cfg = dotenv_values("./qdrant-dev-server/.env")
QDRANT_HOST = qdrant_cfg.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(qdrant_cfg.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = qdrant_cfg.get("QDRANT_COLLECTION", "edges")

arangodb_cfg = dotenv_values("./arangodb-dev-server/.env")
ARANGO_URL = arangodb_cfg.get("ARANGO_URL", "http://localhost:8529")
ARANGO_USER = arangodb_cfg.get("ARANGO_USERNAME", "root")
ARANGO_PASSWORD = arangodb_cfg.get("ARANGO_ROOT_PASSWORD")
ARANGO_DB = arangodb_cfg.get("ARANGO_DB", "test")

# === Initialize clients and model ===
arango_client = ArangoClient(hosts=ARANGO_URL)
arango_db = arango_client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASSWORD)
edges_col = arango_db.collection("relations")
nodes_col = arango_db.collection("concepts")

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Helper: Node name cache ===
node_cache = {}
def get_node_name(handle: str) -> str:
    if handle in node_cache:
        return node_cache[handle]
    try:
        _, key = handle.split('/', 1)
        doc = nodes_col.get(key)
        name = doc.get('name', '<unknown>') if doc else '<missing>'
    except Exception as e:
        logger.error(f"Error fetching node metadata for {handle}: {e}")
        name = '<error>'
    node_cache[handle] = name
    return name

# === Modular functions ===
def search_query(query: str) -> (list, np.ndarray):
    """
    Perform Qdrant vector search and return top-K edge IDs plus the encoded query vector.
    """
    logger.info(f"Searching Qdrant for query: '{query}'")
    start = time.time()
    query_vector = model.encode(query)
    logger.debug(f"Query encoded in {time.time() - start:.2f}s")

    start = time.time()
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector.tolist(),
        limit=TOP_K,
        with_payload=False,
        with_vectors=False
    )
    logger.info(f"Retrieved {len(hits)} hits in {time.time() - start:.2f}s")
    return [hit.id for hit in hits], query_vector


def collect_cascade(seed_id: str, query_vector: np.ndarray) -> list:
    """
    For a starting edge ID, perform a weighted BFS cascade with weighted sampling by (cosine similarity^2 * truth value).
    """
    seed_doc = edges_col.get(seed_id)
    if not seed_doc:
        logger.warning(f"Seed edge {seed_id} not found; skipping.")
        return []

    results = []
    collected_keys = {seed_doc['_key']}
    frontier = [seed_doc['_to']]
    depth = 0
    q_norm = np.linalg.norm(query_vector)

    # Seed line
    seed_pct = int(seed_doc['weight'] * 100)
    seed_sentence = seed_doc['sentence']
    sv = model.encode(seed_sentence)
    cos_sim = np.dot(query_vector, sv) / (q_norm * np.linalg.norm(sv)) if q_norm else 0.0
    line = f"{seed_sentence}\t({seed_pct}% True)\t({int((cos_sim**2)*100)}% Relevant)"
    results.append((seed_doc['_key'], line))
    logger.debug(f"Seed line: {line}")

    # Propagation
    while frontier and len(results) < MAX_EDGES_PER_START and depth < MAX_DEPTH:
        depth += 1
        logger.info(f"Cascade depth {depth}: {len(frontier)} nodes")
        potential = []
        for node in frontier:
            edges = list(arango_db.aql.execute(
                "FOR e IN relations FILTER e._from == @node RETURN e", bind_vars={"node": node}
            ))
            logger.debug(f"Node {get_node_name(node)}: {len(edges)} outgoing edges")
            potential.extend(edges)

        if not potential:
            logger.info("No further edges to propagate.")
            break

        # Compute scores = (cos_sim^2) * weight
        scores = []
        for e in potential:
            sv = model.encode(e['sentence'])
            sim = (np.dot(query_vector, sv) / (q_norm * np.linalg.norm(sv))) if q_norm else 0.0
            sim_sq = sim * sim
            scores.append(sim_sq * e['weight'])
        scores = np.array(scores, dtype=float)
        if not np.any(scores > 0):
            logger.warning("All candidate scores are zero; stopping propagation.")
            break
        probs = scores / scores.sum()

        remaining = MAX_EDGES_PER_START - len(results)
        sample_count = min(N_EDGES_PER_PROP, len(potential), remaining)
        idxs = np.random.choice(len(potential), size=sample_count, replace=False, p=probs)
        sampled = [potential[i] for i in idxs]

        next_frontier = []
        for e in sampled:
            if e['_key'] in collected_keys:
                continue
            collected_keys.add(e['_key'])
            pct = int(e['weight'] * 100)
            sentence = e['sentence']
            sv = model.encode(sentence)
            cos_sim_e = (np.dot(query_vector, sv) / (q_norm * np.linalg.norm(sv))) if q_norm else 0.0
            relevance_pct= int((cos_sim_e**2)*100)
            line = f"{sentence}\t({pct}% True)\t({relevance_pct}% Relevant)"
            if cos_sim_e  * cos_sim_e >= CUTOFF_RELEVANCY and e['weight'] >= CUTOFF_TRUTH:
                results.append((e['_key'], line))
            logger.debug(f"Propagated line: {line}")
            # Probabilistic propagation based on squared similarity
            rand_val = random.random()
            prob_continue = cos_sim_e * cos_sim_e * e['weight']
            if rand_val <= prob_continue:
                next_frontier.append(e['_to'])
                logger.debug(f"Continue propagation (rand={rand_val:.2f} <= sim^2={prob_continue:.2f}) for edge {e['_key']}")
            else:
                logger.info(f"Stop propagation (rand={rand_val:.2f} > sim^2={prob_continue:.2f}) at edge {e['_key']}")

        frontier = list(set(next_frontier))

    return results

@click.command()
@click.option("--query", required=True, help="Query text for semantic search.")
@click.option("--output-file", "-o", default="relations.txt", help="File to write results.")
def main(query, output_file):
    logger.info("Starting inference pipeline...")

    seed_ids, query_vector = search_query(query)

    all_results = []
    for seed_id in tqdm(seed_ids, desc="Processing seeds"):
        logger.info(f"Processing seed edge {seed_id}")
        all_results.extend(collect_cascade(seed_id, query_vector))


    seen_hashes = set()
    unique_result_lines = []

    for _, line in all_results:
        line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()[:HASH_HEX_CHARS]
        if line_hash not in seen_hashes:
            seen_hashes.add(line_hash)
            unique_result_lines.append(line)

    logger.info(f"Writing {len(unique_result_lines)} lines to {output_file}")
    with open(output_file, 'w') as f:
        for line in unique_result_lines:
            f.write(line + "\n")
    logger.info("Output write complete.")
    logger.info("Inference complete.")

if __name__ == '__main__':
    main()
