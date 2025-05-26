import os
import pickle
import uuid
import hashlib
import mysql.connector
from dotenv import dotenv_values
from arango import ArangoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer
import click
from tqdm import tqdm
from qdrant_client.http.models import VectorParams

# === Load configs from separate .env files ===
mysql_cfg    = dotenv_values("./mysql-dev-server/.env")
arangodb_cfg = dotenv_values("./arangodb-dev-server/.env")
qdrant_cfg   = dotenv_values("./qdrant-dev-server/.env")

# === MySQL connection settings ===
MYSQL_CONFIG = {
    "host":             mysql_cfg.get("HOST", "localhost"),
    "port":             int(mysql_cfg.get("PORT", 3306)),
    "user":             mysql_cfg["MYSQL_USER"],
    "password":         mysql_cfg["MYSQL_PASSWORD"],
    "database":         mysql_cfg["MYSQL_DATABASE"],
    "connection_timeout": 86400,
    "client_flags":     [mysql.connector.ClientFlag.LONG_FLAG],
    "pool_name":        "kg_pool",
    "pool_size":        5,
}

# === ArangoDB connection settings ===
ARANGO_URL      = arangodb_cfg.get("ARANGO_URL", "http://localhost:8529")
ARANGO_USER     = arangodb_cfg.get("ARANGO_USERNAME", "root")
ARANGO_PASSWORD = arangodb_cfg["ARANGO_ROOT_PASSWORD"]
ARANGO_DB       = arangodb_cfg.get("ARANGO_DB", "test")

# === Qdrant connection settings ===
QDRANT_HOST       = qdrant_cfg.get("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(qdrant_cfg.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = qdrant_cfg.get("QDRANT_COLLECTION", "edges")

# === Other constants ===
BATCH_SIZE            = 1024
VECTOR_SIZE           = 384  # SentenceTransformer output dim
NODES_CHECKPOINT_FILE = "nodes_inserted.chk"
EDGES_CHECKPOINT_FILE = "edges_checkpoint.txt"

def deterministic_edge_key(start, relation, end):
    """Generate a repeatable key for an edge."""
    return hashlib.md5(f"{start}-{relation}-{end}".encode()).hexdigest()

def load_edge_checkpoint():
    if os.path.exists(EDGES_CHECKPOINT_FILE):
        with open(EDGES_CHECKPOINT_FILE) as f:
            return int(f.read().strip())
    return 0

def save_edge_checkpoint(edge_id):
    with open(EDGES_CHECKPOINT_FILE, "w") as f:
        f.write(str(edge_id))

@click.command()
@click.option(
    "--clear",
    is_flag=True,
    default=False,
    help="Clear All Data."
)
@click.option(
    "--nodes",
    is_flag=True,
    default=False,
    help="If set, run only the node insertion step."
)
@click.option(
    "--edges",
    is_flag=True,
    default=False,
    help="If set, run only the edge processing step."
)
def main(clear, nodes, edges):
    # --- Connect MySQL ---
    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
    mysql_cursor = mysql_conn.cursor()

    # --- Connect ArangoDB ---
    arango_client = ArangoClient(hosts=ARANGO_URL)
    sys_db = arango_client.db("_system", username=ARANGO_USER, password=ARANGO_PASSWORD)
    if not sys_db.has_database(ARANGO_DB):
        sys_db.create_database(ARANGO_DB)
    arango_db = arango_client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASSWORD)

    # Collection names
    NODES_COL = "concepts"
    EDGES_COL = "relations"

    # --- Optional restart: drop collections & checkpoints ---
    if clear:
        if arango_db.has_collection(EDGES_COL):
            arango_db.delete_collection(EDGES_COL)
        if arango_db.has_collection(NODES_COL):
            arango_db.delete_collection(NODES_COL)
        # Delete checkpoint files
        if os.path.exists(NODES_CHECKPOINT_FILE):
            os.remove(NODES_CHECKPOINT_FILE)
        if os.path.exists(EDGES_CHECKPOINT_FILE):
            os.remove(EDGES_CHECKPOINT_FILE)

        if os.path.exists("data/edge_ids.pkl"):
            os.remove("data/edge_ids.pkl")

        print("All arango and checkpoint data and id map data has been cleared.")



    # --- Connect Qdrant ---
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if clear:
        qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=rest_models.Distance.COSINE),
        )
        print("Qdrant data cleared.")
        exit("All data cleared")
    else:
        try:
            qdrant.get_collection(QDRANT_COLLECTION)
        except Exception:
            qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=rest_models.Distance.COSINE),
            )



    # Ensure ArangoDB collections exist
    if not arango_db.has_collection(NODES_COL):
        arango_db.create_collection(NODES_COL)
    if not arango_db.has_collection(EDGES_COL):
        arango_db.create_collection(EDGES_COL, edge=True)

    nodes_col = arango_db.collection(NODES_COL)
    edges_col = arango_db.collection(EDGES_COL)

    # --- Sentence embedding model ---
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Determine which steps to run: default to both if neither flag is set
    run_nodes = nodes or (not nodes and not edges)
    run_edges = edges or (not nodes and not edges)

    # Build unique node set in memory
    starts, ends = set(), set()
    mysql_cursor.execute("SELECT start_node FROM conceptnet_en")
    print("Computing unique start nodes...")
    for (s,) in mysql_cursor:
        starts.add(s)
    mysql_cursor.execute("SELECT end_node FROM conceptnet_en")
    print("Computing unique end nodes...")

    for (e,) in mysql_cursor:
        ends.add(e)
    unique_nodes = starts.union(ends)
    print(f"Found {len(unique_nodes)} total unique nodes.")

    # Insert nodes into ArangoDB
    node_map = {}

    if not os.path.exists("data/edge_ids.pkl"):
        for name in unique_nodes:
            node_key = str(uuid.uuid4())
            node_map[name] = node_key
        with open("data/edge_ids.pkl", "wb") as f:
            pickle.dump(node_map, f)
    else:
        with open("data/edge_ids.pkl", "rb") as f:
           node_map = pickle.load(f)
    
    # === 1. Node insertion step ===
    if run_nodes:
        node_docs = []

        for name in unique_nodes:
            
            node_docs.append({"_key": node_map[name], "name": name})

        if os.path.exists(NODES_CHECKPOINT_FILE):
            print("✅ Nodes already inserted; skipping node step.")
        else:

            print("Inserting nodes in bulk...")
            nodes_col.import_bulk(node_docs)

            # Write checkpoint
            with open(NODES_CHECKPOINT_FILE, "w") as f:
                f.write("done")
            print("✅ Node insertion complete.")

    # === 2. Edge processing step ===
    if run_edges:
        # Load node_map (either just built, or from existing ArangoDB)
        # if 'node_map' not in locals():
        #     node_map = {doc["name"]: doc["_key"] for doc in nodes_col.all()}

        last_edge_id = load_edge_checkpoint()

        # Count total edges
        mysql_cursor.execute("SELECT COUNT(*) FROM conceptnet_en")
        total_edges = mysql_cursor.fetchone()[0]

        # Fetch ordered by id for resumability
        mysql_cursor.execute(
            "SELECT id, start_node, relation, end_node, weight, sentence "
            "FROM conceptnet_en ORDER BY id"
        )

        edge_docs = []
        points_buffer = []

        print(f"Resuming edge processing from ID > {last_edge_id}...")
        for record in tqdm(mysql_cursor, desc="Processing edges", total=total_edges):
            edge_id, s, rel, e, weight, sentence = record
            if edge_id <= last_edge_id:
                continue

            if s not in node_map or e not in node_map:
                print(f"Missing node: {s} or {e}; skipping edge {edge_id}.")
                save_edge_checkpoint(edge_id)
                continue

            # Deterministic key
            # edge_key = deterministic_edge_key(s, rel, e)

            edge_key = str(uuid.uuid4())

            # 3a) ArangoDB edge doc
            edge_docs.append({
                "_key": edge_key,
                "_from": f"{NODES_COL}/{node_map[s]}",
                "_to":   f"{NODES_COL}/{node_map[e]}",
                "relation": rel,
                "weight":   weight,
                "sentence": sentence,
            })

            # 3b) Sentence embedding
            if sentence:
                vec = model.encode(sentence, batch_size=32).tolist()
                points_buffer.append(
                    rest_models.PointStruct(
                        id=edge_key,
                        vector=vec,
                        payload={"arangodb_id": edge_key},
                    )
                )

            # 3c) Batch write + checkpoint update
            if len(edge_docs) >= BATCH_SIZE:
                edges_col.import_bulk(edge_docs)
                qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points_buffer)
                save_edge_checkpoint(edge_id)
                edge_docs.clear()
                points_buffer.clear()

        # Final flush
        if edge_docs:
            edges_col.import_bulk(edge_docs)
        if points_buffer:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points_buffer)
            # Edge_id still holds the last record's ID
            save_edge_checkpoint(edge_id)

        print("✅ Edge processing complete.")

    # --- Cleanup ---
    mysql_cursor.close()
    mysql_conn.close()


if __name__ == "__main__":
    main()
