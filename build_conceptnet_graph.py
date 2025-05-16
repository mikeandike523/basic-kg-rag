import os
import uuid
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
mysql_cfg   = dotenv_values("./mysql-dev-server/.env")
arangodb_cfg = dotenv_values("./arangodb-dev-server/.env")
qdrant_cfg  = dotenv_values("./qdrant-dev-server/.env")

# === MySQL connection settings ===
MYSQL_CONFIG = {
    "host":     mysql_cfg.get("HOST", "localhost"),
    "port":     int(mysql_cfg.get("PORT", 3306)),
    "user":     mysql_cfg["MYSQL_USER"],
    "password": mysql_cfg["MYSQL_PASSWORD"],
    "database": mysql_cfg["MYSQL_DATABASE"],
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
BATCH_SIZE   = 1024
VECTOR_SIZE  = 384  # SentenceTransformer output dim

@click.command()
@click.option(
    "--restart",
    is_flag=True,
    default=False,
    help="If set, drop & recreate ArangoDB & Qdrant collections before ingest.",
)
def main(restart):
    # --- Connect MySQL ---
    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
    mysql_cursor = mysql_conn.cursor()

    # --- Connect ArangoDB ---
    arango_client = ArangoClient(hosts=ARANGO_URL)
    sys_db = arango_client.db(
        "_system", username=ARANGO_USER, password=ARANGO_PASSWORD
    )
    if not sys_db.has_database(ARANGO_DB):
        sys_db.create_database(ARANGO_DB)
    arango_db = arango_client.db(
        ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASSWORD
    )

    # Collection names
    NODES_COL = "nodes"
    EDGES_COL = "edges"

    # (Re)create collections if requested
    if restart:
        if arango_db.has_collection(EDGES_COL):
            arango_db.delete_collection(EDGES_COL)
        if arango_db.has_collection(NODES_COL):
            arango_db.delete_collection(NODES_COL)

    if not arango_db.has_collection(NODES_COL):
        arango_db.create_collection(NODES_COL)
    if not arango_db.has_collection(EDGES_COL):
        arango_db.create_collection(EDGES_COL, edge=True)

    nodes_col = arango_db.collection(NODES_COL)
    edges_col = arango_db.collection(EDGES_COL)

    # --- Connect Qdrant ---
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if restart:
        qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=rest_models.Distance.COSINE
                ),
        )
    else:
        try:
            qdrant.get_collection(QDRANT_COLLECTION)
        except Exception:
            qdrant.create_collection(
               collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=rest_models.Distance.COSINE
                ),
            )
    # --- Sentence embedding model ---
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- 1. Build unique node set in memory ---
    starts=set()
    ends=set()
    # Select DISTINCT was failing for some reason. May be utf-8 or other encoding issue
    # So we compute uniqueness on python side even though VERY slow
    mysql_cursor.execute("SELECT start_node FROM conceptnet_en")
    print("Computing unique start nodes...")
    for r in mysql_cursor:
        starts.add(r[0])
    # Select DISTINCT was failing for some reason. May be utf-8 or other encoding issue
    # So we compute uniqueness on python side even though VERY slow
    mysql_cursor.execute("SELECT end_node FROM conceptnet_en")
    print("Computing unique end nodes...")
    for r in mysql_cursor:
        ends.add(r[0])
    unique_nodes = starts.union(ends)
    click.echo(f"Found {len(unique_nodes)} total unique nodes.")

    # --- 2. Insert nodes into ArangoDB, build name→_key map ---
    node_map = {}
    for name in tqdm(unique_nodes, desc="Inserting nodes"):
        node_key = str(uuid.uuid4())
        nodes_col.insert({"_key": node_key, "name": name})
        node_map[name] = node_key

    # --- 3. Process each edge: ArangoDB edge + Qdrant vector ---
    mysql_cursor.execute(
        "SELECT id, start_node, relation, end_node, weight, sentence FROM conceptnet_en"
    )
    points_buffer = []


    for record in tqdm(mysql_cursor, desc="Processing edges", total=mysql_cursor.rowcount):
        _, s, rel, e, weight, sentence = record

        # 3a) Create ArangoDB edge doc
        edge_key = str(uuid.uuid4())
        if s in node_map and e in node_map:


            edges_col.insert({
                "_key":     edge_key,
                "_from":    f"{NODES_COL}/{node_map[s]}",
                "_to":      f"{NODES_COL}/{node_map[e]}",
                "relation": rel,
                "weight":   weight,
                "sentence": sentence,
            })

        else:
            # Because of our prvious workaround, this should never occur
            # But better safe than sorry for long tasks
            print(f"s or e not in node_map: {s}, {e}")

        # 3b) Embed sentence and buffer for Qdrant
        vec = model.encode(sentence).tolist()
        points_buffer.append(
            rest_models.PointStruct(
                id=edge_key,
                vector=vec,
                payload={"arangodb_id": edge_key},
            )
        )

        # 3c) Flush buffer in batches
        if len(points_buffer) >= BATCH_SIZE:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points_buffer)
            points_buffer.clear()

    # Final flush
    if points_buffer:
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points_buffer)

    # --- Cleanup ---
    mysql_cursor.close()
    mysql_conn.close()
    click.echo("✅ Ingestion complete.")

if __name__ == "__main__":
    main()
