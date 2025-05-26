import os
import json

from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

# ─── CONFIG ────────────────────────────────────────────────────────────────────
repo_root = os.path.dirname(os.path.dirname(__file__))
# loads QDRANT_PORT (default 6333) and optionally QDRANT_HOST
config = dotenv_values(os.path.join(repo_root, "qdrant-dev-server", ".env"))

HOST = config.get("QDRANT_HOST", "localhost")
PORT = int(config.get("QDRANT_PORT", 6333))

COLLECTION_NAME = "flatland"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384
BATCH_SIZE = 100

# ─── LOAD TOPICS ───────────────────────────────────────────────────────────────
with open(os.path.join(repo_root, "story_learner/topicized_flatland_mistral_7b_instruct_quantized_8bit"), "r", encoding="utf-8") as f:
    all_topics = json.load(f)

# ─── SETUP QDRANT CLIENT ───────────────────────────────────────────────────────
client = QdrantClient(host=HOST, port=PORT)

# Delete & recreate the collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)

# ─── LOAD EMBEDDING MODEL ──────────────────────────────────────────────────────
model = SentenceTransformer(EMBEDDING_MODEL)

# ─── ENCODE & UPSERT ──────────────────────────────────────────────────────────
points_batch = []
for topic_idx, topic_data in enumerate(all_topics):
    paragraphs = topic_data["paragraphs"]
    # get embeddings for all paragraphs in this topic
    embeddings = model.encode(paragraphs, show_progress_bar=True)

    for para_idx, embedding in enumerate(embeddings):
        payload = {
            "topic_idx": topic_idx,
            "paragraph_idx": para_idx
        }
        # you can also store the paragraph text itself if you want:
        # payload["text"] = paragraphs[para_idx]

        point = PointStruct(
            id=str(uuid.uuid4()),  # let Qdrant auto-assign an ID
            vector=embedding.tolist(),
            payload=payload
        )
        points_batch.append(point)

        # flush every BATCH_SIZE points
        if len(points_batch) >= BATCH_SIZE:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch
            )
            points_batch = []

# upsert any remaining points
if points_batch:
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points_batch
    )

print(f"✅ Collection '{COLLECTION_NAME}' loaded with {sum(len(t['paragraphs']) for t in all_topics)} vectors.")
