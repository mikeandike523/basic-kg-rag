import os
import json
from collections import Counter, defaultdict


from qdrant_client import QdrantClient
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


TOP_K = 100
CUTOFF_RELEVANCE = 50.0  # Exclude results below this percentage relevance
HIT_PERCENT_THRESHOLD = 2.5


# ─── CONFIG ────────────────────────────────────────────────────────────────────
repo_root = os.path.dirname(os.path.dirname(__file__))
config = dotenv_values(os.path.join(repo_root, "qdrant-dev-server", ".env"))

HOST = config.get("QDRANT_HOST", "localhost")
PORT = int(config.get("QDRANT_PORT", 6333))

COLLECTION_NAME = "flatland"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── LOAD EMBEDDING MODEL ──────────────────────────────────────────────────────
model = SentenceTransformer(EMBEDDING_MODEL)

# ─── SETUP QDRANT CLIENT ───────────────────────────────────────────────────────
client = QdrantClient(host=HOST, port=PORT)

# ─── LOAD TOPIC DATA ───────────────────────────────────────────────────────────
with open(os.path.join(repo_root, "story_learner/topicized_flatland_mistral_7b_instruct_quantized_8bit"), "r", encoding="utf-8") as f:
    all_topics = json.load(f)

# ─── USER QUERY ────────────────────────────────────────────────────────────────
query = input("Enter your query: ")
query_vector = model.encode(query).tolist()

# ─── QDRANT SEARCH ─────────────────────────────────────────────────────────────
search_results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=TOP_K,
    with_payload=True,
    with_vectors=False,
)

# ─── FILTER & GROUP BY TOPIC ───────────────────────────────────────────────────
filtered_results = []
topic_hit_counts = Counter()
topic_relevance_sum = defaultdict(float)

for result in search_results:
    relevance = ((result.score + 1) / 2) * 100
    if relevance >= CUTOFF_RELEVANCE:
        topic_idx = result.payload["topic_idx"]
        topic_hit_counts[topic_idx] += 1
        topic_relevance_sum[topic_idx] += relevance
        filtered_results.append((topic_idx, relevance))

total_filtered = sum(topic_hit_counts.values())

# ─── DISPLAY RESULTS ───────────────────────────────────────────────────────────
print(f"\n📊 Showing results with relevance ≥ {CUTOFF_RELEVANCE:.1f}%")
print(f"🔎 Total filtered hits: {total_filtered}/{TOP_K}\n")

for topic_idx, count in topic_hit_counts.most_common():
    topic_title = all_topics[topic_idx]["topic"]
    avg_relevance = topic_relevance_sum[topic_idx] / count
    percentage = (count / total_filtered) * 100 if total_filtered else 0
    print(f"🔹 {topic_title}")
    print(f"   • Hits: {count} ({percentage:.1f}%)")
    print(f"   • Avg Relevance: {avg_relevance:.1f}%\n")

# Count hits and sum relevance per topic index
topic_hit_counts = Counter()
topic_relevance_sum = defaultdict(float)

for topic_idx, relevance in filtered_results:
    topic_hit_counts[topic_idx] += 1
    topic_relevance_sum[topic_idx] += relevance

# Calculate average relevance per topic
topic_avg_relevance = {
    idx: topic_relevance_sum[idx] / count
    for idx, count in topic_hit_counts.items()
}

# Sort by topic index
sorted_indices = sorted(topic_hit_counts.keys())
hit_counts = [topic_hit_counts[idx] for idx in sorted_indices]
avg_relevances = [topic_avg_relevance[idx] for idx in sorted_indices]

# Compute percentage of hits per topic
hit_percentages = [(topic_hit_counts[idx] / total_filtered) * 100 for idx in sorted_indices]

# Plot percentage histogram
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_indices, hit_percentages)

plt.xlabel("Topic Index")
plt.ylabel("Hit Percentage (%)")
plt.title("Hit Percentage by Topic Index (Filtered by Relevance)")

# Annotate bars with average relevance
for bar, avg_rel in zip(bars, avg_relevances):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{avg_rel:.1f}%",
        ha='center',
        va='bottom',
        fontsize=8
    )

plt.tight_layout()
plt.show()

# ─── PRINT TOP TOPIC TITLE & SUMMARY ───────────────────────────────────────────
if topic_hit_counts:
    # Get the topic index with the most hits (first one if tied)
    top_topic_idx = topic_hit_counts.most_common(1)[0][0]
    top_topic = all_topics[top_topic_idx]

    print("\n🏆 Top Topic Based on Relevance Hits:")
    print(f"📌 Title: {top_topic['topic']}")
    print(f"📝 Summary: {top_topic['summary']}\n")
else:
    print("\n⚠️ No topics met the relevance threshold.")



topic_indices_above_hit_percent_threshold = []

for idx, percentage in enumerate(hit_percentages):
    if percentage >= HIT_PERCENT_THRESHOLD:
        topic_indices_above_hit_percent_threshold.append(idx)

print("Topics above hit percentage threshold:")

for index in topic_indices_above_hit_percent_threshold:
    topic = all_topics[index]
    print(f"�� Topic Index: {index}")
    print(f"�� Title: {topic['topic']}")
    print(f"�� Summary: {topic['summary']}\n")