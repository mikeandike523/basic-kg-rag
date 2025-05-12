import os
import mysql.connector
from tqdm import tqdm
from dotenv import dotenv_values

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# === Load environment variables ===
values = dotenv_values("mysql-dev-server/.env")
PORT = values["PORT"]
MYSQL_ROOT_PASSWORD = values["MYSQL_ROOT_PASSWORD"]
MYSQL_DATABASE = values["MYSQL_DATABASE"]
MYSQL_USER = values["MYSQL_USER"]
MYSQL_PASSWORD = values["MYSQL_PASSWORD"]

# === Load FLAN-T5 ===
print("Loading FLAN-T5 model...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

def generate_sentence(start, start_pos, relation, end, end_pos):
    def normalize(pos):
        if pos.lower() == "adjective satellite":
            return "adjective"
        return pos.lower() if pos else "any"

    start_pos = normalize(start_pos)
    end_pos = normalize(end_pos)

    prompt = f"Convert this structured relation into a sentence:\n{start} ({start_pos}) -> {relation} -> {end} ({end_pos})"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=64)
    sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sentence

# === MySQL Connection ===
print("Connecting to MySQL...")
conn = mysql.connector.connect(
    host="localhost",
    port=PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DATABASE
)
cursor = conn.cursor()

# === Create or Recreate Table ===
cursor.execute("DROP TABLE IF EXISTS conceptnet_en")

cursor.execute("""
CREATE TABLE conceptnet_en (
    id INT AUTO_INCREMENT PRIMARY KEY,
    start_node VARCHAR(255),
    start_pos ENUM('noun', 'verb', 'adjective', 'adverb', 'any'),
    relation VARCHAR(100),
    end_node VARCHAR(255),
    end_pos ENUM('noun', 'verb', 'adjective', 'adverb', 'any'),
    weight FLOAT DEFAULT 1.0,
    sentence TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# === Read File ===
data_file = "data/conceptnet-data-en-formatted.tsv"
valid_pos = {'noun', 'verb', 'adjective', 'adverb', 'any'}

def count_lines(file_path):
    line_count = 0
    total_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc='Counting lines') as pbar:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            line_count += chunk.count(b'\n')
            pbar.update(len(chunk))
    return line_count

print(f"Data file: {data_file}")
print("Counting lines...")
line_count = count_lines(data_file)

# === Process and Insert ===
with open(data_file, 'r', encoding='utf-8') as f, tqdm(total=line_count, desc="Processing lines") as pbar:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 6:
            pbar.update(1)
            continue  # skip malformed lines

        start, start_pos, relation, end, end_pos, weight_str = parts
        start_pos = start_pos.strip().lower().replace("adjective satellite", "adjective")
        end_pos = end_pos.strip().lower().replace("adjective satellite", "adjective")

        if start_pos not in valid_pos:
            start_pos = "any"
        if end_pos not in valid_pos:
            end_pos = "any"

        try:
            weight = float(weight_str)
        except ValueError:
            weight = 1.0

        sentence = generate_sentence(start, start_pos, relation, end, end_pos)

        cursor.execute("""
            INSERT INTO conceptnet_en (start_node, start_pos, relation, end_node, end_pos, weight, sentence)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (start, start_pos, relation, end, end_pos, weight, sentence))

        pbar.update(1)

    conn.commit()

cursor.close()
conn.close()
print("Done.")
