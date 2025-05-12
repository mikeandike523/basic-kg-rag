import json
import os
import re
import mysql.connector
from termcolor import colored
from tqdm import tqdm
from dotenv import dotenv_values
import click
from run_mistral import completion, UnfinishedResponseError

# === Configurable batch size ===
BATCH_SIZE = 4

# === Env & checkpoint constants ===
values = dotenv_values("mysql-dev-server/.env")
PORT = values.get("PORT")
MYSQL_USER = values.get("MYSQL_USER")
MYSQL_PASSWORD = values.get("MYSQL_PASSWORD")
MYSQL_DATABASE = values.get("MYSQL_DATABASE")
CHECKPOINT_FILE = "data/checkpoint_line.txt"
DATA_FILE = "data/conceptnet-data-en-formatted.tsv"
FAILED_LINES_FILE = "data/failed_lines.txt"


def count_lines(file_path):
    """
    Efficiently count the number of lines in a file by reading in binary chunks.
    This is used for progress estimation with tqdm.
    """
    line_count = 0
    total_size = os.path.getsize(file_path)
    with open(file_path, "rb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Counting lines"
    ) as pbar:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            line_count += chunk.count(b"\n")
            pbar.update(len(chunk))
    return line_count


def call_local_llm(endpoint, payload, on_429_alt_sizes=None):
    """
    Send payload to local LLM; on 429, retry with smaller or alternative max_tokens.
    """
    original_max = payload.get("max_tokens")
    alt_sizes = [original_max] + (on_429_alt_sizes or [])

    print("\n----\n")
    print(colored(payload["messages"][0]["content"], "magenta"))
    print(colored(payload["messages"][1]["content"], "blue"))

    for max_tokens in alt_sizes:
        print(f"Attempting with max_tokens = {max_tokens}")
        payload["max_tokens"] = max_tokens
        try:
            resp = completion(payload)
            print(colored(resp.strip(), "green"))
            print("\n----\n")
            return {"role": "assistant", "content": resp.strip()}
        except UnfinishedResponseError as e:
            print(str(e))
    raise Exception("All max_tokens options resulted in incomplete responses.")


def get_truth_value(sentence):
    """
    Ask the LLM to rate truth of a sentence 0–100%. Returns fraction 0.0–1.0 or None.
    """
    chat = [
        {
            "role": "system",
            "content": "On a scale of 0% to 100%, how true is the given sentence? Output only the final percentage.",
        },
        {"role": "user", "content": sentence},
    ]
    payload = {"max_tokens": 16, "temperature": 0.4, "top_p": 0.95, "messages": chat}
    resp = call_local_llm(
        "http://localhost:5000/completion", payload, on_429_alt_sizes=[32, 64]
    )
    match = re.search(r"(\d{1,3}(?:\.\d+)?)%?", resp["content"])
    if match:
        pct = float(match.group(1))
        return min(max(pct, 0), 100) / 100
    return None


def generate_sentence(start, start_pos, relation, end, end_pos):
    """
    Convert a relation tuple into a simple sentence via the LLM.
    """
    relation_string = f"({start}) - ({relation}) - ({end})"
    chat = [
        {
            "role": "system",
            "content": "Convert the structured relation statement into a plain sentence. The relation format is (subject) - (relation) - (object).",
        },
        {"role": "user", "content": relation_string},
    ]
    payload = {"messages": chat, "temperature": 0.2, "top_p": 0.95, "max_tokens": 64}
    resp = call_local_llm(
        "http://localhost:5000/completion", payload, on_429_alt_sizes=[128, 256]
    )
    return resp["content"].strip()


@click.command()
@click.option(
    "--restart",
    is_flag=True,
    default=False,
    help="If set, clear the table and start from the beginning.",
)
def main(restart):
    # 1. Figure out where to start
    
    if not os.path.isfile(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "w") as fl:
            fl.write("0")

    if not os.path.isfile(FAILED_LINES_FILE):
        with open(FAILED_LINES_FILE, "w") as FL:
            fl.write("")
    
    if restart:
        start_line = 0
        with open(FAILED_LINES_FILE, "w") as fl:
            fl.write("")
        with open(CHECKPOINT_FILE, "w") as fl:
            fl.write("0")
    else:
        try:
            with open(CHECKPOINT_FILE, "r") as cf:
                start_line = int(cf.read().strip())
        except (IOError, ValueError):
            start_line = 0

    

    # 2. Connect to MySQL
    print("Connecting to MySQL…")
    conn = mysql.connector.connect(
        host="localhost",
        port=PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )
    cursor = conn.cursor()

    # 3. (Re)create or ensure table
    if restart:
        cursor.execute("DROP TABLE IF EXISTS conceptnet_en")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conceptnet_en (
            id INT AUTO_INCREMENT PRIMARY KEY,
            start_node VARCHAR(255),
            start_pos ENUM('noun','verb','adjective','adverb','any'),
            relation VARCHAR(100),
            end_node VARCHAR(255),
            end_pos ENUM('noun','verb','adjective','adverb','any'),
            weight FLOAT DEFAULT 1.0,
            sentence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()

    insert_sql = """
        INSERT INTO conceptnet_en
          (start_node, start_pos, relation, end_node, end_pos, weight, sentence)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    # 4. Process file in batches, skipping up to start_line
    total_lines = count_lines(DATA_FILE)
    print(f"Processing {total_lines} lines (resuming at {start_line})…")

    records = []
    # with open(DATA_FILE, "r", encoding="utf-8") as f, tqdm(
    #     total=total_lines, desc="Lines processed"
    # ) as pbar:

    with open(DATA_FILE, "r", encoding="utf-8") as f:

        for idx, line in enumerate(f, start=0):
            # pbar.update(1)
            if idx < start_line:
                continue

            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue

            s, sp, rel, e, ep, w_str = parts
            sp = sp.lower().replace("adjective satellite", "adjective")
            ep = ep.lower().replace("adjective satellite", "adjective")

            try:
                original_weight = None
                try:
                    original_weight = float(w_str)
            
                except ValueError:
                    pass

                if original_weight > 1:
                    original_weight = 1.0 

                # weight = get_truth_value(generate_sentence(s, sp, rel, e, ep)) or 0.0

                print(f"Processing line {idx + 1} of {total_lines}...")
                sentence = generate_sentence(s, sp, rel, e, ep)

                # records.append((s, sp, rel, e, ep, weight, sentence))
                records.append((s, sp, rel, e, ep, original_weight, sentence))
            except Exception as e:
                print(colored(f"Error processing line {idx}: {str(e)}", "red"))
                with open(FAILED_LINES_FILE, "a") as fl:
                    fl.write(f"{idx}\n")

            if len(records) >= BATCH_SIZE:
                cursor.executemany(insert_sql, records)
                conn.commit()
                records.clear()
                with open(CHECKPOINT_FILE, "w") as cf:
                    cf.write(str(idx))

        # final partial batch
        if records:
            cursor.executemany(insert_sql, records)
            conn.commit()
            with open(CHECKPOINT_FILE, "w") as cf:
                cf.write(str(idx))

    cursor.close()
    conn.close()
    print("Done. Last processed line:", idx)


if __name__ == "__main__":
    main()
