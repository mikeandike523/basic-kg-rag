import os
import re
import mysql.connector
from termcolor import colored
from tqdm import tqdm
from dotenv import dotenv_values
import click
import inflection

# === Configurable batch size ===
BATCH_SIZE = 1024

# === Env & checkpoint constants ===
values = dotenv_values("mysql-dev-server/.env")
PORT = values.get("PORT")
MYSQL_USER = values.get("MYSQL_USER")
MYSQL_PASSWORD = values.get("MYSQL_PASSWORD")
MYSQL_DATABASE = values.get("MYSQL_DATABASE")
CHECKPOINT_FILE = "data/checkpoint_line.txt"
DATA_FILE = "data/conceptnet-data-en-formatted.tsv"
FAILED_LINES_FILE = "data/failed_lines.txt"


def parse_relation_templates(raw_text):
    template_dict = {}
    # Match lines like: RelationName: "template string"
    pattern = re.compile(r'^([^:]+):\s*"(.+)"\s*$')

    for line in raw_text.strip().splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        key, template = match.groups()
        template_dict[key.strip()] = template.strip()

    return template_dict


relation_templates = {}

with open("unique_relations.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    relation_templates = parse_relation_templates(raw_text)

def format_basic_sentence(start_term, relation_name, end_term):
    if relation_name not in relation_templates:
        return f"{"'"+start_term.replace("_"," ")+"'"} {inflection.underscore(relation_name).lower().replace("_"," ")} {"'"+end_term.replace("_"," ")+"'"}."
    
    template = relation_templates[relation_name]
    value = template.replace("<A>", "'" + start_term.replace("_", " ") + "'").replace(
        "<B>", "'" + end_term.replace("_", " ") + "'"
    )
    return value


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


@click.command()
@click.option(
    "--restart",
    is_flag=True,
    default=False,
    help="If set, clear the table and start from the beginning.",
)
def main(restart):

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
            start_node TEXT,
            relation VARCHAR(64),
            end_node TEXT,
            weight FLOAT,
            sentence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()

    insert_sql = """
        INSERT INTO conceptnet_en
          (start_node, relation, end_node, weight, sentence)
        VALUES (%s, %s, %s, %s, %s)
    """

    # 4. Process file in batches, skipping up to start_line
    total_lines = count_lines(DATA_FILE)
    print(f"Processing {total_lines} lines (resuming at {start_line})…")

    records = []

    with open(DATA_FILE, "r", encoding="utf-8") as f, tqdm(
        dynamic_ncols=True,
        total = total_lines,
        desc="Lines processed",
        unit="lines",
        unit_scale=True,
        leave=False,
        ascii=True,
    ) as pbar:

        for idx, line in enumerate(f, start=0):
        
            if idx < start_line:
                pbar.update(1)
                continue

            try:

                parts = line.strip().split("\t")
                if len(parts) != 6:
                      raise ValueError(f"Invalid line format: {line}")

                s, _, rel, e, _, w_str = parts

                weight = None
                try:
                    weight = float(w_str)
                except ValueError:
                    pass

                if weight > 1:
                    weight = 1.0

                if weight is None:
                    raise ValueError("Invalid weight: {w_str}")

                sentence = format_basic_sentence(s, rel, e)

                records.append((s, rel, e, weight, sentence))
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

            pbar.update(1)

        # final partial batch
        if records:
            cursor.executemany(insert_sql, records)
            conn.commit()
            with open(CHECKPOINT_FILE, "w") as cf:
                cf.write(str(idx))


    cursor.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
