import json
import os
import re
from termcolor import colored
from tqdm import tqdm
from dotenv import dotenv_values
import click


# === Configurable batch size ===
BATCH_SIZE = 4

# === Env & checkpoint constants ===

DATA_FILE = "data/conceptnet-data-en-formatted.tsv"

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

print(relation_templates)

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
def main():
    
    unique_relations = list()
  
    # 4. Process file in batches, skipping up to start_line
    total_lines = count_lines(DATA_FILE)


    with open(DATA_FILE, "r", encoding="utf-8") as f, tqdm(
        total=total_lines, desc="Lines processed",
        dynamic_ncols=True
    ) as pbar:

        for idx, line in enumerate(f, start=0):
            
            pbar.update(1)
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue

            s, sp, rel, e, ep, w_str = parts
            sp = sp.lower().replace("adjective satellite", "adjective")
            ep = ep.lower().replace("adjective satellite", "adjective")

            if rel not in unique_relations:
                unique_relations.append(rel)

    with open ("unique_relations.txt", "w", encoding="utf-8") as f:
        for rel in unique_relations:
            f.write(f"{rel}\n")


if __name__ == "__main__":
    main()
