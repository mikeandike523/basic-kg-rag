import os

from dotenv import dotenv_values
from tqdm import tqdm

values = dotenv_values("mysql/.env")

PORT=values["PORT"]
MYSQL_ROOT_PASSWORD=values["MYSQL_ROOT_PASSWORD"]
MYSQL_DATABASE=values["MYSQL_DATABASE"]
MYSQL_USER=values["MYSQL_USER"]
MYSQL_PASSWORD=values["MYSQL_PASSWORD"]

print(f"""
PORT: {PORT}
MYSQL_ROOT_PASSWORD: {MYSQL_ROOT_PASSWORD}
MYSQL_DATABASE: {MYSQL_DATABASE}
MYSQL_USER: {MYSQL_USER}
MYSQL_PASSWORD: {MYSQL_PASSWORD}
""")

data_file = "data/conceptnet-data-en-formatted.tsv"

def count_lines(file_path):
    """
    Count the number of lines in a file. Reports progress as a progress bar using tqdm.
    We assume utf-8, so we are simply looking for how many \n are present (we can ignore \r).
    Because we assume utf-8, and "\n" is one byte, we can open the file in binary mode and report progress
    as number of bytes processed out of total file size (queried from operating system).
    """
    line_count = 0
    total_size = os.path.getsize(file_path)
    
    with open(file_path, 'rb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc='Counting lines') as pbar:
        while True:
            chunk = f.read(1024 * 1024)  # Read in 1MB chunks
            if not chunk:
                break
            line_count += chunk.count(b'\n')
            pbar.update(len(chunk))

    return line_count

print(f"Data file: {data_file}")

print("Counting lines...")

line_count = count_lines(data_file)

print(f"Total lines: {line_count}")

# stream in text mode, line by line, encoding utf-8

# data is tab delimited

# no header is present

# each line has the following fields:

# - 1. start node (an english word)
# - 2. start part of speech ( a plain string). 
        # Note: ConceptNet itnroduces their own custom POS called "ADJECTIVATE SATELLITE".
        # - We are likey to convert this back to just "ADJECTIVE"
# - 3. relation (a plain string, but may have exotic formatting such as Is, IsA, IsSame, etc.)
# - 4. end node (an english word)
# - 5. end part of speech (a plain string)
    # - same note as before
# - 6. Weight: typically 1.0. Can be >1.0 if relation is semantically significant and supported by multiple data sources.
#       However we are likely to ignore the weight in the end and ask an llm to grade it instead.

# For each line (streamed), using a tqdm progress bar since we already know line count,
    # 1. Parse the tsv line into a record
    # 2. validate the record column coun
    # 3. convert parts of speech to lowercase. Convert "adjective satellite" to "adjective"
    # 4. validate the parts of speech: Can be one of ['noun', 'verb', 'adjective', 'adverb', 'any'] (any means was not specified in original dataset)
    # 5. Use an ultra-lightweight LLM to convert our relation into an english sentence. For instance, we may start by formatting it as

    # <start_node> (<part_of_speech if not any>) -> <relation> -> <end_node> (<part_of_speech if not any>)
    # then, for now, we will just give it a weight of 1 when we insert into database


# two questions toa ddress

# What is the best LLM, maybe bert based?
# We need to make a mysql table to hold the data