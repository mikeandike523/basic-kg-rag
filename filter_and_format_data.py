import csv
import json
import os
from tqdm import tqdm

# === CONFIGURATION ===
# List of two-letter language codes to retain (e.g., ['en', 'es', 'fr'])
LANGUAGES = ['en']
INPUT_FILE = 'data/conceptnet-data.tsv'
# Dynamically generate output filename based on LANGUAGES
langs_suffix = '_'.join(LANGUAGES)
OUTPUT_FILE = f"data/conceptnet-data-{langs_suffix}-formatted.tsv"

# === HELPERS ===
POS_MAP = {
    'n': 'NOUN',
    'v': 'VERB',
    'a': 'ADJECTIVE',
    's': 'ADJECTIVE SATELLITE',
    'r': 'ADVERB'
}

def parse_node(uri):
    """
    Given a concept URI like '/c/en/run/v' or '/c/en/color',
    return (term, pos), where pos is mapped to its full name or 'ANY'.
    """
    parts = uri.strip().split('/')
    # parts: ['', 'c', lang, term, pos?]
    if len(parts) >= 5 and parts[4]:
        term = parts[3]
        pos = POS_MAP.get(parts[4], 'ANY')
    elif len(parts) >= 4:
        term = parts[3]
        pos = 'ANY'
    else:
        term = uri
        pos = 'ANY'
    return term, pos

def uri_language(uri):
    """
    Extract two-letter language code from a concept URI '/c/{lang}/...'.
    Returns the code or None if unavailable.
    """
    parts = uri.strip().split('/')
    if len(parts) >= 3 and parts[1] == 'c':
        return parts[2]
    return None

# === MAIN PROCESSING ===
def reformat_and_normalize(input_file, output_file, languages):
    """
    Two-pass process, restricted to specified languages:
      1) Scan to count total filtered lines, detect max weight, and tally malformed.
      2) Re-read and output filtered, normalized rows.
    """
    # Step 1: initial scan for metrics on filtered rows
    print("Step 1/2: Scanning file for metrics with language filter...")
    total_filtered = 0
    max_weight = 0.0
    malformed_fields = 0
    malformed_json = 0
    missing_weight = 0

    with open(input_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) != 5:
                malformed_fields += 1
                continue
            uri_start, uri_end, meta_str = row[2], row[3], row[4]
            lang_start = uri_language(uri_start)
            lang_end = uri_language(uri_end)
            if lang_start not in languages or lang_end not in languages:
                continue
            total_filtered += 1
            try:
                meta = json.loads(meta_str)
            except Exception:
                malformed_json += 1
                continue
            if 'weight' not in meta:
                missing_weight += 1
                continue
            try:
                w = float(meta['weight'])
            except Exception:
                malformed_json += 1
                continue
            if w > max_weight:
                max_weight = w

    print(f"  Total filtered lines:       {total_filtered:,}")
    print(f"  Malformed (field count):    {malformed_fields:,}")
    print(f"  Malformed (JSON errors):    {malformed_json:,}")
    print(f"  Missing weight field:       {missing_weight:,}")
    print(f"  Maximum weight detected:    {max_weight}")
    if max_weight <= 0:
        raise ValueError("No valid weights found; cannot normalize.")

    # Step 2: reformat & normalize filtered rows
    print("Step 2/2: Reformatting and normalizing data...")
    written = 0
    malformed_second = 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        for row in tqdm(reader, total=total_filtered, unit='lines', desc='Writing normalized rows', dynamic_ncols=True):
            if len(row) != 5:
                malformed_second += 1
                continue
            uri_rel, uri_start, uri_end, meta_str = row[1], row[2], row[3], row[4]
            lang_start = uri_language(uri_start)
            lang_end = uri_language(uri_end)
            if lang_start not in languages or lang_end not in languages:
                continue
            try:
                meta = json.loads(meta_str)
                raw_w = float(meta['weight'])
            except Exception:
                malformed_second += 1
                continue
            relation = uri_rel.replace('/r/', '')
            start_term, start_pos = parse_node(uri_start)
            end_term, end_pos     = parse_node(uri_end)
            # normalized = raw_w / max_weight
            writer.writerow([
                start_term,
                start_pos,
                relation,
                end_term,
                end_pos,
                # f"{normalized:.6f}"
                raw_w
            ])
            written += 1

    valid = total_filtered - malformed_json - missing_weight
    retained_pct = (written / valid * 100) if valid else 0

    print("\n✅ Processing complete:")
    print(f"   Rows written:               {written:,}")
    print(f"   Malformed skipped (scan):   {malformed_fields + malformed_json + missing_weight:,}")
    print(f"   Malformed skipped (pass2):  {malformed_second:,}")
    print(f"   Percentage retained:        {retained_pct:.2f}%")

if __name__ == "__main__":
    reformat_and_normalize(INPUT_FILE, OUTPUT_FILE, LANGUAGES)
