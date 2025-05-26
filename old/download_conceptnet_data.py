import os
import requests
import gzip
import shutil
from tqdm import tqdm
import click
import csv

DOWNLOAD_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
DOWNLOAD_PATH = "data/conceptnet-data.tsv"
TEMP_GZ_FILE = "data/temp_conceptnet.csv.gz"

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def gunzip_file(gzip_path, output_path):
    with gzip.open(gzip_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def view_tsv_portion(start, end, outfile):
    with open(DOWNLOAD_PATH, 'r', newline='', encoding='utf-8') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        
        if outfile:
            out = open(outfile, 'w', newline='', encoding='utf-8')
            writer = csv.writer(out, delimiter='\t')
        else:
            writer = None

        for i, row in enumerate(tsv_reader):
            if i < start:
                continue
            if i >= end:
                break
            if writer:
                writer.writerow(row)
            else:
                click.echo('\t'.join(row))

        if outfile:
            out.close()
            click.echo(f"Output written to {outfile}")

@click.command()
@click.option('--view-start', type=int, help='Start line for viewing TSV')
@click.option('--view-end', type=int, help='End line for viewing TSV')
@click.option('--view-outfile', type=click.Path(), help='Output file for viewed portion')
def main(view_start, view_end, view_outfile):
    """Download and view ConceptNet data."""
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(DOWNLOAD_PATH), exist_ok=True)

    # Download and process the file if it doesn't exist
    if not os.path.exists(DOWNLOAD_PATH):
        click.echo("Downloading ConceptNet data...")
        download_file(DOWNLOAD_URL, TEMP_GZ_FILE)

        click.echo("Unzipping the file...")
        gunzip_file(TEMP_GZ_FILE, DOWNLOAD_PATH)

        # Clean up temporary file
        os.remove(TEMP_GZ_FILE)

        click.echo(f"ConceptNet data has been downloaded and processed to {DOWNLOAD_PATH}")
    else:
        click.echo(f"ConceptNet data already exists at {DOWNLOAD_PATH}")

    # View portion of the TSV if requested
    if view_start is not None and view_end is not None:
        view_tsv_portion(view_start, view_end, view_outfile)

if __name__ == "__main__":
    main()