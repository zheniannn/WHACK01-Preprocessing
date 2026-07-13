"""Stage 1: filter the raw aircraft database down to conventional light
General Aviation aircraft (rules in utils/ga_classification.py), producing
the icao24 whitelist consumed by stage 2.

Usage:
    python scripts/01_preprocessing.py
    python scripts/01_preprocessing.py --input db.csv --output whitelist.csv
"""

import argparse
import os
import sys

import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ga_classification import classify
from utils.io import get_aircraft_db_path, get_whitelist_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter the raw aircraft database to conventional light GA.")
    parser.add_argument("--input", type=str, default=None,
                        help="Raw aircraft database CSV "
                             "(default: <data root>/archive/aircraftDatabase-2022-06.csv).")
    parser.add_argument("--output", type=str, default=None,
                        help="Whitelist CSV to write "
                             "(default: <data root>/active/aircraftDatabase-2022-06-conventionalGA.csv).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or get_aircraft_db_path()
    output_path = args.output or get_whitelist_path()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Required input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    final_mask, stats = classify(df)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df[final_mask].to_csv(output_path, index=False)

    print(f"total rows:               {stats['rows_total']}")
    print(f"matched base brand rules: {stats['rows_base_match']}")
    print(f"removed by strict rules:  {stats['rows_removed_strict']}")
    print(f"removed malformed icao24: {stats['rows_removed_bad_icao24']}")
    print(f"kept (whitelist):         {stats['rows_kept']}")
    print(f"output: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
