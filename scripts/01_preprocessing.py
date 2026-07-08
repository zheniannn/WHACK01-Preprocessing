"""Entry point: filter the raw aircraft database down to conventional light
General Aviation aircraft (see utils/ga_classification.py for the rules) and
write the result to data/active/.

Usage:
    python F01-Preprocessing/scripts/01_preprocessing.py
"""

import os
import sys

import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ga_classification import classify
from utils.io import get_input_path, get_output_path


def main() -> None:
    input_path = get_input_path()
    output_path = get_output_path()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Required input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    final_mask, stats = classify(df)

    result = df[final_mask]
    result.to_csv(output_path, index=False)

    # Required output: one value per line, in this order.
    print(stats["total_rows"])
    print(stats["rows_kept_previous"])
    print(stats["rows_removed_new"])
    print(stats["rows_removed_bad_icao24"])
    print(stats["final_rows"])
    print(os.path.abspath(output_path))


if __name__ == "__main__":
    main()
