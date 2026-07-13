"""Stage 2: filter raw OpenSky ADS-B state-vector files down to the
conventional-GA aircraft whitelist (stage 1's output), then concatenate and
sort into one CSV per day.

Usage:
    python scripts/02_state_filtering.py
    python scripts/02_state_filtering.py --state-dir raw/ --output-dir daily/
"""

import argparse
import os
import sys

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import get_state_files_dir, get_states_dir, get_whitelist_path
from utils.state_filtering import discover_state_files, load_whitelist, process_day


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter raw ADS-B state vectors to whitelisted aircraft, one sorted CSV per day.")
    parser.add_argument("--whitelist", type=str, default=None,
                        help="Conventional-GA whitelist CSV (default: stage 1's output).")
    parser.add_argument("--state-dir", type=str, default=None,
                        help="Directory of raw hourly state-vector CSVs "
                             "(default: <data root>/archive).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for the daily sorted CSVs "
                             "(default: <data root>/active/states).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    whitelist_path = args.whitelist or get_whitelist_path()
    state_dir = args.state_dir or get_state_files_dir()
    output_dir = args.output_dir or get_states_dir()
    os.makedirs(output_dir, exist_ok=True)

    whitelist = load_whitelist(whitelist_path)
    print(f"Loaded {len(whitelist)} whitelisted icao24 values from {whitelist_path}")

    date_to_files = discover_state_files(state_dir, whitelist_path)
    if not date_to_files:
        print("No ADS-B state-vector files with a detectable date were found.")
        return

    summaries = []
    for date in sorted(date_to_files):
        summary = process_day(date, sorted(date_to_files[date]), whitelist, output_dir)
        if summary is None:
            continue

        summaries.append(summary)
        print(f"\n--- {summary['date']} ---")
        print(f"source files used:     {summary['files_used']}")
        print(f"rows before filtering:  {summary['rows_before']}")
        print(f"rows after filtering:   {summary['rows_after']}")
        print(f"unique aircraft:        {summary['unique_aircraft']}")
        print(f"output file:            {summary['output_path']}")

    print("\n=== Final summary ===")
    if not summaries:
        print("No daily output files were created (no usable data found).")
        return

    for summary in summaries:
        print(f"{summary['date']}: {summary['rows_after']} rows, "
              f"{summary['unique_aircraft']} aircraft -> {summary['output_path']}")


if __name__ == "__main__":
    main()
