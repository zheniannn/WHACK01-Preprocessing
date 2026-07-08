"""Entry point: filter raw OpenSky ADS-B state-vector files down to the
conventional-GA aircraft whitelist, then concatenate and sort into one
sorted CSV per day.

Usage:
    python F01-Preprocessing/scripts/02_state_filtering.py
"""

import os
import sys

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import get_state_files_dir, get_states_dir, get_whitelist_path
from utils.state_filtering import discover_state_files, load_whitelist, process_day


def main() -> None:
    whitelist_path = get_whitelist_path()
    state_dir = get_state_files_dir()
    output_dir = get_states_dir()
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
