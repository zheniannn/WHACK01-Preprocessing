"""Filter OpenSky ADS-B state-vector files down to conventional GA aircraft,
then concatenate and sort them into one file per day.

Pipeline: aircraft whitelist -> discover + validate state files -> group by
date -> filter each file in chunks -> concatenate per day -> sort -> save.
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .common import DATE_PATTERN

EXCLUDE_NAME_PATTERN = re.compile(r"conventionalga|sorted|filtered", re.IGNORECASE)
CHUNK_SIZE = 250_000

# Snapshot time preferred for SORTING (deliberately the opposite order to
# stages 3-4's position-time preference -- see common.POSITION_TIME_CANDIDATES).
TIME_COLUMN_CANDIDATES = ["time", "lastposupdate"]

# Force identifier columns to string on every read: an all-digit-hex chunk
# (e.g. icao24 "111111") would otherwise be inferred as int64, truncating
# leading zeros and silently breaking the whitelist match.
READ_DTYPE_OVERRIDES = {"icao24": str, "callsign": str}


def load_whitelist(whitelist_path: str) -> Set[str]:
    """Load the conventional-GA whitelist and return its icao24 values, lowercased.

    Raises FileNotFoundError / ValueError if the file or its icao24 column is missing.
    """
    if not os.path.exists(whitelist_path):
        raise FileNotFoundError(f"Aircraft whitelist not found: {whitelist_path}")

    whitelist_df = pd.read_csv(whitelist_path, dtype={"icao24": str}, low_memory=False)
    if "icao24" not in whitelist_df.columns:
        raise ValueError(f"Whitelist file is missing required column 'icao24': {whitelist_path}")

    icao24 = whitelist_df["icao24"].dropna().astype(str).str.strip().str.lower()
    return set(icao24[icao24 != ""])


def extract_date(filename: str) -> Optional[str]:
    """Return the first YYYY-MM-DD substring found in filename, or None."""
    match = DATE_PATTERN.search(filename)
    return match.group(1) if match else None


def discover_state_files(state_dir: str, whitelist_path: str) -> Dict[str, List[str]]:
    """Scan state_dir for candidate ADS-B CSVs and group their paths by date.

    Skips (with a printed note) the whitelist file itself, any raw aircraft
    database file, already-processed outputs (conventionalGA/sorted/filtered
    in the name), and files with no detectable YYYY-MM-DD date.
    """
    whitelist_abspath = os.path.abspath(whitelist_path)
    date_to_files: Dict[str, List[str]] = defaultdict(list)

    for name in sorted(os.listdir(state_dir)):
        if not name.lower().endswith(".csv"):
            continue

        path = os.path.join(state_dir, name)
        if os.path.abspath(path) == whitelist_abspath:
            continue
        if "aircraftdatabase" in name.lower():
            continue  # the raw/whitelist aircraft database, never a state-vector file
        if EXCLUDE_NAME_PATTERN.search(name):
            continue  # already a pipeline output, never a source file

        date = extract_date(name)
        if date is None:
            print(f"WARNING: no date pattern found in filename '{name}'; skipping.")
            continue

        date_to_files[date].append(path)

    return date_to_files


def _read_header_columns(path: str) -> Optional[List[str]]:
    """Return a file's column names, or None if the file has no data at all."""
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except pd.errors.EmptyDataError:
        return None


def validate_state_file(path: str) -> Optional[str]:
    """Check one state file for the required columns.

    Returns the timestamp column to sort by ("time" or "lastposupdate"), or
    None if the file is completely empty (which is not treated as an error --
    an empty source file simply contributes nothing). Raises ValueError for
    any non-empty file missing icao24, or missing both timestamp columns.
    """
    columns = _read_header_columns(path)
    if columns is None:
        print(f"WARNING: '{path}' is empty; skipping.")
        return None

    if "icao24" not in columns:
        raise ValueError(f"ADS-B file is missing required column 'icao24': {path}")

    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"ADS-B file has neither 'time' nor 'lastposupdate' column: {path}"
    )


def filter_state_file(path: str, whitelist: Set[str]) -> Tuple[pd.DataFrame, int]:
    """Read path in chunks and keep only whitelisted aircraft (case-insensitive).

    Blank/NaN icao24 rows can never match the whitelist (it contains no empty
    values), so a single isin() covers them too. Returns (filtered_df, rows_read).
    """
    rows_read = 0
    kept_chunks = []

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, dtype=READ_DTYPE_OVERRIDES, low_memory=False):
        rows_read += len(chunk)
        icao24 = chunk["icao24"].astype(str).str.strip().str.lower()
        chunk = chunk[icao24.isin(whitelist)]
        if not chunk.empty:
            kept_chunks.append(chunk)

    if kept_chunks:
        return pd.concat(kept_chunks, ignore_index=True), rows_read
    return pd.DataFrame(), rows_read


def process_day(date: str, files: List[str], whitelist: Set[str], output_dir: str) -> Optional[dict]:
    """Filter, concatenate, sort, and save one day's worth of state files.

    Returns a summary dict, or None if none of the day's files had usable data.
    """
    day_frames = []
    rows_before_total = 0
    files_used = 0

    for path in files:
        if validate_state_file(path) is None:
            continue  # empty file, already warned about in validate_state_file

        filtered_df, rows_read = filter_state_file(path, whitelist)
        rows_before_total += rows_read
        files_used += 1

        if not filtered_df.empty:
            day_frames.append(filtered_df)

    if not day_frames:
        print(f"WARNING: no usable data for {date}; skipping output file.")
        return None

    day_df = pd.concat(day_frames, ignore_index=True)

    # "time" is preferred; only fall back to "lastposupdate" if "time" never appears.
    sort_column = "time" if "time" in day_df.columns else "lastposupdate"
    if sort_column not in day_df.columns:
        # Defensive only: validate_state_file guaranteed one of the two exists.
        raise ValueError(f"Neither 'time' nor 'lastposupdate' present in concatenated data for {date}")

    icao24_lower = day_df["icao24"].astype(str).str.strip().str.lower()
    day_df = day_df.assign(_sort_key=icao24_lower).sort_values(
        by=["_sort_key", sort_column], kind="mergesort"
    ).drop(columns="_sort_key")

    output_path = os.path.join(output_dir, f"states_{date}_conventionalGA_sorted.csv")
    day_df.to_csv(output_path, index=False)

    return {
        "date": date,
        "files_used": files_used,
        "rows_before": rows_before_total,
        "rows_after": len(day_df),
        "unique_aircraft": int(icao24_lower.nunique()),
        "output_path": os.path.abspath(output_path),
    }
