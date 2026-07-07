"""Clean and segment daily sorted conventional-GA ADS-B state vectors into
per-flight trajectory segments.

This stage does NOT do ENU conversion and does NOT build training windows --
it only removes bad/stale points and splits each aircraft's daily point
stream into physically-plausible flight segments, ready for those later
stages.

Pipeline per day (see process_day):
  read (chunked) with per-chunk freshness filter -> concatenate
  -> dedupe on (icao24, lastposupdate) -> basic_clean -> re-sort
  -> assign_segments -> remove_glitch_points -> filter_valid_segments -> save.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

INPUT_PREFIX = "states_"
INPUT_SUFFIX = "_conventionalGA_sorted.csv"
OUTPUT_SUFFIX = "_conventionalGA_segments.csv"
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")

CHUNK_SIZE = 500_000

# --- Timestamp / staleness ---------------------------------------------------
TIMESTAMP_CANDIDATES = ["lastposupdate", "time"]   # lastposupdate preferred (see module docstring)
FRESHNESS_MAX_LAG_S = 10.0                          # keep only rows where time - lastposupdate <= this

# --- Physical-plausibility thresholds ----------------------------------------
KNOTS_TO_MPS = 0.514444
MAX_SPEED_KNOTS = 300.0
MAX_SPEED_MPS = MAX_SPEED_KNOTS * KNOTS_TO_MPS       # ~154.33 m/s
MAX_GLITCH_PASSES = 3
PERSISTENT_VIOLATION_FRACTION = 0.05                 # discard segment if >5% of steps still violate

ALT_MIN_M = -400.0
ALT_MAX_M = 12000.0

MIN_VELOCITY_MPS = 15.0                              # drops airborne-flagged idling/taxi artifacts

EARTH_RADIUS_M = 6_371_000.0

REQUIRED_BASE_COLUMNS = ["icao24", "lat", "lon", "onground"]
ALTITUDE_CANDIDATES = ["geoaltitude", "baroaltitude"]  # geoaltitude preferred (GPS-based; needed for slant range later)

# Force these to string dtype on every read: a sorted file's early rows can be
# all-numeric hex codes (e.g. icao24 "111111"), which would otherwise make
# pandas infer int64 for a chunk and silently truncate leading zeros
# (e.g. "010203" -> 10203), corrupting the identifier.
READ_DTYPE_OVERRIDES = {"icao24": str, "callsign": str}


# =============================================================================
# Helper functions (required names per spec)
# =============================================================================

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two (lat, lon) points (vectorized)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    # clip guards against tiny negative values from floating-point rounding at a==0/1
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def find_timestamp_column(columns) -> str:
    """Return 'lastposupdate' if present (canonical), else 'time'; raise if neither exists."""
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Neither 'lastposupdate' nor 'time' present in columns: {list(columns)}")


def find_altitude_columns(columns) -> Tuple[Optional[str], Optional[str]]:
    """Return (geoaltitude_col_or_None, baroaltitude_col_or_None); raise if both absent."""
    geo = "geoaltitude" if "geoaltitude" in columns else None
    baro = "baroaltitude" if "baroaltitude" in columns else None
    if geo is None and baro is None:
        raise ValueError(f"Neither 'geoaltitude' nor 'baroaltitude' present in columns: {list(columns)}")
    return geo, baro


def parse_onground(series: pd.Series) -> pd.Series:
    """Robustly interpret the onground column as booleans, whether it round-tripped
    through CSV as native bool or as text/int representations."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapping = {
        True: True, "True": True, "true": True, 1: True, "1": True,
        False: False, "False": False, "false": False, 0: False, "0": False,
    }
    return series.map(mapping).fillna(False)


def validate_columns(path: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Peek a CSV's header and confirm required columns exist.

    Returns (timestamp_col, geo_col, baro_col). Raises ValueError with a
    clear message if icao24/lat/lon/onground, the timestamp column, or the
    altitude column(s) are missing.
    """
    columns = list(pd.read_csv(path, nrows=0).columns)

    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"Missing required column(s) {missing} in {path}")

    timestamp_col = find_timestamp_column(columns)
    geo_col, baro_col = find_altitude_columns(columns)
    return timestamp_col, geo_col, baro_col


def freshness_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop rows whose 'time' is more than FRESHNESS_MAX_LAG_S ahead of
    'lastposupdate' -- OpenSky repeats a stale position across snapshots
    while 'time' keeps advancing, so a large (time - lastposupdate) gap
    means the row's lat/lon/altitude are old, not a new observation.

    Skipped (no-op) if either column is absent -- we can't judge staleness
    without both. Returns (filtered_df, rows_removed).
    """
    if "time" not in df.columns or "lastposupdate" not in df.columns:
        return df, 0
    rows_before = len(df)
    mask = (df["time"] - df["lastposupdate"]) <= FRESHNESS_MAX_LAG_S
    filtered = df[mask]
    return filtered, rows_before - len(filtered)


def basic_clean(
    df: pd.DataFrame,
    geo_col: Optional[str],
    baro_col: Optional[str],
    verbose_label: str = "",
) -> Tuple[pd.DataFrame, int]:
    """Drop invalid rows and add the coalesced 'alt' column.

    Order: blank/missing icao24 -> missing lat/lon -> invalid coordinates
    (out-of-range or null-island) -> on-ground rows -> altitude coalesce +
    range filter. Each step's removed-row count is printed individually.

    Returns (cleaned_df, onground_rows_removed) -- the latter is returned
    explicitly (not stashed in df.attrs) because attrs don't reliably survive
    the filtering/concat operations used elsewhere in this pipeline.
    """
    def log(step: str, n_removed: int, n_before: int):
        pct = (100.0 * n_removed / n_before) if n_before else 0.0
        print(f"  [{verbose_label}] {step}: removed {n_removed} rows ({pct:.2f}%)")

    df = df.copy()

    n0 = len(df)
    icao24 = df["icao24"].astype(str).str.strip()
    has_icao24 = icao24.notna() & (icao24 != "") & (icao24.str.lower() != "nan")
    df = df[has_icao24]
    log("blank/missing icao24", n0 - len(df), n0)

    n0 = len(df)
    df = df[df["lat"].notna() & df["lon"].notna()]
    log("missing lat/lon", n0 - len(df), n0)

    n0 = len(df)
    lat, lon = df["lat"], df["lon"]
    null_island = (lat.abs() < 1e-6) & (lon.abs() < 1e-6)
    valid_coords = (lat.abs() <= 90) & (lon.abs() <= 180) & (~null_island)
    df = df[valid_coords]
    log("invalid coordinates (out-of-range / null-island)", n0 - len(df), n0)

    n0 = len(df)
    is_ground = parse_onground(df["onground"])
    onground_removed = int(is_ground.sum())
    df = df[~is_ground]
    log("on-ground rows", n0 - len(df), n0)

    # Coalesce altitude: geoaltitude preferred (GPS-based; needed for slant
    # range later), falling back to baroaltitude where geoaltitude is missing.
    # Both original columns are kept untouched in the output.
    n0 = len(df)
    if geo_col and baro_col:
        alt = df[geo_col].where(df[geo_col].notna(), df[baro_col])
    elif geo_col:
        alt = df[geo_col]
    else:
        alt = df[baro_col]
    df["alt"] = alt
    df = df[df["alt"].notna()]
    log("altitude missing (both sources)", n0 - len(df), n0)

    n0 = len(df)
    df = df[(df["alt"] >= ALT_MIN_M) & (df["alt"] <= ALT_MAX_M)]
    log(f"altitude outside [{ALT_MIN_M}, {ALT_MAX_M}] m", n0 - len(df), n0)

    return df, onground_removed


def assign_segments(df: pd.DataFrame, timestamp_col: str, gap_split_s: float) -> pd.DataFrame:
    """Split each aircraft's point stream into segments and label them.

    A new segment starts whenever, relative to the previous point for the
    SAME aircraft:
      1. it's the first point for that aircraft (a hard boundary), or
      2. the time gap since the previous point exceeds gap_split_s, or
      3. the callsign changed (NaN/blank callsigns are treated as equal to
         each other, but not equal to a real callsign) -- a callsign change
         usually means a new flight leg even without a long ground gap.

    Adds 'segment_id' as f"{icao24}_{int(segment_start_time)}" (globally
    unique across days/aircraft) plus 'segment_start_time', 'segment_end_time',
    and 'segment_duration_s'. Does not filter anything out -- that's
    remove_glitch_points()/filter_valid_segments()'s job.
    """
    df = df.sort_values(["icao24", timestamp_col], kind="mergesort").reset_index(drop=True)

    is_new_aircraft = df["icao24"] != df["icao24"].shift(1)

    prev_time = df.groupby("icao24")[timestamp_col].shift(1)
    time_gap = df[timestamp_col] - prev_time
    is_time_gap = time_gap > gap_split_s

    if "callsign" in df.columns:
        callsign_norm = df["callsign"].astype(str).str.strip()
        callsign_norm = callsign_norm.where(~callsign_norm.str.lower().isin(["", "nan", "none"]), "")
        prev_callsign = callsign_norm.groupby(df["icao24"]).shift(1)
        # Two blanks are "equal" (no change); blank vs. real callsign IS a change.
        is_callsign_change = (callsign_norm != prev_callsign) & is_new_aircraft.eq(False)
    else:
        is_callsign_change = pd.Series(False, index=df.index)

    new_segment_flag = is_new_aircraft | is_time_gap.fillna(False) | is_callsign_change.fillna(False)

    # Running count of segment starts within each aircraft gives a stable,
    # temporary per-aircraft segment number, used only to compute each
    # segment's start time before building the final segment_id.
    segment_number = new_segment_flag.groupby(df["icao24"]).cumsum()
    temp_key = df["icao24"].astype(str) + "_" + segment_number.astype(int).astype(str)

    grouped_time = df.groupby(temp_key)[timestamp_col]
    segment_start = grouped_time.transform("min")
    segment_end = grouped_time.transform("max")

    df["segment_id"] = df["icao24"].astype(str) + "_" + segment_start.astype("int64").astype(str)
    df["segment_start_time"] = segment_start
    df["segment_end_time"] = segment_end
    df["segment_duration_s"] = segment_end - segment_start

    return df


def _clean_one_segment(times: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Iteratively drop the single point most responsible for implausible
    implied-speed steps, up to MAX_GLITCH_PASSES times.

    A single GPS glitch corrupts one point but shows up as TWO bad steps
    (into it and out of it), so removing the point that participates in the
    most violating steps -- rather than splitting the segment there -- fixes
    both steps at once without fragmenting an otherwise-good flight.

    Returns (keep_mask over the input arrays, points_removed, final_violation_fraction).
    """
    n = len(times)
    keep = np.ones(n, dtype=bool)
    removed = 0

    violation_fraction = 0.0
    for _ in range(MAX_GLITCH_PASSES):
        idx = np.where(keep)[0]
        if len(idx) < 2:
            break
        dt = np.diff(times[idx])
        dist = haversine_m(lats[idx][:-1], lons[idx][:-1], lats[idx][1:], lons[idx][1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt > 0, dist / dt, 0.0)
        violating = speed > MAX_SPEED_MPS
        n_violations = int(violating.sum())
        violation_fraction = (n_violations / len(violating)) if len(violating) else 0.0
        if n_violations == 0:
            break

        # participation[i] = how many of the (up to 2) adjacent steps touching
        # local point i are violating
        participation = np.zeros(len(idx))
        participation[:-1] += violating
        participation[1:] += violating
        worst_local = int(np.argmax(participation))
        if participation[worst_local] == 0:
            break  # defensive: shouldn't happen given n_violations > 0
        keep[idx[worst_local]] = False
        removed += 1

    return keep, removed, violation_fraction


def remove_glitch_points(df: pd.DataFrame, timestamp_col: str) -> Tuple[pd.DataFrame, int, int]:
    """Per segment, remove GPS-glitch points via _clean_one_segment(); discard
    the entire segment if implausible-speed steps still exceed
    PERSISTENT_VIOLATION_FRACTION after MAX_GLITCH_PASSES attempts.

    This is the one deliberately non-vectorized part of the pipeline: each
    segment needs its own small iterative fixed-point loop, but segments are
    small (tens to low hundreds of points), so total cost stays in the
    seconds-to-low-minutes range even across tens of thousands of segments.

    Returns (cleaned_df, points_removed, segments_discarded).
    """
    keep_mask = np.ones(len(df), dtype=bool)
    points_removed = 0
    segments_discarded = 0

    times = df[timestamp_col].to_numpy()
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    for _, positions in df.groupby("segment_id", sort=False).indices.items():
        # positions are integer offsets into df (and therefore into times/lats/lons)
        seg_keep, removed, frac = _clean_one_segment(times[positions], lats[positions], lons[positions])
        if frac > PERSISTENT_VIOLATION_FRACTION:
            keep_mask[positions] = False
            segments_discarded += 1
        else:
            keep_mask[positions] = seg_keep
            points_removed += removed

    return df[keep_mask].reset_index(drop=True), points_removed, segments_discarded


def filter_valid_segments(df: pd.DataFrame, min_duration_s: float, min_points: int) -> pd.DataFrame:
    """Keep only segments that are long enough, dense enough, and (if a
    velocity column exists) fast enough in the median to be real flight
    rather than an on-the-ground/taxi artifact that slipped past onground.

    Point counts and median velocity are recomputed on the post-glitch-removal
    rows; duration deliberately uses the stored segment_duration_s from
    assign_segments() (the segment's original span), so a segment isn't
    re-penalized for having a glitchy endpoint removed.
    """
    grouped = df.groupby("segment_id")
    duration = grouped["segment_duration_s"].transform("first")
    count = grouped["segment_id"].transform("count")

    keep = (duration >= min_duration_s) & (count >= min_points)

    if "velocity" in df.columns:
        median_velocity = grouped["velocity"].transform("median")
        keep = keep & (median_velocity >= MIN_VELOCITY_MPS)

    return df[keep].reset_index(drop=True)


def discover_input_files(input_dir: str) -> List[Tuple[str, str]]:
    """Return sorted (date, path) pairs for every states_*_conventionalGA_sorted.csv in input_dir."""
    results = []
    for name in sorted(os.listdir(input_dir)):
        if not (name.startswith(INPUT_PREFIX) and name.endswith(INPUT_SUFFIX)):
            continue
        if "_segments" in name:  # belt-and-suspenders; can't happen given the suffix check above
            continue
        match = DATE_PATTERN.search(name)
        if not match:
            print(f"WARNING: no date pattern found in filename '{name}'; skipping.")
            continue
        results.append((match.group(1), os.path.join(input_dir, name)))
    return results


def read_and_freshness_filter(path: str) -> Tuple[pd.DataFrame, int, int]:
    """Chunked read with per-chunk freshness filtering (row-independent, so
    it's safe to apply before concatenating -- this keeps peak memory to one
    ~500k-row chunk instead of the whole raw file).

    basic_clean() deliberately does NOT run per chunk here: the freshness
    filter + the (icao24, lastposupdate) dedup that follows it need to be
    fully applied and counted (as "rows_after_freshness") BEFORE
    basic_clean's drops are counted separately (as "rows_after_basic_cleaning").
    Running basic_clean per chunk first would conflate the two counts.

    Returns (concatenated_freshness_filtered_df, rows_input, rows_removed_freshness).
    """
    rows_input = 0
    rows_removed_freshness = 0
    kept_chunks = []

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, dtype=READ_DTYPE_OVERRIDES, low_memory=False):
        rows_input += len(chunk)
        chunk, n_removed = freshness_filter(chunk)
        rows_removed_freshness += n_removed
        if not chunk.empty:
            kept_chunks.append(chunk)

    combined = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame()
    return combined, rows_input, rows_removed_freshness


def process_day(
    date: str,
    input_path: str,
    output_dir: str,
    gap_split_s: float,
    min_duration_s: float,
    min_points: int,
) -> Dict:
    """Run the full freshness -> clean -> dedupe -> segment -> glitch-removal
    -> filter -> save pipeline for one day. Returns the summary dict for this
    day (also used to build trajectory_prep_summary.csv).
    """
    timestamp_col, geo_col, baro_col = validate_columns(input_path)

    df_fresh, rows_input, rows_removed_freshness = read_and_freshness_filter(input_path)

    unique_aircraft_input = int(df_fresh["icao24"].nunique()) if not df_fresh.empty else 0

    # Dedup needs a global (not per-chunk) view of (icao24, timestamp).
    # timestamp_col is 'lastposupdate' whenever that column exists.
    n_before_dedup = len(df_fresh)
    df_fresh = df_fresh.drop_duplicates(subset=["icao24", timestamp_col], keep="first")
    rows_removed_freshness += n_before_dedup - len(df_fresh)
    rows_after_freshness = len(df_fresh)

    # basic_clean runs once here, on the full (already freshness-filtered and
    # deduped, so already smaller) frame -- this keeps "rows_after_freshness"
    # and "rows_after_basic_cleaning" as genuinely distinct, separately
    # meaningful counts instead of the same number.
    df_clean, onground_removed = basic_clean(df_fresh, geo_col, baro_col, verbose_label=date)
    df_clean = df_clean.sort_values(["icao24", timestamp_col], kind="mergesort").reset_index(drop=True)
    rows_after_basic_cleaning = len(df_clean)

    df_segmented = assign_segments(df_clean, timestamp_col, gap_split_s)
    df_glitch_free, rows_removed_glitch_points, segments_discarded = remove_glitch_points(df_segmented, timestamp_col)

    df_final = filter_valid_segments(df_glitch_free, min_duration_s, min_points)
    rows_after_segmentation = len(df_final)
    rows_removed_short_segments = len(df_glitch_free) - rows_after_segmentation

    unique_aircraft_output = int(df_final["icao24"].nunique()) if not df_final.empty else 0
    num_segments_output = int(df_final["segment_id"].nunique()) if not df_final.empty else 0

    if not df_final.empty:
        per_segment_duration = df_final.groupby("segment_id")["segment_duration_s"].first()
        median_duration = float(per_segment_duration.median())
        mean_duration = float(per_segment_duration.mean())
        max_duration = float(per_segment_duration.max())
    else:
        median_duration = mean_duration = max_duration = 0.0

    output_name = os.path.basename(input_path)[: -len(INPUT_SUFFIX)] + OUTPUT_SUFFIX
    output_path = os.path.join(output_dir, output_name)
    df_final.to_csv(output_path, index=False)

    return {
        "date": date,
        "input_file": input_path,
        "output_file": os.path.abspath(output_path),
        "rows_input": rows_input,
        "rows_after_freshness": rows_after_freshness,
        "rows_after_basic_cleaning": rows_after_basic_cleaning,
        "rows_after_segmentation": rows_after_segmentation,
        "rows_removed_freshness": rows_removed_freshness,
        "rows_removed_basic_cleaning": rows_input - rows_removed_freshness - rows_after_basic_cleaning,
        "rows_removed_glitch_points": rows_removed_glitch_points,
        "segments_discarded_persistent_violations": segments_discarded,
        "rows_removed_short_segments": rows_removed_short_segments,
        "unique_aircraft_input": unique_aircraft_input,
        "unique_aircraft_output": unique_aircraft_output,
        "num_segments_output": num_segments_output,
        "median_segment_duration_s": median_duration,
        "mean_segment_duration_s": mean_duration,
        "max_segment_duration_s": max_duration,
        # kept only for the assertion / validation gate below, not written to the summary CSV
        "_onground_removed": onground_removed,
        "_final_df": df_final,
        "_timestamp_col": timestamp_col,
    }
