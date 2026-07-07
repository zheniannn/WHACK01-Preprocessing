"""Resample cleaned trajectory segments onto a uniform time grid.

Stage 4 consumes stage 3's per-flight segment CSVs and produces, per day,
uniformly-sampled (default 10 s) trajectories ready for later ENU conversion
and dataset construction. It does NOT split train/test, normalize, build ML
windows, or emit ENU tensors -- those belong to stage 5.

Pipeline per day (see process_day):
  read segments -> per segment: sort, dedup timestamps, split at large gaps
  -> unwrap longitude -> linear interpolation of lat/lon/alt onto the grid
  -> optional light smoothing -> speed/accel/turn-rate computation
  -> plausibility filters (dropped trajectories go to an audit file) -> save.

Numerical care taken here, because each mistake silently poisons downstream:
  * heading differences are wrapped to [-180, 180) before the turn rate is
    computed, so a track crossing north (359 -> 1 deg) reads as 2 deg, not 358;
  * longitude is unwrapped per subsegment before interpolation, so a track
    crossing the antimeridian doesn't interpolate across the whole planet;
  * accel_mps2 is longitudinal (d|v|/dt) -- near zero in a steady turn, so the
    accel filter doesn't double-count turning flight that the turn-rate filter
    already polices; the centripetal-inclusive magnitude is also written, as
    accel_vector_mps2, for downstream statistics only (never filtered on).
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

INPUT_PREFIX = "states_"
INPUT_SUFFIX = "_conventionalGA_segments.csv"
OUTPUT_SUFFIX = "_conventionalGA_trajectories_10s.csv"
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")

TIMESTAMP_CANDIDATES = ["lastposupdate", "time"]   # lastposupdate preferred, as in stage 3

REQUIRED_COLUMNS = [
    "icao24", "lat", "lon", "alt", "segment_id",
    "segment_start_time", "segment_end_time", "segment_duration_s",
]

# Carried through unchanged (one value per trajectory) when the input has them.
OPTIONAL_METADATA_COLUMNS = [
    "callsign", "manufacturername", "model", "icaoaircrafttype",
    "registration", "typecode",
]

# Reported ADS-B channels resampled onto the grid (as <name>_interp) when
# present, for reported-vs-derived cross-validation and vertical rate later.
REPORTED_CHANNELS = ["velocity", "heading", "vertrate"]

EARTH_RADIUS_M = 6_371_000.0
KNOTS_TO_MPS = 0.514444
MAX_SPEED_MPS_DEFAULT = 300.0 * KNOTS_TO_MPS   # ~154.33; aligned with stage 3's glitch threshold

# Fraction of samples allowed to violate the accel / turn-rate limits before
# the whole trajectory is dropped (mirrors stage 3's 5% rule).
MAX_VIOLATION_FRACTION = 0.05

# Drop-rule names, in the order they are checked; also the summary-counter keys.
DROP_REASONS = ["duration", "min_points", "speed", "accel", "turn_rate"]


@dataclass
class ResampleConfig:
    """All stage-4 tunables in one place (populated from the CLI)."""
    dt_s: float = 10.0
    max_interp_gap_s: float = 30.0     # never interpolate across a gap longer than this
    min_duration_s: float = 300.0
    min_points: int = 30
    max_speed_mps: float = MAX_SPEED_MPS_DEFAULT
    max_accel_mps2: float = 10.0
    max_turn_rate_deg_s: float = 6.0
    smooth: bool = False               # optional light smoothing, off by default


# =============================================================================
# Discovery / validation
# =============================================================================

def find_timestamp_column(columns) -> str:
    """Return 'lastposupdate' if present (canonical), else 'time'; raise if neither exists."""
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Neither 'lastposupdate' nor 'time' present in columns: {list(columns)}")


def discover_input_files(input_dir: str) -> List[Tuple[str, str]]:
    """Return sorted (date, path) pairs for every stage-3 segments CSV in input_dir."""
    results = []
    for name in sorted(os.listdir(input_dir)):
        if not (name.startswith(INPUT_PREFIX) and name.endswith(INPUT_SUFFIX)):
            continue
        match = DATE_PATTERN.search(name)
        if not match:
            print(f"WARNING: no date pattern found in filename '{name}'; skipping.")
            continue
        results.append((match.group(1), os.path.join(input_dir, name)))
    return results


def validate_columns(path: str) -> Tuple[str, List[str], List[str]]:
    """Peek a CSV's header; return (timestamp_col, metadata_cols, reported_channels)
    actually present. Raises ValueError if any REQUIRED_COLUMNS or both
    timestamp candidates are missing."""
    columns = list(pd.read_csv(path, nrows=0).columns)

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"Missing required column(s) {missing} in {path}")

    timestamp_col = find_timestamp_column(columns)
    metadata_cols = [c for c in OPTIONAL_METADATA_COLUMNS if c in columns]
    reported_cols = [c for c in REPORTED_CHANNELS if c in columns]
    return timestamp_col, metadata_cols, reported_cols


# =============================================================================
# Per-trajectory numerics
# =============================================================================

def wrap_longitude(lon_deg: np.ndarray) -> np.ndarray:
    """Wrap (possibly unwrapped) longitudes back into [-180, 180)."""
    return ((np.asarray(lon_deg) + 180.0) % 360.0) - 180.0


def build_grid(t: np.ndarray, dt_s: float) -> np.ndarray:
    """Uniform dt_s grid anchored at t[0], never extending past t[-1]."""
    return t[0] + np.arange(0.0, (t[-1] - t[0]) + 1e-9, dt_s)


def interp_channel(t_grid, t, y) -> np.ndarray:
    """np.interp that tolerates NaNs in y by interpolating over the finite
    points only (all-NaN or single-point channels come back as all-NaN)."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full(len(t_grid), np.nan)
    return np.interp(t_grid, t[ok], y[ok])


def interp_heading_deg(t_grid, t, heading_deg) -> np.ndarray:
    """Interpolate a compass heading channel: unwrap -> linear -> re-wrap to
    [0, 360), so 359 -> 1 deg interpolates through north, not backwards."""
    h = np.asarray(heading_deg, dtype=float)
    ok = np.isfinite(h)
    if ok.sum() < 2:
        return np.full(len(t_grid), np.nan)
    unwrapped = np.degrees(np.unwrap(np.radians(h[ok])))
    return np.interp(t_grid, t[ok], unwrapped) % 360.0


def is_interpolated_mask(t_grid, t, dt_s: float) -> np.ndarray:
    """True where no real fix lies within dt/2 of the grid point -- i.e. the
    sample is synthetic (constant-velocity by construction). Downstream
    analyses can exclude these to avoid biasing a prior toward straightness."""
    pos = np.searchsorted(t, t_grid)
    left = np.abs(t_grid - t[np.clip(pos - 1, 0, len(t) - 1)])
    right = np.abs(t[np.clip(pos, 0, len(t) - 1)] - t_grid)
    return np.minimum(left, right) > dt_s / 2.0


def smooth_series(x: np.ndarray) -> np.ndarray:
    """Light centered rolling median (window 3); endpoints fall back to
    whatever data the window can see. Deliberately simple -- no Kalman/RTS."""
    return pd.Series(x).rolling(3, center=True, min_periods=1).median().to_numpy()


def compute_motion(t_grid, lat, lon_unwrapped, dt_s: float):
    """speed / longitudinal accel / vector accel / turn rate on the grid.

    Positions are converted to local flat-earth metres relative to the first
    point (E = R*cos(lat0)*dlon, N = R*dlat, angles in radians; lon must be
    UNWRAPPED) -- accurate enough for validation quantities over
    single-flight extents.

    Backward differences: index i describes motion over (i-1, i], so
      speed valid from index 1 (index 0 copies index 1),
      accels and turn rate valid from index 2 (NaN before).
    """
    n = len(t_grid)
    speed = np.full(n, np.nan)
    accel = np.full(n, np.nan)
    accel_vector = np.full(n, np.nan)
    turn_rate = np.full(n, np.nan)
    if n < 2:
        return speed, accel, accel_vector, turn_rate

    lat0 = np.radians(lat[0])
    east = EARTH_RADIUS_M * np.cos(lat0) * np.radians(lon_unwrapped - lon_unwrapped[0])
    north = EARTH_RADIUS_M * np.radians(lat - lat[0])

    v_east = np.diff(east) / dt_s
    v_north = np.diff(north) / dt_s
    speed[1:] = np.hypot(v_east, v_north)
    speed[0] = speed[1]  # copy, so the first row is usable downstream

    if n >= 3:
        accel[2:] = np.diff(speed[1:]) / dt_s                              # longitudinal: d|v|/dt
        accel_vector[2:] = np.hypot(np.diff(v_east), np.diff(v_north)) / dt_s  # includes centripetal
        heading_deg = np.degrees(np.arctan2(v_east, v_north))              # compass-style: 0=N, 90=E
        d_heading = np.diff(heading_deg)
        d_heading = (d_heading + 180.0) % 360.0 - 180.0                    # wrap: 359->1 is 2 deg, not 358
        turn_rate[2:] = d_heading / dt_s

    return speed, accel, accel_vector, turn_rate


def classify_trajectory(t_grid, speed, accel, turn_rate, cfg: ResampleConfig) -> Optional[str]:
    """Return the first violated drop-rule name (see DROP_REASONS), or None
    if the trajectory passes every plausibility filter.

    Only longitudinal accel is filtered on -- see the module docstring.
    NaNs (the warm-up rows of speed/accel/turn rate) are excluded from every
    check so they can never cause a false failure.
    """
    if t_grid[-1] - t_grid[0] < cfg.min_duration_s:
        return "duration"
    if len(t_grid) < cfg.min_points:
        return "min_points"

    finite_speed = speed[np.isfinite(speed)]
    if finite_speed.size and finite_speed.max() > cfg.max_speed_mps:
        return "speed"

    for reason, values, limit in (
        ("accel", accel, cfg.max_accel_mps2),
        ("turn_rate", turn_rate, cfg.max_turn_rate_deg_s),
    ):
        finite = values[np.isfinite(values)]
        if finite.size and (np.abs(finite) > limit).mean() > MAX_VIOLATION_FRACTION:
            return reason
    return None


# =============================================================================
# Per-day orchestration
# =============================================================================

def _split_at_large_gaps(t: np.ndarray, max_gap_s: float) -> List[np.ndarray]:
    """Split positions 0..len(t)-1 into runs whose internal gaps are all
    <= max_gap_s. Interpolation never bridges a larger gap."""
    breaks = np.where(np.diff(t) > max_gap_s)[0] + 1
    return np.split(np.arange(len(t)), breaks)


def process_day(date: str, input_path: str, output_dir: str, cfg: ResampleConfig) -> Dict:
    """Resample one day's segments file. Returns the summary dict for the day
    (also used to build trajectory_resample_summary.csv), including the
    dropped-trajectory audit records under '_dropped_records'."""
    timestamp_col, metadata_cols, reported_cols = validate_columns(input_path)

    read_cols = list(dict.fromkeys(REQUIRED_COLUMNS + [timestamp_col] + metadata_cols + reported_cols))
    string_cols = {c: str for c in ["icao24", "segment_id"] + metadata_cols}
    df = pd.read_csv(input_path, usecols=read_cols, dtype=string_cols, low_memory=False)

    rows_input = len(df)
    input_segments = int(df["segment_id"].nunique())

    t_all = df[timestamp_col].to_numpy(dtype=float)
    lat_all = df["lat"].to_numpy(dtype=float)
    lon_all = df["lon"].to_numpy(dtype=float)
    alt_all = df["alt"].to_numpy(dtype=float)
    reported_all = {c: df[c].to_numpy(dtype=float) for c in reported_cols}

    dropped = {reason: 0 for reason in DROP_REASONS}
    dropped_records: List[Dict] = []
    segments_split = 0
    out_frames: List[pd.DataFrame] = []

    def audit(trajectory_id: str, segment_id: str, reason: str, n_samples: int, duration_s: float):
        dropped[reason] += 1
        dropped_records.append({
            "date": date, "trajectory_id": trajectory_id, "source_segment_id": segment_id,
            "reason": reason, "n_samples": n_samples, "duration_s": duration_s,
        })

    for segment_id, positions in df.groupby("segment_id", sort=False).indices.items():
        order = np.argsort(t_all[positions], kind="mergesort")
        idx = positions[order]
        t = t_all[idx]

        # Drop duplicate timestamps within the segment, keeping the first.
        keep = np.r_[True, np.diff(t) > 0]
        idx, t = idx[keep], t[keep]
        if len(idx) < 2:
            audit(f"{segment_id}_r0", segment_id, "min_points", len(idx),
                  float(t[-1] - t[0]) if len(t) else 0.0)
            continue

        subsegments = _split_at_large_gaps(t, cfg.max_interp_gap_s)
        was_split = len(subsegments) > 1
        if was_split:
            segments_split += 1

        first_row = df.iloc[idx[0]]  # per-segment metadata source

        for k, local in enumerate(subsegments):
            trajectory_id = f"{segment_id}_r{k}"   # k-th resampled subsegment; globally unique
            t_sub = t[local]
            if len(local) < 2:
                audit(trajectory_id, segment_id, "min_points", len(local), float(t_sub[-1] - t_sub[0]))
                continue

            gi = idx[local]
            # Unwrap longitude so an antimeridian crossing (+179.99 -> -179.99)
            # interpolates locally instead of sweeping across the planet.
            lon_unwrapped = np.degrees(np.unwrap(np.radians(lon_all[gi])))

            t_grid = build_grid(t_sub, cfg.dt_s)
            lat_i = np.interp(t_grid, t_sub, lat_all[gi])
            lon_i = np.interp(t_grid, t_sub, lon_unwrapped)   # stays unwrapped until output
            alt_i = np.interp(t_grid, t_sub, alt_all[gi])

            if cfg.smooth:
                lat_s, lon_s, alt_s = smooth_series(lat_i), smooth_series(lon_i), smooth_series(alt_i)
            else:
                lat_s, lon_s, alt_s = lat_i, lon_i, alt_i

            # Motion quantities follow the smoothed track (== interpolated
            # track when smoothing is off), since that's what stage 5 consumes.
            speed, accel, accel_vector, turn_rate = compute_motion(t_grid, lat_s, lon_s, cfg.dt_s)

            reason = classify_trajectory(t_grid, speed, accel, turn_rate, cfg)
            if reason is not None:
                audit(trajectory_id, segment_id, reason, len(t_grid), float(t_grid[-1] - t_grid[0]))
                continue

            n = len(t_grid)
            block = {
                "icao24": first_row["icao24"],
                "segment_id": segment_id,
                "trajectory_id": trajectory_id,
                "source_segment_id": segment_id,
                "sample_idx": np.arange(n),
                "timestamp": t_grid,
                "dt_s": cfg.dt_s,
                "lat_interp": lat_i,
                "lon_interp": wrap_longitude(lon_i),
                "alt_interp": alt_i,
                "lat_smooth": lat_s,
                "lon_smooth": wrap_longitude(lon_s),
                "alt_smooth": alt_s,
                "is_interpolated": is_interpolated_mask(t_grid, t_sub, cfg.dt_s),
                "speed_mps": speed,
                "accel_mps2": accel,
                "accel_vector_mps2": accel_vector,
                "turn_rate_deg_s": turn_rate,
                "segment_start_time": first_row["segment_start_time"],
                "segment_end_time": first_row["segment_end_time"],
                "segment_duration_s": first_row["segment_duration_s"],
                "trajectory_duration_s": t_grid[-1] - t_grid[0],
                "n_samples": n,
                "was_split_by_interp_gap": was_split,
            }
            for col in reported_cols:
                if col == "heading":
                    block["heading_interp"] = interp_heading_deg(t_grid, t_sub, reported_all[col][gi])
                else:
                    block[f"{col}_interp"] = interp_channel(t_grid, t_sub, reported_all[col][gi])
            for col in metadata_cols:
                block[col] = first_row[col]
            out_frames.append(pd.DataFrame(block))

    df_final = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

    # Column order: identity, grid, positions, provenance, motion, reported
    # channels, segment/trajectory metadata, then carried aircraft metadata.
    if not df_final.empty:
        cols = ["icao24"] + (["callsign"] if "callsign" in metadata_cols else [])
        cols += [
            "segment_id", "trajectory_id", "source_segment_id", "sample_idx",
            "timestamp", "dt_s",
            "lat_interp", "lon_interp", "alt_interp",
            "lat_smooth", "lon_smooth", "alt_smooth",
            "is_interpolated",
            "speed_mps", "accel_mps2", "accel_vector_mps2", "turn_rate_deg_s",
        ]
        cols += [f"{c}_interp" for c in reported_cols]
        cols += [
            "segment_start_time", "segment_end_time", "segment_duration_s",
            "trajectory_duration_s", "n_samples", "was_split_by_interp_gap",
        ]
        cols += [c for c in metadata_cols if c != "callsign"]
        df_final = df_final[cols]

    output_name = os.path.basename(input_path)[: -len(INPUT_SUFFIX)] + OUTPUT_SUFFIX
    output_path = os.path.join(output_dir, output_name)
    df_final.to_csv(output_path, index=False)

    # Distribution stats for the summary CSV; accel / turn rate are reported
    # as magnitudes, matching how the plausibility limits are applied.
    def _median_p95(values: np.ndarray) -> Tuple[float, float]:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return float("nan"), float("nan")
        return float(np.median(finite)), float(np.percentile(finite, 95))

    if not df_final.empty:
        median_speed, p95_speed = _median_p95(df_final["speed_mps"].to_numpy())
        median_accel, p95_accel = _median_p95(np.abs(df_final["accel_mps2"].to_numpy()))
        median_turn, p95_turn = _median_p95(np.abs(df_final["turn_rate_deg_s"].to_numpy()))
        output_trajectories = int(df_final["trajectory_id"].nunique())
    else:
        median_speed = p95_speed = median_accel = p95_accel = median_turn = p95_turn = float("nan")
        output_trajectories = 0

    return {
        "date": date,
        "input_rows": rows_input,
        "input_segments": input_segments,
        "output_rows": len(df_final),
        "output_trajectories": output_trajectories,
        "segments_split_by_interp_gap": segments_split,
        "dropped_duration": dropped["duration"],
        "dropped_min_points": dropped["min_points"],
        "dropped_speed": dropped["speed"],
        "dropped_accel": dropped["accel"],
        "dropped_turn_rate": dropped["turn_rate"],
        "median_speed_mps": median_speed,
        "p95_speed_mps": p95_speed,
        "median_accel_mps2": median_accel,
        "p95_accel_mps2": p95_accel,
        "median_turn_rate_deg_s": median_turn,
        "p95_turn_rate_deg_s": p95_turn,
        "output_file": os.path.abspath(output_path),
        # kept only for the validation gate / audit file, not written to the summary CSV
        "_final_df": df_final,
        "_dropped_records": dropped_records,
    }
