"""Resample cleaned trajectory segments onto a uniform time grid.

Stage 4 consumes stage 3's per-flight segment CSVs and produces, per day,
uniformly-sampled (default 10 s) trajectory CSVs. It performs no coordinate
conversion, normalization, or windowing.

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
    accel_vector_mps2, for downstream statistics only (never filtered on);
  * accel / turn-rate exceedances FLAG a trajectory rather than dropping it
    (unless cfg.drop_dynamics) -- dropping would bias the dataset toward
    benign, steady flight, exactly the wrong prior for discriminating real
    maneuvering targets from clutter;
  * grid-derived motion is low-passed by linear interpolation onto the grid,
    so native-rate dynamics computed from the RAW fixes are carried alongside
    (raw_speed_max_mps / raw_accel_max_mps2 / raw_turn_rate_max_deg_s per grid
    interval, plus n_raw_fixes and raw_update_median_s) -- downstream should
    treat those, not the grid differences, as the truth for maneuver intensity.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import EARTH_RADIUS_M, MAX_SPEED_MPS, discover_input_files as _discover, find_timestamp_column
from .io import dt_tag

INPUT_SUFFIX = "_conventionalGA_segments.csv"


def output_suffix(dt_s: float) -> str:
    """Per-day output filename suffix, tagged with the grid spacing (see io.dt_tag)."""
    return f"_conventionalGA_trajectories_{dt_tag(dt_s)}.csv"


REQUIRED_COLUMNS = [
    "icao24", "lat", "lon", "alt", "segment_id",
    "segment_start_time", "segment_end_time", "segment_duration_s",
]

# Carried through unchanged (one value per trajectory) when the input has it.
# Only callsign survives the upstream stages: state vectors carry no
# aircraft-database metadata and stage 2 performs no whitelist join.
OPTIONAL_METADATA_COLUMNS = ["callsign"]

# Reported ADS-B channels resampled onto the grid (as <name>_interp) when
# present, so reported values can be cross-checked against derived ones.
REPORTED_CHANNELS = ["velocity", "heading", "vertrate"]

MAX_SPEED_MPS_DEFAULT = MAX_SPEED_MPS   # same 300 kt ceiling as stage 3's glitch threshold

# Fraction of samples allowed to violate the accel / turn-rate limits before
# the whole trajectory is flagged -- or dropped, under cfg.drop_dynamics
# (mirrors stage 3's 5% rule).
MAX_VIOLATION_FRACTION = 0.05

# Drop-rule names, in the order they are checked; also the summary-counter keys.
# duration/min_points/speed are always hard drops (unusable or glitch-corrupted
# data); accel/turn_rate only drop under cfg.drop_dynamics, else they flag.
DROP_REASONS = ["duration", "min_points", "speed", "accel", "turn_rate"]
DYNAMICS_FLAGS = ["accel", "turn_rate"]


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
    drop_dynamics: bool = False        # True restores pre-flag behavior: accel/turn-rate exceedances drop


# =============================================================================
# Discovery / validation
# =============================================================================

def discover_input_files(input_dir: str) -> List[Tuple[str, str]]:
    """Sorted (date, path) pairs for every stage-3 segments CSV in input_dir."""
    return _discover(input_dir, INPUT_SUFFIX)


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
    points only (all-NaN or single-point channels come back as all-NaN).

    Note: np.interp clamps outside the finite span, so a leading/trailing
    NaN run yields constant edge values rather than NaN. Positions are
    unaffected (lat/lon/alt are NaN-free after basic_clean); this only
    touches the reported cross-check channels."""
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


def raw_dynamics_per_interval(
    t_grid: np.ndarray,
    t_raw: np.ndarray,
    lat_raw: np.ndarray,
    lon_unwrapped_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Native-rate motion aggregated onto the grid, from the RAW fixes.

    Linear interpolation onto the grid low-passes the true dynamics (at
    ~55 m/s a 6 deg/s turn sweeps 60 deg between 10 s samples), so the
    grid-derived channels understate maneuver intensity. These channels
    preserve it: for each grid sample i, over the interval
    (t_grid[i-1], t_grid[i]], report the max |speed| / |accel| / |turn rate|
    of the raw inter-fix steps whose midpoint time falls in that interval,
    plus the raw-fix count. NaN where the interval contains no raw step --
    broadly aligned with is_interpolated, though the two use different
    criteria (step midpoint in interval vs nearest fix within dt/2) and can
    disagree near interval edges.

    Returns (raw_speed_max, raw_accel_max, raw_turn_rate_max, n_raw_fixes).
    """
    n = len(t_grid)
    speed_max = np.full(n, np.nan)
    accel_max = np.full(n, np.nan)
    turn_max = np.full(n, np.nan)
    n_fixes = np.zeros(n, dtype=int)

    # side="left": a value at exactly t_grid[i] belongs to interval i.
    fix_bins = np.clip(np.searchsorted(t_grid, t_raw, side="left"), 0, n - 1)
    np.add.at(n_fixes, fix_bins, 1)

    if len(t_raw) < 2:
        return speed_max, accel_max, turn_max, n_fixes

    # Per-step quantities at step-midpoint times (same flat-earth frame as
    # compute_motion, but on the nonuniform raw timestamps).
    dt = np.diff(t_raw)                              # > 0: timestamps are deduped upstream
    lat0 = np.radians(lat_raw[0])
    east = EARTH_RADIUS_M * np.cos(lat0) * np.radians(lon_unwrapped_raw - lon_unwrapped_raw[0])
    north = EARTH_RADIUS_M * np.radians(lat_raw - lat_raw[0])
    v_east = np.diff(east) / dt
    v_north = np.diff(north) / dt
    speed = np.hypot(v_east, v_north)
    t_mid = (t_raw[:-1] + t_raw[1:]) / 2.0

    def scatter_max(target: np.ndarray, times: np.ndarray, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        if not finite.any():
            return
        bins = np.clip(np.searchsorted(t_grid, times[finite], side="left"), 0, n - 1)
        acc = np.full(n, -np.inf)
        np.maximum.at(acc, bins, np.abs(values[finite]))
        seen = acc > -np.inf
        target[seen] = acc[seen]

    scatter_max(speed_max, t_mid, speed)

    if len(speed) >= 2:
        dt_mid = np.diff(t_mid)                      # > 0 since t_raw strictly increases
        accel = np.diff(speed) / dt_mid
        heading_deg = np.degrees(np.arctan2(v_east, v_north))
        d_heading = (np.diff(heading_deg) + 180.0) % 360.0 - 180.0
        turn = d_heading / dt_mid
        t_mid2 = (t_mid[:-1] + t_mid[1:]) / 2.0
        scatter_max(accel_max, t_mid2, accel)
        scatter_max(turn_max, t_mid2, turn)

    return speed_max, accel_max, turn_max, n_fixes


def classify_trajectory(
    t_grid, speed, accel, turn_rate, cfg: ResampleConfig
) -> Tuple[Optional[str], Dict[str, bool]]:
    """Return (hard_drop_reason_or_None, dynamics_flags).

    duration / min_points / speed are always hard drops -- they mark a
    trajectory as unusable or glitch-corrupted. accel / turn_rate
    exceedances (>MAX_VIOLATION_FRACTION of samples over the limit) are
    returned as flags and only become drops under cfg.drop_dynamics:
    maneuver-rich flights are real targets, and silently removing them
    would bias a motion prior toward benign, steady flight.

    Only longitudinal accel is flagged on -- see the module docstring.
    NaNs (the warm-up rows of speed/accel/turn rate) are excluded from every
    check so they can never cause a false failure.
    """
    flags = {name: False for name in DYNAMICS_FLAGS}

    if t_grid[-1] - t_grid[0] < cfg.min_duration_s:
        return "duration", flags
    if len(t_grid) < cfg.min_points:
        return "min_points", flags

    finite_speed = speed[np.isfinite(speed)]
    if finite_speed.size and finite_speed.max() > cfg.max_speed_mps:
        return "speed", flags

    for reason, values, limit in (
        ("accel", accel, cfg.max_accel_mps2),
        ("turn_rate", turn_rate, cfg.max_turn_rate_deg_s),
    ):
        finite = values[np.isfinite(values)]
        flags[reason] = bool(finite.size and (np.abs(finite) > limit).mean() > MAX_VIOLATION_FRACTION)

    if cfg.drop_dynamics:
        for reason in DYNAMICS_FLAGS:
            if flags[reason]:
                return reason, flags
    return None, flags


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
    flagged = {name: 0 for name in DYNAMICS_FLAGS}
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
            # track when smoothing is off), since that is the track consumers read.
            speed, accel, accel_vector, turn_rate = compute_motion(t_grid, lat_s, lon_s, cfg.dt_s)

            reason, dyn_flags = classify_trajectory(t_grid, speed, accel, turn_rate, cfg)
            if reason is not None:
                audit(trajectory_id, segment_id, reason, len(t_grid), float(t_grid[-1] - t_grid[0]))
                continue
            for name in DYNAMICS_FLAGS:
                flagged[name] += int(dyn_flags[name])

            # Native-rate dynamics from the raw fixes (pre-interpolation), so
            # downstream sees true maneuver intensity, not the low-passed grid.
            raw_speed_max, raw_accel_max, raw_turn_max, n_raw_fixes = raw_dynamics_per_interval(
                t_grid, t_sub, lat_all[gi], lon_unwrapped)

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
                "raw_speed_max_mps": raw_speed_max,
                "raw_accel_max_mps2": raw_accel_max,
                "raw_turn_rate_max_deg_s": raw_turn_max,
                "n_raw_fixes": n_raw_fixes,
                "raw_update_median_s": float(np.median(np.diff(t_sub))),
                "exceeds_accel_limit": dyn_flags["accel"],
                "exceeds_turn_rate_limit": dyn_flags["turn_rate"],
                # segment_* fields describe the PARENT stage-3 segment
                # (provenance); trajectory_* fields describe THIS resampled
                # subsegment and are the ones consumers should use.
                "segment_start_time": first_row["segment_start_time"],
                "segment_end_time": first_row["segment_end_time"],
                "segment_duration_s": first_row["segment_duration_s"],
                "trajectory_start_time": t_grid[0],
                "trajectory_end_time": t_grid[-1],
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
            "raw_speed_max_mps", "raw_accel_max_mps2", "raw_turn_rate_max_deg_s",
            "n_raw_fixes", "raw_update_median_s",
        ]
        cols += [f"{c}_interp" for c in reported_cols]
        cols += [
            "segment_start_time", "segment_end_time", "segment_duration_s",
            "trajectory_start_time", "trajectory_end_time", "trajectory_duration_s",
            "n_samples", "was_split_by_interp_gap",
            "exceeds_accel_limit", "exceeds_turn_rate_limit",
        ]
        cols += [c for c in metadata_cols if c != "callsign"]
        df_final = df_final[cols]

    output_name = os.path.basename(input_path)[: -len(INPUT_SUFFIX)] + output_suffix(cfg.dt_s)
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
        "flagged_accel": flagged["accel"],
        "flagged_turn_rate": flagged["turn_rate"],
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
