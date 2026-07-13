"""Stage 4: resample stage 3's cleaned trajectory segments onto a uniform
time grid (default 10 s), producing per-day trajectory CSVs (rules in
utils/resample_trajectories.py).

Outputs are tagged with the grid spacing (trajectories_10s/, *_10s.csv;
--dt 4 gives trajectories_4s/, *_4s.csv), so resampling at a radar's actual
scan period is a fresh run from the stage-3 segments -- never a second
interpolation of an already-resampled product, which would compound the
low-pass smoothing.

Accel / turn-rate exceedances FLAG trajectories (exceeds_accel_limit /
exceeds_turn_rate_limit columns) rather than dropping them, so maneuver-rich
flights stay in the dataset; pass --drop-dynamics to drop them instead.

Usage:
    python scripts/04_resample_trajectories.py
    python scripts/04_resample_trajectories.py --dt 5 --smooth
    python scripts/04_resample_trajectories.py --dt 4   # radar scan period
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import (
    get_dropped_audit_path,
    get_resample_summary_path,
    get_segments_dir,
    get_trajectories_dir,
)
from utils.resample_trajectories import (
    MAX_SPEED_MPS_DEFAULT,
    ResampleConfig,
    discover_input_files,
    process_day,
)

SUMMARY_COLUMNS = [
    "date", "input_rows", "input_segments", "output_rows", "output_trajectories",
    "segments_split_by_interp_gap",
    "dropped_duration", "dropped_min_points", "dropped_speed", "dropped_accel", "dropped_turn_rate",
    "flagged_accel", "flagged_turn_rate",
    "median_speed_mps", "p95_speed_mps",
    "median_accel_mps2", "p95_accel_mps2",
    "median_turn_rate_deg_s", "p95_turn_rate_deg_s",
    "output_file",
]

# Plausible range for the combined median ground speed of light GA cruise.
ANCHOR_SPEED_RANGE_MPS = (25.0, 90.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resample cleaned conventional-GA trajectory segments onto a uniform time grid.")
    parser.add_argument("--input-dir", type=str, default=None,
                         help="Directory containing states_*_conventionalGA_segments.csv files "
                              "(default: this project's data/active/segments).")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Directory for the trajectory CSVs and summary "
                              "(default: this project's data/active/trajectories_<dt>s, "
                              "tagged with the grid spacing).")
    parser.add_argument("--dt", type=float, default=10.0,
                         help="Grid spacing in seconds (default: 10). Set this to the simulated "
                              "radar's scan period and rerun from the stage-3 segments rather than "
                              "re-resampling an existing grid.")
    parser.add_argument("--max-interp-gap-s", type=float, default=30.0,
                         help="Never interpolate across a gap longer than this; split instead (default: 30).")
    parser.add_argument("--min-duration-s", type=float, default=300.0,
                         help="Minimum resampled trajectory duration in seconds to keep (default: 300).")
    parser.add_argument("--min-points", type=int, default=30,
                         help="Minimum number of grid samples a trajectory must have to keep (default: 30).")
    parser.add_argument("--max-speed-mps", type=float, default=MAX_SPEED_MPS_DEFAULT,
                         help="Drop a trajectory if any sample's speed exceeds this "
                              "(default: ~154.33, i.e. 300 kt -- aligned with stage 3's glitch threshold).")
    parser.add_argument("--max-accel-mps2", type=float, default=10.0,
                         help="Flag a trajectory (exceeds_accel_limit) if >5%% of samples exceed "
                              "this |acceleration| (default: 10); drops instead under --drop-dynamics.")
    parser.add_argument("--max-turn-rate-deg-s", type=float, default=6.0,
                         help="Flag a trajectory (exceeds_turn_rate_limit) if >5%% of samples exceed "
                              "this |turn rate| (default: 6); drops instead under --drop-dynamics.")
    parser.add_argument("--drop-dynamics", action="store_true",
                         help="Drop (rather than flag) trajectories exceeding the accel / turn-rate "
                              "limits. Off by default: dropping biases the dataset toward benign, "
                              "steady flight, which is the wrong prior for target-vs-clutter work.")
    parser.add_argument("--smooth", action="store_true",
                         help="Apply a light centered rolling median (window 3) to the "
                              "interpolated positions (default: off).")
    return parser.parse_args()


# =============================================================================
# Validation gate -- run after processing, before declaring success
# =============================================================================

def _fail(message: str) -> None:
    raise ValueError(f"Stage 04 validation failed: {message}")


def _check_uniform_spacing(day_results: list, dt_s: float) -> None:
    """Every consecutive within-trajectory timestamp pair must be exactly dt apart."""
    for r in day_results:
        df = r["_final_df"]
        if df.empty:
            continue
        same_traj = df["trajectory_id"] == df["trajectory_id"].shift(1)
        steps = df["timestamp"].diff()[same_traj]
        if not np.allclose(steps, dt_s, atol=1e-6):
            bad = int((~np.isclose(steps, dt_s, atol=1e-6)).sum())
            _fail(f"{r['date']}: {bad} within-trajectory steps deviate from dt={dt_s}s")
    print(f"  uniform {dt_s}s spacing: OK")


def _check_thresholds(day_results: list, min_duration_s: float, min_points: int) -> None:
    for r in day_results:
        df = r["_final_df"]
        if df.empty:
            continue
        if df["trajectory_duration_s"].min() < min_duration_s:
            _fail(f"{r['date']}: trajectory shorter than {min_duration_s}s survived the filter")
        if df["n_samples"].min() < min_points:
            _fail(f"{r['date']}: trajectory with fewer than {min_points} samples survived the filter")
    print(f"  all trajectories >= {min_duration_s}s and >= {min_points} samples: OK")


def _check_motion_stats(day_results: list, cfg: ResampleConfig) -> None:
    """Combined-across-days motion statistics.

    The percentile table is REPORT-ONLY: the filters just enforced ~p95
    compliance at these exact thresholds, so asserting on them would be
    circular. The one hard check here is the median-speed anchor, which the
    filters do not enforce and which genuinely tests the pipeline.
    """
    frames = [r["_final_df"] for r in day_results if not r["_final_df"].empty]
    if not frames:
        _fail("no output trajectories to validate")

    channels = {
        "speed (m/s)": np.concatenate([f["speed_mps"].to_numpy() for f in frames]),
        "|accel| (m/s^2)": np.abs(np.concatenate([f["accel_mps2"].to_numpy() for f in frames])),
        "|accel_vec| (m/s^2)": np.abs(np.concatenate([f["accel_vector_mps2"].to_numpy() for f in frames])),
        "|turn| (deg/s)": np.abs(np.concatenate([f["turn_rate_deg_s"].to_numpy() for f in frames])),
    }

    print("\n  combined motion statistics (report-only):")
    print(f"  {'channel':>20} | {'p50':>8} | {'p95':>8} | {'p99':>8}")
    for name, values in channels.items():
        p50, p95, p99 = np.nanpercentile(values, [50, 95, 99])
        print(f"  {name:>20} | {p50:>8.3f} | {p95:>8.3f} | {p99:>8.3f}")

    median_speed = float(np.nanmedian(channels["speed (m/s)"]))
    lo, hi = ANCHOR_SPEED_RANGE_MPS
    print(f"\n  anchor check: combined median speed {median_speed:.2f} m/s (expected {lo}-{hi})")
    if not lo <= median_speed <= hi:
        _fail(f"combined median speed {median_speed:.2f} m/s outside plausible GA range {lo}-{hi}")


def _spot_check_trajectories(day_results: list) -> None:
    """Print min/max stats for 3 random trajectories across all days."""
    print("\n--- Spot-check: 3 random trajectories ---")
    rng = np.random.default_rng()

    catalog = []
    for r in day_results:
        df = r["_final_df"]
        if not df.empty:
            catalog.extend([(r["date"], tid) for tid in df["trajectory_id"].unique()])
    if not catalog:
        print("  no trajectories available to spot-check.")
        return

    for i in rng.choice(len(catalog), size=min(3, len(catalog)), replace=False):
        date, tid = catalog[i]
        df = next(r["_final_df"] for r in day_results if r["date"] == date)
        traj = df[df["trajectory_id"] == tid]
        print(f"  {tid} ({date}): n_samples={len(traj)}, "
              f"duration={traj['trajectory_duration_s'].iloc[0]:.0f}s, "
              f"max speed={np.nanmax(traj['speed_mps']):.2f} m/s, "
              f"max |accel|={np.nanmax(np.abs(traj['accel_mps2'])):.3f} m/s^2, "
              f"max |turn|={np.nanmax(np.abs(traj['turn_rate_deg_s'])):.3f} deg/s")


def main() -> None:
    args = parse_args()
    cfg = ResampleConfig(
        dt_s=args.dt,
        max_interp_gap_s=args.max_interp_gap_s,
        min_duration_s=args.min_duration_s,
        min_points=args.min_points,
        max_speed_mps=args.max_speed_mps,
        max_accel_mps2=args.max_accel_mps2,
        max_turn_rate_deg_s=args.max_turn_rate_deg_s,
        smooth=args.smooth,
        drop_dynamics=args.drop_dynamics,
    )

    input_dir = args.input_dir or get_segments_dir()
    output_dir = args.output_dir or get_trajectories_dir(cfg.dt_s)
    os.makedirs(output_dir, exist_ok=True)

    day_files = discover_input_files(input_dir)
    if not day_files:
        print(f"No states_*_conventionalGA_segments.csv files found in {input_dir}")
        return

    day_results = []
    for date, input_path in day_files:
        result = process_day(date, input_path, output_dir, cfg)
        day_results.append(result)

        print(f"\n--- {result['date']} ---")
        print(f"input rows:                        {result['input_rows']}")
        print(f"input segments:                    {result['input_segments']}")
        print(f"output rows:                       {result['output_rows']}")
        print(f"output trajectories:               {result['output_trajectories']}")
        print(f"segments split (interp gap):       {result['segments_split_by_interp_gap']}")
        print(f"dropped: duration:                 {result['dropped_duration']}")
        print(f"dropped: min points:               {result['dropped_min_points']}")
        print(f"dropped: speed:                    {result['dropped_speed']}")
        print(f"dropped: acceleration:             {result['dropped_accel']}")
        print(f"dropped: turn rate:                {result['dropped_turn_rate']}")
        print(f"flagged (kept): acceleration:      {result['flagged_accel']}")
        print(f"flagged (kept): turn rate:         {result['flagged_turn_rate']}")
        print(f"output path:                       {result['output_file']}")

    summary_rows = [{k: v for k, v in r.items() if k in SUMMARY_COLUMNS} for r in day_results]
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_path = get_resample_summary_path(output_dir)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to: {os.path.abspath(summary_path)}")

    # Audit trail of every dropped trajectory (id, reason, size), so filter
    # losses -- especially maneuver-rich flights hitting the turn-rate rule --
    # can be inspected instead of vanishing silently.
    audit_records = [rec for r in day_results for rec in r["_dropped_records"]]
    audit_df = pd.DataFrame(
        audit_records,
        columns=["date", "trajectory_id", "source_segment_id", "reason", "n_samples", "duration_s"],
    )
    audit_path = get_dropped_audit_path(output_dir)
    audit_df.to_csv(audit_path, index=False)
    print(f"Dropped-trajectory audit ({len(audit_df)} rows) written to: {os.path.abspath(audit_path)}")

    # --- Validation gate ---
    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)
    if not any(os.path.exists(r["output_file"]) for r in day_results):
        _fail("no output trajectory file was created")
    print(f"  output files created: {sum(os.path.exists(r['output_file']) for r in day_results)}")
    _check_uniform_spacing(day_results, cfg.dt_s)
    _check_thresholds(day_results, cfg.min_duration_s, cfg.min_points)
    _check_motion_stats(day_results, cfg)
    _spot_check_trajectories(day_results)

    print("\n04_resample_trajectories completed successfully.")


if __name__ == "__main__":
    main()
