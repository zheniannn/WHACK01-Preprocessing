"""Stage 3: clean and segment the daily sorted conventional-GA ADS-B
state-vector files into per-flight trajectory segments (rules in
utils/trajectory_prep.py).

Usage:
    python scripts/03_trajectory_prep.py
    python scripts/03_trajectory_prep.py --gap-split 90 --min-points 30
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import get_segments_dir, get_states_dir, get_summary_path
from utils.trajectory_prep import (
    MAX_SPEED_MPS,
    discover_input_files,
    haversine_m,
    process_day,
)

SUMMARY_COLUMNS = [
    "date", "input_file", "output_file",
    "rows_input", "rows_after_freshness", "rows_after_basic_cleaning", "rows_after_segmentation",
    "rows_removed_freshness", "rows_removed_basic_cleaning", "rows_removed_glitch_points",
    "segments_discarded_persistent_violations", "rows_removed_short_segments",
    "unique_aircraft_input", "unique_aircraft_output", "num_segments_output",
    "median_segment_duration_s", "mean_segment_duration_s", "max_segment_duration_s",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and segment daily conventional-GA ADS-B state vectors.")
    parser.add_argument("--gap-split", type=float, default=60.0,
                        help="Seconds between consecutive points beyond which a new segment starts (default: 60).")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing states_*_conventionalGA_sorted.csv files "
                              "(default: this project's data/active/states).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for the segment CSVs and summary "
                              "(default: this project's data/active/segments).")
    parser.add_argument("--min-duration", type=float, default=300.0,
                        help="Minimum segment duration in seconds to keep (default: 300).")
    parser.add_argument("--min-points", type=int, default=20,
                        help="Minimum number of points a segment must have to keep (default: 20).")
    parser.add_argument("--min-median-velocity", type=float, default=15.0,
                        help="Minimum median reported velocity in m/s for a segment to be kept "
                              "(default: 15, which biases toward cruise; lower it to retain "
                              "slow-flight regimes like pattern work and approaches).")
    return parser.parse_args()


# =============================================================================
# Validation gate -- run after processing, before declaring success
# =============================================================================

def _fail(message: str) -> None:
    raise ValueError(f"Stage 03 validation failed: {message}")


def _gap_sensitivity_report(day_result: dict, gap_split_s: float, min_duration_s: float,
                            min_points: int, min_velocity_mps: float) -> None:
    """Rerun segmentation in-memory on one already-cleaned day at the
    configured gap-split and at 2x / 3x, and report num_segments /
    points_kept / median_duration for each. Informational only.

    The report re-segments the day's FINAL points, so it can only coarsen:
    gaps below the configured split can't be exercised (those boundaries are
    already cut), and rows dropped by the configured run stay dropped -- the
    coarser rows therefore read slightly low.
    """
    from utils.trajectory_prep import assign_segments, filter_valid_segments, remove_glitch_points

    # assign_segments() re-derives boundaries from scratch regardless of any
    # pre-existing segment_id, so this genuinely exercises gap sensitivity.
    base_df = day_result["_final_df"]
    timestamp_col = day_result["_timestamp_col"]

    print(f"\n--- Gap-split sensitivity report (day {day_result['date']}, informational only) ---")
    print(f"{'gap_split_s':>12} | {'num_segments':>12} | {'total_points_kept':>18} | {'median_duration_s':>18}")
    for gap in (gap_split_s, 2 * gap_split_s, 3 * gap_split_s):
        resegmented = assign_segments(base_df.drop(columns=["segment_id", "segment_start_time",
                                                              "segment_end_time", "segment_duration_s"]),
                                       timestamp_col, gap)
        cleaned, _, _ = remove_glitch_points(resegmented, timestamp_col)
        final = filter_valid_segments(cleaned, min_duration_s, min_points, min_velocity_mps)
        if final.empty:
            print(f"{gap:>12g} | {0:>12} | {0:>18} | {0.0:>18}")
            continue
        n_segments = final["segment_id"].nunique()
        n_points = len(final)
        median_dur = float(final.groupby("segment_id")["segment_duration_s"].first().median())
        print(f"{gap:>12g} | {n_segments:>12} | {n_points:>18} | {median_dur:>18.1f}")


def _cross_day_consistency_check(summary_df: pd.DataFrame) -> None:
    """Flag any day whose total-removed-row proportion deviates more than 2x
    from the median across all processed days.
    """
    print("\n--- Cross-day consistency check ---")
    print(summary_df[["date", "rows_input", "rows_after_segmentation"]].to_string(index=False))

    removed_fraction = 1.0 - (summary_df["rows_after_segmentation"] / summary_df["rows_input"])
    median_fraction = removed_fraction.median()
    print(f"\nremoved-row fraction per day: {dict(zip(summary_df['date'], removed_fraction.round(4)))}")
    print(f"median removed-row fraction across days: {median_fraction:.4f}")

    for date, frac in zip(summary_df["date"], removed_fraction):
        if median_fraction > 0 and (frac > 2 * median_fraction or frac < median_fraction / 2):
            print(f"  FLAG: {date} removed-row fraction ({frac:.4f}) deviates >2x from the median ({median_fraction:.4f})")


def _spot_check_segments(day_results: list) -> None:
    """Pick 3 random segments across all processed days and print
    min/median/max implied speed, confirming max < MAX_SPEED_MPS.

    Note: a segment can legitimately still contain UP TO 5% violating steps
    (that's the discard threshold, not a hard zero-violation guarantee), so
    an occasional flagged segment here is expected, not necessarily a bug.
    """
    print("\n--- Spot-check: 3 random segments ---")
    rng = np.random.default_rng()

    all_segment_ids = []
    for r in day_results:
        df = r["_final_df"]
        if not df.empty:
            all_segment_ids.extend([(r["date"], r["_timestamp_col"], sid) for sid in df["segment_id"].unique()])

    if not all_segment_ids:
        print("  no segments available to spot-check.")
        return

    chosen = rng.choice(len(all_segment_ids), size=min(3, len(all_segment_ids)), replace=False)
    for i in chosen:
        date, timestamp_col, segment_id = all_segment_ids[i]
        df = next(r["_final_df"] for r in day_results if r["date"] == date)
        seg = df[df["segment_id"] == segment_id].sort_values(timestamp_col)

        t = seg[timestamp_col].to_numpy()
        lat = seg["lat"].to_numpy()
        lon = seg["lon"].to_numpy()
        dt = np.diff(t)
        dist = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt > 0, dist / dt, 0.0)

        status = "OK" if speed.max() < MAX_SPEED_MPS else "FLAG (residual violation within the 5% allowance)"
        print(f"  segment {segment_id} ({date}, n={len(seg)}): "
              f"min={speed.min():.2f} median={np.median(speed):.2f} max={speed.max():.2f} m/s -> {status}")


def _anchor_check(day_results: list) -> None:
    """Median reported velocity across ALL days' final output should land in
    45-75 m/s (typical cruise speed for light GA); flag if outside.
    """
    print("\n--- Anchor check: combined median reported velocity ---")
    frames = [r["_final_df"] for r in day_results if not r["_final_df"].empty]
    if not frames or "velocity" not in frames[0].columns:
        print("  velocity column absent; skipping anchor check.")
        return

    combined_velocity = pd.concat([f["velocity"] for f in frames], ignore_index=True).dropna()
    median_velocity = float(combined_velocity.median())
    in_range = 45.0 <= median_velocity <= 75.0
    print(f"  median reported velocity across all days: {median_velocity:.2f} m/s "
          f"({'within' if in_range else 'OUTSIDE'} expected 45-75 m/s range)")
    if not in_range:
        print("  FLAG: combined median velocity outside the expected range -- investigate upstream filters.")


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir or get_states_dir()
    output_dir = args.output_dir or get_segments_dir()
    os.makedirs(output_dir, exist_ok=True)

    day_files = discover_input_files(input_dir)
    if not day_files:
        print(f"No states_*_conventionalGA_sorted.csv files found in {input_dir}")
        return

    day_results = []
    for date, input_path in day_files:
        result = process_day(date, input_path, output_dir, args.gap_split, args.min_duration,
                             args.min_points, args.min_median_velocity)
        day_results.append(result)

        print(f"\n--- {result['date']} ---")
        print(f"input file:                  {result['input_file']}")
        print(f"rows before cleaning:        {result['rows_input']}")
        print(f"rows after freshness filter: {result['rows_after_freshness']}")
        print(f"rows after basic cleaning:   {result['rows_after_basic_cleaning']}")
        print(f"final rows saved:            {result['rows_after_segmentation']}")
        print(f"unique aircraft saved:       {result['unique_aircraft_output']}")
        print(f"number of segments:          {result['num_segments_output']}")
        print(f"median segment duration:     {result['median_segment_duration_s']:.1f} s")
        print(f"output path:                 {result['output_file']}")

    # Summary is written BEFORE the validation gate, so a gate failure never
    # destroys the diagnostics needed to debug it.
    summary_rows = [{k: v for k, v in r.items() if k in SUMMARY_COLUMNS} for r in day_results]
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_path = get_summary_path(output_dir)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to: {os.path.abspath(summary_path)}")

    # --- Validation gate ---
    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)
    # The onground filter must remove rows on at least one day, else
    # parse_onground() is likely misinterpreting the column.
    onground_removed_by_day = {r["date"]: r["_onground_removed"] for r in day_results}
    for date, n in onground_removed_by_day.items():
        print(f"onground rows removed ({date}): {n}")
    if max(onground_removed_by_day.values()) <= 0:
        _fail(f"onground filter removed 0 rows on every day {onground_removed_by_day} -- "
              f"parse_onground() is likely misinterpreting the column.")
    _gap_sensitivity_report(day_results[0], args.gap_split, args.min_duration,
                            args.min_points, args.min_median_velocity)
    _cross_day_consistency_check(summary_df)
    _spot_check_segments(day_results)
    _anchor_check(day_results)

    print("\n03_trajectory_prep completed successfully.")


if __name__ == "__main__":
    main()
