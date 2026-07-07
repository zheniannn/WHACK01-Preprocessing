# 01_preprocessing

Four-stage preprocessing pipeline that turns the raw aircraft database and
raw OpenSky ADS-B state-vector files into a homogeneous, filtered dataset of
**conventional, light, fixed-wing General Aviation aircraft and their
uniformly-sampled flight trajectories**, for use as a light-GA motion-prior
training set.

## Structure

```
01_preprocessing/
├── scripts/
│   ├── 01_preprocessing.py           # stage 1: aircraft DB -> GA whitelist
│   ├── 02_state_filtering.py         # stage 2: state vectors -> filtered daily files
│   ├── 03_trajectory_prep.py         # stage 3: daily files -> trajectory segments
│   └── 04_resample_trajectories.py   # stage 4: segments -> uniform 10s trajectories
└── utils/
    ├── io.py                          # input/output path resolution for all stages
    ├── ga_classification.py           # stage 1 rules: manufacturer/model classification
    ├── state_filtering.py             # stage 2 rules: discovery, validation, filtering, sorting
    ├── trajectory_prep.py             # stage 3 rules: cleaning, segmentation, filtering
    └── resample_trajectories.py       # stage 4 rules: gap splitting, interpolation, motion checks
```

## Usage

Run the stages in order — each depends on the previous stage's output:

```bash
python 01_preprocessing/scripts/01_preprocessing.py
python 01_preprocessing/scripts/02_state_filtering.py
python 01_preprocessing/scripts/03_trajectory_prep.py
python 01_preprocessing/scripts/04_resample_trajectories.py
```

All can be run from any working directory — paths are resolved relative to
the project root, not the shell's cwd.

---

## Stage 1 — `01_preprocessing.py`

Filters the full aircraft database down to conventional GA aircraft.

- **Input:** `data/archive/aircraftDatabase-2022-06.csv`
- **Output:** `data/active/aircraftDatabase-2022-06-conventionalGA.csv`
  (overwritten on each run; all original columns are preserved unchanged)

Prints six lines: total input rows, rows kept by the base manufacturer/model
rules, rows removed by the stricter homogeneity rules, rows removed for a
malformed `icao24`, final row count, and the output file's absolute path.

### Method

Classification is driven by `manufacturername` and `model` text, **not**
`icaoaircrafttype` — that column is often blank or mistagged in the source
data (e.g. a real Cessna 172 tagged as a helicopter), so it's used only as a
secondary safety net, never as the primary signal.

The pipeline runs in three internal passes (`utils/ga_classification.py::classify`):

1. **Base conventional-GA rules** (`build_base_mask`) — one keep/exclude
   regex pair per manufacturer (Cessna, Piper, Beech/Raytheon, Cirrus,
   Mooney, Socata, Robin, Diamond, Bellanca/Champion, Aeronca, Taylorcraft,
   Luscombe, Stinson), matching normal piston singles/twins while excluding
   agricultural, military/warbird, turbine/jet, and rotorcraft/glider
   variants that share a manufacturer with the aircraft we want to keep.

2. **Stricter homogeneity rules** (`build_strict_exclude_mask`), layered on
   top of pass 1 to narrow the set further:
   - Military Cub/Super Cub designators (L-4, L-18, L-21, C-145)
   - Rows tagged helicopter/jet/turboprop in `icaoaircrafttype`, unless the
     model is unambiguously one of an explicit allow-list of piston families
     (protects against source-data mistagging)
   - Large business/utility piston twins (Navajo/Chieftain, Cessna
     401/402/404/411/414/421, Beech Duke/Queen Air)

3. **icao24 format filter** (`valid_icao24_mask`) — drops any surviving row
   whose `icao24` is not exactly six hex characters. The raw database
   contains occasional malformed entries (e.g. a truncated 5-char code
   duplicating an aircraft that also has a valid row); such codes can never
   match real ADS-B traffic in stage 2, so they'd only be dead weight in the
   whitelist.

Both passes are manufacturer-scoped: a regex like `\bBARON\b` only applies
within rows already matched to the Beech brand, so it can't accidentally
keep or exclude an unrelated manufacturer's model.

---

## Stage 2 — `02_state_filtering.py`

Filters raw OpenSky ADS-B state-vector files down to the conventional-GA
aircraft whitelist produced by stage 1, then concatenates and sorts them
into one file per day.

- **Whitelist:** `data/active/aircraftDatabase-2022-06-conventionalGA.csv`
  (its `icao24` column; matched case-insensitively)
- **Input:** every `*.csv` in `data/archive/` whose filename contains a
  `YYYY-MM-DD` date, except the aircraft database files themselves and any
  file already named with `conventionalGA`, `sorted`, or `filtered`
- **Output:** `data/active/states/states_YYYY-MM-DD_conventionalGA_sorted.csv`,
  one per date, overwritten on each run; all original ADS-B columns and
  values are preserved (only row order and row membership change)

### Method

For each date found across the input filenames:

1. Read every matching file for that date in chunks (`state_filtering.CHUNK_SIZE`).
2. Drop rows with a missing/blank `icao24`.
3. Keep only rows whose `icao24`, lowercased, is in the whitelist — the
   original `icao24` casing is preserved in the output.
4. Concatenate all of that date's filtered rows into one DataFrame.
5. Sort by `icao24` (case-insensitively) then by `time` — falling back to
   `lastposupdate` only if `time` is absent from that day's data.
6. Save as `states_<date>_conventionalGA_sorted.csv`.

A file is only skipped silently if it's an already-processed pipeline output
or the aircraft database itself. Anything else that's malformed is treated
as a hard failure:

- Missing `icao24` column, or missing both `time` and `lastposupdate` →
  raises `ValueError` naming the offending file.
- No detectable `YYYY-MM-DD` date in the filename → warns and skips just
  that file.
- Completely empty file (0 bytes, no header) → warns and skips just that
  file; it isn't counted toward that day's "source files used".
- A date where every source file turned out empty → warns and skips the
  whole day (no output file is written for it).

---

## Stage 3 — `03_trajectory_prep.py`

Cleans and segments the daily sorted conventional-GA state-vector files
(stage 2's output) into per-flight trajectory segments. This stage does
**not** do ENU conversion and does **not** build training windows — it only
removes bad/stale points and splits each aircraft's daily point stream into
physically-plausible flight segments, ready for those later stages.

```bash
python 01_preprocessing/scripts/03_trajectory_prep.py
python 01_preprocessing/scripts/03_trajectory_prep.py --gap-split 90 --min-points 30
```

CLI flags: `--gap-split` (default 60s), `--min-duration` (default 300s),
`--min-points` (default 20), `--input-dir` (default: this project's
`data/active/states`), and `--output-dir` (default: `data/active/segments`;
the summary CSV is written alongside the segment files).

- **Input:** every `states_*_conventionalGA_sorted.csv` in `data/active/states/`
  (stage 2's output); files already containing `_segments` are ignored
- **Output:** `data/active/segments/states_YYYY-MM-DD_conventionalGA_segments.csv`,
  one per day, plus `data/active/segments/trajectory_prep_summary.csv`
  summarizing all days. Original sorted inputs are never modified.

### Method

For each day (`utils/trajectory_prep.py::process_day`), in order:

1. **Validate columns** — confirm `icao24`, `lat`, `lon`, `onground` exist;
   pick `lastposupdate` as the canonical timestamp (falling back to `time`
   only if `lastposupdate` is absent); pick `geoaltitude` as the altitude
   source (falling back to `baroaltitude`).

2. **Freshness filter** — rows with stale position updates are removed by
   requiring the position timestamp to be within 10 seconds of the
   state-vector timestamp (`time − lastposupdate ≤ 10s`). OpenSky repeats
   a stale `lat`/`lon` across multiple snapshots while `time` keeps
   advancing, so a large gap here means the row's position is old, not a
   new observation.

3. **Duplicate removal** — a separate repeated-state cleanup step, run
   right after the freshness filter but logically distinct from it: exact
   duplicate `(icao24, lastposupdate)` pairs are dropped, keeping the first
   occurrence. The freshness filter and duplicate removal together produce
   the "rows after freshness" count reported per day.

4. **`basic_clean`** — one pass over the freshness-filtered, deduplicated
   frame, each step's removed-row count logged separately: drop blank/missing
   `icao24`; drop missing `lat`/`lon`; drop invalid coordinates (`|lat|>90`,
   `|lon|>180`, or null-island); drop `onground == True` rows; coalesce
   `alt` = `geoaltitude` where present else `baroaltitude` and drop rows
   missing both; drop `alt` outside a generous `[-400, 12000]` m range.

5. **Re-sort** by `(icao24, timestamp)`.

6. **`assign_segments`** — segments are split primarily by aircraft identity
   and time gaps, with callsign changes used as an additional, more
   conservative split criterion (blank/NaN callsigns are treated as equal to
   each other, so missing callsigns never force a spurious split). A new
   segment starts whenever:
   - it's the aircraft's first point (a hard boundary), or
   - the time gap since the previous point exceeds `--gap-split` (default
     60s), or
   - the callsign changed from the previous point.

7. **`remove_glitch_points`** — computes implied ground speed between
   consecutive points via haversine distance / time gap. A single GPS
   glitch corrupts one point but creates two impossible speed jumps, so
   rather than splitting the segment there, the point most responsible for
   violating steps (`> 300 kt`, ~154.33 m/s) is iteratively removed and the
   segment re-checked, up to 3 passes. If more than 5% of steps still
   violate the threshold after 3 passes, the entire segment is discarded as
   untrustworthy rather than partially patched.

8. **`filter_valid_segments`** — keeps only segments spanning at least
   `--min-duration` (default 300s) **and** containing at least
   `--min-points` (default 20) points **and**, since a `velocity` column is
   present, with median reported velocity ≥ 15 m/s (~29 kt) — low enough not
   to remove normal airborne GA cruise/climb/descent, high enough to drop
   stationary clutter or taxi-like fragments that slipped past `onground`.

9. Save the result with all original ADS-B columns intact plus `alt`,
   `segment_id` (`{icao24}_{int(segment_start_time)}`), `segment_start_time`,
   `segment_end_time`, and `segment_duration_s` appended.

After all days are processed, a **validation gate** runs: an assertion that
the on-ground filter removed a nonzero number of rows on at least one day
(a sanity check on `parse_onground`), a gap-split sensitivity report (60 /
120 / 180s, informational only — the configured default is never changed
automatically), a cross-day consistency check flagging any day whose
removed-row fraction deviates more than 2x from the four-day median, a
spot-check of 3 random segments' implied-speed distribution, and an anchor
check that the combined median reported velocity falls in the expected
45–75 m/s range for light GA cruise.

---

## Stage 4 — `04_resample_trajectories.py`

Resamples stage 3's cleaned segments onto a uniform 10-second grid,
producing per-day trajectory CSVs ready for later ENU conversion and
dataset construction. This stage does **not** split train/test, normalize,
build ML windows, heading-align, or emit ENU tensors — those belong to
stage 5.

```bash
python 01_preprocessing/scripts/04_resample_trajectories.py
python 01_preprocessing/scripts/04_resample_trajectories.py --dt 5 --smooth
```

CLI flags: `--dt` (default 10s), `--max-interp-gap-s` (default 30s),
`--min-duration-s` (default 300s), `--min-points` (default 30),
`--max-speed-mps` (default ~154.33, i.e. 300 kt — aligned with stage 3's
glitch threshold so the two stages agree about the same physics),
`--max-accel-mps2` (default 10), `--max-turn-rate-deg-s` (default 6),
`--smooth` (off by default), `--input-dir` (default:
`data/active/segments`), and `--output-dir` (default:
`data/active/trajectories_10s`; the summary and audit CSVs are written
alongside the trajectory files).

- **Input:** every `states_*_conventionalGA_segments.csv` in
  `data/active/segments/` (stage 3's output)
- **Output:** `data/active/trajectories_10s/states_YYYY-MM-DD_conventionalGA_trajectories_10s.csv`,
  one per day, plus `trajectory_resample_summary.csv` (one row per day) and
  `trajectory_resample_dropped.csv` (audit trail: every dropped trajectory's
  id, source segment, drop reason, and size — so filter losses, especially
  maneuver-rich flights hitting the turn-rate rule, can be inspected rather
  than vanishing silently). Original segment files are never modified.

### Method

For each day (`utils/resample_trajectories.py::process_day`), per `segment_id`:

1. **Sort and deduplicate** — points are sorted by the canonical timestamp
   (`lastposupdate`, falling back to `time`); duplicate timestamps within a
   segment are dropped keeping the first. Segments left with fewer than 2
   points are discarded.

2. **Split at large gaps** — interpolation never bridges a time gap longer
   than `--max-interp-gap-s` (default 30s). The segment is split at every
   such gap and each resulting subsegment is resampled independently; the
   `was_split_by_interp_gap` output flag records whether a split occurred.

3. **Resample** — a uniform grid anchored at the subsegment's first
   timestamp (`t_grid = start + arange(0, duration + 1e-9, dt)`) with
   linear interpolation of `lat`/`lon`/`alt`. The grid never extends past
   the last original point, so there is no extrapolation. Longitude is
   unwrapped per subsegment before interpolating (and wrapped back to
   [-180, 180) on output), so an antimeridian crossing interpolates
   locally instead of sweeping across the planet. Linear interpolation is
   deliberate: splines can overshoot around sparse maneuvers, and inventing
   dynamics is worse than under-modeling them across a ≤30s hole. Each
   sample also gets an `is_interpolated` flag (true when no real fix lies
   within `dt/2` of the grid point) — synthetic samples inside a hole have
   constant velocity by construction, and downstream analyses need to be
   able to exclude them to avoid biasing a learned prior toward
   straightness.

4. **Optional light smoothing** — off by default; `--smooth` applies a
   centered rolling median (window 3) to the interpolated positions.
   Whether on or off, both `*_interp` and `*_smooth` columns are written
   (identical when smoothing is off). Deliberately no Kalman/RTS smoother
   yet — the `*_smooth` columns leave the seam for adding one later without
   schema changes.

5. **Motion quantities** — positions are converted to local flat-earth
   metres relative to the trajectory's first point (E = R·cos(lat₀)·Δlon,
   N = R·Δlat, R = 6,371,000 m) and differentiated on the grid.
   `accel_mps2` is **longitudinal** (Δ|v|/dt) — near zero in a steady turn,
   so the accel filter doesn't double-count turning flight that the
   turn-rate filter already polices; `accel_vector_mps2` (|Δv⃗|/dt,
   including the centripetal term) is also written for downstream
   statistics but never filtered on. Heading differences are wrapped to
   [-180°, 180°) before dividing by dt, so a track crossing north
   (359°→1°) reads as 2°, not 358°. Backward differences mean the first
   speed row copies the second, and the first two accel/turn-rate rows are
   NaN — these NaNs are excluded from all checks. Reported ADS-B channels
   (`velocity`, `heading`, `vertrate`) are interpolated onto the grid as
   `*_interp` columns when present (heading via unwrap→interpolate→rewrap),
   for reported-vs-derived cross-validation later.

6. **Plausibility filters** — a resampled trajectory is dropped (counted
   under its first failed rule, and logged to the audit CSV) if:
   duration < `--min-duration-s`; samples < `--min-points`; **any** speed
   > `--max-speed-mps`; more than 5% of |accel| values >
   `--max-accel-mps2`; or more than 5% of |turn rate| values >
   `--max-turn-rate-deg-s`. Note the dynamics limits police genuine
   aggressive flight, not just errors (a steep pattern turn can reach
   15–20 °/s) — the audit file exists precisely so those losses stay
   inspectable.

7. **Save** — one row per grid sample with identity (`icao24`, `callsign`,
   `segment_id`, `trajectory_id` = `{segment_id}_r{k}` for the k-th
   subsegment, `source_segment_id`), grid (`sample_idx`, `timestamp`,
   `dt_s`), positions, provenance (`is_interpolated`), motion quantities,
   reported channels, and segment/trajectory metadata. Aircraft metadata
   columns (`manufacturername`, `model`, `icaoaircrafttype`,
   `registration`, `typecode`) are carried through when present in the
   input.

After all days are processed, a **validation gate** runs and raises an
error on failure. Hard checks are the genuinely independent ones: at least
one output file exists; every within-trajectory step equals `dt` (via
`np.isclose` — float grids from `arange` aren't exactly equal); no
surviving trajectory violates the duration or sample-count minimums; and
the combined median speed lands in 25–90 m/s. The p50/p95/p99 table for
speed / |accel| / |accel_vec| / |turn rate| is **report-only** — asserting
p95 < threshold would be circular, since the filters just enforced ~p95
compliance at those exact values. It ends with a 3-trajectory random
spot-check.

## Extending

Each stage's rules live entirely in its own `utils/` module —
`ga_classification.py` for stage 1, `state_filtering.py` for stage 2,
`trajectory_prep.py` for stage 3 (edit the constants at its top:
`FRESHNESS_MAX_LAG_S`, `MAX_SPEED_KNOTS`, `MAX_GLITCH_PASSES`,
`PERSISTENT_VIOLATION_FRACTION`, `MIN_VELOCITY_MPS`, altitude bounds), and
`resample_trajectories.py` for stage 4 (thresholds arrive via the CLI; the
shared 5% violation allowance is `MAX_VIOLATION_FRACTION`). Edit those
directly; `io.py` and the entry-point scripts only need to change if the
underlying file/folder layout changes.
