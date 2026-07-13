"""Filesystem path helpers shared by all pipeline stages.

Paths are resolved from the repository's own location, never the caller's
working directory, so every script behaves the same regardless of where it
is launched from.

Data layout (override the root with the WHACK_DATA_ROOT environment variable;
default is a `data/` directory next to the repository):

    <data root>/
    ├── archive/               # raw inputs, never modified
    └── active/                # everything the pipeline writes
        ├── states/            # stage 2 output
        ├── segments/          # stage 3 output
        └── trajectories_10s/  # stage 4 output (dir name follows --dt)
"""

import os

# Repository root: one level above utils/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_root() -> str:
    """$WHACK_DATA_ROOT if set, else `data/` beside the repository."""
    return os.environ.get("WHACK_DATA_ROOT") or os.path.join(os.path.dirname(_REPO_ROOT), "data")


def get_archive_dir() -> str:
    """Raw, never-modified inputs: the aircraft database and hourly state files."""
    return os.path.join(get_data_root(), "archive")


def get_active_dir() -> str:
    """Root for everything the pipeline writes."""
    return os.path.join(get_data_root(), "active")


# --- Stage 1: aircraft database -> conventional-GA whitelist ---

def get_aircraft_db_path() -> str:
    """Raw aircraft database CSV (stage 1 input)."""
    return os.path.join(get_archive_dir(), "aircraftDatabase-2022-06.csv")


def get_whitelist_path() -> str:
    """Conventional-GA whitelist CSV (stage 1 output, stage 2 input)."""
    return os.path.join(get_active_dir(), "aircraftDatabase-2022-06-conventionalGA.csv")


# --- Stage 2: ADS-B state vectors -> filtered, sorted daily files ---

def get_state_files_dir() -> str:
    """Directory scanned for raw hourly ADS-B state-vector CSVs."""
    return get_archive_dir()


def get_states_dir() -> str:
    """Daily filtered/sorted state-vector CSVs (stage 2 output, stage 3 input)."""
    return os.path.join(get_active_dir(), "states")


# --- Stage 3: sorted daily states -> per-flight trajectory segments ---

def get_segments_dir() -> str:
    """Per-day segment CSVs and their summary (stage 3 output, stage 4 input)."""
    return os.path.join(get_active_dir(), "segments")


def get_summary_path(output_dir: str = "") -> str:
    """Cross-day trajectory-prep summary CSV, kept next to the segment files."""
    return os.path.join(output_dir or get_segments_dir(), "trajectory_prep_summary.csv")


# --- Stage 4: segments -> uniform-grid trajectories ---

def dt_tag(dt_s: float) -> str:
    """Grid-spacing tag for stage-4 names: 10.0 -> '10s', 4.0 -> '4s', 2.5 -> '2p5s'.

    Tagging keeps products resampled at different scan periods side by side
    instead of overwriting each other.
    """
    return f"{dt_s:g}".replace(".", "p") + "s"


def get_trajectories_dir(dt_s: float = 10.0) -> str:
    """Per-day uniform-grid trajectory CSVs and their summary (stage 4 output)."""
    return os.path.join(get_active_dir(), f"trajectories_{dt_tag(dt_s)}")


def get_resample_summary_path(output_dir: str = "") -> str:
    """Cross-day resampling summary CSV, kept next to the trajectory files."""
    return os.path.join(output_dir or get_trajectories_dir(), "trajectory_resample_summary.csv")


def get_dropped_audit_path(output_dir: str = "") -> str:
    """Dropped-trajectory audit CSV, kept next to the trajectory files."""
    return os.path.join(output_dir or get_trajectories_dir(), "trajectory_resample_dropped.csv")
