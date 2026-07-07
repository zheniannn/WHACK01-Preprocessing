"""Filesystem path helpers for the preprocessing pipeline (all three stages).

Paths are resolved from this file's location rather than the current working
directory, so scripts behave the same no matter where they're launched from.
"""

import os


def get_project_root() -> str:
    """Return the repo root (two levels above utils/, which lives inside 01_preprocessing/)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Stage 1 (01_preprocessing.py): aircraft database -> conventional-GA whitelist ---

def get_input_path() -> str:
    """Path to the raw, unfiltered aircraft database CSV."""
    return os.path.join(get_project_root(), "data", "archive", "aircraftDatabase-2022-06.csv")


def get_output_path() -> str:
    """Path to the filtered conventional-GA aircraft CSV produced by stage 1."""
    return os.path.join(get_project_root(), "data", "active", "aircraftDatabase-2022-06-conventionalGA.csv")


# --- Stage 2 (02_state_filtering.py): ADS-B state vectors -> filtered daily files ---

def get_whitelist_path() -> str:
    """Path to the conventional-GA aircraft whitelist (stage 1's output, read as stage 2's input)."""
    return get_output_path()


def get_state_files_dir() -> str:
    """Directory containing the raw OpenSky ADS-B state-vector CSV files."""
    return os.path.join(get_project_root(), "data", "archive")


def get_states_dir() -> str:
    """Directory of the daily, filtered, sorted state-vector CSVs (stage 2's output, stage 3's input)."""
    return os.path.join(get_project_root(), "data", "active", "states")


# --- Stage 3 (03_trajectory_prep.py): sorted daily states -> trajectory segments ---

def get_segments_dir() -> str:
    """Directory for the per-day segmented CSVs and the summary CSV."""
    return os.path.join(get_project_root(), "data", "active", "segments")


def get_summary_path(output_dir: str = "") -> str:
    """Path to the cross-day trajectory-prep summary CSV (kept next to the segment files)."""
    return os.path.join(output_dir or get_segments_dir(), "trajectory_prep_summary.csv")


# --- Stage 4 (04_resample_trajectories.py): segments -> uniform 10 s trajectories ---

def get_trajectories_dir() -> str:
    """Directory for the per-day uniform-grid trajectory CSVs and their summary."""
    return os.path.join(get_project_root(), "data", "active", "trajectories_10s")


def get_resample_summary_path(output_dir: str = "") -> str:
    """Path to the cross-day resampling summary CSV (kept next to the trajectory files)."""
    return os.path.join(output_dir or get_trajectories_dir(), "trajectory_resample_summary.csv")


def get_dropped_audit_path(output_dir: str = "") -> str:
    """Path to the dropped-trajectory audit CSV (kept next to the trajectory files)."""
    return os.path.join(output_dir or get_trajectories_dir(), "trajectory_resample_dropped.csv")
