"""Constants and helpers shared by more than one pipeline stage.

Everything here used to be duplicated per stage; a single definition
removes the drift hazard (e.g. two independently-written 300 kt
expressions that must stay equal).
"""

import os
import re
from typing import List, Tuple

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
INPUT_PREFIX = "states_"

EARTH_RADIUS_M = 6_371_000.0

# Physical-plausibility speed ceiling, shared by stage 3 (glitch removal)
# and stage 4 (trajectory drop rule) so the two stages police the same limit.
KNOTS_TO_MPS = 0.514444
MAX_SPEED_KNOTS = 300.0
MAX_SPEED_MPS = MAX_SPEED_KNOTS * KNOTS_TO_MPS       # ~154.33 m/s

# Position-time preference for stages 3-4: 'lastposupdate' is when the
# position was actually measured, 'time' is the snapshot time. (Stage 2
# deliberately sorts by the OPPOSITE order -- snapshot time first -- see
# state_filtering.TIME_COLUMN_CANDIDATES.)
POSITION_TIME_CANDIDATES = ["lastposupdate", "time"]


def find_timestamp_column(columns) -> str:
    """Return 'lastposupdate' if present (canonical), else 'time'; raise if neither exists."""
    for candidate in POSITION_TIME_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Neither 'lastposupdate' nor 'time' present in columns: {list(columns)}")


def discover_input_files(input_dir: str, suffix: str) -> List[Tuple[str, str]]:
    """Sorted (date, path) pairs for every states_*<suffix> file in input_dir."""
    results = []
    for name in sorted(os.listdir(input_dir)):
        if not (name.startswith(INPUT_PREFIX) and name.endswith(suffix)):
            continue
        match = DATE_PATTERN.search(name)
        if not match:
            print(f"WARNING: no date pattern found in filename '{name}'; skipping.")
            continue
        results.append((match.group(1), os.path.join(input_dir, name)))
    return results
