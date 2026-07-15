"""Report figures for the preprocessing pipeline (stage-4 output).

Reads the uniform-grid trajectory CSVs and renders four PNGs:

  1_coverage_map.png       log-density map of every trajectory sample,
                           all days -- where the ground truth lives.
  2_trajectory_gallery.png 5x5 gallery of individual ground tracks --
                           the shape variety the whitelist captures.
  3_regime_profile.png     duration / altitude / speed / turn-rate
                           distributions -- the dynamic regimes covered.
  4_resampling_grid.png    one trajectory on the 10 s grid, gap-filled
                           samples highlighted -- what stage 4 produces.

Days are streamed one at a time (files are ~1.2 GB each); the coverage
map accumulates a fixed-bin histogram so memory stays bounded.

Usage:
    python scripts/make_figures.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import get_plot_dir, get_trajectories_dir

# Palette shared with WHACK02-Radar so the two repos' figures match.
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
C_TARGET = "#2a78d6"; C_ACCENT = "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12,
})

FILE_SUFFIX = "_conventionalGA_trajectories_10s.csv"
GALLERY_SEED = 20220606
KM_PER_DEG = 111.32   # great-circle km per degree of latitude


def day_paths(traj_dir):
    """Sorted (date, path) pairs for every per-day trajectory CSV."""
    names = sorted(n for n in os.listdir(traj_dir) if n.endswith(FILE_SUFFIX))
    return [(n.split("_")[1], os.path.join(traj_dir, n)) for n in names]


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {name} -> {path}")


def coverage_map(days, out_dir):
    """Fig 1: log-density of all trajectory samples, accumulated per day."""
    # Fixed world bins so days can be histogrammed independently and summed.
    lon_edges = np.linspace(-180, 180, 1441)   # 0.25 deg cells
    lat_edges = np.linspace(-90, 90, 721)
    h = np.zeros((len(lon_edges) - 1, len(lat_edges) - 1))
    n_rows = 0
    for date, path in days:
        df = pd.read_csv(path, usecols=["lat_interp", "lon_interp"])
        h += np.histogram2d(df["lon_interp"], df["lat_interp"],
                            bins=[lon_edges, lat_edges])[0]
        n_rows += len(df)

    # Crop to the occupied area (with margin) so the map fills the frame.
    occ_lon = np.where(h.sum(axis=1) > 0)[0]
    occ_lat = np.where(h.sum(axis=0) > 0)[0]
    i0, i1 = max(occ_lon[0] - 8, 0), min(occ_lon[-1] + 8, h.shape[0])
    j0, j1 = max(occ_lat[0] - 8, 0), min(occ_lat[-1] + 8, h.shape[1])

    cmap = LinearSegmentedColormap.from_list("density", [SURFACE, C_TARGET, INK])
    fig, ax = plt.subplots(figsize=(11, 6.5))
    m = ax.imshow(h[i0:i1, j0:j1].T, origin="lower", aspect="auto", cmap=cmap,
                  norm=LogNorm(vmin=1),
                  extent=[lon_edges[i0], lon_edges[i1], lat_edges[j0], lat_edges[j1]])
    cb = fig.colorbar(m, ax=ax, shrink=0.85, label="10 s samples per 0.25° cell")
    cb.ax.yaxis.label.set_color(INK2)
    ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)")
    ax.set_title(f"Ground-truth coverage — {n_rows:,} samples across {len(days)} days\n"
                 "conventional-GA trajectories on the uniform 10 s grid", color=INK)
    _save(fig, out_dir, "1_coverage_map.png")


def trajectory_gallery(day, out_dir):
    """Fig 2: 5x5 gallery of ground tracks, spread across duration deciles."""
    date, path = day
    df = pd.read_csv(path, usecols=["trajectory_id", "lat_interp", "lon_interp",
                                    "trajectory_duration_s"])
    per = df.groupby("trajectory_id")["trajectory_duration_s"].first().sort_values()
    # Deterministic spread: evenly spaced ranks, jittered inside their band.
    n_cells = min(25, len(per))
    rng = np.random.default_rng(GALLERY_SEED)
    ranks = (np.linspace(0.04, 0.96, n_cells) + rng.uniform(-0.015, 0.015, n_cells)) * (len(per) - 1)
    ids = per.index[np.clip(ranks.astype(int), 0, len(per) - 1)]

    fig, axes = plt.subplots(5, 5, figsize=(11, 11))
    for ax, tid in zip(axes.flat, ids):
        g = df[df["trajectory_id"] == tid]
        lat0 = g["lat_interp"].mean()
        e = (g["lon_interp"] - g["lon_interp"].mean()) * np.cos(np.radians(lat0)) * KM_PER_DEG
        n = (g["lat_interp"] - lat0) * KM_PER_DEG
        ax.plot(e, n, color=C_TARGET, lw=0.9)
        ax.plot(e.iloc[0], n.iloc[0], "o", color="#1baf7a", ms=3.5)   # start
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        span = max(e.max() - e.min(), n.max() - n.min(), 1.0)
        ax.set_title(f"{per[tid] / 60:.0f} min · {span:.0f} km", fontsize=8, color=INK2)
    fig.suptitle(f"Trajectory shape gallery — 25 of {len(per):,} trajectories ({date})\n"
                 "spread across duration deciles; green dot marks the start",
                 color=INK, y=0.995)
    _save(fig, out_dir, "2_trajectory_gallery.png")


def regime_profile(day, out_dir):
    """Fig 3: distributions of the dynamic quantities the radar will see."""
    date, path = day
    df = pd.read_csv(path, usecols=["trajectory_id", "trajectory_duration_s",
                                    "alt_interp", "speed_mps", "turn_rate_deg_s"])
    dur_min = df.groupby("trajectory_id")["trajectory_duration_s"].first() / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        (axes[0, 0], dur_min, "trajectory duration (min)", (0, 240), 60),
        (axes[0, 1], df["alt_interp"], "altitude (m MSL)", (0, 6000), 60),
        (axes[1, 0], df["speed_mps"], "ground speed (m/s)", (0, 120), 60),
        (axes[1, 1], df["turn_rate_deg_s"].abs(), "|turn rate| (deg/s)", (0, 6), 60),
    ]
    for ax, x, label, rng_, bins in panels:
        ax.hist(x.dropna(), bins=bins, range=rng_, color=C_TARGET, alpha=0.85)
        ax.set_xlabel(label); ax.set_yscale("log")
        ax.set_ylabel("count (log)")
    fig.suptitle(f"Dynamic-regime profile — {dur_min.size:,} trajectories ({date})\n"
                 "straight cruise dominates, but turns and low-and-slow flight are retained",
                 color=INK, y=0.99)
    _save(fig, out_dir, "3_regime_profile.png")


def resampling_grid(day, out_dir):
    """Fig 4: one trajectory on the 10 s grid, gap-filled samples marked."""
    date, path = day
    df = pd.read_csv(path, usecols=["trajectory_id", "lat_interp", "lon_interp",
                                    "is_interpolated", "n_samples"])
    # A mid-sized trajectory with a visible (but not dominant) gap-filled share;
    # fall back to the most-gap-filled, then to the first, if the band is empty.
    frac = df.groupby("trajectory_id").agg(
        f=("is_interpolated", "mean"), n=("n_samples", "first"))
    pick = frac[(frac["f"].between(0.05, 0.25)) & (frac["n"].between(150, 400))]
    tid = pick.index[0] if len(pick) else frac["f"].idxmax()
    g = df[df["trajectory_id"] == tid]
    lat0 = g["lat_interp"].mean()
    e = (g["lon_interp"] - g["lon_interp"].mean()) * np.cos(np.radians(lat0)) * KM_PER_DEG
    n = (g["lat_interp"] - lat0) * KM_PER_DEG
    syn = g["is_interpolated"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.plot(e, n, color=GRID, lw=1.0, zorder=1)
    ax.scatter(e[~syn], n[~syn], s=10, color=C_TARGET, lw=0, zorder=3,
               label=f"anchored by a real fix ({int((~syn).sum())})")
    ax.scatter(e[syn], n[syn], s=14, facecolor="none", edgecolor=C_ACCENT, lw=1.0,
               zorder=4, label=f"gap-filled, constant-velocity ({int(syn.sum())})")
    ax.set_aspect("equal")
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    leg = ax.legend(loc="best", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title(f"Stage-4 product — one trajectory on the uniform 10 s grid ({date})\n"
                 "every sample sits on the grid; samples far from any real fix are flagged",
                 color=INK)
    _save(fig, out_dir, "4_resampling_grid.png")


def main():
    parser = argparse.ArgumentParser(description="Render the report figures.")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Stage-4 trajectory dir (default: active/trajectories_10s).")
    args = parser.parse_args()

    traj_dir = args.input_dir or get_trajectories_dir()
    days = day_paths(traj_dir)
    if not days:
        raise FileNotFoundError(f"No stage-4 trajectory CSVs found in {traj_dir}")
    out_dir = get_plot_dir()
    print(f"rendering figures from {len(days)} day(s) -> {out_dir}")

    coverage_map(days, out_dir)
    trajectory_gallery(days[0], out_dir)
    regime_profile(days[0], out_dir)
    resampling_grid(days[0], out_dir)
    print("make_figures completed successfully.")


if __name__ == "__main__":
    main()
