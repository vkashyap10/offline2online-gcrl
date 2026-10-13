"""Plot offline-to-online results from the metrics.csv files written by training runs.

Usage:
    python scripts/plot_results.py --runs_dir=<save_dir>/<project_name> --out_dir=docs/figures

Expects runs at <runs_dir>/<experiment_name>/seed<N>/metrics.csv.
Writes learning_curves.png (success per epoch) and success_change.png (online mean minus offline level per setting).
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette, fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

CURVES = [
    ("gcivl_r0.0_all", "Online data only, all episodes"),
    ("gcivl_r0.0_succ", "Online data only, successful episodes"),
    ("gcivl_r0.5_all", "50% offline data, all episodes"),
    ("gcivl_r1.0_all", "Offline data only (online data unused)"),
]
BARS = [
    ("gcivl_r0.0_all", "GCIVL, online only, all"),
    ("gcivl_r0.0_succ", "GCIVL, online only, successful"),
    ("gcivl_r0.5_all", "GCIVL, 50% offline, all"),
    ("gcivl_r0.5_succ", "GCIVL, 50% offline, successful"),
    ("gcivl_r1.0_all", "GCIVL, offline only"),
    ("gcivl_r0.0_all_utd1", "GCIVL, online only, all, 4x updates"),
    ("iql_r0.0_all", "IQL, online only, all"),
    ("iql_r0.5_all", "IQL, 50% offline, all"),
]


def load(path):
    rows = list(csv.DictReader(open(path)))
    success = np.array([float(r["eval/episode_success"]) for r in rows])
    online = np.array([float(r["training/is_online"]) for r in rows]) > 0
    return success, online


def runs_for(runs_dir, name):
    """Load each seed of a setting. Runs restored from an offline checkpoint contain only online epochs; their offline
    epochs are taken from the matching offline job (<runs_dir>/offline*/seed<N>), so every run has both phases."""
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, name, "seed*", "metrics.csv"))):
        success, online = load(path)
        if online.all():
            seed_dir = os.path.basename(os.path.dirname(path))
            offline_name = "offline_fulldata" if name.endswith("_fulldata") else "offline"
            offline_path = os.path.join(runs_dir, offline_name, seed_dir, "metrics.csv")
            if not os.path.exists(offline_path):
                continue
            off_success, _ = load(offline_path)
            success = np.concatenate([off_success, success])
            online = np.concatenate([np.zeros(len(off_success), bool), online])
        runs.append((success, online))
    return runs


def style(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(runs_dir, out):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    style(ax)
    num_offline = None
    for color, (name, label) in zip(COLORS, CURVES):
        runs = [r for r in runs_for(runs_dir, name) if r[1].sum() == 10]  # completed runs only
        if not runs:
            continue
        num_offline = int((~runs[0][1]).sum())
        curves = np.stack([s for s, _ in runs])
        epochs = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(0)
        if len(runs) > 1:
            ax.fill_between(epochs, curves.min(0), curves.max(0), color=color, alpha=0.12, linewidth=0)
        ax.plot(epochs, mean, color=color, linewidth=2, label=f"{label} (n={len(runs)})")
    if num_offline:
        ax.axvline(num_offline + 0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(num_offline / 2 + 0.5, 1.04, "offline training", ha="center", color=MUTED, fontsize=9)
        ax.text(num_offline + 5.5, 1.04, "online fine-tuning", ha="center", color=MUTED, fontsize=9)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("Epoch", color=INK, fontsize=10)
    ax.set_ylabel("Success rate", color=INK, fontsize=10)
    ax.set_xticks(range(1, 21, 2) if num_offline else ax.get_xticks())
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(out, facecolor="#fcfcfb")


def plot_change(runs_dir, out):
    entries = []
    for name, label in BARS:
        runs = [r for r in runs_for(runs_dir, name) if r[1].sum() == 10]
        if runs:
            changes = [s[online].mean() - s[~online][-5:].mean() for s, online in runs]
            entries.append((label, changes))
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(entries) + 1.2), dpi=150)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for i, (label, changes) in enumerate(entries):
        ax.barh(i, np.mean(changes), height=0.55, color=COLORS[0], alpha=0.85)
        ax.scatter(changes, [i] * len(changes), s=24, color=INK, zorder=3, linewidths=0)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(entries)), [f"{label} (n={len(c)})" for label, c in entries], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Change in success: online mean minus last 5 offline epochs (dots = seeds)", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, facecolor="#fcfcfb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out_dir", default="docs/figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_curves(args.runs_dir, os.path.join(args.out_dir, "learning_curves.png"))
    plot_change(args.runs_dir, os.path.join(args.out_dir, "success_change.png"))
    print(f"Wrote figures to {args.out_dir}")
