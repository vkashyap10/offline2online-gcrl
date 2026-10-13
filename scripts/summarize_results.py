"""Print a summary table of offline-to-online runs from their metrics.csv files.

Usage:
    python scripts/summarize_results.py <save_dir>/<project_name>

For each run: offline level (mean of the last 5 offline epochs), online success per epoch, online mean, and the change
from the offline level. Runs restored from an offline checkpoint take their offline level from <offline*>/seed<N>.
"""

import csv
import glob
import os
import sys


def success(path):
    rows = list(csv.DictReader(open(path)))
    return [float(r["eval/episode_success"]) for r in rows], [float(r["training/is_online"]) > 0 for r in rows]


def main(runs_dir):
    for path in sorted(glob.glob(os.path.join(runs_dir, "offline*", "seed*", "metrics.csv"))):
        s, _ = success(path)
        name = os.path.relpath(os.path.dirname(path), runs_dir)
        print(f"{name:32s} offline {len(s):2d} epochs: " + " ".join(f"{x:.2f}" for x in s) + f" | last5 {sum(s[-5:]) / len(s[-5:]):.2f}")
    print()
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "seed*", "metrics.csv")), key=os.path.getmtime):
        name = os.path.relpath(os.path.dirname(path), runs_dir)
        if name.startswith("offline"):
            continue
        s, online = success(path)
        on = [x for x, o in zip(s, online) if o]
        off = [x for x, o in zip(s, online) if not o]
        if not off:  # restored run: offline level from the matching offline job
            offline_name = "offline_fulldata" if "fulldata" in name else "offline"
            offline_path = os.path.join(runs_dir, offline_name, os.path.basename(os.path.dirname(path)), "metrics.csv")
            off = success(offline_path)[0] if os.path.exists(offline_path) else []
        if not on or not off:
            print(f"{name:32s} (no online epochs yet)")
            continue
        base, mean = sum(off[-5:]) / len(off[-5:]), sum(on) / len(on)
        print(f"{name:32s} start {base:.2f} | online {len(on):2d} epochs mean {mean:.2f} | change {mean - base:+.0%} | " + " ".join(f"{x:.2f}" for x in on))


if __name__ == "__main__":
    main(sys.argv[1])
