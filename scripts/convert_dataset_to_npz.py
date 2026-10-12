"""Convert a legacy offline dataset (.pkl list of per-transition dicts) to a compact .npz of stacked arrays.

Usage: python scripts/convert_dataset_to_npz.py path/to/dataset.pkl [more.pkl ...]
Writes path/to/dataset.npz next to each input.
"""

import os
import pickle
import sys

import numpy as np

from offline2online_gcrl.impls.utils.buffer_setup import stack_transitions

for pkl_path in sys.argv[1:]:
    with open(pkl_path, "rb") as f:
        transitions = pickle.load(f)
    data = stack_transitions(transitions)
    del transitions
    npz_path = os.path.splitext(pkl_path)[0] + ".npz"
    np.savez(npz_path, **data)
    print(f"{pkl_path} -> {npz_path}: " + ", ".join(f"{k} {v.shape} {v.dtype}" for k, v in data.items()), flush=True)
