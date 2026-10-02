"""Ensemble the members in outputs/ensemble_runs/*/probs.npz.

Everything is chosen on VALIDATION only (member subset by greedy forward selection, flip-TTA per member);
the test split is touched once, at the end, for reporting:
  * test_all_original_labels - comparable with every earlier run
  * test_audited_subset      - test images flagged by the out-of-fold label audit excluded (not human-verified)
  * test_corrected_labels    - the manually reviewed labels in outputs/label_review/label_corrections.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "outputs" / "ensemble_runs"
CLASS_NAMES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

members = {}
for f in sorted(RUNS.glob("*/probs.npz")):
    members[f.parent.name] = dict(np.load(f))
assert members, "no finished members yet"
ref = next(iter(members.values()))
yv, yt = ref["y_val"], ref["y_test"]
for d in members.values():
    assert (d["val_idx"] == ref["val_idx"]).all() and (d["test_idx"] == ref["test_idx"]).all()

