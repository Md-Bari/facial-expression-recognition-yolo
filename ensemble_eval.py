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

acc = lambda p, y, m=None: float((p.argmax(1) == y)[slice(None) if m is None else m].mean())

# per member: flip-TTA chosen on validation
view = {}
rows = []
for n, d in members.items():
    view[n] = "_flip" if acc(d["val_flip"], yv) > acc(d["val"], yv) else ""
    rows.append({"member": n, "val": acc(d["val"], yv), "val_flip": acc(d["val_flip"], yv), "tta": bool(view[n]),
                 "test_all": acc(d["test" + view[n]], yt)})
print(pd.DataFrame(rows).round(4).to_string(index=False))

# greedy forward selection with replacement (Caruana et al., 2004) on validation log-likelihood + accuracy
V = {n: d["val" + view[n]] for n, d in members.items()}
T = {n: d["test" + view[n]] for n, d in members.items()}
score = lambda p: (acc(p, yv), float(np.log(np.clip(p[np.arange(len(yv)), yv], 1e-9, 1)).mean()))
chosen, cur = [], None
for _ in range(10):
    best = max(V, key=lambda n: score(V[n] if cur is None else (cur * len(chosen) + V[n]) / (len(chosen) + 1)))
    new = V[best] if cur is None else (cur * len(chosen) + V[best]) / (len(chosen) + 1)
    if cur is not None and score(new) <= score(cur):
        break
    chosen.append(best)
    cur = new
w = pd.Series(chosen).value_counts() / len(chosen)
print("\nselected on validation:", w.round(3).to_dict(), "| val acc", round(acc(cur, yv), 4))
pt = sum(T[n] * wi for n, wi in w.items())
pt_all = sum(T.values()) / len(T)


def report(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(CLASS_NAMES)))
    return {"images": int(len(y)), "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "per_class_accuracy": {c: float(cm[i, i] / max(1, cm[i].sum())) for i, c in enumerate(CLASS_NAMES)},
            "confusion_matrix": cm.tolist()}


corr = pd.read_csv(ROOT / "outputs" / "label_review" / "label_corrections.csv").set_index("idx")
di = pd.Series(ref["dataset_idx_test"])
action = di.map(corr.action).fillna("keep").values
y_corr = np.array([CLASS_NAMES.index(nl) if ac == "relabel" else y
                   for ac, nl, y in zip(action, di.map(corr.new_label).values, yt)])
keep = action != "remove"

