"""Noise-robust fine-tune of an Ultralytics classification backbone (same recipe as
yolo11m_finetune_noise_robust.ipynb), as a script so several backbones / sizes can be run and ensembled.

Saves in --out/<name>/: probs.npz (val + test class probabilities, single view and flip-TTA, using the best
validation epoch), results.csv, best_ema_state.pt, weights/best.pt (Ultralytics-loadable), args.json.

Images are kept in pinned CPU memory (batches are copied to the GPU), so it can share the GPU with an idle kernel.
"""
import argparse
import copy
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

CLASS_NAMES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
NC = len(CLASS_NAMES)
ROOT = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--weights", default=str(ROOT / "yolo11m-cls.pt"))
p.add_argument("--imgsz", type=int, default=256)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--epochs", type=int, default=45)
p.add_argument("--lr0", type=float, default=5e-4)
p.add_argument("--lr_min", type=float, default=1e-6)
p.add_argument("--weight_decay", type=float, default=0.05)
p.add_argument("--warmup_epochs", type=float, default=2.0)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--label_smoothing", type=float, default=0.1)
p.add_argument("--mixup_prob", type=float, default=0.4)
p.add_argument("--mixup_alpha", type=float, default=0.4)
p.add_argument("--cutmix_prob", type=float, default=0.2)
p.add_argument("--cutmix_alpha", type=float, default=1.0)
p.add_argument("--crop_scale_min", type=float, default=0.6)
p.add_argument("--degrees", type=float, default=10.0)
p.add_argument("--erasing", type=float, default=0.25)
p.add_argument("--ema_decay", type=float, default=0.999)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--dataset", default=str(ROOT / "outputs" / "_shared_work" / "dataset"))
p.add_argument("--split_csv", default=str(ROOT / "outputs" / "yolo11m" / "split.csv"))
p.add_argument("--audit_csv", default=str(ROOT / "outputs" / "yolov8m_finetuned" / "runs" /
                                          "yolov8m_noise_robust_nb_20261111_191527" / "label_audit.csv"))
p.add_argument("--out", default=str(ROOT / "outputs" / "ensemble_runs"))
a = p.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
torch.backends.cudnn.benchmark = True
RUN = Path(a.out) / a.name
(RUN / "weights").mkdir(parents=True, exist_ok=True)
json.dump(vars(a), open(RUN / "args.json", "w"), indent=2)


# ---------------------------------------------------------------- data
def build_index(dataset: Path, split_csv: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        for li, cname in enumerate(CLASS_NAMES):
            for f in sorted((dataset / split / cname).iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    rows.append((str(f), split, li, int(f.name.split("_")[0])))
    df = pd.DataFrame(rows, columns=["crop_path", "split", "y", "idx"])
    src = pd.read_csv(split_csv).reset_index().rename(columns={"index": "idx"})
    df = df.merge(src[["idx", "split"]], on="idx", suffixes=("", "_src"), validate="1:1")
    assert (df.split == df.split_src).all(), "dataset folder does not match split.csv"
    return df.drop(columns=["split_src"])


