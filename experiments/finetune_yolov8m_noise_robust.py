"""Noise-robust fine-tuning of YOLOv8m-cls for 6-class face-expression classification.

Why this script exists
----------------------
The earlier Ultralytics fine-tunes (320 px run, 256 px refine) plateaued at ~90 % because:
  * the training loss was driven to ~0.03 (99.8 % train accuracy) while validation loss rose - the
    network memorised mislabeled / ambiguous images;
  * several of the "tuned" arguments are no-ops for Ultralytics classification (mosaic, close_mosaic,
    translate, mixup, cutmix; hsv_* is disabled whenever auto_augment is set) and label smoothing was
    removed from Ultralytics, so there was no regularisation against label noise at all.

This script trains the same YOLOv8m-cls network with a custom loop that adds
label smoothing, mixup / cutmix, face-safe geometric augmentation, weight EMA and a 5-fold
out-of-fold (OOF) label audit (confident learning).

Stages
------
1. OOF audit : 5 group-aware folds over ALL images -> every image gets a prediction from a model that
               never saw it. Images whose given label is confidently contradicted are *flagged*
               (nothing is deleted; see label_audit.csv).
2. Final fit : train on the original train split minus flagged train images, select the epoch on
               validation, evaluate the original test split.

Test accuracy is reported twice: on ALL original test labels, and on the audited subset
(flagged test images excluded). Only the first is comparable with the earlier runs.
"""
from __future__ import annotations

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
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

CLASS_NAMES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
NC = len(CLASS_NAMES)
ROOT = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "outputs" / "_shared_work" / "dataset"))
    ap.add_argument("--split-csv", default=str(ROOT / "outputs" / "yolov8m" / "split.csv"))
    ap.add_argument("--weights", default=str(ROOT / "yolov8m-cls.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "yolov8m_finetuned" / "runs"))
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--oof-epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr0", type=float, default=5e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup-prob", type=float, default=0.4)
    ap.add_argument("--mixup-alpha", type=float, default=0.4)
    ap.add_argument("--cutmix-prob", type=float, default=0.2)
    ap.add_argument("--cutmix-alpha", type=float, default=1.0)
    ap.add_argument("--crop-scale-min", type=float, default=0.6, help="min area fraction of random crop")
    ap.add_argument("--degrees", type=float, default=10.0)
    ap.add_argument("--erasing", type=float, default=0.25)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-csv", default="", help="reuse an existing label_audit.csv and skip the OOF stage")
    ap.add_argument("--no-audit", action="store_true", help="skip OOF audit; train on all original train labels")
    ap.add_argument("--name", default="")
    return ap.parse_args()


# ----------------------------------------------------------------------------- data
def build_index(dataset: Path, split_csv: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        for li, cname in enumerate(CLASS_NAMES):
            for f in sorted((dataset / split / cname).iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    rows.append((str(f), split, li, int(f.name.split("_")[0])))
    df = pd.DataFrame(rows, columns=["crop_path", "split", "y", "idx"])
    src = pd.read_csv(split_csv).reset_index().rename(columns={"index": "idx"})
    df = df.merge(src[["idx", "path", "group", "split", "label"]], on="idx", suffixes=("", "_src"), validate="1:1")
    assert (df.split == df.split_src).all() and (df.label == df.y.map(lambda i: CLASS_NAMES[i])).all(), \
        "dataset folder does not match split.csv"
    return df.drop(columns=["split_src", "label"]).rename(columns={"path": "source_path"})


