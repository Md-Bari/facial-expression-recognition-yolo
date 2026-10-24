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


def load_image(path_size):
    path, s = path_size
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        r = s / min(w, h)  # same policy as Ultralytics classify val: short side -> s, centre crop
        nw, nh = max(s, round(w * r)), max(s, round(h * r))
        im = im.resize((nw, nh), Image.BILINEAR)
        l, t = (nw - s) // 2, (nh - s) // 2
        return np.asarray(im.crop((l, t, l + s, t + s)), dtype=np.uint8)


def load_all(df: pd.DataFrame, s: int) -> torch.Tensor:
    t0 = time.time()
    with ThreadPoolExecutor(12) as ex:
        arrs = list(ex.map(load_image, [(p, s) for p in df.crop_path]))
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()  # N,3,S,S uint8
    print(f"loaded {len(df)} images at {s}px in {time.time() - t0:.0f}s ({x.numel() / 1e9:.1f} GB)", flush=True)
    return x


# ----------------------------------------------------------------------------- augmentation (GPU)
def gpu_augment(xb: torch.Tensor, a) -> torch.Tensor:
    """xb: B,3,S,S float in [0,1]. Face-safe: mild zoom/aspect/rotation/shift, h-flip, brightness/contrast, erasing."""
    B, dev = xb.shape[0], xb.device
    area = torch.empty(B, device=dev).uniform_(a.crop_scale_min, 1.0)
    logr = torch.empty(B, device=dev).uniform_(math.log(0.85), math.log(1 / 0.85))
    sx = (area * torch.exp(logr)).sqrt().clamp(max=1.0)
    sy = (area / torch.exp(logr)).sqrt().clamp(max=1.0)
    ang = torch.empty(B, device=dev).uniform_(-a.degrees, a.degrees) * math.pi / 180
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - sx)
    ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - sy)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([
        torch.stack([cos * sx * flip, -sin * sy, tx], 1),
        torch.stack([sin * sx * flip, cos * sy, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, xb.shape, align_corners=False)
    xb = F.grid_sample(xb, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # photometric
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.12, 0.12)
    m = xb.mean(dim=(1, 2, 3), keepdim=True)
    xb = ((xb - m) * c + m + b).clamp_(0, 1)
    # random erasing
    if a.erasing > 0:
        S = xb.shape[-1]
        for i in torch.nonzero(torch.rand(B, device=dev) < a.erasing).flatten().tolist():
            ar = np.random.uniform(0.02, 0.2) * S * S
            rr = math.exp(np.random.uniform(math.log(0.4), math.log(2.5)))
            h, w = int(min(S, round(math.sqrt(ar * rr)))), int(min(S, round(math.sqrt(ar / rr))))
            t, l = np.random.randint(0, S - h + 1), np.random.randint(0, S - w + 1)
            xb[i, :, t:t + h, l:l + w] = torch.rand(3, 1, 1, device=dev)
    return xb


def mix(xb, yb, a):
    """Returns mixed images and soft targets (label smoothing applied)."""
    tgt = F.one_hot(yb, NC).float() * (1 - a.label_smoothing) + a.label_smoothing / NC
    r = np.random.rand()
    if r < a.mixup_prob:
        lam = np.random.beta(a.mixup_alpha, a.mixup_alpha)
        perm = torch.randperm(xb.shape[0], device=xb.device)
        xb = lam * xb + (1 - lam) * xb[perm]
        tgt = lam * tgt + (1 - lam) * tgt[perm]
    elif r < a.mixup_prob + a.cutmix_prob:
        lam = np.random.beta(a.cutmix_alpha, a.cutmix_alpha)
        perm = torch.randperm(xb.shape[0], device=xb.device)
        S = xb.shape[-1]
        h, w = int(S * math.sqrt(1 - lam)), int(S * math.sqrt(1 - lam))
        t, l = np.random.randint(0, S - h + 1), np.random.randint(0, S - w + 1)
        xb = xb.clone()
        xb[:, :, t:t + h, l:l + w] = xb[perm][:, :, t:t + h, l:l + w]
        lam = 1 - h * w / (S * S)
        tgt = lam * tgt + (1 - lam) * tgt[perm]
    return xb, tgt


