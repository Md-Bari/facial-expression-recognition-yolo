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


def load_image(path_size):
    path, s = path_size
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        r = s / min(w, h)
        nw, nh = max(s, round(w * r)), max(s, round(h * r))
        im = im.resize((nw, nh), Image.BILINEAR)
        l, t = (nw - s) // 2, (nh - s) // 2
        return np.asarray(im.crop((l, t, l + s, t + s)), dtype=np.uint8)


df = build_index(Path(a.dataset), Path(a.split_csv))
t0 = time.time()
with ThreadPoolExecutor(12) as ex:
    arrs = list(ex.map(load_image, [(q, a.imgsz) for q in df.crop_path]))
X = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous().pin_memory()
del arrs
print(f"loaded {len(df)} images at {a.imgsz}px in {time.time() - t0:.0f}s ({X.numel() / 1e9:.1f} GB, pinned CPU)",
      flush=True)
Y = df.y.values.astype(np.int64)
idx_tr, idx_va, idx_te = (np.where(df.split.values == s)[0] for s in ("train", "val", "test"))
audit = pd.read_csv(a.audit_csv)
flag = df.idx.map(audit.set_index("idx").flagged).fillna(False).values.astype(bool)
tr_clean, va_clean = idx_tr[~flag[idx_tr]], idx_va[~flag[idx_va]]
print(f"train {len(idx_tr)} -> {len(tr_clean)} | val {len(idx_va)} -> {len(va_clean)}", flush=True)


def fetch(idx):
    """uint8 CPU rows -> float GPU batch in [0,1]."""
    idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    return X.index_select(0, idx).pin_memory().cuda(non_blocking=True).float().div_(255)

