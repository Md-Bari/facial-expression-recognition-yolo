"""Build a class-balanced dataset from the images a trained model classifies correctly.

Steps
-----
1. Run a previously trained classifier (default: YOLO11l, the best base run) on every face crop in
   outputs/_shared_work/dataset/{train,val,test}/<class>/.
2. Keep only the correctly classified images and copy them to the output folder (same split / class layout).
3. For every class whose correct-image count is below --min-per-class (default 3000), create augmented
   copies until the class reaches that count. Classes already at or above it are left untouched.

Augmented copies are generated ONLY from training images and written ONLY to the train split, so
validation / test keep real images and no augmented twin of a val / test image leaks into training.

Outputs (in --out):
    train|val|test/<class>/*.jpg      balanced dataset, directly usable as an Ultralytics classify dataset
    manifest.csv                      one row per output image (source, split, class, original / augmented)
    class_counts.csv                  per-class counts before and after balancing

Usage
-----
    python balance_dataset.py
    python balance_dataset.py --model outputs/yolov8m/runs/yolov8m_70_15_15/weights/best.pt --min-per-class 3000
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "outputs" / "_shared_work" / "dataset"),
                    help="source dataset with train/val/test/<class>/ folders")
    ap.add_argument("--model", default=str(ROOT / "outputs" / "yolo11l" / "runs" / "yolo11l_70_15_15" / "weights" / "best.pt"),
                    help="trained Ultralytics classification checkpoint used to decide which images are correct")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "balanced_dataset"))
    ap.add_argument("--min-per-class", type=int, default=3000)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="delete --out first if it already exists")
    return ap.parse_args()


def list_images(dataset: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for class_dir in sorted(p for p in (dataset / split).iterdir() if p.is_dir()):
            for f in sorted(class_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append((str(f), split, class_dir.name))
    if not rows:
        raise FileNotFoundError(f"No images found under {dataset}")
    return pd.DataFrame(rows, columns=["path", "split", "label"])


def classify(df: pd.DataFrame, model_path: str, imgsz: int, batch: int) -> pd.DataFrame:
    """Adds predicted label, confidence and correct flag using the trained model."""
    import torch
    from ultralytics import YOLO

