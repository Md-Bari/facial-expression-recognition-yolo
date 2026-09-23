"""Show which face regions a trained YOLO classification model uses for its prediction.

Two independent methods are drawn side by side for each image:
  * Grad-CAM            - gradient of the predicted class score w.r.t. the last convolutional feature map
                          (fast; shows where the evidence FOR the predicted class is located).
  * Occlusion sensitivity - a grey patch is slid over the face and the drop in the predicted-class
                          probability is recorded (slow but model-agnostic; shows which regions the
                          prediction actually DEPENDS on).
If both maps highlight the same area (e.g. mouth for happy, eyes/brows for angry), the explanation is reliable.

Outputs (in --out):
    <class>_examples.jpg        rows of: face | Grad-CAM | occlusion map, for a few test images per class
    class_mean_gradcam.jpg      average Grad-CAM per class over many correctly classified faces
                                (the crops are face-aligned, so this is the typical region used per emotion)

Usage
-----
    python explain_face_regions.py
    python explain_face_regions.py --model outputs/yolov8m/runs/yolov8m_70_15_15/weights/best.pt --images a.jpg b.jpg
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "outputs" / "yolo11l" / "runs" / "yolo11l_70_15_15" / "weights" / "best.pt"))
    ap.add_argument("--dataset", default=str(ROOT / "outputs" / "_shared_work" / "dataset"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "explanations"))
    ap.add_argument("--images", nargs="*", default=None, help="explain these image files instead of sampling the dataset")
    ap.add_argument("--examples-per-class", type=int, default=6)
    ap.add_argument("--mean-per-class", type=int, default=150, help="images averaged per class for the mean map")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--patch", type=int, default=48, help="occlusion patch size in pixels")
    ap.add_argument("--stride", type=int, default=16, help="occlusion stride in pixels")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

