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

