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

    model = YOLO(model_path)
    names = [model.names[i] for i in range(len(model.names))]
    missing = set(df.label) - set(names)
    if missing:
        raise ValueError(f"Dataset classes {sorted(missing)} are not in the model classes {names}")
    device = 0 if torch.cuda.is_available() else "cpu"
    preds, confs = [], []
    paths = df.path.tolist()
    for start in range(0, len(paths), batch):
        results = model.predict(paths[start:start + batch], imgsz=imgsz, device=device, batch=batch, verbose=False)
        preds += [names[r.probs.top1] for r in results]
        confs += [float(r.probs.top1conf) for r in results]
        if (start // batch) % 20 == 0:
            print(f"  classified {min(start + batch, len(paths))}/{len(paths)}", flush=True)
    out = df.copy()
    out["predicted"], out["confidence"] = preds, confs
    out["correct"] = out.predicted == out.label
    return out


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    """Label-preserving face augmentation: flip, small rotation, mild zoom/shift, brightness, contrast.

    Every copy gets a rotation and a zoom, so no augmented image is pixel-identical to its source."""
    image = image.convert("RGB")
    w, h = image.size
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    angle = rng.choice((-1, 1)) * rng.uniform(3.0, 12.0)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=tuple(
        int(v) for v in np.asarray(image.resize((1, 1), Image.BILINEAR)).reshape(3)))
    keep = rng.uniform(0.86, 0.96)  # centre-ish crop keeps 86-96 % of each side, then resize back
    cw, ch = int(w * keep), int(h * keep)
    left, top = rng.randint(0, w - cw), rng.randint(0, h - ch)
    image = image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    return image


def main():
    a = parse_args()
    rng = random.Random(a.seed)
    dataset, out = Path(a.dataset), Path(a.out)
    if out.exists() and any(out.iterdir()):
        if not a.overwrite:
            raise SystemExit(f"{out} already exists and is not empty. Pass --overwrite to rebuild it.")
        shutil.rmtree(out)

    print(f"Model  : {a.model}\nDataset: {dataset}\nOutput : {out}")
    df = classify(list_images(dataset), a.model, a.imgsz, a.batch)
    classes = sorted(df.label.unique())
    correct = df[df.correct].reset_index(drop=True)
    print(f"\nModel accuracy on the source dataset: {df.correct.mean():.4f} "
          f"({len(correct)} of {len(df)} images correctly classified)")

    # 1) copy correctly classified images
    manifest = []
    for r in correct.itertuples():
        dst = out / r.split / r.label / Path(r.path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.path, dst)
        manifest.append((str(dst), r.path, r.split, r.label, "original", r.confidence))

    # 2) top up classes below the minimum with augmented copies of their correct TRAIN images
    for cname in classes:
        have = int((correct.label == cname).sum())
        need = a.min_per_class - have
        if need <= 0:
            print(f"{cname:9s}: {have} correct images >= {a.min_per_class} -> left as is")
            continue
        pool = correct[(correct.label == cname) & (correct.split == "train")].path.tolist()
        if not pool:
            raise RuntimeError(f"Class {cname} has no correctly classified training image to augment")
        print(f"{cname:9s}: {have} correct images -> adding {need} augmented copies (from {len(pool)} train images)")
        # cycle through a shuffled pool so sources are used as evenly as possible
        order, sources = [], []
        while len(sources) < need:
            if not order:
                order = pool[:]
                rng.shuffle(order)
            sources.append(order.pop())
        used = {}
        for src in sources:
            k = used[src] = used.get(src, 0) + 1
            dst = out / "train" / cname / f"aug{k:02d}_{Path(src).stem}.jpg"
            with Image.open(src) as im:
                augment(im, rng).save(dst, quality=95)
            manifest.append((str(dst), src, "train", cname, "augmented", np.nan))

    man = pd.DataFrame(manifest, columns=["path", "source_path", "split", "label", "kind", "model_confidence"])
    man.to_csv(out / "manifest.csv", index=False)

    counts = pd.DataFrame({
        "source_total": df.label.value_counts(),
        "correct": correct.label.value_counts(),
        "augmented_added": man[man.kind == "augmented"].label.value_counts(),
        "balanced_total": man.label.value_counts(),
    }).reindex(classes).fillna(0).astype(int)
    for s in SPLITS:
        counts[f"balanced_{s}"] = man[man.split == s].label.value_counts().reindex(classes).fillna(0).astype(int)
    counts.to_csv(out / "class_counts.csv")
    print("\n" + counts.to_string())
    assert (counts.balanced_total >= a.min_per_class).all()
    on_disk = sum(1 for p in out.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    assert on_disk == len(man), f"{on_disk} files on disk but {len(man)} in manifest"
    print(f"\nSaved {len(man)} images to {out}")


if __name__ == "__main__":
    main()
