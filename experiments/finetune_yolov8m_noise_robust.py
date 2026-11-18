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


# ----------------------------------------------------------------------------- model
def new_model(a):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import ClassificationModel

    yolo = YOLO(a.weights)
    m = yolo.model
    ClassificationModel.reshape_outputs(m, NC)
    m.names = dict(enumerate(CLASS_NAMES))
    m.nc = NC
    if isinstance(getattr(m, "yaml", None), dict):
        m.yaml["nc"] = NC
    for p in m.parameters():
        p.requires_grad_(True)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = a.dropout
    return yolo, m.float().cuda()  # NCHW: channels_last is ~5x slower for this net on this GPU


class EMA:
    def __init__(self, model, decay):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = self.decay * (1 - math.exp(-self.n / 500))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def predict(model, X, idx, flip_tta=False, bs=256):
    """Class probabilities for X[idx] (uint8 tensor on GPU)."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        xb = X[idx[i:i + bs]].float().div_(255)
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(xb)
            o = o[0] if isinstance(o, (tuple, list)) else o  # eval -> (softmax probs, logits)
            if flip_tta:
                o2 = model(xb.flip(-1))
                o = (o + (o2[0] if isinstance(o2, (tuple, list)) else o2)) / 2
        out.append(o.float())
    p = torch.cat(out)
    if not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-2):
        p = p.softmax(1)
    return p.cpu().numpy()


def train_one(a, X, Y, tr_idx, epochs, val_idx=None, tag=""):
    """Returns (yolo wrapper, best/final EMA model, history). Selects best epoch on val_idx if given."""
    yolo, model = new_model(a)
    ema = EMA(model, a.ema_decay)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr0, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda")
    steps_per_epoch = len(tr_idx) // a.batch
    total, warm = epochs * steps_per_epoch, int(a.warmup_epochs * steps_per_epoch)
    tr = torch.as_tensor(tr_idx, device="cuda")
    Yg = torch.as_tensor(Y, device="cuda")
    best_acc, best_state, best_epoch, hist, step = -1.0, None, 0, [], 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), device="cuda")]
        run_loss = 0.0
        for bi in range(steps_per_epoch):  # drop last
            lr = a.lr0 * (step + 1) / warm if step < warm else \
                a.lr_min + 0.5 * (a.lr0 - a.lr_min) * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = lr
            ib = perm[bi * a.batch:(bi + 1) * a.batch]
            xb = gpu_augment(X[ib].float().div_(255), a)
            xb, tgt = mix(xb, Yg[ib], a)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
            loss = torch.sum(-tgt * F.log_softmax(logits.float(), 1), 1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            run_loss += loss.item()
            step += 1
        row = {"epoch": ep, "time": time.time() - t0, "train_loss": run_loss / steps_per_epoch, "lr": lr}
        if val_idx is not None:
            pv = predict(ema.ema, X, val_idx)
            acc = float((pv.argmax(1) == Y[val_idx]).mean())
            vloss = float(-np.log(np.clip(pv[np.arange(len(val_idx)), Y[val_idx]], 1e-9, 1)).mean())
            row.update(val_acc=acc, val_loss=vloss)
            if acc > best_acc:
                best_acc, best_epoch = acc, ep
                best_state = copy.deepcopy(ema.ema.state_dict())
        hist.append(row)
        print(f"[{tag}] " + " ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
    if best_state is not None:
        ema.ema.load_state_dict(best_state)
    return yolo, ema.ema, pd.DataFrame(hist), best_epoch


# ----------------------------------------------------------------------------- audit
def confident_learning_flags(P: np.ndarray, Y: np.ndarray):
    """Confident-learning rule (Northcutt et al., 2021): flag i when its given label is below that class's
    mean self-confidence threshold AND another class is at/above its own threshold."""
    thr = np.array([P[Y == k, k].mean() for k in range(NC)])
    above = P >= thr[None]
    above[np.arange(len(Y)), Y] = False
    masked = np.where(above, P, -1.0)
    sugg = masked.argmax(1)
    flag = above.any(1) & (P[np.arange(len(Y)), Y] < thr[Y])
    return flag, sugg, thr


def contact_sheet(df, path, n=96, cols=12, s=128):
    df = df.head(n)
    rows = max(1, (len(df) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * s, rows * (s + 26)), "white")
    d = ImageDraw.Draw(canvas)
    for i, r in enumerate(df.itertuples()):
        im = Image.open(r.crop_path).convert("RGB").resize((s, s))
        x, y = (i % cols) * s, (i // cols) * (s + 26)
        canvas.paste(im, (x, y))
        d.text((x + 2, y + s + 1), f"label: {r.given_label}", fill="green")
        d.text((x + 2, y + s + 13), f"oof: {r.suggested_label} {r.oof_prob_suggested:.2f}", fill="red")
    canvas.save(path, quality=88)


def wilson(acc, n, z=1.959963984540054):
    den = 1 + z * z / n
    c = (acc + z * z / (2 * n)) / den
    h = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def report(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    cm = confusion_matrix(y, pred, labels=np.arange(NC))
    return {
        "images": int(len(y)),
        "accuracy": acc,
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy_wilson_95_ci": wilson(acc, len(y)),
        "per_class_accuracy": {c: float(cm[i, i] / max(1, cm[i].sum())) for i, c in enumerate(CLASS_NAMES)},
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y, pred, labels=np.arange(NC), target_names=CLASS_NAMES,
                                                       output_dict=True, zero_division=0),
    }


# ----------------------------------------------------------------------------- main
def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    run_dir = Path(a.out) / (a.name or time.strftime("yolov8m_noise_robust_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(run_dir / "args.json", "w"), indent=2)
    print("run dir:", run_dir, flush=True)

    df = build_index(Path(a.dataset), Path(a.split_csv))
    X = load_all(df, a.imgsz).cuda()
    Y = df.y.values.astype(np.int64)
    split = df.split.values
    idx_tr, idx_va, idx_te = (np.where(split == s)[0] for s in ("train", "val", "test"))

    # ---- stage 1: out-of-fold label audit
    if a.no_audit:
        flag = np.zeros(len(df), bool)
    elif a.audit_csv:
        au = pd.read_csv(a.audit_csv).set_index("idx")
        flag = df.idx.map(au.flagged).fillna(False).values.astype(bool)
        print(f"reusing audit {a.audit_csv}: {flag.sum()} flagged", flush=True)
    else:
        P = np.zeros((len(df), NC), np.float32)
        skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        for k, (tr, ho) in enumerate(skf.split(np.zeros(len(df)), Y, groups=df.group.values)):
            _, m, _, _ = train_one(a, X, Y, tr, a.oof_epochs, val_idx=None, tag=f"oof{k}")
            P[ho] = predict(m, X, ho, flip_tta=True)
            print(f"[oof{k}] held-out acc = {(P[ho].argmax(1) == Y[ho]).mean():.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
        flag, sugg, thr = confident_learning_flags(P, Y)
        audit = pd.DataFrame({
            "idx": df.idx, "crop_path": df.crop_path, "source_path": df.source_path, "split": df.split,
            "given_label": [CLASS_NAMES[i] for i in Y], "oof_pred": [CLASS_NAMES[i] for i in P.argmax(1)],
            "oof_prob_given": P[np.arange(len(Y)), Y], "flagged": flag,
            "suggested_label": np.where(flag, np.array(CLASS_NAMES)[sugg], ""),
            "oof_prob_suggested": np.where(flag, P[np.arange(len(Y)), sugg], np.nan),
        })
        for i, c in enumerate(CLASS_NAMES):
            audit[f"oof_p_{c}"] = P[:, i]
        audit.to_csv(run_dir / "label_audit.csv", index=False)
        print("OOF accuracy (all images):", float((P.argmax(1) == Y).mean()), flush=True)
        print("class thresholds:", dict(zip(CLASS_NAMES, thr.round(3))), flush=True)
        print(audit.groupby("split").flagged.agg(["sum", "mean"]), flush=True)
        print(audit.groupby("given_label").flagged.agg(["sum", "mean"]), flush=True)
        for s in ("val", "test", "train"):
            sub = audit[(audit.split == s) & audit.flagged].sort_values("oof_prob_suggested", ascending=False)
            contact_sheet(sub, run_dir / f"flagged_{s}_most_confident.jpg")
            if len(sub) > 96:
                contact_sheet(sub.sample(96, random_state=0), run_dir / f"flagged_{s}_random.jpg")

    # ---- stage 2: final model
    tr_clean = idx_tr[~flag[idx_tr]]
    va_clean = idx_va[~flag[idx_va]]
    print(f"final fit: train {len(idx_tr)} -> {len(tr_clean)} | val for selection {len(idx_va)} -> {len(va_clean)}",
          flush=True)
    yolo, model, hist, best_epoch = train_one(a, X, Y, tr_clean, a.epochs, val_idx=va_clean, tag="final")
    hist.to_csv(run_dir / "results.csv", index=False)

    pv, pv_f = predict(model, X, idx_va), predict(model, X, idx_va, flip_tta=True)
    acc = lambda p, idx, m=None: float((p.argmax(1) == Y[idx])[m if m is not None else slice(None)].mean())
    vmask, tmask = ~flag[idx_va], ~flag[idx_te]
    use_tta = acc(pv_f, idx_va, vmask) > acc(pv, idx_va, vmask)
    pt = predict(model, X, idx_te, flip_tta=use_tta)

    metrics = {
        "model": "yolov8m-cls", "init_weights": a.weights, "img_size": a.imgsz, "best_epoch": int(best_epoch),
        "epochs_run": int(len(hist)), "train_minutes": float(hist.time.iloc[-1] / 60),
        "flip_tta_selected_on_validation": bool(use_tta),
        "train_images_used": int(len(tr_clean)), "train_images_flagged_removed": int(len(idx_tr) - len(tr_clean)),
        "validation": {
            "all_labels_single_view": acc(pv, idx_va), "all_labels_flip_tta": acc(pv_f, idx_va),
            "audited_single_view": acc(pv, idx_va, vmask), "audited_flip_tta": acc(pv_f, idx_va, vmask),
            "flagged": int((~vmask).sum()),
        },
        "test_all_original_labels": report(Y[idx_te], pt),
        "test_audited_subset": report(Y[idx_te], pt, tmask),
        "test_flagged_excluded": int((~tmask).sum()),
        "note": "test_all_original_labels is the only figure comparable with earlier runs. test_audited_subset "
                "excludes test images flagged by out-of-fold confident learning (models that never trained on "
                "them); flagged images were not human-verified, so treat it as an optimistic estimate.",
        "baseline_test_accuracy": 0.9056, "target_test_accuracy": 0.95,
    }
    metrics["target_achieved_all_original_labels"] = metrics["test_all_original_labels"]["accuracy"] >= 0.95
    metrics["target_achieved_audited_subset"] = metrics["test_audited_subset"]["accuracy"] >= 0.95
    json.dump(metrics, open(run_dir / "metrics.json", "w"), indent=2)

