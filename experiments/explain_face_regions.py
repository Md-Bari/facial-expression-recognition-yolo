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


def load(path, s):
    """Same preprocessing as Ultralytics classify inference: short side -> s, centre crop, RGB in [0, 1]."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        r = s / min(w, h)
        nw, nh = max(s, round(w * r)), max(s, round(h * r))
        im = im.resize((nw, nh), Image.BILINEAR)
        l, t = (nw - s) // 2, (nh - s) // 2
        arr = np.asarray(im.crop((l, t, l + s, t + s)), dtype=np.float32) / 255
    return torch.from_numpy(arr).permute(2, 0, 1)


class Explainer:
    def __init__(self, model_path):
        from ultralytics import YOLO

        yolo = YOLO(model_path)
        self.names = [yolo.names[i] for i in range(len(yolo.names))]
        self.model = yolo.model.float().eval().cuda()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.head = self.model.model[-1]
        # last conv feature map before global pooling (falls back to the layer feeding the head)
        target = self.head.conv if hasattr(self.head, "conv") else self.model.model[-2]
        self.act = None
        target.register_forward_hook(lambda m, i, o: setattr(self, "act", o))

    def scores(self, x):
        """Returns (probabilities, logits) for a batch."""
        o = self.model(x)
        if isinstance(o, (tuple, list)):
            a, b = o[0], o[1]
            probs, logits = (a, b) if torch.allclose(a.sum(1), torch.ones_like(a[:, 0]), atol=1e-3) else (b, a)
            return probs, logits
        return o, torch.log(o.clamp_min(1e-9))

    def gradcam(self, x):
        """x: B,3,S,S on GPU. Returns (cam B,S,S in [0,1], predicted class, probability)."""
        x = x.clone().requires_grad_(True)
        probs, logits = self.scores(x)
        cls = probs.argmax(1)
        act = self.act
        grads = torch.autograd.grad(logits.gather(1, cls[:, None]).sum(), act)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-9)
        return cam.detach().cpu().numpy(), cls.cpu().numpy(), probs.detach().gather(1, cls[:, None])[:, 0].cpu().numpy()

    @torch.no_grad()
    def occlusion(self, x, cls, patch, stride):
        """x: 3,S,S. Map of the drop in p(cls) when each region is covered by a grey patch."""
        S = x.shape[-1]
        base = self.scores(x[None])[0][0, cls].item()
        pos = [(t, l) for t in range(0, S - patch + 1, stride) for l in range(0, S - patch + 1, stride)]
        drop, count = torch.zeros(S, S, device=x.device), torch.zeros(S, S, device=x.device)
        for i in range(0, len(pos), 64):
            chunk = pos[i:i + 64]
            xb = x[None].repeat(len(chunk), 1, 1, 1)
            for j, (t, l) in enumerate(chunk):
                xb[j, :, t:t + patch, l:l + patch] = 0.5
            p = self.scores(xb)[0][:, cls]
            for (t, l), pj in zip(chunk, p):
                drop[t:t + patch, l:l + patch] += base - pj
                count[t:t + patch, l:l + patch] += 1
        m = (drop / count.clamp_min(1)).clamp_min(0)
        return (m / m.max().clamp_min(1e-9)).cpu().numpy()


def overlay(ax, img, heat, title):
    ax.imshow(img)
    ax.imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


