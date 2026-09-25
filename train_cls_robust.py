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

