#!/bin/sh
# Trains the ensemble members one after another (never in parallel - they would share the GPU).
cd "$(dirname "$0")"
W26="../combined 3 models/yolo26m-cls.pt"; W11L="../combined 3 models/yolo11l-cls.pt"; W11M="yolo11m-cls.pt"
run() { [ -f "outputs/ensemble_runs/$1/probs.npz" ] && { echo "skip $1"; return; }; shift_name=$1; shift; python train_cls_robust.py --name "$shift_name" "$@" 2>&1 | grep -v -i warning; }
run y26m_256 --weights "$W26" --imgsz 256
run y11l_256 --weights "$W11L" --imgsz 256
run y11m_320 --weights "$W11M" --imgsz 320
run y11m_256 --weights "$W11M" --imgsz 256 --seed 1
run y26m_320 --weights "$W26" --imgsz 320 --seed 1
run y11l_320 --weights "$W11L" --imgsz 320
run y26m_384 --weights "$W26" --imgsz 384 --seed 2
run y11m_384 --weights "$W11M" --imgsz 384 --seed 2
echo QUEUE_DONE
