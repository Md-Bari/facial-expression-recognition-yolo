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

