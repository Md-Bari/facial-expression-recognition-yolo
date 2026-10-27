# Experimental Parameters – Face Expression Classification (emp_edu)

All values below are taken from the saved `args.yaml`, `metrics.json` and `deployment_metadata.json` files in `outputs/`.

## 1. Models applied

| # | Model | Pretrained weights | Run name | Params (M) | Weights (MB) |
|---|-------|--------------------|----------|-----------|--------------|
| 1 | YOLO11s-cls | yolo11s-cls.pt | yolo11s_70_15_15 | 5.45 | 11.0 |
| 2 | YOLO11m-cls | yolo11m-cls.pt | yolo11m_70_15_15 | 10.36 | 20.9 |
| 3 | YOLO11l-cls | yolo11l-cls.pt | yolo11l_70_15_15 | 12.84 | 25.9 |
| 4 | YOLOv8m-cls | yolov8m-cls.pt | yolov8m_70_15_15 | 15.78 | 31.7 |
| 5 | YOLOv8m-cls fine-tune A (320 px) | yolov8m-cls.pt | yolov8m_320_finetune_20261110_002800 | 15.78 | – |
| 6 | YOLOv8m-cls fine-tune B (256 px refine) | best.pt of model 4 | yolov8m_256_refine_20261110_021209 | 15.78 | – |

## 2. Dataset and split

| Parameter | Value |
|-----------|-------|
| Dataset | Multi-culture / emp_edu |
| Classes (6) | angry, fear, happy, neutral, sad, surprise |
| Split | 70 / 15 / 15 (train / val / test), stratified, grouped by image hash (`group` column in split.csv) |
| Seed | 42 |
| Face detector | YuNet (`face_detection_yunet_2023mar.onnx`) |
| Face crop margin | 0.2 |
| Crop rule | short side resized to S, centre crop |
| Input format | float32 [N,3,S,S], RGB, 0–1 |
| Grayscale | No |
| Total images in split.csv | 21,392 (YOLO11s); 21,541 (YOLO11m / YOLO11l / YOLOv8m) |
| No-face-detected images | 772 (YOLO11s split); 1,657 (YOLOv8m / YOLO11m / YOLO11l splits) |
| Crop-index entries | 21,541 (`detected` / fallback crop mode) |

### Images per split

| Model | Train | Val | Test |
|-------|-------|-----|------|
| YOLO11s | 14,974 | 3,208 | 3,210 |
| YOLO11m | 15,078 | 3,232 | 3,231 |
| YOLO11l | 15,078 | 3,232 | 3,231 |
| YOLOv8m | 15,078 | 3,232 | 3,231 |

Test class support (YOLOv8m, 3,231 images): angry 495, fear 325, happy 732, neutral 623, sad 593, surprise 463.

## 3. Training hyper-parameters (base runs, models 1–4)

Identical for all four models; only the weights file differs.

