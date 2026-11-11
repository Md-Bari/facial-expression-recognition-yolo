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

| Parameter | Value |
|-----------|-------|
| Task / mode | classify / train |
| Epochs (max) | 60 |
| Early-stop patience | 15 |
| Batch size | 32 |
| Image size (imgsz) | 256 |
| Optimizer | AdamW |
| Initial LR (lr0) | 0.0003 |
| Final LR factor (lrf) | 0.01 |
| Cosine LR (cos_lr) | True |
| Momentum | 0.937 |
| Weight decay | 0.05 |
| Warm-up epochs | 3 |
| Warm-up momentum | 0.8 |
| Warm-up bias LR | 0.1 |
| Dropout | 0.3 |
| Nominal batch size (nbs) | 64 |
| Pretrained | True |
| AMP (mixed precision) | True |
| Seed | 42 |
| Deterministic | False |
| Device | GPU 0 |
| Workers | 8 |
| Cache | RAM |
| Freeze layers | None |
| Close mosaic (last N epochs) | 10 |
| Multi-scale | 0.0 |
| Validation each epoch | True (split = val) |
| Loss gains (cls) | 0.5 |

### Augmentation (base runs)

| Parameter | Value |
|-----------|-------|
| hsv_h / hsv_s / hsv_v | 0.015 / 0.5 / 0.4 |
| degrees | 0.0 |
| translate | 0.1 |
| scale | 0.3 |
| shear | 0.0 |
| perspective | 0.0 |
| flipud | 0.0 |
| fliplr | 0.5 |
| bgr | 0.0 |
| mosaic | 1.0 |
| mixup | 0.0 |
| cutmix | 0.0 |
| copy_paste | 0.0 |
| auto_augment | randaugment |
| erasing | 0.1 |

### Inference / test-time settings

| Parameter | Value |
|-----------|-------|
| TTA (flip, inside exported model) | True |
| Export format (args.yaml) | torchscript; ONNX deployment model also exported |
| ONNX output | class probabilities |

## 4. Fine-tuning runs (YOLOv8m)

| Parameter | A: 320 px fine-tune | B: 256 px refine |
|-----------|--------------------|------------------|
| Start weights | yolov8m-cls.pt (COCO/ImageNet pretrained) | best.pt of YOLOv8m base run |
| Epochs (max) | 100 | 40 |
| Patience | 20 | 10 |
| Batch size | 16 | 32 |
| Image size | 320 | 256 |
| Optimizer | AdamW | AdamW |
| lr0 | 0.00015 | 0.00005 |
| lrf | 0.01 | 0.1 |
| Cosine LR | True | True |
| Weight decay | 0.03 | 0.05 |
| Warm-up epochs | 3 | 3 |
| Dropout | 0.35 | 0.3 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 | 0.015 / 0.5 / 0.4 |
| translate | 0.1 | 0.1 |
| scale | 0.5 | 0.3 |
| fliplr | 0.5 | 0.5 |
| mixup / cutmix | 0.0 / 0.0 | 0.0 / 0.0 |
| auto_augment | randaugment | randaugment |
| erasing | 0.1 | 0.1 |
| close_mosaic | 10 | 10 |
| Freeze | None | None |
| Seed | 42 | 42 |
| Flip-TTA selected on validation | Yes | No |
| Epochs run | 73 | 40 |
| Best epoch (val top-1) | 50 (0.90439) | 31 (0.91058) |
| Training time (last epoch timestamp) | 4893.9 s (~81.6 min) | 2077.0 s (~34.6 min) |
| Final train loss | 0.03322 | 0.02624 |
| Final val loss | 0.47278 | 0.50988 |
| Target test accuracy | 0.95 (not achieved) | 0.95 (not achieved) |

## 5. Training outcome per model (base runs)

| Model | Epochs run | Best epoch | Train time (min) | Final train loss | Final val loss | Final val top-1 | Final val top-5 |
|-------|-----------|-----------|------------------|------------------|----------------|-----------------|-----------------|
| YOLO11s | 56 | 55 | 12.1 | 0.04719 | 0.44343 | 0.90337 | 0.99844 |
| YOLO11m | 49 | 39 | 13.4 | 0.04292 | 0.46718 | 0.89882 | 0.99722 |
| YOLO11l | 60 | 47 | 20.3 | 0.02993 | 0.46014 | 0.90687 | 0.99907 |
| YOLOv8m | 44 | 29 | 10.0 | 0.03887 | 0.47354 | 0.90563 | 0.99814 |

(Best epoch for YOLO11s is 55 in metrics.json; epochs_run 56.)

## 6. Results – accuracy and macro-F1

| Model | Train acc | Val acc | Test acc | Test macro-F1 | Test 95% CI | ONNX test acc | ms/image |
|-------|-----------|---------|----------|---------------|-------------|---------------|----------|
| YOLO11s | 0.9991 | 0.9074 | 0.8947 | 0.8839 | 0.884–0.906 | 0.8916 | 8.27 |
| YOLO11m | 0.9971 | 0.9072 | 0.8969 | 0.8914 | 0.887–0.907 | 0.8997 | 8.80 |
| YOLO11l | 0.9995 | 0.9121 | 0.9068 | 0.9013 | 0.897–0.917 | 0.9059 | 9.00 |
| YOLOv8m | 0.9978 | 0.9109 | 0.9056 | 0.9013 | 0.895–0.916 | 0.9019 | 5.65 |
| YOLOv8m fine-tune A (320) | – | 0.9066 (flip-TTA) / 0.9038 (single) | 0.9006 | 0.8967 | 0.890–0.910 | – | – |
| YOLOv8m fine-tune B (256) | – | 0.9106 (single) / 0.9097 (flip-TTA) | 0.9025 | 0.8972 | 0.892–0.912 | – | – |

Validation macro-F1 (base runs): YOLO11s 0.9002, YOLO11m 0.9016, YOLO11l 0.9065, YOLOv8m 0.9062.
Train macro-F1: YOLO11s 0.9991, YOLO11m 0.9972, YOLO11l 0.9995, YOLOv8m 0.9976.

## 7. Per-class test accuracy

| Model | angry | fear | happy | neutral | sad | surprise |
|-------|-------|------|-------|---------|-----|----------|
| YOLO11s | 0.8730 | 0.7715 | 0.9508 | 0.8973 | 0.9140 | 0.8815 |
| YOLO11m | 0.8970 | 0.8338 | 0.9481 | 0.8989 | 0.8803 | 0.8790 |
| YOLO11l | 0.9051 | 0.8431 | 0.9508 | 0.9101 | 0.8988 | 0.8898 |
| YOLOv8m | 0.8869 | 0.8738 | 0.9563 | 0.8989 | 0.8904 | 0.8963 |
| YOLOv8m fine-tune A | 0.8889 | 0.8738 | 0.9495 | 0.8957 | 0.8769 | 0.8920 |
| YOLOv8m fine-tune B | 0.9010 | 0.8338 | 0.9440 | 0.8973 | 0.9022 | 0.8942 |

## 8. Per-class accuracy – validation split (base runs)

| Model | angry | fear | happy | neutral | sad | surprise |
|-------|-------|------|-------|---------|-----|----------|
| YOLO11s | 0.8848 | 0.8416 | 0.9590 | 0.9069 | 0.9020 | 0.9006 |
| YOLO11m | 0.8770 | 0.8554 | 0.9672 | 0.9181 | 0.8885 | 0.8901 |
| YOLO11l | 0.8810 | 0.8646 | 0.9699 | 0.9246 | 0.8936 | 0.8944 |
| YOLOv8m | 0.8669 | 0.8738 | 0.9781 | 0.9085 | 0.8936 | 0.9030 |

