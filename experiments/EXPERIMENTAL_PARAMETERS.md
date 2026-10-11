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

