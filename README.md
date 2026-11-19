# Facial Expression Recognition with YOLO Classifiers

Six-class facial expression recognition (**angry, fear, happy, neutral, sad, surprise**) on the multi-culture *emp_edu* face dataset. The project benchmarks Ultralytics YOLO11 and YOLOv8 classification models and then tries to push accuracy further with noise-robust fine-tuning, label auditing, and model ensembling. It also checks with explainability maps which face regions each prediction relies on.

## Highlights

- **Benchmark of 4 backbones:** YOLO11s/m/l-cls and YOLOv8m-cls, trained with an identical recipe on a leakage-free 70/15/15 split.
- **Best test accuracy: 90.68 %** (YOLO11l). **Best macro-F1: 0.901** (YOLO11l, YOLOv8m). **Fastest: 5.65 ms/image** (YOLOv8m, ONNX).
- **Face-crop pipeline:** YuNet face detection with a 0.2 crop margin, and a fallback crop when no face is detected.
- **Noise-robust fine-tuning:** a custom training loop with label smoothing, mixup/cutmix, face-safe augmentation, and weight EMA. It also runs a 5-fold out-of-fold *confident-learning* label audit.
- **Validation-only ensembling:** members and flip-TTA are chosen by greedy forward selection on the validation split. The test split is used once, for the final report.
- **Explainability:** two independent saliency methods are drawn side by side to show which face regions drive each class.
- **Deployment:** models are exported to TorchScript and ONNX, with flip-TTA built in.

## Results (test split, 3,231 images)

| Model | Params (M) | Test acc | Test macro-F1 | 95 % CI | ONNX acc | ms / image |
|-------|-----------:|---------:|--------------:|---------|---------:|-----------:|
| YOLO11s-cls | 5.45 | 0.8947 | 0.8839 | 0.884–0.906 | 0.8916 | 8.27 |
| YOLO11m-cls | 10.36 | 0.8969 | 0.8914 | 0.887–0.907 | 0.8997 | 8.80 |
| **YOLO11l-cls** | 12.84 | **0.9068** | **0.9013** | 0.897–0.917 | 0.9059 | 9.00 |
| YOLOv8m-cls | 15.78 | 0.9056 | 0.9013 | 0.895–0.916 | 0.9019 | **5.65** |
| YOLOv8m fine-tune (320 px) | 15.78 | 0.9006 | 0.8967 | 0.890–0.910 | – | – |
| YOLOv8m refine (256 px) | 15.78 | 0.9025 | 0.8972 | 0.892–0.912 | – | – |

*Happy* is the easiest class (about 95 % accuracy). *Fear* is the hardest: it is most often confused with *surprise* and *sad*. Standard Ultralytics fine-tuning plateaued at about 90 %, which led to the noise-robust and ensemble experiments below. The full hyper-parameters, per-class metrics, and confusion matrices are in [experiments/EXPERIMENTAL_PARAMETERS.md](experiments/EXPERIMENTAL_PARAMETERS.md).

## Method

1. **Data preparation:** faces are detected with YuNet and cropped with a 0.2 margin, then resized on the short side and centre-cropped. The data is split 70/15/15 (stratified, seed 42). Images are grouped by hash so that duplicates never cross splits.
2. **Base training:** the following settings are shared by all four backbones.
   - 256 px input, AdamW (lr 3e-4, cosine schedule, weight decay 0.05), dropout 0.3.
   - Augmentation: RandAugment, random erasing, horizontal flip, and HSV jitter.
   - Up to 60 epochs, with early stopping (patience 15).
3. **Fine-tuning:** YOLOv8m was fine-tuned at 320 px, then refined at 256 px from the best base checkpoint.
4. **Noise-robust training:** a 5-fold out-of-fold audit flags images whose labels are likely wrong. The final model trains on the cleaned training split, with EMA weights and selection on the validation split.
5. **Ensembling:** several backbones (YOLO11m, YOLO11l, and YOLO26m) are trained at multiple resolutions (256, 320, and 384 px) and seeds, then combined on validation probabilities.

## Repository structure

| Path | Description |
|------|-------------|
| `yolo11s/11m/11l_face_expression_emp_edu.ipynb`, `yolov8m_face_expression_emp_edu.ipynb` | End-to-end base pipelines: face cropping, split, training, evaluation, ONNX export |
| `yolov8m_finetune_accuracy.ipynb` | YOLOv8m resolution fine-tune / refine experiments |
| `yolov8m_finetune_noise_robust.ipynb`, `yolo11m_finetune_noise_robust.ipynb` | Noise-robust fine-tuning notebooks |
| `yolo11m_finetune_corrected_labels.ipynb` | Retraining on audited (corrected) labels |
| `train_cls_robust.py` | Script version of the noise-robust recipe, for training ensemble members (saves val/test probabilities) |
| `run_ensemble_queue.sh` | Trains the ensemble members one after another on a single GPU |
| `ensemble_eval.py` | Greedy validation-only ensemble selection and final test report |
| `experiments/finetune_yolov8m_noise_robust.py` | Custom YOLOv8m loop with OOF confident-learning label audit |
| `experiments/balance_dataset.py` | Builds a class-balanced dataset from correctly classified images (train-only augmentation, no leakage) |
| `experiments/explain_face_regions.py` | Side-by-side saliency maps of the face regions used per prediction |
| `experiments/onnx_correctly_classified_images.ipynb` | ONNX inference and analysis of correctly classified images |
| `experiments/EXPERIMENTAL_PARAMETERS.md` | Full hyper-parameters and results |

## Getting started

```bash
pip install ultralytics opencv-python onnxruntime scikit-learn pandas matplotlib
```

The dataset, run outputs (`outputs/`), and model weights (`*.pt`, `weights/`) are not included in this repository. Put the face-expression dataset where the notebooks expect it, then run a base notebook. To train and evaluate the ensemble:

```bash
sh run_ensemble_queue.sh   # train ensemble members
python ensemble_eval.py    # select the ensemble on validation, report on test
```

## Tech stack

Python · PyTorch · Ultralytics YOLO11 / YOLOv8 · OpenCV (YuNet) · ONNX Runtime · scikit-learn · pandas · matplotlib
