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

