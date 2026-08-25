#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InceptionV3 — Multimodal training for Chinese Medicinal Herb classification.
Training strategy STRICTLY follows MNV2_text.py (MobileNetV2 multimodal baseline).

Architecture (identical to MNV2_text.py):
  Visual branch: InceptionV3 (299×299) → GAP → BN → Dense(512, swish)
  Text branch:   Embedding → SpatialDropout1D → BiLSTM → GAP1D+GMP1D → LayerNorm → Dense(512, swish)
  Fusion:        Concat → Gate(Dense 1024, sigmoid) → Multiply → Dense(512, swish) → BN → Dropout(0.5) → Softmax

Training (identical to MNV2_text.py):
  - Single-phase joint training, Adam(lr=1e-4), 30 epochs
  - EarlyStopping(patience=8) + ReduceLROnPlateau
  - Class-balanced weights, BN layers frozen, ~77% backbone frozen
  - 5-fold Stratified CV, 15% val split, Chinese text via jieba
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_multimodal_config import MULTIMODAL_CONFIG, run_multimodal_cv


def build_inceptionv3_backbone(input_tensor=None):
    """Build InceptionV3 backbone, optionally with a specific input_tensor."""
    kwargs = dict(include_top=False, weights='imagenet')
    if input_tensor is not None:
        kwargs['input_tensor'] = input_tensor
    else:
        kwargs['input_shape'] = (*IMAGE_SIZE, 3)
    return tf.keras.applications.InceptionV3(**kwargs)


# InceptionV3 native input size
IMAGE_SIZE = (299, 299)

# InceptionV3 preprocessing: scales pixels to [-1, 1]
PREPROCESS_FN = tf.keras.applications.inception_v3.preprocess_input

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "output_inceptionv3_multimodal")


if __name__ == "__main__":
    print("=" * 70)
    print("InceptionV3 MULTIMODAL Training")
    print("Strategy: IDENTICAL to MobileNetV2 multimodal (MNV2_text.py)")
    print("=" * 70)
    print(f"Backbone: InceptionV3 (ImageNet pretrained)")
    print(f"Input size: {IMAGE_SIZE}  (native 299×299)")
    print(f"Text branch: BiLSTM(256) + GAP1D/GMP1D pooling")
    print(f"Fusion: Gated multimodal fusion (same as baseline)")
    print(f"Batch size: {MULTIMODAL_CONFIG['batch_size']}")
    print(f"5-Fold Stratified CV | Single-phase joint training")
    print(f"Fine-tune ratio: {MULTIMODAL_CONFIG['fine_tune_ratio']}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    run_multimodal_cv(
        model_name="InceptionV3_Multimodal",
        backbone_builder=build_inceptionv3_backbone,
        preprocess_fn=PREPROCESS_FN,
        image_size=IMAGE_SIZE,
        output_dir=OUTPUT_DIR
    )
