#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InceptionV3 — Image-only training for Chinese Medicinal Herb classification.
Training strategy STRICTLY follows baseline_model_strict.py (MobileNetV2).

Backbone: InceptionV3 (ImageNet pretrained, include_top=False)
Native input: 299×299  (uses model's recommended size, not forced to 224)
Preprocessing: inception_v3.preprocess_input (scales to [-1, 1])
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_config import CONFIG, run_cv


def build_inceptionv3_backbone():
    backbone = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMAGE_SIZE, 3)
    )
    return backbone


# InceptionV3 native input size
IMAGE_SIZE = (299, 299)

# InceptionV3 preprocessing: scales pixels to [-1, 1]
PREPROCESS_FN = tf.keras.applications.inception_v3.preprocess_input

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_inceptionv3")


if __name__ == "__main__":
    print("=" * 70)
    print("InceptionV3 Training — Strict Comparison with MobileNetV2 Baseline")
    print("=" * 70)
    print(f"Backbone: InceptionV3 (ImageNet pretrained)")
    print(f"Input size: {IMAGE_SIZE}  (native resolution)")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"5-Fold Stratified CV | 2-Phase Training")
    print(f"Fine-tune ratio: {CONFIG['fine_tune_ratio']} (auto-computed per layer count)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    run_cv(
        model_name="InceptionV3",
        backbone_builder=build_inceptionv3_backbone,
        preprocess_fn=PREPROCESS_FN,
        image_size=IMAGE_SIZE,
        output_dir=OUTPUT_DIR
    )
