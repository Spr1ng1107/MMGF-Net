#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet50 — Image-only training for Chinese Medicinal Herb classification.
Training strategy STRICTLY follows baseline_model_strict.py (MobileNetV2).

NOTE: ResNet18 is NOT available in tf.keras.applications.
      Using ResNet50 as the closest standard alternative.
      ResNet50 (25.6M params) > ResNet18 (11.7M params) > MobileNetV2 (3.5M params).

Backbone: ResNet50 (ImageNet pretrained, include_top=False)
Native input: 224×224
Preprocessing: resnet50.preprocess_input (BGR zero-center)
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_config import CONFIG, run_cv


def build_resnet50_backbone():
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMAGE_SIZE, 3)
    )
    return backbone


# ResNet50 native input size
IMAGE_SIZE = (224, 224)

# ResNet50 preprocessing: RGB → BGR, zero-center each channel
PREPROCESS_FN = tf.keras.applications.resnet50.preprocess_input

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_resnet50")


if __name__ == "__main__":
    print("=" * 70)
    print("ResNet50 Training — Strict Comparison with MobileNetV2 Baseline")
    print("=" * 70)
    print(f"NOTE: ResNet18 not in keras.applications — using ResNet50 instead")
    print(f"Backbone: ResNet50 (ImageNet pretrained)")
    print(f"Input size: {IMAGE_SIZE}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"5-Fold Stratified CV | 2-Phase Training")
    print(f"Fine-tune ratio: {CONFIG['fine_tune_ratio']} (auto-computed per layer count)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    run_cv(
        model_name="ResNet50",
        backbone_builder=build_resnet50_backbone,
        preprocess_fn=PREPROCESS_FN,
        image_size=IMAGE_SIZE,
        output_dir=OUTPUT_DIR
    )
