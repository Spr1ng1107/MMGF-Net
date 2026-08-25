#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGG16 — Image-only training for Chinese Medicinal Herb classification.
Training strategy STRICTLY follows baseline_model_strict.py (MobileNetV2).

Backbone: VGG16 (ImageNet pretrained, include_top=False)
Native input: 224×224
Preprocessing: vgg16.preprocess_input (BGR zero-center)
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_config import CONFIG, run_cv


def build_vgg16_backbone():
    backbone = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMAGE_SIZE, 3)
    )
    return backbone


# VGG16 native input size
IMAGE_SIZE = (224, 224)

# VGG16-specific preprocessing: RGB → BGR, zero-center each channel
PREPROCESS_FN = tf.keras.applications.vgg16.preprocess_input

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_vgg16")


if __name__ == "__main__":
    print("=" * 70)
    print("VGG16 Training — Strict Comparison with MobileNetV2 Baseline")
    print("=" * 70)
    print(f"Backbone: VGG16 (ImageNet pretrained)")
    print(f"Input size: {IMAGE_SIZE}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"5-Fold Stratified CV | 2-Phase Training")
    print(f"Fine-tune ratio: {CONFIG['fine_tune_ratio']} (auto-computed per layer count)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    run_cv(
        model_name="VGG16",
        backbone_builder=build_vgg16_backbone,
        preprocess_fn=PREPROCESS_FN,
        image_size=IMAGE_SIZE,
        output_dir=OUTPUT_DIR
    )
