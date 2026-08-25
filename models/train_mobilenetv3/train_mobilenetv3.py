#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV3Large — Image-only training for Chinese Medicinal Herb classification.
Training strategy STRICTLY follows baseline_model_strict.py (MobileNetV2).

Backbone: MobileNetV3Large (ImageNet pretrained, include_top=False)
Native input: 224×224
Preprocessing: mobilenet_v3.preprocess_input (scales to [-1, 1])
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_config import CONFIG, run_cv


def build_mobilenetv3_backbone():
    backbone = tf.keras.applications.MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMAGE_SIZE, 3)
    )
    return backbone


# MobileNetV3Large native input size
IMAGE_SIZE = (224, 224)

# MobileNetV3 preprocessing: scales pixels to [-1, 1]
PREPROCESS_FN = tf.keras.applications.mobilenet_v3.preprocess_input

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_mobilenetv3")


if __name__ == "__main__":
    print("=" * 70)
    print("MobileNetV3Large Training — Strict Comparison with MobileNetV2 Baseline")
    print("=" * 70)
    print(f"Backbone: MobileNetV3Large (ImageNet pretrained)")
    print(f"Input size: {IMAGE_SIZE}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"5-Fold Stratified CV | 2-Phase Training")
    print(f"Fine-tune ratio: {CONFIG['fine_tune_ratio']} (auto-computed per layer count)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    run_cv(
        model_name="MobileNetV3Large",
        backbone_builder=build_mobilenetv3_backbone,
        preprocess_fn=PREPROCESS_FN,
        image_size=IMAGE_SIZE,
        output_dir=OUTPUT_DIR
    )
