#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================================
中药饮片识别 — 文献复现基线训练+预测（论文表 2）
===========================================================================
模型:
  ① 改进ResNet50    [19] 葛琪等   — ResNet50 + CBAM (通道+空间注意力)
  ② CCNNet   [15] Hu et al.    — 轻量级深度可分离卷积网络
  ③ ConvNeXt [16] 兰邵琨等     — ConvNeXtTiny (现代化CNN)
  ④ SFE-CA   [18] Zhang et al. — EfficientNetB0 + SE + 通道注意力

数据:
  labels.csv (image_name, Task_Chinese_medicinal_herb)
  images 目录

流程:
  1. 加载 labels.csv, 按图片存在性过滤
  2. 5折分层交叉验证训练 (与 MobileNetV2 基线策略一致)
  3. 用训练好的5折模型对全部数据预测 (soft voting)
  4. 输出预测 CSV + 分类报告 + 混淆矩阵

用法:
  python train_and_predict_5models.py                     # 训练全部5个模型
  python train_and_predict_5models.py --only improved_resnet50,convnext
  python train_and_predict_5models.py --skip ccnnet
  python train_and_predict_5models.py --predict-only       # 仅预测(不训练)
===========================================================================
"""

import os, sys, gc, json, argparse, warnings, shutil
import numpy as np
import pandas as pd
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score, roc_auc_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 路径配置 — 服务器路径
# ═══════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "labels_csv":  "/mnt/data1/spring/multimodal_project/labels.csv",
    "images_dir":  "/mnt/data1/spring/multimodal_project/data/images",
    "output_root": os.path.join(_SCRIPT_DIR, "model_predictions"),

    # --- 训练超参 (与 MobileNetV2 基线一致) ---
    "image_size":    (224, 224),
    "batch_size":    16,
    "cv_folds":      5,
    "val_split":     0.15,
    "seed":          42,

    "phase1_lr":       1e-4,
    "phase1_epochs":   12,
    "phase1_patience": 5,

    "phase2_lr":             5e-6,
    "phase2_epochs":         20,
    "phase2_patience":       8,
    "phase2_reduce_patience": 4,
    "phase2_reduce_factor":   0.5,
    "phase2_min_lr":          1e-7,

    "dense_units":      512,
    "dense_activation": "swish",
    "fine_tune_ratio":  0.77,
}


# GPU
for d in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(d, True)


# ═══════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """
    加载 labels.csv, 构建绝对路径, 过滤不存在的图片。
    返回: df, le, num_classes, class_names
    """
    df = pd.read_csv(CONFIG["labels_csv"])

    # 用 image_name 拼接绝对路径
    df['image_path'] = df['image_name'].apply(
        lambda x: os.path.join(CONFIG["images_dir"], x))

    before = len(df)
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    after = len(df)
    if before != after:
        logger.warning(f"Filtered out {before - after} missing images, {after} remain.")

    if len(df) == 0:
        raise RuntimeError(
            f"No images found! Check:\n"
            f"  labels_csv: {CONFIG['labels_csv']}\n"
            f"  images_dir: {CONFIG['images_dir']}")

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Task_Chinese_medicinal_herb'])
    num_classes = len(le.classes_)
    class_names = [str(c) for c in le.classes_]

    logger.info(f"Loaded {len(df)} samples | {num_classes} classes | labels: {class_names}")
    return df, le, num_classes, class_names


# ═══════════════════════════════════════════════════════════════════════════
# 数据生成器
# ═══════════════════════════════════════════════════════════════════════════

class ImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, preprocess_fn, image_size, augment=False):
        self.df = df.reset_index(drop=True)
        self.preprocess_fn = preprocess_fn
        self.image_size = image_size
        self.augment = augment
        self.img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20, horizontal_flip=True, brightness_range=[0.9, 1.1]
        ) if augment else None

    def __len__(self):
        return int(np.ceil(len(self.df) / CONFIG["batch_size"]))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * CONFIG["batch_size"]:(idx + 1) * CONFIG["batch_size"]]
        imgs, labels = [], []
        for _, row in batch.iterrows():
            img = load_img(row['image_path'], target_size=self.image_size)
            img_arr = img_to_array(img)
            if self.augment:
                img_arr = self.img_gen.random_transform(img_arr)
            img_arr = self.preprocess_fn(img_arr)
            imgs.append(img_arr)
            labels.append(row['label_encoded'])
        return np.array(imgs), np.array(labels)


# ═══════════════════════════════════════════════════════════════════════════
# 断点续训 (Checkpoint/Resume) — 双层检查: 磁盘文件(主力) + JSON(辅助)
# ═══════════════════════════════════════════════════════════════════════════

def check_fold_completed_from_disk(fold_dir):
    """
    按折检查磁盘上是否存在完整的输出文件 (主力方法, 不依赖 JSON).
    返回: (is_completed: bool, acc_from_report: float or None)
    """
    # 检查模型文件 (Keras 格式或权重格式)
    keras_model = os.path.join(fold_dir, "best_model.keras")
    weights_model = os.path.join(fold_dir, "best_model.weights.h5")
    has_model = (os.path.exists(keras_model) and os.path.getsize(keras_model) > 0) or \
                (os.path.exists(weights_model) and os.path.getsize(weights_model) > 0)

    # 检查分类报告
    report_path = os.path.join(fold_dir, "classification_report.txt")
    has_report = os.path.exists(report_path) and os.path.getsize(report_path) > 0

    acc_val = None
    if has_report:
        try:
            import re
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            acc_match = re.search(r'accuracy\s+([\d.]+)', content)
            if acc_match:
                acc_val = float(acc_match.group(1))
        except Exception:
            pass

    return (has_model and has_report), acc_val


def scan_completed_folds_from_disk(output_dir, cv_folds):
    """
    扫描所有折目录, 直接从磁盘判断完成状态.
    返回: (completed_folds: list, recovered_metrics: dict)
    """
    completed = []
    recovered = {'acc': [], 'macro_prec': [], 'macro_rec': [],
                 'macro_f1': [], 'macro_auc': []}

    for fold in range(1, cv_folds + 1):
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        if not os.path.isdir(fold_dir):
            continue
        is_done, acc = check_fold_completed_from_disk(fold_dir)
        if is_done:
            completed.append(fold)
            if acc is not None:
                recovered['acc'].append(acc)
                # 尝试从分类报告提取更完整的指标
                report_path = os.path.join(fold_dir, "classification_report.txt")
                try:
                    import re
                    with open(report_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    macro = re.search(r'macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
                    weight = re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
                    if macro:
                        recovered['macro_prec'].append(float(macro.group(1)))
                        recovered['macro_rec'].append(float(macro.group(2)))
                        recovered['macro_f1'].append(float(macro.group(3)))
                    if weight:
                        recovered['macro_auc'].append(float('nan'))  # AUC不在分类报告中
                except Exception:
                    pass

    if completed:
        logger.info(f"Disk scan: {len(completed)}/{cv_folds} folds have complete artifacts "
                    f"(folds {completed})")
    return completed, recovered




# ── JSON 断点 (辅助: 保留指标精度/AUC 等磁盘报告不含的信息) ──

def save_ckpt(ckpt_path, state):
    """保存训练检查点到 JSON (辅助, 保留 AUC 等磁盘报告缺的指标)"""
    try:
        serializable = {}
        for k, v in state.items():
            if isinstance(v, dict):
                serializable[k] = {
                    sk: [float(x) if hasattr(x, 'item') else x for x in sv]
                    if isinstance(sv, list) else
                    float(sv) if hasattr(sv, 'item') else sv
                    for sk, sv in v.items()
                }
            elif isinstance(v, list):
                serializable[k] = [
                    float(x) if hasattr(x, 'item') else x for x in v
                ]
            else:
                serializable[k] = v
        with open(ckpt_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"Checkpoint saved → {ckpt_path}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_ckpt(ckpt_path, model_name, cv_folds):
    """加载 JSON 检查点 (辅助方法, 用于恢复AUC等精密指标)"""
    if not os.path.exists(ckpt_path):
        return None, None
    try:
        with open(ckpt_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        if state.get('model_name') != model_name:
            logger.warning(f"Checkpoint model mismatch: '{state.get('model_name')}' vs '{model_name}'")
            return None, None
        if state.get('total_folds') != cv_folds:
            logger.warning(f"Checkpoint folds mismatch: {state.get('total_folds')} vs {cv_folds}")
            return None, None
        completed = state.get('completed_folds', [])
        metrics = state.get('metrics_summary', {})
        for key in ['acc', 'macro_prec', 'macro_rec', 'macro_f1', 'macro_auc']:
            if key not in metrics:
                metrics[key] = []
            else:
                metrics[key] = [float(v) for v in metrics[key]]
        logger.info(f"JSON checkpoint: {len(completed)}/{cv_folds} folds done "
                    f"(folds {completed})")
        return completed, metrics
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Checkpoint corrupted ({e}), starting fresh")
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# 分类头
# ═══════════════════════════════════════════════════════════════════════════

def build_head(backbone, num_classes, model_name="model"):
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(CONFIG["dense_units"], activation=CONFIG["dense_activation"])(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=backbone.input, outputs=output, name=model_name)


# ═══════════════════════════════════════════════════════════════════════════
# ① 改进ResNet50: ResNet50 + CBAM
# ═══════════════════════════════════════════════════════════════════════════

def cbam_channel_attention(x, ratio=8, name="ca"):
    channels = x.shape[-1]
    avg = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    maxv = layers.GlobalMaxPooling2D(name=f"{name}_gmp")(x)
    reduced = max(1, channels // ratio)
    c1 = layers.Conv2D(reduced, 1, activation='relu', use_bias=False, name=f"{name}_c1")
    c2 = layers.Conv2D(channels, 1, use_bias=False, name=f"{name}_c2")
    avg = layers.Reshape((1, 1, channels), name=f"{name}_gap_r")(avg)
    maxv = layers.Reshape((1, 1, channels), name=f"{name}_gmp_r")(maxv)
    att = layers.Add(name=f"{name}_add")([c2(c1(avg)), c2(c1(maxv))])
    att = layers.Activation('sigmoid', name=f"{name}_sig")(att)
    return layers.Multiply(name=f"{name}_scale")([x, att])


def cbam_spatial_attention(x, kernel_size=7, name="sa"):
    avg = tf.reduce_mean(x, axis=-1, keepdims=True)
    maxv = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1, name=f"{name}_cat")([avg, maxv])
    att = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid',
                        use_bias=False, name=f"{name}_conv")(concat)
    return layers.Multiply(name=f"{name}_scale")([x, att])


def build_improved_resnet50_backbone(input_shape=(224, 224, 3)):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = cbam_channel_attention(base.output, ratio=16, name="cbam_ca")
    x = cbam_spatial_attention(x, kernel_size=7, name="cbam_sa")
    backbone = models.Model(inputs=base.input, outputs=x, name="ImprovedResNet50_Backbone")
    logger.info(f"改进ResNet50 backbone: {backbone.count_params():,} params")
    return backbone


# ═══════════════════════════════════════════════════════════════════════════
# ② CCNNet: 轻量级深度可分离卷积网络
# ═══════════════════════════════════════════════════════════════════════════

def se_block(x, reduction=4, name="se"):
    channels = x.shape[-1]
    reduced = max(1, channels // reduction)
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Reshape((1, 1, channels), name=f"{name}_r")(se)
    se = layers.Conv2D(reduced, 1, activation='relu', name=f"{name}_fc1")(se)
    se = layers.Conv2D(channels, 1, activation='sigmoid', name=f"{name}_fc2")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def ds_conv_block(x, filters, stride=1, expansion=1, name="dsconv"):
    in_channels = x.shape[-1]
    expanded = in_channels * expansion
    out = x
    if expansion > 1:
        out = layers.Conv2D(expanded, 1, use_bias=False, name=f"{name}_expand")(out)
        out = layers.BatchNormalization(name=f"{name}_expand_bn")(out)
        out = layers.ReLU(name=f"{name}_expand_relu")(out)
    out = layers.DepthwiseConv2D(3, strides=stride, padding='same',
                                 use_bias=False, name=f"{name}_dw")(out)
    out = layers.BatchNormalization(name=f"{name}_dw_bn")(out)
    out = layers.ReLU(name=f"{name}_dw_relu")(out)
    out = layers.Conv2D(filters, 1, use_bias=False, name=f"{name}_project")(out)
    out = layers.BatchNormalization(name=f"{name}_project_bn")(out)
    out = se_block(out, name=f"{name}_se")
    if stride == 1 and in_channels == filters:
        out = layers.Add(name=f"{name}_add")([x, out])
    return out


def build_ccnnet_backbone(input_shape=(224, 224, 3)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = ds_conv_block(x, 32, stride=1, expansion=1, name="s1_b0")
    x = ds_conv_block(x, 64, stride=2, expansion=4, name="s1_b1")
    x = ds_conv_block(x, 64, stride=1, expansion=4, name="s2_b0")
    x = ds_conv_block(x, 128, stride=2, expansion=4, name="s2_b1")
    x = ds_conv_block(x, 128, stride=1, expansion=4, name="s3_b0")
    x = ds_conv_block(x, 128, stride=1, expansion=4, name="s3_b1")
    x = ds_conv_block(x, 256, stride=2, expansion=4, name="s3_b2")
    x = ds_conv_block(x, 256, stride=1, expansion=4, name="s4_b0")
    x = ds_conv_block(x, 256, stride=1, expansion=4, name="s4_b1")
    x = ds_conv_block(x, 512, stride=2, expansion=4, name="s4_b2")
    x = ds_conv_block(x, 512, stride=1, expansion=2, name="s5_b0")
    backbone = models.Model(inputs=inp, outputs=x, name="CCNNet_Backbone")
    logger.info(f"CCNNet backbone: {backbone.count_params():,} params")
    return backbone


# ═══════════════════════════════════════════════════════════════════════════
# ③ ConvNeXt: ConvNeXtTiny
# ═══════════════════════════════════════════════════════════════════════════

def build_convnext_backbone(input_shape=(224, 224, 3)):
    try:
        from tensorflow.keras.applications import ConvNeXtTiny
        backbone = ConvNeXtTiny(weights='imagenet', include_top=False,
                                input_shape=input_shape)
        logger.info(f"ConvNeXtTiny backbone: {backbone.count_params():,} params")
    except ImportError:
        logger.warning("ConvNeXtTiny not available, fallback to EfficientNetV2B0")
        from tensorflow.keras.applications import EfficientNetV2B0
        backbone = EfficientNetV2B0(weights='imagenet', include_top=False,
                                    input_shape=input_shape)
        logger.info(f"ConvNeXt(fallback) backbone: {backbone.count_params():,} params")
    return backbone


# ═══════════════════════════════════════════════════════════════════════════
# ⑤ SFE-CA: EfficientNetB0 + SE + Channel Attention
# ═══════════════════════════════════════════════════════════════════════════

def se_module(x, reduction=16, name="se"):
    channels = x.shape[-1]
    reduced = max(1, channels // reduction)
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Dense(reduced, activation='relu', name=f"{name}_d1")(se)
    se = layers.Dense(channels, activation='sigmoid', name=f"{name}_d2")(se)
    se = layers.Reshape((1, 1, channels), name=f"{name}_r")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def channel_attention_module(x, name="ca"):
    channels = x.shape[-1]
    att = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    att = layers.Dense(max(1, channels // 8), activation='relu', name=f"{name}_fc1")(att)
    att = layers.Dense(channels, activation='sigmoid', name=f"{name}_fc2")(att)
    att = layers.Reshape((1, 1, channels), name=f"{name}_r")(att)
    return layers.Multiply(name=f"{name}_scale")([x, att])


def build_sfeca_backbone(input_shape=(224, 224, 3)):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base.output
    x = layers.Conv2D(256, 1, use_bias=False, name="proj_conv")(x)
    x = layers.BatchNormalization(name="proj_bn")(x)
    x = layers.ReLU(name="proj_relu")(x)
    x = se_module(x, reduction=8, name="se_enhance")
    x = channel_attention_module(x, name="ca_enhance")
    backbone = models.Model(inputs=base.input, outputs=x, name="SFECA_Backbone")
    logger.info(f"SFE-CA backbone: {backbone.count_params():,} params")
    return backbone


# ═══════════════════════════════════════════════════════════════════════════
# 预处理函数映射
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_resnet50(x):
    from tensorflow.keras.applications.resnet50 import preprocess_input
    return preprocess_input(x)

def preprocess_efficientnet(x):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(x)

def preprocess_convnext(x):
    try:
        from tensorflow.keras.applications.convnext import preprocess_input
        return preprocess_input(x)
    except ImportError:
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        return preprocess_input(x)

def preprocess_ccnnet(x):
    return (x / 127.5) - 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 模型注册表
# ═══════════════════════════════════════════════════════════════════════════

MODELS = {
    "Improved_ResNet50": {
        "backbone_builder": build_improved_resnet50_backbone,
        "preprocess_fn":    preprocess_resnet50,
        "image_size":       (224, 224),
        "paper":            "[19] 葛琪等 — ResNet50 + CBAM",
    },
    "CCNNet": {
        "backbone_builder": build_ccnnet_backbone,
        "preprocess_fn":    preprocess_ccnnet,
        "image_size":       (224, 224),
        "paper":            "[15] Hu et al. — CCNNet",
    },
    "ConvNeXt": {
        "backbone_builder": build_convnext_backbone,
        "preprocess_fn":    preprocess_convnext,
        "image_size":       (224, 224),
        "paper":            "[16] 兰邵琨等 — ConvNeXtTiny",
    },
    "SFE-CA": {
        "backbone_builder": build_sfeca_backbone,
        "preprocess_fn":    preprocess_efficientnet,
        "image_size":       (224, 224),
        "paper":            "[18] Zhang et al. — EfficientNetB0+SE+CA",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Keras 模型训练 (5折CV)
# ═══════════════════════════════════════════════════════════════════════════

def train_keras_model(model_name, model_info, df, num_classes, class_names):
    output_dir = os.path.join(CONFIG["output_root"], model_name.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)

    backbone_builder = model_info["backbone_builder"]
    preprocess_fn   = model_info["preprocess_fn"]
    image_size      = model_info["image_size"]

    tmp = backbone_builder()
    total_layers = len(tmp.layers)
    fine_tune_at = int(total_layers * CONFIG["fine_tune_ratio"])
    logger.info(f"[{model_name}] Total layers: {total_layers}, fine_tune_at={fine_tune_at}")
    del tmp; gc.collect()

    # ── 断点续训 (双层: 磁盘主力 + JSON辅助) ──
    ckpt_path = os.path.join(output_dir, "checkpoint.json")

    # 第一层: 磁盘文件扫描 (主力)
    disk_completed, disk_metrics = scan_completed_folds_from_disk(
        output_dir, CONFIG["cv_folds"])

    # 第二层: JSON 断点 (辅助, 用于补全磁盘报告缺的指标)
    ckpt_folds, json_metrics = load_ckpt(ckpt_path, model_name, CONFIG["cv_folds"])
    if ckpt_folds is None:
        ckpt_folds = []
    if json_metrics is None:
        json_metrics = {'acc': [], 'macro_prec': [], 'macro_rec': [],
                        'macro_f1': [], 'macro_auc': []}

    # 合并: 磁盘为主, JSON 为辅
    completed_folds = list(set(disk_completed) | set(ckpt_folds))
    metrics_summary = {'acc': [], 'macro_prec': [], 'macro_rec': [],
                       'macro_f1': [], 'macro_auc': []}
    for key in metrics_summary:
        # 优先从 JSON 取 (有AUC), 磁盘报告补充
        merged = disk_metrics.get(key, []) + json_metrics.get(key, [])
        if key == 'macro_auc':
            # AUC 只在 JSON 中有
            metrics_summary[key] = json_metrics.get(key, [])
        else:
            metrics_summary[key] = merged

    fold_models = []

    # 恢复已有的 fold_models 路径 (Keras 格式 + weights 格式)
    known_folds = set()
    for f in sorted(completed_folds):
        fold_dir = os.path.join(output_dir, f"fold_{f}")
        keras_path = os.path.join(fold_dir, "best_model.keras")
        weights_path = os.path.join(fold_dir, "best_model.weights.h5")
        if os.path.exists(keras_path) and os.path.getsize(keras_path) > 0:
            fold_models.append(keras_path)
            known_folds.add(f)
        elif os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
            fold_models.append(weights_path)
            known_folds.add(f)

    # 清理: 去掉 completed_folds 中无实际模型文件的 (磁盘说了算)
    completed_folds = sorted(known_folds)
    logger.info(f"Progress: {len(completed_folds)}/{CONFIG['cv_folds']} folds ready "
                f"(disk={disk_completed}, ckpt={ckpt_folds}, with_model={completed_folds})")

    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])

    for fold, (train_val_idx, test_idx) in enumerate(
        skf.split(df, df['label_encoded']), 1
    ):
        # 跳过已完成折
        if fold in completed_folds:
            logger.info(f"[{model_name}] Fold {fold}/{CONFIG['cv_folds']} — already done, skipping.")
            continue

        logger.info(f"{'='*60}")
        logger.info(f"[{model_name}] Fold {fold}/{CONFIG['cv_folds']}")
        logger.info(f"{'='*60}")

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_val_df = df.iloc[train_val_idx]
        test_fold_df = df.iloc[test_idx]

        train_fold_df, val_df = train_test_split(
            train_val_df, test_size=CONFIG["val_split"],
            stratify=train_val_df['label_encoded'], random_state=CONFIG["seed"]
        )
        logger.info(f"Train: {len(train_fold_df)} | Val: {len(val_df)} | Test: {len(test_fold_df)}")

        train_gen = ImageGenerator(train_fold_df, preprocess_fn, image_size, augment=True)
        val_gen   = ImageGenerator(val_df,        preprocess_fn, image_size, augment=False)
        test_gen  = ImageGenerator(test_fold_df,  preprocess_fn, image_size, augment=False)

        backbone = backbone_builder()
        model = build_head(backbone, num_classes, model_name=model_name)

        # Phase 1: 冻结骨干
        logger.info("Phase 1: Training head (backbone frozen)")
        backbone.trainable = False
        model.compile(
            optimizer=optimizers.Adam(CONFIG["phase1_lr"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        p1_ckpt = os.path.join(fold_dir, "phase1_best.weights.h5")
        csv_log = os.path.join(fold_dir, "training_log.csv")
        hist1 = model.fit(
            train_gen, validation_data=val_gen,
            epochs=CONFIG["phase1_epochs"],
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss',
                                        patience=CONFIG["phase1_patience"],
                                        restore_best_weights=True),
                callbacks.ModelCheckpoint(filepath=p1_ckpt, monitor='val_loss',
                                          save_best_only=True, save_weights_only=True),
                callbacks.CSVLogger(csv_log, append=True),
            ], verbose=1
        )

        # Phase 2: 微调
        logger.info(f"Phase 2: Fine-tuning from layer {fine_tune_at}/{total_layers}")
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        model.optimizer.learning_rate.assign(CONFIG["phase2_lr"])
        p2_ckpt = os.path.join(fold_dir, "phase2_best.weights.h5")
        hist2 = model.fit(
            train_gen, validation_data=val_gen,
            epochs=CONFIG["phase2_epochs"],
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss',
                                        patience=CONFIG["phase2_patience"],
                                        restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            factor=CONFIG["phase2_reduce_factor"],
                                            patience=CONFIG["phase2_reduce_patience"],
                                            min_lr=CONFIG["phase2_min_lr"]),
                callbacks.ModelCheckpoint(filepath=p2_ckpt, monitor='val_loss',
                                          save_best_only=True, save_weights_only=True),
                callbacks.CSVLogger(csv_log, append=True),
            ], verbose=1
        )

        # 保存模型 (weights-first: 先保存权重防止崩溃丢失)
        weights_path = os.path.join(fold_dir, "best_model.weights.h5")
        model.save_weights(weights_path)
        logger.info(f"Weights saved → {weights_path}")

        keras_path = os.path.join(fold_dir, "best_model.keras")
        try:
            model.save(keras_path)
            model_save_path = keras_path
            logger.info(f"Full model saved → {keras_path}")
        except Exception as e:
            logger.warning(f"Full model save failed ({e}), using weights-only")
            model_save_path = weights_path  # fallback
        fold_models.append(model_save_path)

        # 评估
        y_true = test_fold_df['label_encoded'].values
        y_probs = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(y_probs, axis=1)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')

        metrics_summary['acc'].append(acc)
        metrics_summary['macro_prec'].append(prec)
        metrics_summary['macro_rec'].append(rec)
        metrics_summary['macro_f1'].append(f1)
        metrics_summary['macro_auc'].append(auc)

        logger.info(f"Fold {fold} → Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        with open(os.path.join(fold_dir, "classification_report.txt"), 'w') as f:
            f.write(report)

        # ── 断点保存 ──
        completed_folds.append(fold)
        save_ckpt(ckpt_path, {
            'model_name': model_name,
            'total_folds': CONFIG["cv_folds"],
            'completed_folds': completed_folds,
            'metrics_summary': metrics_summary,
            'output_dir': output_dir,
        })

        del model, backbone, train_gen, val_gen, test_gen
        gc.collect()
        tf.keras.backend.clear_session()

    # 汇总 — 删除断点文件
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info(f"Checkpoint deleted (training complete).")

    summary_path = os.path.join(output_dir, f"Final_{model_name}_Summary.txt")
    lines = [
        f"{'='*60}",
        f"{model_name} — 5-Fold CV Summary",
        f"Paper: {model_info['paper']}",
        f"{'='*60}",
        f"Samples: {len(df)} | Classes: {num_classes}",
        f"{'-'*60}",
    ]
    for name, key in [("Accuracy", 'acc'), ("Macro Precision", 'macro_prec'),
                      ("Macro Recall", 'macro_rec'), ("Macro F1", 'macro_f1'),
                      ("Macro AUC", 'macro_auc')]:
        mean_v = np.mean(metrics_summary[key]) * 100
        std_v = np.std(metrics_summary[key]) * 100
        lines.append(f"{name:20s}: {mean_v:6.2f}% ± {std_v:.2f}%")
    lines.append(f"{'='*60}")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Summary → {summary_path}")

    return metrics_summary, fold_models, output_dir




# ═══════════════════════════════════════════════════════════════════════════
# 预测: 5折 soft voting 对全量数据预测
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 安全模型加载 (解决自定义层/激活函数反序列化失败问题)
# ═══════════════════════════════════════════════════════════════════════════

def _get_safe_custom_objects():
    """
    构建加载模型所需的 custom_objects 字典.
    覆盖: 自定义层、旧版 TF 可能缺失的激活函数.
    """
    import tensorflow.keras.layers as kl
    import tensorflow.keras.activations as ka
    co = {}

    # 确保内建激活函数被正确解析为函数 (非字符串)
    for act_name in ['swish', 'relu', 'sigmoid', 'softmax', 'tanh',
                     'elu', 'selu', 'gelu', 'linear']:
        try:
            fn = ka.get(act_name) if hasattr(ka, 'get') else getattr(ka, act_name, None)
            if fn is not None:
                co[act_name] = fn
        except Exception:
            pass

    # AddConv 匹配某些 SavedModel 中可能出现的辅助层
    for name in ['Add', 'Multiply', 'Concatenate']:
        try:
            co[name] = getattr(kl, name)
        except AttributeError:
            pass

    return co


def safe_load_model(model_path, backbone_builder=None, num_classes=None,
                    model_name="model", compile=False):
    """
    安全加载模型:
      1) 先尝试 load_model + custom_objects
      2) 失败则用 backbone_builder 重建架构 + 加载 weights
      3) 权重文件 (.weights.h5) 永远走重建路线
    """
    # 权重文件必须走重建路线
    if model_path.endswith('.weights.h5'):
        if backbone_builder is None:
            raise ValueError(
                f"Weights-only file '{model_path}' requires backbone_builder to reconstruct.")
        backbone = backbone_builder()
        model = build_head(backbone, num_classes, model_name=model_name)
        model.load_weights(model_path)
        logger.info(f"Model reconstructed from scratch + weights loaded: {model_path}")
        return model

    # .keras 或 SavedModel 格式: 先尝试直接加载
    custom_objects = _get_safe_custom_objects()
    try:
        model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=compile)
        logger.info(f"Model loaded successfully: {model_path}")
        return model
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        logger.warning(f"Direct load_model failed ({type(e).__name__}: {e}), "
                       f"falling back to architecture reconstruction...")
        # 回退: 重建架构 + 从 .keras 中加载权重
        if backbone_builder is None:
            raise RuntimeError(
                f"Cannot reconstruct model from '{model_path}': "
                f"load_model failed and no backbone_builder provided.") from e

        backbone = backbone_builder()
        model = build_head(backbone, num_classes, model_name=model_name)
        try:
            # .keras 文件是 zip, 可以用 load_weights 加载
            model.load_weights(model_path)
            logger.info(f"Model reconstructed + weights loaded from .keras: {model_path}")
            return model
        except Exception as e2:
            logger.error(f"Reconstruction also failed: {e2}")
            raise


def predict_keras_on_all(model_name, model_info, fold_models, df,
                         num_classes, class_names, output_dir):
    if not fold_models:
        logger.warning(f"[{model_name}] No trained models to predict with.")
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"[{model_name}] Predicting on ALL {len(df)} images (5-fold voting)")
    logger.info(f"{'='*60}")

    preprocess_fn = model_info["preprocess_fn"]
    image_size    = model_info["image_size"]
    gen = ImageGenerator(df, preprocess_fn, image_size, augment=False)

    all_probs = []
    for fold_idx, model_path in enumerate(fold_models, 1):
        logger.info(f"  Fold {fold_idx}: {model_path}")
        # 使用安全加载: 自动处理自定义层/激活函数反序列化失败
        model = safe_load_model(
            model_path,
            backbone_builder=model_info["backbone_builder"],
            num_classes=num_classes,
            model_name=model_name,
            compile=False
        )
        probs = model.predict(gen, verbose=1)
        all_probs.append(probs)
        del model; gc.collect(); tf.keras.backend.clear_session()

    avg_probs = np.mean(all_probs, axis=0)
    y_pred_idx = np.argmax(avg_probs, axis=1)
    y_true = df['label_encoded'].values
    confidences = np.max(avg_probs, axis=1)

    results = pd.DataFrame({
        'image_name':     df['image_name'].values,
        'image_path':     df['image_path'].values,
        'true_label':     df['Task_Chinese_medicinal_herb'].values,
        'true_label_enc': y_true,
        'pred_label_enc': y_pred_idx,
        'confidence':     np.round(confidences, 6),
        'correct':        (y_true == y_pred_idx).astype(int),
    })
    for i in range(avg_probs.shape[1]):
        results[f'prob_class_{i}'] = np.round(avg_probs[:, i], 6)

    safe_name = model_name.replace(" ", "_")
    pred_csv = os.path.join(output_dir, f"{safe_name}_all_predictions.csv")
    results.to_csv(pred_csv, index=False, encoding='utf-8-sig')
    logger.info(f"Saved → {pred_csv}")

    acc = accuracy_score(y_true, y_pred_idx)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_idx, average='macro', zero_division=0)

    report = classification_report(y_true, y_pred_idx, target_names=class_names, digits=4)
    with open(os.path.join(output_dir, f"{safe_name}_all_report.txt"), 'w') as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred_idx)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} — Confusion Matrix (All Data)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_name}_all_confusion.png"))
    plt.close()

    logger.info(f"Test Set → Acc={acc:.4f} | Macro F1={f1:.4f} | "
                f"Correct={int(results['correct'].sum())}/{len(df)}")

    return {
        'model': model_name,
        'paper': model_info['paper'],
        'n_samples': len(df),
        'n_correct': int(results['correct'].sum()),
        'accuracy': round(acc, 6),
        'macro_precision': round(prec, 6),
        'macro_recall': round(rec, 6),
        'macro_f1': round(f1, 6),
        'csv_path': pred_csv,
    }




# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def print_progress():
    """扫描所有模型进度并打印 (基于磁盘文件, 不依赖 JSON 断点)"""
    root = CONFIG["output_root"]
    if not os.path.isdir(root):
        return

    print(f"\n{'='*80}")
    print(f"  训练进度检查 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    for model_name in MODELS:
        safe = model_name.replace(" ", "_")
        model_dir = os.path.join(root, safe)

        if not os.path.isdir(model_dir):
            print(f"\n  ⚪ {model_name:<10s} — 未开始")
            continue

        # 按折检查磁盘 (主力: 直接检查每折的输出文件)
        disk_completed = []
        for fold in range(1, 6):
            fd = os.path.join(model_dir, f"fold_{fold}")
            if not os.path.isdir(fd):
                continue
            # Keras: 使用 check_fold_completed_from_disk
            is_done, _ = check_fold_completed_from_disk(fd)
            if is_done:
                disk_completed.append(fold)

        # 检查汇总/预测
        has_summary = os.path.exists(os.path.join(
            model_dir, f"Final_{model_name}_Summary.txt"))
        has_predict = os.path.exists(os.path.join(
            model_dir, f"{safe}_all_predictions.csv"))

        # 图标
        if has_summary and has_predict:
            icon = "✅"
        elif has_summary:
            icon = "🟡"
        elif disk_completed:
            icon = "🔄"
        else:
            icon = "🔴"

        bar = "".join("█" if f in disk_completed else "░" for f in range(1, 6))
        print(f"\n  {icon} {model_name:<10s} [{bar}] {len(disk_completed)}/5 folds (disk)")

        # 每折详情
        for fold in range(1, 6):
            fd = os.path.join(model_dir, f"fold_{fold}")
            if not os.path.isdir(fd):
                continue
            marker = " ←" if fold in disk_completed else ""
            # 读该折准确率
            report_path = os.path.join(fd, "classification_report.txt")
            acc_str = ""
            if os.path.exists(report_path):
                try:
                    for line in open(report_path):
                        if "accuracy" in line.lower():
                            acc_str = "  " + line.strip()[:70]
                            break
                except Exception:
                    pass
            # 模型文件状态
            if os.path.exists(os.path.join(fd, "best_model.keras")):
                state = "✓ model"
            elif os.path.exists(os.path.join(fd, "best_model.weights.h5")):
                state = "✓ wts"
            elif os.path.exists(os.path.join(fd, "phase2_best.weights.h5")):
                state = "○ p2"
            elif os.path.exists(os.path.join(fd, "phase1_best.weights.h5")):
                state = "◐ p1"
            else:
                state = "·"
            print(f"      Fold {fold}: [{state}]{marker}{acc_str}")

    # 汇总
    all_done = 0
    for model_name in MODELS:
        safe = model_name.replace(" ", "_")
        md = os.path.join(root, safe)
        s = os.path.exists(os.path.join(md, f"Final_{model_name}_Summary.txt"))
        p = os.path.exists(os.path.join(md, f"{safe}_all_predictions.csv"))
        if s and p:
            all_done += 1
    print(f"\n  ─────────────────────────────")
    print(f"  完成: {all_done}/{len(MODELS)} 模型")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Train & Predict 5 herb models")
    parser.add_argument("--skip", type=str, default="",
                        help="Models to skip, comma-separated (e.g. ccnnet,sfeca)")
    parser.add_argument("--only", type=str, default="",
                        help="Models to run ONLY, comma-separated")
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training, only predict (requires trained models)")
    args = parser.parse_args()

    skip = set(s.strip().lower().replace("-", "").replace(" ", "")
               for s in args.skip.split(",") if s.strip())
    only = set(s.strip().lower().replace("-", "").replace(" ", "")
               for s in args.only.split(",") if s.strip())

    selected = {}
    for key, info in MODELS.items():
        key_norm = key.lower().replace("-", "").replace(" ", "")
        if only:
            if key_norm in only:
                selected[key] = info
        else:
            if key_norm not in skip:
                selected[key] = info

    if not selected:
        print(f"No models selected. Available: {', '.join(MODELS.keys())}")
        return

    # ── 先检查进度 ──
    print_progress()

    print("=" * 70)
    print("  中药饮片识别 — 5模型训练+预测")
    print(f"  Models: {', '.join(selected.keys())}")
    print(f"  Labels: {CONFIG['labels_csv']}")
    print(f"  Images: {CONFIG['images_dir']}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'PREDICT ONLY' if args.predict_only else 'TRAIN + PREDICT'}")
    print("=" * 70)

    print("\n[1/3] Loading data...")
    df, le, num_classes, class_names = load_data()

    os.makedirs(CONFIG["output_root"], exist_ok=True)
    all_predictions = []

    for model_name, model_info in selected.items():
        print(f"\n{'#'*70}")
        print(f"#  {model_name} — {model_info['paper']}")
        print(f"{'#'*70}")

        output_dir = os.path.join(CONFIG["output_root"], model_name.replace(" ", "_"))
        os.makedirs(output_dir, exist_ok=True)

        if args.predict_only:
            fold_models = []
            for fold in range(1, CONFIG["cv_folds"] + 1):
                fd = os.path.join(output_dir, f"fold_{fold}")
                for candidate in ["best_model.keras", "best_model.weights.h5"]:
                    mp = os.path.join(fd, candidate)
                    if os.path.exists(mp):
                        fold_models.append(mp)
                        break
            if not fold_models:
                logger.warning(f"No {model_name} models found. Skipping.")
                continue
        else:
            _, fold_models, _ = train_keras_model(
                model_name, model_info, df, num_classes, class_names)
        result = predict_keras_on_all(
            model_name, model_info, fold_models, df, num_classes, class_names, output_dir)

        if result:
            all_predictions.append(result)

    # 汇总
    print(f"\n{'='*70}")
    print("  预测结果汇总")
    print(f"{'='*70}")

    if all_predictions:
        summary_df = pd.DataFrame(all_predictions)
        summary_csv = os.path.join(CONFIG["output_root"], "all_models_summary.csv")
        summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

        print(f"\n{'Model':<12s} {'Accuracy':>10s} {'Macro F1':>10s} {'Correct':>14s}")
        print("-" * 55)
        for _, row in summary_df.iterrows():
            print(f"{row['model']:<12s} {row['accuracy']:>10.4f} {row['macro_f1']:>10.4f} "
                  f"{int(row['n_correct']):>6d}/{int(row['n_samples']):<5d}")

        print(f"\n  Summary: {summary_csv}")
        for r in all_predictions:
            print(f"  {r['model']}: {r['csv_path']}")

    print(f"\n{'='*70}")
    print(f"  Done! Output: {CONFIG['output_root']}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
