#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gc
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import StratifiedKFold, train_test_split   # 补充 train_test_split 导入
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# ---------- 1. 配置 (与多模态模型严格对齐) ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    "images_dir": '/home/shify_ug/multimodal_project/data/images',
    "captions_path": '/home/shify_ug/multimodal_project/text_slices/image_captions.json',
    "labels_path": '/home/shify_ug/multimodal_project/labels.csv',
    "image_size": (224, 224),
    "batch_size": 16,                # 与多模态一致
    "max_text_len": 128,            # 虽不用，保留占位
    "max_words": 10000,
    "embedding_dim": 256,
    "oov_token": "<OOV>",
    "cv_folds": 5,
    "output_dir": "./baseline_strict",   # 新输出目录
    "fine_tune_at": 120,
}

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# ---------- 2. 图像数据生成器 (仅图像，无文本) ----------
class ImageOnlyGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, le, augment=False):
        self.df = df.reset_index(drop=True)
        self.le = le
        self.augment = augment
        self.img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1]
        ) if augment else None

    def __len__(self):
        return int(np.ceil(len(self.df) / CONFIG["batch_size"]))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * CONFIG["batch_size"] : (idx + 1) * CONFIG["batch_size"]]
        imgs, labels = [], []
        for _, row in batch_df.iterrows():
            img = load_img(row['image_path'], target_size=CONFIG["image_size"])
            img_arr = img_to_array(img)                     # 0-255
            if self.augment:
                img_arr = self.img_gen.random_transform(img_arr)
            img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
            imgs.append(img_arr)
            labels.append(row['label_encoded'])
        return np.array(imgs), np.array(labels)

# ---------- 3. 纯图像模型架构 (与多模态图像分支完全一致) ----------
def build_image_model(num_classes):
    img_input = layers.Input(shape=(*CONFIG["image_size"], 3), name="img_input")
    v_backbone = tf.keras.applications.MobileNetV2(
        input_tensor=img_input, include_top=False, weights='imagenet'
    )
    v_backbone.trainable = False   # 初期冻结

    v_feat = layers.GlobalAveragePooling2D()(v_backbone.output)
    v_feat = layers.BatchNormalization()(v_feat)
    v_feat = layers.Dense(512, activation='swish')(v_feat)

    # 直接分类 (无文本分支)
    output = layers.Dense(num_classes, activation='softmax')(v_feat)

    model = models.Model(inputs=img_input, outputs=output)
    return model, v_backbone

# ---------- 4. 可视化 (同多模态模型) ----------
def plot_training_history(hist1, hist2, save_path, fold):
    if hist2 is not None:
        loss = hist1.history['loss'] + hist2.history['loss']
        val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
        acc = hist1.history['accuracy'] + hist2.history['accuracy']
        val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
        phase1_end = len(hist1.history['loss'])
    else:
        loss = hist1.history['loss']
        val_loss = hist1.history['val_loss']
        acc = hist1.history['accuracy']
        val_acc = hist1.history['val_accuracy']
        phase1_end = len(loss)

    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    if hist2 is not None:
        plt.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1 End')
    plt.title(f'Fold {fold} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'r-', label='Val Acc')
    if hist2 is not None:
        plt.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1 End')
    plt.title(f'Fold {fold} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, fold):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Fold {fold} - Confusion Matrix (Test Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---------- 5. 基线模型训练管道 (严格对齐多模态流程) ----------
def run_baseline_cv():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ---- 加载和对齐数据 (与多模态相同的数据源) ----
    with open(CONFIG["captions_path"], 'r', encoding='utf-8') as f:
        captions = json.load(f)   # 我们不需要文本，但用此确保图像存在
    df = pd.read_csv(CONFIG["labels_path"])
    # 只保留有对应图像的样本 (与多模态一致)
    valid_images = set(captions.keys())
    df = df[df['image_name'].isin(valid_images)]
    df['image_path'] = df['image_name'].apply(lambda x: os.path.join(CONFIG["images_dir"], x))
    df = df.dropna(subset=['image_path']).reset_index(drop=True)

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Task_Chinese_medicinal_herb'])
    num_classes = len(le.classes_)
    class_names_str = [str(c) for c in le.classes_]
    logger.info(f"Baseline: Clean samples: {len(df)} | classes: {num_classes}")

    metrics_summary = {
        'acc': [], 'macro_prec': [], 'macro_rec': [], 'macro_f1': [], 'macro_auc': []
    }

    # 使用与多模态完全相同的 StratifiedKFold 划分
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=42)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df['label_encoded']), 1):
        logger.info("=" * 60)
        logger.info(f"Baseline Fold {fold}/{CONFIG['cv_folds']}")
        logger.info("=" * 60)

        fold_dir = os.path.join(CONFIG["output_dir"], f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        # 从 train_val 中随机切出验证集 (与多模态相同比例 15%)
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.15,
            stratify=train_val_df['label_encoded'], random_state=42
        )
        logger.info(f"Fold {fold} split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # 生成器 (无类别权重)
        train_gen = ImageOnlyGenerator(train_df, le, augment=True)
        val_gen   = ImageOnlyGenerator(val_df,   le, augment=False)
        test_gen  = ImageOnlyGenerator(test_df,  le, augment=False)

        # 构建模型
        model, v_backbone = build_image_model(num_classes)

        # ---------- Phase 1: 训练顶层 (与多模态一致) ----------
        logger.info(f"[Fold {fold}] Phase 1: Training top layers...")
        model.compile(optimizer=optimizers.Adam(1e-4),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        hist1 = model.fit(
            train_gen, validation_data=val_gen, epochs=12,
            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=1
        )

        # ---------- Phase 2: 解冻部分骨干微调 (与多模态相同) ----------
        logger.info(f"[Fold {fold}] Phase 2: Fine-tuning top layers of backbone...")
        v_backbone.trainable = True
        for layer in v_backbone.layers[:CONFIG["fine_tune_at"]]:
            layer.trainable = False
        for layer in v_backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        model.optimizer.learning_rate.assign(5e-6)
        hist2 = model.fit(
            train_gen, validation_data=val_gen, epochs=20,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7)
            ],
            verbose=1
        )

        plot_training_history(hist1, hist2, os.path.join(fold_dir, f"training_curves_fold{fold}.png"), fold)
        model.save(os.path.join(fold_dir, f"best_model_fold{fold}"))

        # ---------- 评估 ----------
        logger.info(f"[Fold {fold}] Evaluating on test set...")
        y_true = test_df['label_encoded'].values
        y_pred_probs = model.predict(test_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)

        report = classification_report(y_true, y_pred, target_names=class_names_str, digits=4)
        with open(os.path.join(fold_dir, f"classification_report_fold{fold}.txt"), 'w') as f:
            f.write(report)
        plot_confusion_matrix(y_true, y_pred, class_names_str,
                              os.path.join(fold_dir, f"confusion_matrix_fold{fold}.png"), fold)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)
        macro_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)

        metrics_summary['acc'].append(test_acc)
        metrics_summary['macro_prec'].append(precision)
        metrics_summary['macro_rec'].append(recall)
        metrics_summary['macro_f1'].append(f1)
        metrics_summary['macro_auc'].append(macro_auc)

        logger.info(f"[Fold {fold}] Results: Acc={test_acc:.4f}, F1={f1:.4f}, AUC={macro_auc:.4f}")

        del model, v_backbone, train_gen, val_gen, test_gen
        gc.collect()
        tf.keras.backend.clear_session()

    # 汇总 5 折
    logger.info("\n" + "=" * 60)
    logger.info("Baseline 5-Fold CV Summary")
    logger.info("=" * 60)
    final_summary = "Baseline Image-Only 5-Fold CV Performance Report\n"
    final_summary += "=" * 50 + "\n"
    for name, key in [("Accuracy", 'acc'), ("Macro Precision", 'macro_prec'),
                      ("Macro Recall", 'macro_rec'), ("Macro F1", 'macro_f1'),
                      ("Macro AUC", 'macro_auc')]:
        mean_val = np.mean(metrics_summary[key]) * 100
        std_val = np.std(metrics_summary[key]) * 100
        result_str = f"{name}: {mean_val:.2f}% +/- {std_val:.2f}%"
        logger.info(result_str)
        final_summary += result_str + "\n"
    with open(os.path.join(CONFIG["output_dir"], "Final_Baseline_Summary.txt"), "w") as f:
        f.write(final_summary)
    logger.info(f"Baseline experiments completed. Results saved to {CONFIG['output_dir']}.")

if __name__ == "__main__":
    run_baseline_cv()