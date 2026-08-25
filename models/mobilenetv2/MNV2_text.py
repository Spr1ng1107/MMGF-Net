#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gc
import json
import numpy as np
import pandas as pd
import jieba

# Fix for headless server plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import logging

# ---------- 1. Logging & Config ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    "images_dir": '/home/shify_ug/multimodal_project/data/images',
    "captions_path": '/home/shify_ug/multimodal_project/text_slices/image_captions.json',
    "labels_path": '/home/shify_ug/multimodal_project/labels.csv',
    "image_size": (224, 224),
    "batch_size": 16,
    "max_text_len": 320,
    "max_words": 10000,
    "embedding_dim": 256,
    "oov_token": "<OOV>",
    "cv_folds": 5,
    "output_dir": "./output3",
    "fine_tune_at": 120,
    "lstm_units": 256,
    "text_dropout_rate": 0.4,       # 降低 Dropout 防止联合训练时模态失衡
    "epochs": 30,                   # 单阶段联合训练总轮数
    "learning_rate": 1e-4           # 联合训练统一初始学习率
}

# ---------- GPU memory growth ----------
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    logger.info("GPU memory growth enabled")

# ---------- 2. Chinese word segmentation ----------
def segment_text(text):
    return ' '.join(jieba.cut(str(text)))

# ---------- 3. Model architecture (Joint Training & Gated Fusion) ----------
def build_stable_mnv2_multimodal(num_classes):
    # Visual branch
    img_input = layers.Input(shape=(*CONFIG["image_size"], 3), name="img_input")
    v_backbone = tf.keras.applications.MobileNetV2(
        input_tensor=img_input, include_top=False, weights='imagenet'
    )
    
    # 提前在构建阶段解冻图像高层网络，直接为联合训练做准备
    v_backbone.trainable = True
    for layer in v_backbone.layers[:CONFIG["fine_tune_at"]]:
        layer.trainable = False
    # 保持 BN 层冻结，防止在 batch 较小时破坏预训练的统计数据
    for layer in v_backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    v_feat = layers.GlobalAveragePooling2D()(v_backbone.output)
    v_feat = layers.BatchNormalization()(v_feat)
    v_feat = layers.Dense(512, activation='swish', name='v_dense_proj')(v_feat)

    # Text branch
    text_input = layers.Input(shape=(CONFIG["max_text_len"],), name="text_input")
    t_feat = layers.Embedding(CONFIG["max_words"], CONFIG["embedding_dim"])(text_input)
    t_feat = layers.SpatialDropout1D(CONFIG["text_dropout_rate"])(t_feat)
    
    # 改进：输出完整序列，使用池化捕获更长远的上下文信息
    t_seq = layers.Bidirectional(
        layers.LSTM(CONFIG["lstm_units"], return_sequences=True)
    )(t_feat)
    
    avg_pool = layers.GlobalAveragePooling1D()(t_seq)
    max_pool = layers.GlobalMaxPooling1D()(t_seq)
    t_feat = layers.concatenate([avg_pool, max_pool])
    
    t_feat = layers.LayerNormalization()(t_feat)
    t_feat = layers.Dense(512, activation='swish', name='t_dense_proj')(t_feat)

    # 主流改进：Gated Multimodal Fusion (门控多模态融合)
    merged = layers.concatenate([v_feat, t_feat], name='concat_features')
    
    # 门控机制：让模型自适应学习应该多大程度信任图像或文本特征
    gate = layers.Dense(1024, activation='sigmoid', name='fusion_gate')(merged)
    gated_merged = layers.multiply([merged, gate], name='gated_fusion')
    
    x = layers.Dense(512, activation='swish')(gated_merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    output = layers.Dense(num_classes, activation='softmax', name='classifier_out')(x)

    model = models.Model(inputs=[img_input, text_input], outputs=output)
    return model

# ---------- 4. Robust multimodal data generator ----------
class RobustGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, tokenizer, le, augment=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
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
        imgs, texts, labels = [], [], []

        for _, row in batch_df.iterrows():
            img = load_img(row['image_path'], target_size=CONFIG["image_size"])
            img_arr = img_to_array(img)
            if self.augment:
                img_arr = self.img_gen.random_transform(img_arr)
            img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
            imgs.append(img_arr)

            texts.append(row['caption_seg'])
            labels.append(row['Task_Chinese_medicinal_herb'])

        X_text = pad_sequences(
            self.tokenizer.texts_to_sequences(texts),
            maxlen=CONFIG["max_text_len"]
        )
        y = self.le.transform(labels)
        return [np.array(imgs), X_text], np.array(y)

# ---------- 5. Visualization functions (Updated for single phase) ----------
def plot_training_history(hist, save_path, fold):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    plt.title(f'Fold {fold} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'r-', label='Val Acc')
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

# ---------- 6. Rigorous training pipeline ----------
def run_rigorous_cv():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1. Load and clean data
    with open(CONFIG["captions_path"], 'r', encoding='utf-8') as f:
        captions = json.load(f)
    df = pd.read_csv(CONFIG["labels_path"])
    df['caption_raw'] = df['image_name'].map(captions)
    df = df.dropna(subset=['caption_raw']).reset_index(drop=True)
    df['caption_seg'] = df['caption_raw'].apply(segment_text)
    df['image_path'] = df['image_name'].apply(lambda x: os.path.join(CONFIG["images_dir"], x))
    df = df.dropna(subset=['image_path']).reset_index(drop=True)

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Task_Chinese_medicinal_herb'])
    num_classes = len(le.classes_)
    class_names_str = [str(c) for c in le.classes_]
    
    logger.info(f"Clean samples: {len(df)} | classes: {num_classes}")

    metrics_summary = {
        'acc': [],
        'macro_prec': [],
        'macro_rec': [],
        'macro_f1': [],
        'macro_auc': []
    }

    # 2. Stratified 5-fold CV
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=42)
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df['label_encoded']), 1):
        logger.info("=" * 60)
        logger.info(f"Starting Fold {fold} / {CONFIG['cv_folds']}")
        logger.info("=" * 60)

        fold_dir = os.path.join(CONFIG["output_dir"], f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        train_df, val_df = train_test_split(
            train_val_df, test_size=0.15, 
            stratify=train_val_df['label_encoded'], random_state=42
        )
        
        logger.info(f"Fold {fold} split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        tok = Tokenizer(num_words=CONFIG["max_words"], oov_token=CONFIG["oov_token"])
        tok.fit_on_texts(train_df['caption_seg'])

        y_train_labels = train_df['label_encoded'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        class_weight_dict = dict(enumerate(class_weights))

        train_gen = RobustGenerator(train_df, tok, le, augment=True)
        val_gen = RobustGenerator(val_df, tok, le, augment=False)
        test_gen = RobustGenerator(test_df, tok, le, augment=False)

        model = build_stable_mnv2_multimodal(num_classes)

        # 联合训练：摒弃分阶段，使用全局学习率与回调函数自动管理
        logger.info(f"[Fold {fold}] Training branches jointly...")
        model.compile(optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
                      
        hist = model.fit(
            train_gen, 
            validation_data=val_gen, 
            epochs=CONFIG["epochs"],
            class_weight=class_weight_dict,
            callbacks=[
                # 提前停止，保留最优权重
                callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                # 学习率动态衰减，当 val_loss 停滞时减小学习率
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
            ],
            verbose=1
        )

        plot_training_history(hist, os.path.join(fold_dir, f"training_curves_fold{fold}.png"), fold)
        model.save(os.path.join(fold_dir, f"best_model_fold{fold}"))

        # Evaluate on test set
        logger.info(f"[Fold {fold}] Evaluating on hold-out test set...")
        y_true = test_df['label_encoded'].values
        y_pred_probs = model.predict(test_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)

        report = classification_report(y_true, y_pred, target_names=class_names_str, digits=4)
        with open(os.path.join(fold_dir, f"classification_report_fold{fold}.txt"), 'w') as f:
            f.write(report)
        plot_confusion_matrix(y_true, y_pred, class_names_str, os.path.join(fold_dir, f"confusion_matrix_fold{fold}.png"), fold)

        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        macro_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
        
        metrics_summary['acc'].append(test_acc)
        metrics_summary['macro_prec'].append(precision)
        metrics_summary['macro_rec'].append(recall)
        metrics_summary['macro_f1'].append(f1)
        metrics_summary['macro_auc'].append(macro_auc)
        
        logger.info(
            f"[Fold {fold}] Results: Acc={test_acc:.4f}, Macro P={precision:.4f}, "
            f"Macro R={recall:.4f}, Macro F1={f1:.4f}, Macro AUC={macro_auc:.4f}"
        )

        # 内存释放与清理机制
        del model, train_gen, val_gen, test_gen
        gc.collect()
        tf.keras.backend.clear_session()

    # 3. Final 5-fold summary
    logger.info("\n" + "=" * 60)
    logger.info("5-Fold Cross-Validation Multimodal Summary")
    logger.info("=" * 60)
    
    final_summary = "Multimodal 5-Fold CV Performance Report\n"
    final_summary += "=" * 50 + "\n"
    
    metrics_list = [
        ("Accuracy", 'acc'),
        ("Macro Precision", 'macro_prec'),
        ("Macro Recall", 'macro_rec'),
        ("Macro F1", 'macro_f1'),
        ("Macro AUC", 'macro_auc'),
    ]
    
    for name, key in metrics_list:
        mean_val = np.mean(metrics_summary[key]) * 100
        std_val = np.std(metrics_summary[key]) * 100
        result_str = f"{name}: {mean_val:.2f}% +/- {std_val:.2f}%"
        logger.info(result_str)
        final_summary += result_str + "\n"

    with open(os.path.join(CONFIG["output_dir"], "Final_5Fold_CrossValidation_Summary.txt"), "w") as f:
        f.write(final_summary)
        
    logger.info(f"All experiments completed. Results saved to {CONFIG['output_dir']}.")

if __name__ == "__main__":
    jieba.setLogLevel(logging.INFO)
    run_rigorous_cv()
