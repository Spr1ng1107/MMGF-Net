#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared multimodal training configuration, data generator, model builder, and CV pipeline.
STRICTLY follows the strategy from MNV2_text.py (MobileNetV2 multimodal baseline).

Architecture (identical to MNV2_text.py):
  Visual branch: backbone → GAP → BN → Dense(512, swish)
  Text branch:   Embedding → SpatialDropout1D → BiLSTM → GAP1D+GMP1D → LayerNorm → Dense(512, swish)
  Fusion:        Concat → Gate(Dense 1024, sigmoid) → Multiply → Dense(512, swish) → BN → Dropout(0.5) → Softmax

Training strategy:
  - Single-phase joint training (backbone partially frozen from start)
  - Adam(lr=1e-4), 30 epochs
  - EarlyStopping(patience=8, restore_best_weights=True)
  - ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
  - Class weights (balanced) for imbalance handling
  - BN layers always frozen (prevents batch-size-induced statistics corruption)
  - 5-fold StratifiedKFold (shuffle=True, random_state=42)
  - Train/Val split: 15% stratified

Data pipeline:
  - Images: model-specific size + preprocessing
  - Text: Chinese captions via jieba segmentation
  - Tokenizer per fold, max_words=10000, max_text_len=320
  - Augmentation: rotation ±20°, horizontal flip, brightness [0.9, 1.1]
  - Filtered by captions.json keys (ensures same image set as multimodal baseline)
"""
import os
import gc
import json
import numpy as np
import pandas as pd
import jieba

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Path resolution ───
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


def _resolve_path(env_var, *candidates):
    """Resolve a path: env var first, then try each candidate, return first that exists."""
    env_val = os.environ.get(env_var)
    if env_val and os.path.exists(env_val):
        return env_val
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ""


# ─── Multimodal CONFIG (STRICTLY matches MNV2_text.py) ───
MULTIMODAL_CONFIG = {
    # --- Paths ---
    "labels_csv": _resolve_path(
        'HERB_LABELS_CSV',
        os.path.join(_PROJECT_ROOT, "labels.csv"),
        os.path.join(_SCRIPT_DIR, "labels.csv"),
    ),
    "captions_json": _resolve_path(
        'HERB_CAPTIONS_JSON',
        os.path.join(_PROJECT_ROOT, "text_slices", "image_captions.json"),
        os.path.join(_SCRIPT_DIR, "image_captions.json"),
    ),
    "images_dir": _resolve_path(
        'HERB_IMAGES_DIR',
        os.path.join(_PROJECT_ROOT, "data", "images"),
        os.path.join(_SCRIPT_DIR, "images"),
    ),

    # --- Training hyperparameters (IDENTICAL to MNV2_text.py) ---
    "batch_size": 16,
    "cv_folds": 5,
    "max_text_len": 320,
    "max_words": 10000,
    "embedding_dim": 256,
    "oov_token": "<OOV>",
    "lstm_units": 256,
    "text_dropout_rate": 0.4,
    "epochs": 30,
    "learning_rate": 1e-4,

    # --- Fine-tune proportion (MNV2: 120/155 ≈ 77%) ---
    "fine_tune_ratio": 0.77,

    # --- Per-model (set by each train script) ---
    "image_size": None,
    "preprocess_fn": None,
    "output_dir": None,
    "backbone_builder": None,
    "model_name": None,

    # --- Checkpoint ---
    "checkpoint_file": "cv_checkpoint_multimodal.json",
}

# GPU memory growth (identical to baseline)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    logger.info("GPU memory growth enabled")


# ─── Chinese word segmentation (identical to MNV2_text.py) ───
def segment_text(text):
    return ' '.join(jieba.cut(str(text)))


# ─── Multimodal model builder (STRICTLY matches MNV2_text.py architecture) ───
def build_multimodal_model(backbone_fn, num_classes, fine_tune_at, image_size):
    """
    Build the full multimodal model (image + text + gated fusion).
    Architecture IDENTICAL to MNV2_text.py build_stable_mnv2_multimodal().

    Args:
        backbone_fn: callable(input_tensor) -> backbone Keras Model (include_top=False)
        num_classes: int
        fine_tune_at: int, layers before this index are frozen
        image_size: tuple (H, W)

    Returns:
        Keras Model with two inputs: [img_input, text_input]
    """
    # ── Visual branch (identical to MNV2_text.py) ──
    img_input = layers.Input(shape=(*image_size, 3), name="img_input")
    v_backbone = backbone_fn(input_tensor=img_input)

    # Freeze early layers (same strategy as MNV2: partial backbone freezing)
    v_backbone.trainable = True
    for layer in v_backbone.layers[:fine_tune_at]:
        layer.trainable = False
    # Freeze all BN layers (prevents statistics corruption with small batch size)
    for layer in v_backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    v_feat = layers.GlobalAveragePooling2D()(v_backbone.output)
    v_feat = layers.BatchNormalization()(v_feat)
    v_feat = layers.Dense(512, activation='swish', name='v_dense_proj')(v_feat)

    # ── Text branch (identical to MNV2_text.py) ──
    text_input = layers.Input(shape=(MULTIMODAL_CONFIG["max_text_len"],), name="text_input")
    t_feat = layers.Embedding(
        MULTIMODAL_CONFIG["max_words"],
        MULTIMODAL_CONFIG["embedding_dim"]
    )(text_input)
    t_feat = layers.SpatialDropout1D(MULTIMODAL_CONFIG["text_dropout_rate"])(t_feat)

    t_seq = layers.Bidirectional(
        layers.LSTM(MULTIMODAL_CONFIG["lstm_units"], return_sequences=True)
    )(t_feat)

    avg_pool = layers.GlobalAveragePooling1D()(t_seq)
    max_pool = layers.GlobalMaxPooling1D()(t_seq)
    t_feat = layers.concatenate([avg_pool, max_pool])

    t_feat = layers.LayerNormalization()(t_feat)
    t_feat = layers.Dense(512, activation='swish', name='t_dense_proj')(t_feat)

    # ── Gated Multimodal Fusion (identical to MNV2_text.py) ──
    merged = layers.concatenate([v_feat, t_feat], name='concat_features')

    gate = layers.Dense(1024, activation='sigmoid', name='fusion_gate')(merged)
    gated_merged = layers.multiply([merged, gate], name='gated_fusion')

    x = layers.Dense(512, activation='swish')(gated_merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(num_classes, activation='softmax', name='classifier_out')(x)

    model = models.Model(inputs=[img_input, text_input], outputs=output)
    return model


# ─── Robust multimodal data generator (STRICTLY matches MNV2_text.py) ───
class RobustMultimodalGenerator(tf.keras.utils.Sequence):
    """
    Image + Text data generator.
    IDENTICAL to MNV2_text.py RobustGenerator.
    Loads images, applies model-specific preprocessing, tokenizes text.
    """
    def __init__(self, df, tokenizer, le, preprocess_fn, image_size, augment=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = le
        self.preprocess_fn = preprocess_fn
        self.image_size = image_size
        self.augment = augment
        self.img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1]
        ) if augment else None

    def __len__(self):
        return int(np.ceil(len(self.df) / MULTIMODAL_CONFIG["batch_size"]))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[
            idx * MULTIMODAL_CONFIG["batch_size"]:
            (idx + 1) * MULTIMODAL_CONFIG["batch_size"]
        ]
        imgs, texts, labels = [], [], []

        for _, row in batch_df.iterrows():
            img = load_img(row['image_path'], target_size=self.image_size)
            img_arr = img_to_array(img)
            if self.augment:
                img_arr = self.img_gen.random_transform(img_arr)
            img_arr = self.preprocess_fn(img_arr)
            imgs.append(img_arr)

            texts.append(row['caption_seg'])
            labels.append(row['Task_Chinese_medicinal_herb'])

        X_text = pad_sequences(
            self.tokenizer.texts_to_sequences(texts),
            maxlen=MULTIMODAL_CONFIG["max_text_len"]
        )
        y = self.le.transform(labels)
        return [np.array(imgs), X_text], np.array(y)


# ─── Visualization functions (identical to MNV2_text.py) ───
def plot_training_history(hist, save_path, fold):
    """Plot training curves for single-phase training."""
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
    """Plot confusion matrix."""
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


# ─── Data loading with caption filtering (STRICTLY matches MNV2_text.py) ───
def load_and_prepare_multimodal_data():
    """
    Load labels.csv, filter by captions.json, segment Chinese text, encode labels.
    STRICTLY matches MNV2_text.py run_rigorous_cv() data pipeline.
    """
    # Filter by captions.json
    captions_path = MULTIMODAL_CONFIG.get("captions_json", "")
    if os.path.exists(captions_path):
        with open(captions_path, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        logger.info(f"Loaded captions.json: {len(captions)} entries")
    else:
        logger.error(f"captions.json not found at '{captions_path}'")
        raise FileNotFoundError(f"captions.json not found at '{captions_path}'")

    df = pd.read_csv(MULTIMODAL_CONFIG["labels_csv"])
    df['caption_raw'] = df['image_name'].map(captions)
    df = df.dropna(subset=['caption_raw']).reset_index(drop=True)
    df['caption_seg'] = df['caption_raw'].apply(segment_text)
    df['image_path'] = df['image_name'].apply(
        lambda x: os.path.join(MULTIMODAL_CONFIG["images_dir"], x))
    df = df[df['image_path'].apply(os.path.exists)]
    df = df.dropna(subset=['image_path']).reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(
            f"No valid image-text pairs found! Check:\n"
            f"  labels_csv: {MULTIMODAL_CONFIG['labels_csv']}\n"
            f"  captions_json: {MULTIMODAL_CONFIG['captions_json']}\n"
            f"  images_dir: {MULTIMODAL_CONFIG['images_dir']}")

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Task_Chinese_medicinal_herb'])
    num_classes = len(le.classes_)
    class_names_str = [str(c) for c in le.classes_]

    logger.info(f"Clean multimodal samples: {len(df)} | classes: {num_classes}")
    return df, le, num_classes, class_names_str


# ─── Checkpoint utilities (for resuming interrupted training) ───
def save_checkpoint(checkpoint_path, state):
    """Save training checkpoint to JSON."""
    try:
        serializable_state = {}
        for key, value in state.items():
            if isinstance(value, dict):
                serializable_state[key] = {
                    k: [float(v) if hasattr(v, 'item') else v for v in v_list]
                    if isinstance(v_list, list) else float(v_list) if hasattr(v_list, 'item') else v_list
                    for k, v_list in value.items()
                }
            elif isinstance(value, list):
                serializable_state[key] = [
                    float(v) if hasattr(v, 'item') else v for v in value
                ]
            else:
                serializable_state[key] = value
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=2, ensure_ascii=False)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_path):
    """Load training checkpoint. Returns state dict or None."""
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoint found at {checkpoint_path} — starting fresh.")
        return None
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        required_keys = ['completed_folds', 'metrics_summary', 'model_name', 'total_folds']
        for key in required_keys:
            if key not in state:
                logger.warning(f"Checkpoint missing key '{key}' — ignoring.")
                return None
        logger.info(f"Checkpoint loaded: {len(state['completed_folds'])}/{state['total_folds']} "
                    f"folds completed (folds {state['completed_folds']})")
        return state
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Checkpoint corrupted ({e}) — starting fresh.")
        return None


def delete_checkpoint(checkpoint_path):
    """Remove checkpoint after successful completion."""
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info(f"Checkpoint {checkpoint_path} deleted (training complete).")
        except OSError as e:
            logger.warning(f"Failed to delete checkpoint: {e}")


# ─── Main multimodal CV pipeline (STRICTLY matches MNV2_text.py run_rigorous_cv) ───
def run_multimodal_cv(model_name, backbone_builder, preprocess_fn, image_size, output_dir):
    """
    Run 5-fold multimodal cross-validation.

    STRICTLY matches MNV2_text.py training strategy:
      - Single-phase joint training (backbone partially frozen from start)
      - Adam(lr=1e-4), 30 epochs
      - EarlyStopping(patience=8, restore_best_weights=True)
      - ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
      - Class weights (balanced)
      - BN layers always frozen
      - fine_tune_at auto-computed from fine_tune_ratio (~77%)

    Args:
        model_name: str, for logging and file naming
        backbone_builder: callable() -> backbone Keras Model (include_top=False)
        preprocess_fn: model-specific preprocessing (e.g. inception_v3.preprocess_input)
        image_size: tuple (H, W), model's native input size
        output_dir: str, output directory
    """
    # Update shared config
    MULTIMODAL_CONFIG["image_size"] = image_size
    MULTIMODAL_CONFIG["preprocess_fn"] = preprocess_fn
    MULTIMODAL_CONFIG["output_dir"] = output_dir
    MULTIMODAL_CONFIG["backbone_builder"] = backbone_builder
    MULTIMODAL_CONFIG["model_name"] = model_name

    os.makedirs(output_dir, exist_ok=True)

    # ─── Checkpoint setup ───
    checkpoint_path = os.path.join(output_dir, MULTIMODAL_CONFIG["checkpoint_file"])
    completed_folds = []

    # ─── Load data ───
    df, le, num_classes, class_names_str = load_and_prepare_multimodal_data()

    # ─── Compute fine_tune_at (build without input_tensor for layer count) ───
    _tmp_backbone = backbone_builder(input_tensor=None)
    total_layers = len(_tmp_backbone.layers)
    fine_tune_at = int(total_layers * MULTIMODAL_CONFIG["fine_tune_ratio"])
    logger.info(f"[{model_name}] Backbone layers: {total_layers}, "
                f"fine_tune_at={fine_tune_at} (ratio={MULTIMODAL_CONFIG['fine_tune_ratio']})")
    del _tmp_backbone

    # ─── Metrics tracking ───
    metrics_summary = {
        'acc': [], 'macro_prec': [], 'macro_rec': [], 'macro_f1': [], 'macro_auc': []
    }

    # ─── Resume from checkpoint ───
    ckpt = load_checkpoint(checkpoint_path)
    if ckpt is not None:
        if ckpt.get('model_name') != model_name:
            logger.warning(f"Checkpoint model_name mismatch: "
                           f"'{ckpt.get('model_name')}' vs '{model_name}' — ignoring.")
        elif ckpt.get('total_folds') != MULTIMODAL_CONFIG["cv_folds"]:
            logger.warning(f"Checkpoint cv_folds mismatch — ignoring.")
        else:
            completed_folds = ckpt.get('completed_folds', [])
            saved_metrics = ckpt.get('metrics_summary', {})
            for key in metrics_summary:
                if key in saved_metrics:
                    metrics_summary[key] = [float(v) for v in saved_metrics[key]]
            logger.info(f"Resuming: {len(completed_folds)}/{MULTIMODAL_CONFIG['cv_folds']} folds done.")

    # ─── 5-Fold Stratified CV ───
    skf = StratifiedKFold(n_splits=MULTIMODAL_CONFIG["cv_folds"], shuffle=True, random_state=42)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df['label_encoded']), 1):
        if fold in completed_folds:
            logger.info(f"[{model_name}] Fold {fold}/{MULTIMODAL_CONFIG['cv_folds']} — already done, skipping.")
            continue

        logger.info("=" * 60)
        logger.info(f"[{model_name}] Starting Fold {fold}/{MULTIMODAL_CONFIG['cv_folds']}")
        logger.info("=" * 60)

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # CSV logger for per-epoch metrics
        csv_logger_path = os.path.join(fold_dir, f"training_log_fold{fold}.csv")

        train_val_df = df.iloc[train_val_idx]
        test_fold_df = df.iloc[test_idx]

        # Train/Val split: 15% val (identical to MNV2_text.py)
        train_fold_df, val_df = train_test_split(
            train_val_df, test_size=0.15,
            stratify=train_val_df['label_encoded'], random_state=42
        )
        logger.info(f"Fold {fold} split - Train: {len(train_fold_df)}, "
                    f"Val: {len(val_df)}, Test: {len(test_fold_df)}")

        # Tokenizer per fold (identical to MNV2_text.py)
        tok = Tokenizer(num_words=MULTIMODAL_CONFIG["max_words"],
                        oov_token=MULTIMODAL_CONFIG["oov_token"])
        tok.fit_on_texts(train_fold_df['caption_seg'])

        # Class weights (balanced) — identical to MNV2_text.py
        y_train_labels = train_fold_df['label_encoded'].values
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        class_weight_dict = dict(enumerate(class_weights))

        # Data generators
        train_gen = RobustMultimodalGenerator(
            train_fold_df, tok, le, preprocess_fn, image_size, augment=True)
        val_gen = RobustMultimodalGenerator(
            val_df, tok, le, preprocess_fn, image_size, augment=False)
        test_gen = RobustMultimodalGenerator(
            test_fold_df, tok, le, preprocess_fn, image_size, augment=False)

        # ─── Build model (backbone_fn creates backbone with input_tensor) ───
        def _backbone_with_input(input_tensor=None):
            return backbone_builder(input_tensor=input_tensor)
        model = build_multimodal_model(_backbone_with_input, num_classes, fine_tune_at, image_size)

        # ─── Single-phase joint training (identical to MNV2_text.py) ───
        logger.info(f"[{model_name} Fold {fold}] Training jointly (single-phase)...")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=MULTIMODAL_CONFIG["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        hist = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=MULTIMODAL_CONFIG["epochs"],
            class_weight=class_weight_dict,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss', patience=8,
                    restore_best_weights=True),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5,
                    patience=3, min_lr=1e-6, verbose=1),
                callbacks.CSVLogger(csv_logger_path, append=False),
            ],
            verbose=1
        )

        # ─── Visualization ───
        plot_training_history(
            hist,
            os.path.join(fold_dir, f"training_curves_fold{fold}.png"),
            fold
        )
        model.save(os.path.join(fold_dir, f"best_model_fold{fold}.keras"))

        # ─── Evaluation ───
        logger.info(f"[{model_name} Fold {fold}] Evaluating on hold-out test set...")

        y_true = test_fold_df['label_encoded'].values
        y_pred_probs = model.predict(test_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=class_names_str, digits=4)
        with open(os.path.join(fold_dir, f"classification_report_fold{fold}.txt"),
                  'w', encoding='utf-8') as f:
            f.write(report)

        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred, class_names_str,
            os.path.join(fold_dir, f"confusion_matrix_fold{fold}.png"),
            fold
        )

        # Metrics
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)
        macro_auc = roc_auc_score(
            y_true, y_pred_probs, multi_class='ovr', average='macro')

        metrics_summary['acc'].append(test_acc)
        metrics_summary['macro_prec'].append(precision)
        metrics_summary['macro_rec'].append(recall)
        metrics_summary['macro_f1'].append(f1)
        metrics_summary['macro_auc'].append(macro_auc)

        logger.info(
            f"[{model_name} Fold {fold}] Results: Acc={test_acc:.4f}, "
            f"Macro P={precision:.4f}, Macro R={recall:.4f}, "
            f"Macro F1={f1:.4f}, Macro AUC={macro_auc:.4f}"
        )

        # ─── Save checkpoint ───
        completed_folds.append(fold)
        checkpoint_state = {
            'model_name': model_name,
            'total_folds': MULTIMODAL_CONFIG["cv_folds"],
            'completed_folds': completed_folds,
            'metrics_summary': metrics_summary,
            'output_dir': output_dir,
        }
        save_checkpoint(checkpoint_path, checkpoint_state)

        # Cleanup
        del model, train_gen, val_gen, test_gen
        gc.collect()
        tf.keras.backend.clear_session()

    # ─── Final 5-Fold Summary ───
    logger.info("\n" + "=" * 60)
    logger.info(f"{model_name} Multimodal 5-Fold CV Summary")
    logger.info("=" * 60)

    final_summary = f"{model_name} Multimodal 5-Fold CV Performance Report\n"
    final_summary += "=" * 60 + "\n"
    final_summary += "Training strategy: IDENTICAL to MobileNetV2 multimodal (MNV2_text.py)\n"
    final_summary += "5-Fold Stratified CV | Single-phase joint training\n"
    final_summary += "Gated multimodal fusion | Class-balanced weights\n"
    final_summary += "-" * 60 + "\n"

    metrics_list = [
        ("Accuracy", 'acc'),
        ("Macro Precision", 'macro_prec'),
        ("Macro Recall", 'macro_rec'),
        ("Macro F1", 'macro_f1'),
        ("Macro AUC", 'macro_auc'),
    ]

    for name, key in metrics_list:
        if metrics_summary[key]:
            mean_val = np.mean(metrics_summary[key]) * 100
            std_val = np.std(metrics_summary[key]) * 100
            result_str = f"{name}: {mean_val:.2f}% +/- {std_val:.2f}%"
        else:
            result_str = f"{name}: No data (all folds failed or skipped)"
        logger.info(result_str)
        final_summary += result_str + "\n"

    with open(os.path.join(output_dir, f"Final_{model_name}_Multimodal_Summary.txt"),
              'w', encoding='utf-8') as f:
        f.write(final_summary)

    logger.info(f"[{model_name}] All multimodal experiments completed. "
                f"Results saved to {output_dir}.")

    # Delete checkpoint on success
    delete_checkpoint(checkpoint_path)

    return metrics_summary
