#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared training configuration, data generator, and utilities.
Strictly follows the strategy from baseline_model_strict.py (MobileNetV2).
All comparison models (VGG16, ResNet50, InceptionV3, MobileNetV3) use this.
"""
import os
import gc
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import StratifiedKFold, train_test_split
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Path resolution ───
# Project root: parent of the directory containing train_config.py
# Baseline layout: project_root/{labels.csv, text_slices/image_captions.json, data/images/}
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # go up from models/ or data1/

def _resolve_path(env_var, *candidates):
    """Resolve a path: env var first, then try each candidate, return first that exists."""
    env_val = os.environ.get(env_var)
    if env_val and os.path.exists(env_val):
        return env_val
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # Fall back to first candidate even if it doesn't exist (will get clear error later)
    return candidates[0]

# ─── Configuration (training hyperparams strictly matching baseline) ───
CONFIG = {
    # --- Paths (STRICTLY match baseline: labels.csv + captions.json filtering) ---
    "labels_csv": _resolve_path(
        'HERB_LABELS_CSV',
        os.path.join(_PROJECT_ROOT, "labels.csv"),          # peer of models/
        os.path.join(_SCRIPT_DIR, "labels.csv"),             # same dir (flat layout)
    ),
    "captions_json": _resolve_path(
        'HERB_CAPTIONS_JSON',
        os.path.join(_PROJECT_ROOT, "text_slices", "image_captions.json"),  # peer of models/
        os.path.join(_SCRIPT_DIR, "image_captions.json"),                   # same dir (flat)
    ),
    "images_dir": _resolve_path(
        'HERB_IMAGES_DIR',
        os.path.join(_PROJECT_ROOT, "data", "images"),       # peer of models/
        os.path.join(_SCRIPT_DIR, "images"),                  # same dir (flat)
    ),

    # --- Training hyperparameters (IDENTICAL to baseline) ---
    "batch_size": 16,
    "cv_folds": 5,

    # --- Phase 1 ---
    "phase1_lr": 1e-4,
    "phase1_epochs": 12,
    "phase1_patience": 5,

    # --- Phase 2 (fine-tuning) ---
    "phase2_lr": 5e-6,
    "phase2_epochs": 20,
    "phase2_patience": 8,
    "phase2_reduce_patience": 4,
    "phase2_reduce_factor": 0.5,
    "phase2_min_lr": 1e-7,

    # --- Architecture head ---
    "dense_units": 512,
    "dense_activation": "swish",

    # --- Fine-tune proportion (same as baseline: MNV2 ~155 layers, fine_tune_at=120 → 77.4%) ---
    "fine_tune_ratio": 0.77,

    # --- Output ---
    "output_dir": None,  # Set per-model

    # --- Checkpoint (resume interrupted training) ---
    "checkpoint_file": "cv_checkpoint.json",  # Set per-model in run_cv()
}

# GPU memory growth (same as baseline)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


class ImageOnlyGenerator(tf.keras.utils.Sequence):
    """
    Image-only data generator.
    STRICTLY matches the baseline ImageOnlyGenerator from baseline_model_strict.py.
    Differences: preprocess_fn is a parameter; image_size is a parameter (per-model native size).
    """
    def __init__(self, df, le, preprocess_fn, image_size, augment=False):
        self.df = df.reset_index(drop=True)
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
        return int(np.ceil(len(self.df) / CONFIG["batch_size"]))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * CONFIG["batch_size"] : (idx + 1) * CONFIG["batch_size"]]
        imgs, labels = [], []
        for _, row in batch_df.iterrows():
            img = load_img(row['image_path'], target_size=self.image_size)
            img_arr = img_to_array(img)  # 0-255
            if self.augment:
                img_arr = self.img_gen.random_transform(img_arr)
            img_arr = self.preprocess_fn(img_arr)
            imgs.append(img_arr)
            labels.append(row['label_encoded'])
        return np.array(imgs), np.array(labels)


def build_head_model(backbone, num_classes, model_name="model"):
    """
    Build the full classification model given a backbone.
    Architecture strictly matches baseline:
        GAP → BatchNormalization → Dense(512, swish) → Dense(n_classes, softmax)
    """
    img_input = backbone.input
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(CONFIG["dense_units"], activation=CONFIG["dense_activation"])(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=img_input, outputs=output, name=model_name)
    return model


# ─── Visualization (identical to baseline) ───
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


def load_and_prepare_data():
    """
    Load labels.csv, filter by captions.json (strict baseline alignment),
    encode labels. STRICTLY matches baseline_model_strict.py data pipeline.
    """
    #   Filter by captions.json — identical to baseline
    #   This ensures the EXACT SAME image set as the multimodal experiments.
    captions_path = CONFIG.get("captions_json", "")
    if os.path.exists(captions_path):
        with open(captions_path, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        valid_images = set(captions.keys())
        logger.info(f"Filtering by captions.json: {len(valid_images)} valid image keys")
    else:
        logger.warning(f"captions.json not found at '{captions_path}' — "
                       f"falling back to image file existence check")
        valid_images = None

    df = pd.read_csv(CONFIG["labels_csv"])

    # Build image paths
    df['image_path'] = df['image_name'].apply(
        lambda x: os.path.join(CONFIG["images_dir"], x))

    # Filter: either by captions.json (baseline mode) or by file existence (fallback)
    if valid_images is not None:
        df = df[df['image_name'].isin(valid_images)]
    df = df[df['image_path'].apply(os.path.exists)]
    df = df.dropna(subset=['image_path']).reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(
            f"No valid images found! Check:\n"
            f"  labels_csv: {CONFIG['labels_csv']}\n"
            f"  captions_json: {CONFIG.get('captions_json', 'N/A')}\n"
            f"  images_dir: {CONFIG['images_dir']}")

    # Label encoding (identical to baseline)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Task_Chinese_medicinal_herb'])

    num_classes = len(le.classes_)
    class_names_str = [str(c) for c in le.classes_]

    logger.info(f"Clean samples: {len(df)} | classes: {num_classes}")
    return df, le, num_classes, class_names_str


# ─── Checkpoint utilities (for resuming interrupted training) ───
def save_checkpoint(checkpoint_path, state):
    """
    Save training checkpoint to a JSON file.
    Records completed folds, accumulated metrics, and model identity.
    """
    try:
        # Convert numpy values to Python native types for JSON serialization
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
    """
    Load training checkpoint from a JSON file.
    Returns the state dict if valid, None otherwise.
    Validates that the checkpoint belongs to the expected model and CONFIG matches.
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoint found at {checkpoint_path} — starting fresh.")
        return None

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        # Basic validation
        required_keys = ['completed_folds', 'metrics_summary', 'model_name', 'total_folds']
        for key in required_keys:
            if key not in state:
                logger.warning(f"Checkpoint missing key '{key}' — ignoring checkpoint.")
                return None
        logger.info(f"Checkpoint loaded: {len(state['completed_folds'])}/{state['total_folds']} folds completed "
                    f"(folds {state['completed_folds']})")
        return state
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Checkpoint file corrupted ({e}) — starting fresh.")
        return None


def delete_checkpoint(checkpoint_path):
    """Remove the checkpoint file after successful completion."""
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info(f"Checkpoint {checkpoint_path} deleted (training complete).")
        except OSError as e:
            logger.warning(f"Failed to delete checkpoint: {e}")


def run_cv(model_name, backbone_builder, preprocess_fn, image_size, output_dir):
    """
    Run 5-fold cross-validation for a given backbone model.

    Training hyperparameters strictly match baseline_model_strict.py:
    - 5-fold StratifiedKFold (shuffle=True, random_state=42)
    - Train/Val split: 15% validation (stratified, random_state=42)
    - Phase 1: backbone frozen, Adam(1e-4), epochs=12, EarlyStopping(patience=5)
    - Phase 2: backbone partially unfrozen (CONFIG["fine_tune_ratio"] = 77%),
               lr=5e-6, epochs=20, EarlyStopping(patience=8), ReduceLROnPlateau
    - BN layers kept frozen during fine-tuning

    Model-specific settings (not forced to match baseline):
    - image_size: each model's native input size
    - preprocess_fn: each model's own preprocess_input
    - fine_tune_at: auto-computed as int(total_layers * fine_tune_ratio),
                    matching baseline MNV2 proportion (120/155 ≈ 77%)

    Args:
        model_name: str, display name for logging
        backbone_builder: callable that returns a backbone model
        preprocess_fn: callable for model-specific image preprocessing
        image_size: tuple (H, W), model's native input size
        output_dir: str, directory for outputs
    """
    CONFIG["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ─── Checkpoint setup ───
    checkpoint_path = os.path.join(output_dir, CONFIG["checkpoint_file"])
    completed_folds = []

    df, le, num_classes, class_names_str = load_and_prepare_data()

    # Build a temporary backbone to compute fine_tune_at proportion
    _tmp_backbone = backbone_builder()
    total_layers = len(_tmp_backbone.layers)
    fine_tune_at = int(total_layers * CONFIG["fine_tune_ratio"])
    logger.info(f"[{model_name}] Total backbone layers: {total_layers}, "
                f"fine_tune_at={fine_tune_at} (ratio={CONFIG['fine_tune_ratio']})")
    del _tmp_backbone

    metrics_summary = {
        'acc': [], 'macro_prec': [], 'macro_rec': [], 'macro_f1': [], 'macro_auc': []
    }

    # ─── Resume from checkpoint if exists ───
    ckpt = load_checkpoint(checkpoint_path)
    if ckpt is not None:
        # Validate model identity
        if ckpt.get('model_name') != model_name:
            logger.warning(f"Checkpoint model_name mismatch: "
                           f"'{ckpt.get('model_name')}' vs '{model_name}' — ignoring checkpoint.")
        elif ckpt.get('total_folds') != CONFIG["cv_folds"]:
            logger.warning(f"Checkpoint cv_folds mismatch: "
                           f"{ckpt.get('total_folds')} vs {CONFIG['cv_folds']} — ignoring checkpoint.")
        else:
            completed_folds = ckpt.get('completed_folds', [])
            # Restore accumulated metrics
            saved_metrics = ckpt.get('metrics_summary', {})
            for key in metrics_summary:
                if key in saved_metrics:
                    metrics_summary[key] = [float(v) for v in saved_metrics[key]]
            logger.info(f"Resuming from checkpoint: {len(completed_folds)}/{CONFIG['cv_folds']} folds already done.")
            logger.info(f"Already completed folds: {completed_folds}")

    # Same StratifiedKFold split as baseline
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=42)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df['label_encoded']), 1):
        # ─── Skip already-completed folds ───
        if fold in completed_folds:
            logger.info(f"[{model_name}] Fold {fold}/{CONFIG['cv_folds']} — already completed, skipping.")
            continue

        logger.info("=" * 60)
        logger.info(f"[{model_name}] Fold {fold}/{CONFIG['cv_folds']}")
        logger.info("=" * 60)

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # CSV logger to record per-epoch metrics (for post-mortem analysis if interrupted mid-fold)
        csv_logger_path = os.path.join(fold_dir, f"training_log_fold{fold}.csv")

        train_val_df = df.iloc[train_val_idx]
        test_fold_df = df.iloc[test_idx]

        # Train/Val split: 15% val (identical to baseline)
        train_fold_df, val_df = train_test_split(
            train_val_df, test_size=0.15,
            stratify=train_val_df['label_encoded'], random_state=42
        )
        logger.info(f"Fold {fold} split - Train: {len(train_fold_df)}, Val: {len(val_df)}, Test: {len(test_fold_df)}")

        # Data generators (pass image_size)
        train_gen = ImageOnlyGenerator(train_fold_df, le, preprocess_fn, image_size, augment=True)
        val_gen   = ImageOnlyGenerator(val_df,        le, preprocess_fn, image_size, augment=False)
        test_gen  = ImageOnlyGenerator(test_fold_df,  le, preprocess_fn, image_size, augment=False)

        # Build model
        backbone = backbone_builder()
        model = build_head_model(backbone, num_classes, model_name=model_name)

        # ═══════ Phase 1: Train top layers only ═══════
        logger.info(f"[{model_name} Fold {fold}] Phase 1: Training top layers (backbone frozen)...")
        backbone.trainable = False
        model.compile(
            optimizer=optimizers.Adam(CONFIG["phase1_lr"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        # Phase 1 callbacks: EarlyStopping + ModelCheckpoint + CSVLogger
        phase1_checkpoint_path = os.path.join(fold_dir, f"phase1_best_fold{fold}.weights.h5")
        hist1 = model.fit(
            train_gen, validation_data=val_gen,
            epochs=CONFIG["phase1_epochs"],
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss', patience=CONFIG["phase1_patience"],
                    restore_best_weights=True),
                callbacks.ModelCheckpoint(
                    filepath=phase1_checkpoint_path,
                    monitor='val_loss', save_best_only=True, save_weights_only=True,
                    verbose=1),
                callbacks.CSVLogger(csv_logger_path, append=True),
            ],
            verbose=1
        )

        # ═══════ Phase 2: Fine-tune backbone ═══════
        logger.info(f"[{model_name} Fold {fold}] Phase 2: Fine-tuning from layer {fine_tune_at}/{total_layers}...")
        backbone.trainable = True
        # Freeze early layers
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False
        # Freeze all BatchNormalization layers (same as baseline)
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        model.optimizer.learning_rate.assign(CONFIG["phase2_lr"])
        # Phase 2 callbacks: EarlyStopping + ReduceLROnPlateau + ModelCheckpoint + CSVLogger
        phase2_checkpoint_path = os.path.join(fold_dir, f"phase2_best_fold{fold}.weights.h5")
        hist2 = model.fit(
            train_gen, validation_data=val_gen,
            epochs=CONFIG["phase2_epochs"],
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss', patience=CONFIG["phase2_patience"],
                    restore_best_weights=True),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=CONFIG["phase2_reduce_factor"],
                    patience=CONFIG["phase2_reduce_patience"],
                    min_lr=CONFIG["phase2_min_lr"]),
                callbacks.ModelCheckpoint(
                    filepath=phase2_checkpoint_path,
                    monitor='val_loss', save_best_only=True, save_weights_only=True,
                    verbose=1),
                callbacks.CSVLogger(csv_logger_path, append=True),
            ],
            verbose=1
        )

        # Visualization
        plot_training_history(hist1, hist2,
                              os.path.join(fold_dir, f"training_curves_fold{fold}.png"), fold)
        model.save(os.path.join(fold_dir, f"best_model_fold{fold}"))

        # ═══════ Evaluation ═══════
        logger.info(f"[{model_name} Fold {fold}] Evaluating on test set...")

        y_true = test_fold_df['label_encoded'].values
        y_pred_probs = model.predict(test_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Per-fold classification report
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

        logger.info(f"[{model_name} Fold {fold}] Results: Acc={test_acc:.4f}, F1={f1:.4f}, AUC={macro_auc:.4f}")

        # ─── Save checkpoint after fold completion ───
        completed_folds.append(fold)
        checkpoint_state = {
            'model_name': model_name,
            'total_folds': CONFIG["cv_folds"],
            'completed_folds': completed_folds,
            'metrics_summary': metrics_summary,
            'output_dir': output_dir,
        }
        save_checkpoint(checkpoint_path, checkpoint_state)

        # Cleanup
        del model, backbone, train_gen, val_gen, test_gen
        gc.collect()
        tf.keras.backend.clear_session()

    # ═══════ Final 5-fold summary ═══════
    logger.info("\n" + "=" * 60)
    logger.info(f"{model_name} 5-Fold CV Summary")
    logger.info("=" * 60)
    final_summary = f"{model_name} Image-Only 5-Fold CV Performance Report\n"
    final_summary += "=" * 50 + "\n"
    final_summary += f"Training strategy: Identical to MobileNetV2 baseline\n"
    final_summary += f"5-Fold Stratified CV, Train/Val/Test split, 2-phase training\n"
    final_summary += "-" * 50 + "\n"
    for name, key in [("Accuracy", 'acc'), ("Macro Precision", 'macro_prec'),
                      ("Macro Recall", 'macro_rec'), ("Macro F1", 'macro_f1'),
                      ("Macro AUC", 'macro_auc')]:
        mean_val = np.mean(metrics_summary[key]) * 100
        std_val = np.std(metrics_summary[key]) * 100
        result_str = f"{name}: {mean_val:.2f}% +/- {std_val:.2f}%"
        logger.info(result_str)
        final_summary += result_str + "\n"
    with open(os.path.join(output_dir, f"Final_{model_name}_Summary.txt"), "w") as f:
        f.write(final_summary)
    logger.info(f"{model_name} experiments completed. Results saved to {output_dir}.")

    # ─── Delete checkpoint on successful completion ───
    delete_checkpoint(checkpoint_path)

    return metrics_summary
