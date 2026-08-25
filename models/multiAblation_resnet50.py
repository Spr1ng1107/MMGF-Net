#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  消融实验 v3-ResNet50 (FIXED) — 以 ResNet50 为极限模型(Full_Model) 的系统消融
  基线: ResNet50 多模态极限模型 (Single-Phase Joint Training + Gated Fusion, 98.03%)
================================================================================
  与 multiAblation_resnet50.py 相比的修复项:
   [F1] 数据加载增加 os.path.exists 过滤 —— 与主模型 train_multimodal_config.py
        (df[df['image_path'].apply(os.path.exists)]) 严格一致, 保证 Full_Model
        与主模型跑在**完全相同的样本集**上 (1119 张)。
   [F2] 随机种子修复为"每折重置": 每折开始时 `_reset_seed(seed + fold)`。
        好处: 所有消融配置在相同折上拥有**完全一致的权重初始化与数据增强序列**,
        真正实现 one-factor-at-a-time —— 除被消融组件外所有随机性对齐。
        (原版只在模块导入时 set_seed 一次, 各配置沿同一 RNG 流先后取数,
         配置间的差异混入了随机噪声, ΔAcc 口径不干净。)
        GPU 说明: 因 cuDNN 非确定性, 跨进程无法逐比特复现, 但同机同环境
        可稳定复现; 如需更强确定性, 设环境变量 HERB_TF_DETERMINISTIC=1。
   [F3] 修正注释: ResNet50 include_top=False 实际 175 层 (TF2.20), fine_tune_at
        = int(175*0.77) = 134 (原注释误写 176→135; 代码本就是动态计算, 无影响)。
   [F4] fold_results.csv 写入 config 列; 续跑时只复用**属于当前配置**的折结果,
        避免旧配置的残留 CSV 被静默复用。
   [F5] 顶层断点 completed_experiments.txt 增加 fold_results.csv 完整性复核:
        仅当该配置 5 折结果齐全时才跳过, 防止"标记完成但 CSV 被删"漏跑。
   [F6] 新增命令行参数 --only Full_Model,Image_Only 可先只跑参考行核验,
        再放开跑全部 16 项配置 (Full_Model + 15 项消融)。
   [F7] Full_Model 完成后自动与主模型记录值 98.03% 对比, |Δ|>0.5pp 时告警,
        并打印 Full_Model / Image_Only 逐折准确率, 供复算配对 t 检验。
   [F8] 两阶段协议参数显式化 (CONFIG.phase1_epochs/phase2_epochs/phase2_lr),
        与论文 4.2 节"先冻结骨干 12 轮, 再 5e-6 微调 20 轮"严格对应。
   [F9] 新增每折断点记忆:
        (a) 每折保存最优模型 fold_{N}/best_model_fold{N}.keras (与主模型命名一致,
            可离线重评估 / 重绘 图3 图4);
        (b) 训练期间经 ModelCheckpoint 实时落盘最优权重 (仅权重 .weights.h5,
            进程被杀也不丢; 使用 save_weights_only=True 以规避 Keras3 在原生
            .keras 全模型保存时报 "options not supported" 的兼容性问题);
        (c) fold_{N}/.inprogress 标记: 续跑时检测上次中断的折并告警, 从零干净重跑;
        (d) 已完成折仍以 fold_results.csv 为准自动跳过。
   [F10] 移除 Scratch(随机初始化)预训练消融 —— 论文不再验证预训练权重,
        配置总数由 17 减至 16 (Full_Model + 15 项), 表6 中 Scratch 行同步删除。
   [F11] 启动即打印断点续跑状态 (每个配置已完折/待续折), 服务器停止重启后
        可据此确认从第几个组件(配置)第几折继续; 中断折由 .inprogress 标记
        自动识别并从零干净重跑。
        ★ 断点续跑要点 (重启不会从头):
           - 已完成折: 依据 fold_results.csv 自动跳过 (逐折记忆, 折级断点);
           - 中断折:   依据 fold_{N}/.inprogress 标记识别, 重启该折;
           - 已完配置: 依据 completed_experiments.txt + 5折齐全 跳过整个配置;
           - 同一命令重复执行即可续跑: python multiAblation_resnet50.py

  ResNet50 多模态架构 (与主模型完全一致, 逐参数核对见论文附录 A.4):
   - Image: ResNet50(imagenet, 224×224) -> GAP -> BN -> Dense(512, swish)
   - Text:  Embedding(10000→256) -> SpatialDropout1D(0.4)
            -> BiLSTM(256, return_sequences=True)
            -> GlobalAvgPool1D + GlobalMaxPool1D -> Concat
            -> LayerNorm -> Dense(512, swish)
   - Fusion: Concat(1024) -> Dense(1024, sigmoid Gate) -> Multiply
             -> Dense(512, swish) -> BN -> Dropout(0.5) -> Dense(num_classes, softmax)
   - Training: Joint (single-phase), Adam(1e-4), 30 epochs
               EarlyStopping(p=8), ReduceLROnPlateau(p=3, f=0.5, min_lr=1e-7)
================================================================================
"""
import os, gc, json, sys, warnings, random
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import jieba
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
jieba.setLogLevel(logging.ERROR)

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
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)

# ============================================================
# GPU 配置 — 必须在任何模型/算子创建之前调用
# ============================================================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# [F7] 可选强确定性 (会显著变慢, 部分算子可能不支持, 仅在显式开启时启用)
if os.environ.get('HERB_TF_DETERMINISTIC', '0') == '1':
    try:
        tf.config.experimental.enable_op_determinism()
        logger.info("TF op determinism ENABLED (slower)")
    except Exception as e:
        logger.warning(f"enable_op_determinism failed: {e}")

# ============================================================
# 全局配置
# ============================================================
def _resolve_path(env_var, *candidates):
    """优先使用环境变量, 否则依次尝试候选路径, 返回第一个存在的; 全不存在返回首个。"""
    env_val = os.environ.get(env_var)
    if env_val and os.path.exists(env_val):
        return env_val
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ""

# 默认服务端路径 (与主模型所在项目一致); 本地/其他服务器用环境变量覆盖:
#   HERB_IMAGES_DIR, HERB_CAPTIONS_JSON, HERB_LABELS_CSV
CONFIG = {
    # -------- 数据路径 --------
    "images_dir":   _resolve_path(
        'HERB_IMAGES_DIR',
        '/mnt/data1/spring/multimodal_project/data/images',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'images'),
    ),
    "captions_path": _resolve_path(
        'HERB_CAPTIONS_JSON',
        '/mnt/data1/spring/multimodal_project/text_slices/image_captions.json',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'text_slices', 'image_captions.json'),
    ),
    "labels_path":  _resolve_path(
        'HERB_LABELS_CSV',
        '/mnt/data1/spring/multimodal_project/labels.csv',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'labels.csv'),
    ),

    # -------- 图像参数 --------
    "image_size": (224, 224),
    "batch_size": 16,

    # -------- 文本参数 --------
    "max_text_len": 320,
    "max_words": 10000,
    "embedding_dim": 256,
    "oov_token": "<OOV>",

    # -------- 训练参数 (与主模型 train_multimodal_config.py 完全一致) --------
    "cv_folds": 5,
    "fine_tune_ratio": 0.77,       # ResNet50(include_top=False) 实测 175 层 -> int(175*0.77)=134
    "lstm_units": 256,
    "text_dropout_rate": 0.4,
    "epochs": 30,
    "learning_rate": 1e-4,
    "seed": 42,

    # [F8] 两阶段消融协议 (论文 4.2 节: 冻结骨干训练 12 轮 -> 5e-6 全局微调 20 轮)
    "phase1_epochs": 12,
    "phase2_epochs": 20,
    "phase2_lr": 5e-6,

    # -------- 输出 --------
    "output_dir": "./ablation_resnet50_final",
}

# [F2] 种子重置: 每折开始时调用, 使所有配置同折共享一致的初始化与增强序列
def _reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 动态计算 ResNet50 解冻层数 (实测 175 层 -> int(175*0.77)=134)
def _compute_fine_tune_at():
    """用 weights=None 实例化 ResNet50 统计层数, 无需下载权重。"""
    tmp = tf.keras.applications.ResNet50(
        include_top=False, weights=None, input_shape=(*CONFIG["image_size"], 3))
    n_layers = len(tmp.layers)
    ft = int(n_layers * CONFIG["fine_tune_ratio"])
    logger.info(f"ResNet50 total layers: {n_layers}, fine_tune_at={ft} "
                f"(ratio={CONFIG['fine_tune_ratio']:.2f})")
    del tmp
    return ft

FINE_TUNE_AT = _compute_fine_tune_at()

_reset_seed(CONFIG["seed"])


def segment_text(text):
    return ' '.join(jieba.cut(str(text)))


# ============================================================
# 模型构建 — 返回 (model, v_backbone)
# ============================================================
def build_model(num_classes, ablation_config):
    """
    构建模型并返回 (model, v_backbone) 引用。

    视觉分支 (与主模型 build_multimodal_model 完全一致):
      ResNet50(imagenet), 冻结前{FINE_TUNE_AT}层+BN -> GAP -> BN -> Dense(512, swish)
    文本分支:
      Embedding(10000→256) -> SpatialDropout1D(0.4)
      -> BiLSTM(256, return_sequences=True) -> AvgPool+MaxPool Concat
      -> LayerNorm -> Dense(512, swish)
    融合:
      Concat -> Dense(1024, sigmoid Gate) -> Multiply
      -> Dense(512, swish) -> BN -> Dropout(0.5) -> Output
    """
    use_text = ablation_config.get('use_text', True)
    use_pretrained = ablation_config.get('use_pretrained', True)
    skip_vision = ablation_config.get('skip_vision', False)

    # ---- 图像分支 ----
    if skip_vision:
        img_input = None
        v_backbone = None
        features = []
    else:
        img_input = layers.Input(shape=(*CONFIG["image_size"], 3), name="img_input")
        if use_pretrained:
            v_backbone = tf.keras.applications.ResNet50(
                input_tensor=img_input, include_top=False, weights='imagenet'
            )
        else:
            v_backbone = tf.keras.applications.ResNet50(
                input_tensor=img_input, include_top=False, weights=None
            )

        joint_training = not ablation_config.get('two_phase', False)
        if not use_pretrained:
            # Scratch: 全部层从头训练, 不冻结
            v_backbone.trainable = True
        elif joint_training:
            # 主模型方式: 冻结前 FINE_TUNE_AT 层 + 全部 BN
            v_backbone.trainable = True
            for layer in v_backbone.layers[:FINE_TUNE_AT]:
                layer.trainable = False
            for layer in v_backbone.layers:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
        else:
            # Two-phase: 初始完全冻结, run_cv 中第二阶段再解冻
            v_backbone.trainable = False

        v_feat = layers.GlobalAveragePooling2D()(v_backbone.output)
        v_feat = layers.BatchNormalization()(v_feat)
        v_feat = layers.Dense(512, activation='swish', name='v_dense_proj')(v_feat)
        features = [v_feat]

    # ---- 文本分支 (可消融) ----
    if use_text:
        text_input = layers.Input(shape=(CONFIG["max_text_len"],), name="text_input")
        t_feat = layers.Embedding(CONFIG["max_words"], CONFIG["embedding_dim"])(text_input)

        drop_rate = ablation_config.get('text_dropout_rate', CONFIG["text_dropout_rate"])
        if drop_rate > 0:
            t_feat = layers.SpatialDropout1D(drop_rate)(t_feat)

        lstm_units = ablation_config.get('lstm_units', CONFIG["lstm_units"])
        return_seq = ablation_config.get('lstm_return_sequences', True)

        if return_seq:
            t_seq = layers.Bidirectional(
                layers.LSTM(lstm_units, return_sequences=True)
            )(t_feat)
            pool_mode = ablation_config.get('pool_mode', 'avgmax')
            if pool_mode == 'avgmax':
                t_pool = layers.concatenate([
                    layers.GlobalAveragePooling1D()(t_seq),
                    layers.GlobalMaxPooling1D()(t_seq)
                ])
            elif pool_mode == 'avg':
                t_pool = layers.GlobalAveragePooling1D()(t_seq)
            elif pool_mode == 'max':
                t_pool = layers.GlobalMaxPooling1D()(t_seq)
            else:
                raise ValueError(f"Unknown pool_mode: {pool_mode}")
            t_feat_proj = layers.LayerNormalization()(t_pool)
        else:
            t_seq = layers.Bidirectional(
                layers.LSTM(lstm_units, return_sequences=False)
            )(t_feat)
            t_feat_proj = layers.LayerNormalization()(t_seq)

        t_feat_proj = layers.Dense(512, activation='swish', name='t_dense_proj')(t_feat_proj)
        features.append(t_feat_proj)

    # ---- 融合与分类 ----
    if len(features) == 2:
        merged = layers.concatenate(features, name='concat_features')

        use_gate = ablation_config.get('use_gate', True)
        if use_gate:
            gate = layers.Dense(1024, activation='sigmoid', name='fusion_gate')(merged)
            merged = layers.multiply([merged, gate], name='gated_fusion')

        fusion_head = ablation_config.get('fusion_head', 'papermix')
        if fusion_head == 'papermix':
            x = layers.Dense(512, activation='swish', name='classifier_dense')(merged)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            output = layers.Dense(num_classes, activation='softmax', name='classifier_out')(x)
        elif fusion_head == 'deep':
            x = layers.Dense(1024, activation='swish', name='fusion_dense_1')(merged)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(512, activation='swish', name='fusion_dense_2')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            output = layers.Dense(num_classes, activation='softmax')(x)
        else:
            raise ValueError(f"Unknown fusion_head: {fusion_head}")

    elif len(features) == 1:
        # 单模态: 直接分类
        x = layers.Dense(512, activation='swish')(features[0])
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(num_classes, activation='softmax')(x)

    elif len(features) == 0:
        raise ValueError("No features to build model from")

    # 构建模型
    if use_text and not skip_vision:
        model = models.Model(inputs=[img_input, text_input], outputs=output)
    elif use_text and skip_vision:
        model = models.Model(inputs=text_input, outputs=output)
    else:
        model = models.Model(inputs=img_input, outputs=output)

    return model, v_backbone


# ============================================================
# 数据生成器 (ResNet50 预处理, 其余完全匹配主模型)
# ============================================================
class RobustGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, tokenizer, le, augment=False, use_text=True, text_column='caption_seg', skip_vision=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = le
        self.augment = augment
        self.use_text = use_text
        self.text_column = text_column
        self.skip_vision = skip_vision
        self.img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20, horizontal_flip=True, brightness_range=[0.9, 1.1]
        ) if augment else None

    def __len__(self):
        return int(np.ceil(len(self.df) / CONFIG["batch_size"]))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * CONFIG["batch_size"] : (idx + 1) * CONFIG["batch_size"]]
        imgs, texts, labels = [], [], []
        for _, row in batch_df.iterrows():
            if not self.skip_vision:
                img = load_img(row['image_path'], target_size=CONFIG["image_size"])
                img_arr = img_to_array(img)
                if self.augment:
                    img_arr = self.img_gen.random_transform(img_arr)
                # ResNet50 预处理: RGB -> BGR, 各通道减去 ImageNet 均值
                img_arr = tf.keras.applications.resnet50.preprocess_input(img_arr)
                imgs.append(img_arr)
            if self.use_text:
                texts.append(row[self.text_column])
            labels.append(row['Task_Chinese_medicinal_herb'])
        y = self.le.transform(labels)
        if self.use_text:
            X_text = pad_sequences(
                self.tokenizer.texts_to_sequences(texts), maxlen=CONFIG["max_text_len"]
            )
            if self.skip_vision:
                return X_text, y
            else:
                return [np.array(imgs), X_text], y
        else:
            return np.array(imgs), y


# ============================================================
# 训练曲线可视化
# ============================================================
def plot_training_history(hist, save_path, config_name, fold, two_phase=False):
    if two_phase:
        hist1, hist2 = hist
        loss = hist1.history['loss'] + hist2.history['loss']
        val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
        acc = hist1.history['accuracy'] + hist2.history['accuracy']
        val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
        phase1_end = len(hist1.history['loss'])
    else:
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        acc = hist.history['accuracy']
        val_acc = hist.history['val_accuracy']
        phase1_end = None

    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    if phase1_end:
        plt.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1 End')
    plt.title(f'{config_name} - Fold {fold} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'r-', label='Val Acc')
    if phase1_end:
        plt.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1 End')
    plt.title(f'{config_name} - Fold {fold} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# CV Runner — 训练策略精确匹配 ResNet50 主模型
# ============================================================
def run_cv_for_config(df, le, class_names, ablation_config, output_subdir):
    os.makedirs(output_subdir, exist_ok=True)
    num_classes = len(class_names)
    cfg_name = ablation_config['name']

    # ── 断点续跑: fold_results.csv 为完成折的权威记录; .inprogress 标记用于
    #    检测上次运行被中断的折 ──
    fold_csv = os.path.join(output_subdir, "fold_results.csv")

    metrics_summary = {'acc': [], 'macro_prec': [], 'macro_rec': [], 'macro_f1': [], 'macro_auc': []}
    completed_folds = set()

    if os.path.exists(fold_csv):
        exist_df = pd.read_csv(fold_csv)
        # [F4] 只读取属于当前配置的行
        if 'config' in exist_df.columns:
            exist_df = exist_df[exist_df['config'] == cfg_name]
        stale = len(exist_df) > 0
        if stale:
            for _, row in exist_df.iterrows():
                f_num = int(row['fold'])
                completed_folds.add(f_num)
                metrics_summary['acc'].append(row['accuracy'])
                metrics_summary['macro_prec'].append(row['macro_precision'])
                metrics_summary['macro_rec'].append(row['macro_recall'])
                metrics_summary['macro_f1'].append(row['macro_f1'])
                metrics_summary['macro_auc'].append(row['macro_auc'])
        logger.info(f"  [{cfg_name}] {len(completed_folds)}/{CONFIG['cv_folds']} folds in {fold_csv}")

    # [F9] 检测上次运行中断的折: 有 .inprogress 标记但未写入 fold_results.csv
    for d in sorted(os.listdir(output_subdir)):
        if d.startswith('fold_') and os.path.isdir(os.path.join(output_subdir, d)):
            marker = os.path.join(output_subdir, d, '.inprogress')
            if os.path.exists(marker):
                try:
                    f_num = int(d.split('_')[1])
                except (IndexError, ValueError):
                    os.remove(marker)   # 无效标记, 直接清理
                    continue
                if f_num not in completed_folds:
                    logger.warning(f"  [{cfg_name}] Fold {f_num} 上次运行被中断 "
                                   f"(.inprogress 标记存在且未完成), 现将从零重跑该折.")
                os.remove(marker)   # 清理残留标记 (已完成折的遗留标记也一并清掉)

    # 如果全部折已完成, 直接从 CSV 汇总返回
    if len(completed_folds) >= CONFIG["cv_folds"]:
        results = {}
        for key in metrics_summary:
            arr = np.array(metrics_summary[key])
            results[f'{key}_mean'] = arr.mean()
            results[f'{key}_std'] = arr.std()
        results['name'] = cfg_name
        logger.info(f"  [{cfg_name}] All folds completed, loading from checkpoint.")
        return results

    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df['label_encoded']), 1):
        if fold in completed_folds:
            logger.info(f"  [{cfg_name}] Fold {fold}/{CONFIG['cv_folds']} already completed, skipping.")
            continue

        # [F2] 每折重置种子: seed + fold, 所有配置同折共享一致初始化/增强序列
        _reset_seed(CONFIG["seed"] + fold)

        # [F9] 每折断点记忆: 建本折目录 + 训练中标记 + 最优模型路径 (主模型同命名)
        fold_dir = os.path.join(output_subdir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        inprogress_file = os.path.join(fold_dir, ".inprogress")
        with open(inprogress_file, 'w') as f:
            f.write(cfg_name + '\n')
        # 训练中 ModelCheckpoint 只存权重 (.weights.h5): 避免 Keras3 全模型保存
        # 在原生 .keras 格式下报 "options not supported" 错误
        ckpt_weights_path = os.path.join(fold_dir, f"best_weights_fold{fold}.weights.h5")
        # 折末完整模型 (与主模型 best_model_fold{N}.keras 命名一致, 供重评估/重绘图)
        best_model_path = os.path.join(fold_dir, f"best_model_fold{fold}.keras")

        logger.info(f"  [{cfg_name}] Fold {fold}/{CONFIG['cv_folds']}")

        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.15,
            stratify=train_val_df['label_encoded'], random_state=CONFIG["seed"]
        )

        use_text = ablation_config.get('use_text', True)
        use_segmentation = ablation_config.get('use_segmentation', True)
        text_column = 'caption_seg' if (use_text and use_segmentation) else ('caption_raw' if use_text else None)

        # Tokenizer (num_words=10000, 与主模型一致; 每折基于 train 拟合)
        tok = Tokenizer(num_words=CONFIG["max_words"], oov_token=CONFIG["oov_token"])
        if use_text:
            tok.fit_on_texts(train_df[text_column])
        else:
            tok.fit_on_texts(['<OOV>'])

        if use_text:
            all_seq = tok.texts_to_sequences(train_df[text_column])
            max_idx = max((max(seq) for seq in all_seq if seq), default=0)
            assert max_idx < CONFIG['max_words'], \
                f"Tokenizer index {max_idx} >= Embedding input_dim {CONFIG['max_words']}"

        y_train_labels = train_df['label_encoded'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        class_weight_dict = dict(enumerate(class_weights))

        skip_vision = ablation_config.get('skip_vision', False)

        train_gen = RobustGenerator(train_df, tok, le, augment=True, use_text=use_text,
                                    text_column=text_column, skip_vision=skip_vision)
        val_gen   = RobustGenerator(val_df,   tok, le, augment=False, use_text=use_text,
                                    text_column=text_column, skip_vision=skip_vision)
        test_gen  = RobustGenerator(test_df,  tok, le, augment=False, use_text=use_text,
                                    text_column=text_column, skip_vision=skip_vision)

        # 构建模型, 获得 backbone 引用
        model, v_backbone = build_model(num_classes, ablation_config)

        # --- 训练策略 ---
        two_phase = ablation_config.get('two_phase', False)
        verbose = ablation_config.get('verbose', 0)

        if two_phase:
            # ---- 两阶段训练 (论文 4.2 / Baseline-I 协议) ----
            model.compile(optimizer=optimizers.Adam(CONFIG["learning_rate"]),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            logger.info(f"    [{cfg_name}] Phase 1 (frozen backbone, {CONFIG['phase1_epochs']} epochs)")
            hist1 = model.fit(train_gen, validation_data=val_gen, epochs=CONFIG["phase1_epochs"],
                              class_weight=class_weight_dict,
                              callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                              verbose=verbose)

            # Phase 2: 解冻 backbone 上层 (前 FINE_TUNE_AT 层+BN 仍冻结)
            if v_backbone is not None:
                v_backbone.trainable = True
                for layer in v_backbone.layers[:FINE_TUNE_AT]:
                    layer.trainable = False
                for layer in v_backbone.layers:
                    if isinstance(layer, layers.BatchNormalization):
                        layer.trainable = False

            model.optimizer.learning_rate.assign(CONFIG["phase2_lr"])
            logger.info(f"    [{cfg_name}] Phase 2 (partially unfrozen, lr={CONFIG['phase2_lr']}, {CONFIG['phase2_epochs']} epochs)")
            # [F9] Phase 2 期间实时保存最优权重 (仅权重, 崩溃也不丢)
            cb_list = [
                callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                callbacks.ModelCheckpoint(filepath=ckpt_weights_path, monitor='val_loss',
                                          save_best_only=True, save_weights_only=True, verbose=0),
            ]
            if not ablation_config.get('fixed_lr', False):
                cb_list.append(callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=verbose))
            hist2 = model.fit(train_gen, validation_data=val_gen, epochs=CONFIG["phase2_epochs"],
                              class_weight=class_weight_dict, callbacks=cb_list, verbose=verbose)
            hist = (hist1, hist2)

        else:
            # ---- 单阶段联合训练 (ResNet50 主模型标准) ----
            model.compile(optimizer=optimizers.Adam(CONFIG["learning_rate"]),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            cb_list = [
                callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=verbose
                ),
                # [F9] 训练期间实时保存最优权重 (仅权重, 崩溃也不丢)
                callbacks.ModelCheckpoint(filepath=ckpt_weights_path, monitor='val_loss',
                                          save_best_only=True, save_weights_only=True, verbose=0),
            ]
            hist = model.fit(train_gen, validation_data=val_gen, epochs=CONFIG["epochs"],
                             class_weight=class_weight_dict, callbacks=cb_list, verbose=verbose)

        # 保存训练曲线
        plot_training_history(hist, os.path.join(output_subdir, f"training_curves_fold{fold}.png"),
                              cfg_name, fold, two_phase=two_phase)

        # 评估测试集
        y_true = test_df['label_encoded'].values
        y_pred_probs = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        macro_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)

        metrics_summary['acc'].append(test_acc)
        metrics_summary['macro_prec'].append(precision)
        metrics_summary['macro_rec'].append(recall)
        metrics_summary['macro_f1'].append(f1)
        metrics_summary['macro_auc'].append(macro_auc)

        # 保存每折报告
        fold_dir = os.path.join(output_subdir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        with open(os.path.join(fold_dir, "classification_report.txt"), 'w') as f:
            f.write(report)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{cfg_name} - Fold {fold} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

        logger.info(f"    [{cfg_name}] Fold {fold} Acc={test_acc:.4f}, F1={f1:.4f}, AUC={macro_auc:.4f}")

        # 保存每折详细结果到 CSV ([F4] 附加 config 列, 防止跨配置串用)
        fold_data = {
            'config': cfg_name,
            'fold': fold,
            'accuracy': test_acc,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'macro_auc': macro_auc,
            'test_loss': test_loss,
        }
        fold_df = pd.DataFrame([fold_data])
        if not os.path.exists(fold_csv):
            fold_df.to_csv(fold_csv, index=False)
        else:
            fold_df.to_csv(fold_csv, mode='a', header=False, index=False)

        # [F9] 落盘本折最优模型 (EarlyStopping 已 restore_best_weights, 内存即最优;
        #      ModelCheckpoint 训练中已实时保存, 此处再显式 save 确保存在)
        try:
            model.save(best_model_path)
        except Exception as e:
            logger.warning(f"    [{cfg_name}] 保存 fold {fold} 模型失败: {e}")
        if os.path.exists(inprogress_file):
            os.remove(inprogress_file)

        del model, v_backbone, train_gen, val_gen, test_gen
        gc.collect()
        tf.keras.backend.clear_session()

    results = {}
    for key in metrics_summary:
        arr = np.array(metrics_summary[key])
        results[f'{key}_mean'] = arr.mean()
        results[f'{key}_std'] = arr.std()
    results['name'] = cfg_name
    return results


# ============================================================
# 消融实验配置 — 每次仅改变一个组件 (15项配置 + Full_Model = 16, 论文表6去掉 Scratch 预训练行)
# ============================================================
CATEGORY_OF = {
    'Full_Model':       '参考',
    'Image_Only':       '模态',
    'Image_Only_Joint': '模态',
    'Text_Only':        '模态',
    'Two_Phase':        '训练',
    'FixedLR':          '训练',
    'NoSegmentation':   '文本分支',
    'LSTM128':          '文本分支',
    'LSTM512':          '文本分支',
    'LSTM_NoSeq':       '文本分支',
    'Pool_Avg_Only':    '文本分支',
    'Pool_Max_Only':    '文本分支',
    'No_TextDropout':   '文本分支',
    'Low_TextDropout':  '文本分支',
    'No_Gate':          '融合',
    'Deep_Fusion':      '融合',
}

ablation_configs = [
    # ======== 1. 参考基准 (精确复现 ResNet50 多模态主模型, 期望 ~98.03%) ========
    {
        'name': 'Full_Model',
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 1,
    },

    # ======== 2. 模态消融 (验证多模态必要性) ========
    {
        # Image_Only = 纯图像两阶段基线 (复现 Baseline-I / 论文表6 Image_Only, 96.60%)
        'name': 'Image_Only',
        'use_text': False, 'use_segmentation': False, 'use_gate': False,
        'pool_mode': 'avgmax', 'two_phase': True, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },
    {
        # Image_Only_Joint = 纯图像单阶段联合训练 (Baseline-II, 分离训练协议贡献与文本独立贡献)
        'name': 'Image_Only_Joint',
        'use_text': False, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },
    {
        'name': 'Text_Only',           # 移除图像 → 纯文本
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': True,
        'verbose': 0,
    },

    # ======== 3. 文本处理消融 ========
    {
        'name': 'NoSegmentation',      # 无 jieba 分词 (使用原始文本)
        'use_text': True, 'use_segmentation': False, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },

    # ======== 4. 融合机制消融 ========
    {
        'name': 'No_Gate',             # 无门控融合 (纯拼接)
        'use_text': True, 'use_segmentation': True, 'use_gate': False,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },
    {
        'name': 'Deep_Fusion',         # 深层融合头 (D1024 → D512)
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'deep', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },

    # ======== 5. 池化策略消融 ========
    {
        'name': 'Pool_Avg_Only',
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avg', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },
    {
        'name': 'Pool_Max_Only',
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'max', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },

    # ======== 6. LSTM 容量消融 ========
    {
        'name': 'LSTM128',
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 128, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },
    {
        'name': 'LSTM512',
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 512, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },
    {
        'name': 'LSTM_NoSeq',          # return_sequences=False (无池化, 取末态)
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': False,
        'skip_vision': False,
        'verbose': 0,
    },

    # ======== 7. 正则化消融 ========
    {
        'name': 'No_TextDropout',      # 文本分支无 Dropout
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.0, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },
    {
        'name': 'Low_TextDropout',     # 低文本 Dropout (0.1 对比 0.4)
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.1, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },

    # ======== 8. 训练策略消融 ========
    {
        'name': 'Two_Phase',           # 两阶段训练 (与单阶段联合训练对比)
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': True, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': False,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },
    {
        'name': 'FixedLR',             # 固定学习率 (无 ReduceLROnPlateau)
        'use_text': True, 'use_segmentation': True, 'use_gate': True,
        'pool_mode': 'avgmax', 'two_phase': False, 'use_pretrained': True,
        'text_dropout_rate': 0.4, 'lstm_units': 256, 'fixed_lr': True,
        'fusion_head': 'papermix', 'lstm_return_sequences': True,
        'skip_vision': False,
        'verbose': 0,
    },
]


def report_table_for_paper(results_list):
    """
    按论文表6的格式输出消融汇总 (可直接复制到 Word/论文)。
    表6 列: 类别 | 配置 | 准确率/% | 宏平均F1/% | 宏平均AUC/% | ΔAcc/pp
    """
    by_name = {r['name']: r for r in results_list}
    full = by_name.get('Full_Model')
    if full is None:
        logger.warning("Full_Model 未在结果中, 无法计算 ΔAcc")
        ref_acc = 0.0
    else:
        ref_acc = full['acc_mean']

    order = ['Full_Model', 'Image_Only', 'Image_Only_Joint', 'Text_Only',
             'Two_Phase', 'FixedLR',
             'NoSegmentation', 'LSTM128', 'LSTM512', 'LSTM_NoSeq',
             'Pool_Avg_Only', 'Pool_Max_Only', 'No_TextDropout', 'Low_TextDropout',
             'No_Gate', 'Deep_Fusion']

    lines = []
    lines.append("表6  消融实验结果汇总 (ResNet50 基线, 5折均值±标准差)")
    lines.append("| 类别 | 配置 | 准确率/% | 宏平均F1/% | 宏平均AUC/% | ΔAcc/pp |")
    lines.append("|---|---|---|---|---|---|")
    for name in order:
        if name not in by_name:
            lines.append(f"| ? | {name} | 未完成 | 未完成 | 未完成 | ? |")
            continue
        r = by_name[name]
        delta = (r['acc_mean'] - ref_acc) * 100
        delta_s = '—' if name == 'Full_Model' else (f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}")
        lines.append(f"| {CATEGORY_OF[name]} | {name} | "
                     f"{r['acc_mean']*100:.2f}±{r['acc_std']*100:.2f} | "
                     f"{r['macro_f1_mean']*100:.2f}±{r['macro_f1_std']*100:.2f} | "
                     f"{r['macro_auc_mean']*100:.2f}±{r['macro_auc_std']*100:.2f} | {delta_s} |")
    return "\n".join(lines)


def _config_complete_on_disk(output_subdir, cv_folds=5):
    """检查某配置的 fold_results.csv 是否已有全部 cv_folds 折结果。"""
    fold_csv = os.path.join(output_subdir, "fold_results.csv")
    if not os.path.exists(fold_csv):
        return False
    df = pd.read_csv(fold_csv)
    if 'config' in df.columns:
        df = df[df['config'] == os.path.basename(output_subdir)]
    return len(df) >= cv_folds


def _resume_status_lines(configs, output_dir):
    """
    [F11] 返回断点续跑状态行: 逐配置报告"已完折 / 待续折 / 未开始"。
    服务器停止重启后, 据此可确认会从第几个组件(配置)的第几折继续。
    """
    cv = CONFIG['cv_folds']
    done, partial, fresh = [], [], []
    for cfg in configs:
        name = cfg['name']
        fold_csv = os.path.join(output_dir, name, 'fold_results.csv')
        folds_done = set()
        if os.path.exists(fold_csv):
            try:
                df = pd.read_csv(fold_csv)
                if 'config' in df.columns:
                    df = df[df['config'] == name]
                folds_done = {int(f) for f in df['fold']}
            except Exception as e:
                logger.warning(f"  读取 {name} 断点失败: {e}")
        if len(folds_done) >= cv:
            done.append(name)
        elif folds_done:
            remain = sorted(set(range(1, cv + 1)) - folds_done)
            partial.append((name, sorted(folds_done), remain))
        else:
            fresh.append(name)

    lines = [f"  已完成(将跳过): {', '.join(done) if done else '无'}",
             f"  未开始(从头跑): {', '.join(fresh) if fresh else '无'}"]
    if partial:
        lines.append("  部分完成(将续跑):")
        for name, fd, rem in partial:
            lines.append(f"    - {name}: 已完折 {fd} | 待跑折 {rem}")
    else:
        lines.append("  部分完成(将续跑): 无")
    return lines


# ============================================================
# 主程序
# ============================================================
def main():
    # [F6] 命令行参数: --only Full_Model,Image_Only 可先只跑参考行核验
    only = None
    if '--only' in sys.argv:
        idx = sys.argv.index('--only')
        if idx + 1 < len(sys.argv):
            only = {x.strip() for x in sys.argv[idx + 1].split(',') if x.strip()}
            logger.info(f"Running only: {sorted(only)}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 加载数据
    if not os.path.exists(CONFIG["labels_path"]):
        raise FileNotFoundError(
            f"labels.csv 不存在: {CONFIG['labels_path']}\n"
            f"请设置环境变量 HERB_LABELS_CSV / HERB_CAPTIONS_JSON / HERB_IMAGES_DIR 指向服务器数据。")
    with open(CONFIG["captions_path"], 'r', encoding='utf-8') as f:
        captions = json.load(f)
    df = pd.read_csv(CONFIG["labels_path"])
    df['caption_raw'] = df['image_name'].map(captions)
    df = df.dropna(subset=['caption_raw']).reset_index(drop=True)
    df['caption_seg'] = df['caption_raw'].apply(segment_text)
    df['image_path'] = df['image_name'].apply(lambda x: os.path.join(CONFIG["images_dir"], x))
    # [F1] 与主模型一致: 过滤掉实际不存在的图像文件 (保证与主模型样本集完全相同)
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    df = df.dropna(subset=['image_path']).reset_index(drop=True)

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Task_Chinese_medicinal_herb'])
    class_names = [str(c) for c in le.classes_]
    logger.info(f"Total clean samples: {len(df)}, classes: {len(class_names)}")

    # 顶层断点 (完整结果在 ablation_summary.csv)
    ckpt_file = os.path.join(CONFIG["output_dir"], "completed_experiments.txt")
    completed = set()
    if os.path.exists(ckpt_file):
        with open(ckpt_file, 'r') as f:
            completed = set(line.strip() for line in f if line.strip())

    csv_path = os.path.join(CONFIG["output_dir"], "ablation_summary.csv")
    if os.path.exists(csv_path):
        all_results_df = pd.read_csv(csv_path)
        all_results = all_results_df.to_dict('records')
    else:
        all_results = []

    configs_to_run = ablation_configs
    if only:
        configs_to_run = [c for c in ablation_configs if c['name'] in only]
        missing = only - {c['name'] for c in ablation_configs}
        if missing:
            logger.warning(f"Unknown configs in --only: {sorted(missing)}")

    # [F11] 启动即打印断点续跑状态: 每个配置已完折 / 待续折, 便于确认重启接续点
    logger.info("===== 断点续跑状态 =====")
    for _line in _resume_status_lines(configs_to_run, CONFIG["output_dir"]):
        logger.info(_line)
    logger.info("=========================")

    for config in configs_to_run:
        name = config['name']
        subdir = os.path.join(CONFIG["output_dir"], name)

        # [F5] 仅当 5 折结果齐全时才跳过 (防止标记完成但 CSV 被删)
        if name in completed and _config_complete_on_disk(subdir):
            logger.info(f"Skipping already completed: {name}")
            continue

        logger.info(f"\n{'='*60}\nRunning ablation: {name}\n{'='*60}")

        try:
            results = run_cv_for_config(df, le, class_names, config, subdir)
        except Exception as e:
            logger.error(f"Experiment {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            with open(os.path.join(CONFIG["output_dir"], "failed_experiments.txt"), 'a') as f:
                f.write(f"{name}: {e}\n")
            continue

        # 合并进总结果 (按 name 覆盖, 避免重复)
        all_results = [r for r in all_results if r.get('name') != name] + [results]
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        logger.info(f"Completed {name}: Acc={results['acc_mean']:.4f}+/-{results['acc_std']:.4f}, "
                     f"F1={results['macro_f1_mean']:.4f}+/-{results['macro_f1_std']:.4f}, "
                     f"AUC={results['macro_auc_mean']:.4f}+/-{results['macro_auc_std']:.4f}")

        with open(ckpt_file, 'a') as f:
            f.write(name + '\n')
        completed.add(name)

    # 最终输出
    logger.info(f"All experiments done. Summary saved to {csv_path}")
    final_df = pd.DataFrame(all_results)
    if not final_df.empty:
        print("\n" + "=" * 70)
        print("      Ablation Study v3 — ResNet50 Reproduction (FIXED)")
        print("=" * 70)
        cols = ['name', 'acc_mean', 'acc_std', 'macro_f1_mean', 'macro_f1_std', 'macro_auc_mean', 'macro_auc_std']
        print(final_df[cols].to_string(index=False))
        print("=" * 70)

        # 按准确率排序
        sorted_df = final_df.sort_values('acc_mean', ascending=False)
        print("\n--- Ranked by Accuracy ---")
        for _, row in sorted_df.iterrows():
            print(f"  {row['name']:25s}  Acc={row['acc_mean']:.2%} ± {row['acc_std']:.2%}  "
                  f"F1={row['macro_f1_mean']:.2%}  AUC={row['macro_auc_mean']:.2%}")

        # [F7] Full_Model 与主模型记录值 98.03% 一致性核验
        by_name = {r['name']: r for r in all_results}
        if 'Full_Model' in by_name:
            fm = by_name['Full_Model']
            ref = 0.9803
            diff_pp = (fm['acc_mean'] - ref) * 100
            print("\n--- Full_Model 与主模型一致性核验 ---")
            print(f"  Full_Model Acc = {fm['acc_mean']*100:.2f}%  vs  主模型记录 98.03%  → Δ = {diff_pp:+.2f}pp")
            if abs(diff_pp) > 0.5:
                print("  ⚠ 警告: 偏差超过 0.5pp, 请检查数据路径/样本集是否与主模型完全一致!")
            else:
                print("  ✓ 在噪声范围内一致 (主模型未设种子, 消融已设种子, 存在随机波动属正常)")

        # [F7] 打印 Full_Model / Image_Only 逐折准确率 (供复算配对 t 检验)
        print("\n--- 逐折准确率 (配对 t 检验用) ---")
        for nm in ['Full_Model', 'Image_Only']:
            if nm in by_name:
                fold_csv_path = os.path.join(CONFIG["output_dir"], nm, "fold_results.csv")
                if os.path.exists(fold_csv_path):
                    fdf = pd.read_csv(fold_csv_path)
                    if 'config' in fdf.columns:
                        fdf = fdf[fdf['config'] == nm]
                    fdf = fdf.sort_values('fold')
                    accs = [f"{v*100:.2f}" for v in fdf['accuracy']]
                    print(f"  {nm:12s} folds acc: {', '.join(accs)}")

        # 论文表6格式输出
        print("\n" + "=" * 70)
        print("  论文表6 格式汇总 (可直接复制进论文)")
        print("=" * 70)
        print(report_table_for_paper(all_results))

        # 论文 4.x 节关键口径提示
        print("\n" + "=" * 70)
        print("  论文口径提示 (运行后按真实值回填)")
        print("=" * 70)
        if 'Image_Only' in by_name and 'Image_Only_Joint' in by_name and 'Two_Phase' in by_name:
            io = by_name['Image_Only']['acc_mean'] * 100
            ioj = by_name['Image_Only_Joint']['acc_mean'] * 100
            tp = by_name['Two_Phase']['acc_mean'] * 100
            print(f"  文本模态独立净贡献      = 98.03 − Image_Only_Joint = {98.03 - ioj:.2f} pp")
            print(f"  训练协议贡献(BaselineII) = Image_Only_Joint − Image_Only = {ioj - io:.2f} pp")
            print(f"  Two_Phase 折损          = 98.03 − Two_Phase = {98.03 - tp:.2f} pp (若负值需改写论文 5.2 节)")


if __name__ == "__main__":
    main()
