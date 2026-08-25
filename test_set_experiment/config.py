# -*- coding: utf-8 -*-
"""
===========================================================================
独立测试集实验 — 共享配置
Independent Test-Set Experiment — Shared Config
===========================================================================
本配置集中管理：
  1. 路径（默认按服务器布局，可用环境变量覆盖）
  2. 训练超参（严格与已完成的 5 折分层交叉验证实验一致）
  3. 全部模型的注册表（骨干 / 多模态 / 对比方法）

服务器默认项目根目录：
  /home/shify_ug/multimodal_project        （labels.csv 中 image_path 所指）
  另一台机器上见  /mnt/data1/spring/multimodal_project/data/images

环境变量（可覆盖默认路径）：
  HERB_PROJECT_ROOT   项目根目录
  HERB_IMAGES_DIR     1257 张图像所在目录
  HERB_LABELS_CSV     1119 张训练图像标签 (labels.csv)
  HERB_TEST_LABELS_CSV 独立测试集标签（138 行）
  HERB_TEST_CAPTIONS_JSON  测试集图像描述（多模态模型用）
  HERB_CAPTIONS_JSON      训练用图像描述（多模态模型用，默认 image_captions.json）
  HERB_OUTPUT_DIR     实验结果输出目录
  DASHSCOPE_API_KEY   （prepare_captions.py 生成测试集描述时使用）
===========================================================================
"""
import os


# ─── 工具：返回第一个存在的路径，否则返回第一个候选 ──────────────────────────
def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0] if paths else ""


# ─── 项目根目录解析 ───────────────────────────────────────────────────────
# 本包会被拷贝到服务器上，放在项目根目录下（或其子目录中）。
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.environ.get("HERB_PROJECT_ROOT")
if not _PROJECT_ROOT:
    _PROJECT_ROOT = _first_existing(
        "/home/shify_ug/multimodal_project",          # 服务器（labels.csv 布局）
        "/mnt/data1/spring/multimodal_project",       # 另一台服务器（预测脚本路径）
        os.path.dirname(os.path.dirname(_PKG_DIR)),   # 本包位于 <root>/test_set_experiment
        os.path.dirname(_PKG_DIR),                    # 本包位于 <root>/xxx/test_set_experiment
    )
PROJECT_ROOT = _PROJECT_ROOT


# ─── 数据集路径 ───────────────────────────────────────────────────────────
IMAGES_DIR = os.environ.get(
    "HERB_IMAGES_DIR") or _first_existing(
    os.path.join(PROJECT_ROOT, "data", "images"),   # 服务器：1257 张图像
    os.path.join(PROJECT_ROOT, "images"),           # labels.csv 中 image_path 所示
    "E:/Herb/data",                                 # 本地拷贝（纯 Windows 复现用）
)
if not os.path.isabs(IMAGES_DIR):
    IMAGES_DIR = os.path.join(PROJECT_ROOT, IMAGES_DIR)

LABELS_CSV = os.environ.get("HERB_LABELS_CSV") or _first_existing(
    os.path.join(PROJECT_ROOT, "labels.csv"),        # 1119 张训练图像标签
    "E:/Herb/labels.csv",
)

# 比赛原始数据（仅用于 build_split.py 自动回填测试集标签，服务器上若无则手动填标签）
COMPETITION_TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "train.csv")
COMPETITION_TEST_CSV  = os.path.join(PROJECT_ROOT, "data", "test.csv")
COMPETITION_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "data1", "images")

# 图像描述（药典式），多模态模型训练所用（text_slices/image_captions.json）
CAPTIONS_JSON = os.environ.get("HERB_CAPTIONS_JSON") or os.path.join(
    PROJECT_ROOT, "text_slices", "image_captions.json")
TEST_CAPTIONS_JSON = os.environ.get("HERB_TEST_CAPTIONS_JSON") or os.path.join(
    os.path.dirname(__file__), "test_captions.json")

# 划分文件（build_split.py 生成）
TRAIN_LABELS_CSV = os.environ.get("HERB_TRAIN_LABELS_CSV") or os.path.join(
    os.path.dirname(__file__), "train_labels.csv")
TEST_LABELS_CSV = os.environ.get("HERB_TEST_LABELS_CSV") or os.path.join(
    os.path.dirname(__file__), "test_labels.csv")

# 输出目录
OUTPUT_DIR = os.environ.get("HERB_OUTPUT_DIR") or os.path.join(
    PROJECT_ROOT, "test_set_results")


# ─── 训练超参（严格与已完成实验一致） ───────────────────────────────────────
BATCH_SIZE   = 16
CV_FOLDS     = 5
SEED         = 42
VAL_FRACTION = 0.15          # 每折内部再划 15% 作为验证集
MAX_TEXT_LEN = 320           # 多模态文本分支序列长度
MAX_WORDS    = 10000         # 分词器词表大小
OOV_TOKEN    = "<OOV>"
EMBEDDING_DIM = 256
LSTM_UNITS   = 256


# ─── 图像尺寸与预处理（按各模型原生输入） ───────────────────────────────────
# 各模型训练脚本中已确认的 IMAGE_SIZE
IMAGE_SIZES = {
    "resnet50":      (224, 224),
    "mobilenetv2":   (224, 224),
    "mobilenetv3":   (224, 224),
    "inceptionv3":   (299, 299),
    "vgg16":         (224, 224),
    "convnext":      (224, 224),
    "efficientnet":  (224, 224),
    "ccnnet":        (224, 224),
}


def get_preprocess_fn(name):
    """按训练脚本一致的方式返回预处理函数（惰性导入，避免无 TF 环境报错）。"""
    import tensorflow as tf
    ka = tf.keras.applications
    if name == "resnet50":
        return ka.resnet50.preprocess_input
    if name == "mobilenetv2":
        return ka.mobilenet_v2.preprocess_input
    if name == "mobilenetv3":
        return ka.mobilenet_v3.preprocess_input
    if name == "inceptionv3":
        return ka.inception_v3.preprocess_input
    if name == "vgg16":
        return ka.vgg16.preprocess_input
    if name == "efficientnet":
        return ka.efficientnet.preprocess_input
    if name == "convnext":
        try:
            from tensorflow.keras.applications.convnext import preprocess_input
        except ImportError:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        return preprocess_input
    if name == "ccnnet":
        # CCNNet 专用预处理：归一化到 [-1, 1]（与 train_ccnnet_multimodal.py 一致）
        def _ccnnet_preprocess(x):
            return (x / 127.5) - 1.0
        return _ccnnet_preprocess
    raise KeyError(f"Unknown preprocess name: {name}")


# ─── 模型注册表 ───────────────────────────────────────────────────────────
# 每个条目：
#   key         标识（输出子目录名）
#   paper       论文/展示名
#   family      image_only | multimodal
#   rel_dir     相对于 PROJECT_ROOT 的输出目录（已训练的各折模型所在）
#   fold_model  每折 SavedModel 目录名模板（{k}=折号）
#   image_size  模型输入尺寸
#   preprocess  预处理函数名（见 get_preprocess_fn）
#   note        备注
MODELS = [
    # ── 表3：五种骨干网络的图像单模态（image-only）实验 ──
    dict(key="resnet50",      paper="ResNet50",            family="image_only",
         rel_dir="models/train_resnet50/output_resnet50",
         fold_model="best_model_fold{k}", image_size=IMAGE_SIZES["resnet50"],
         preprocess="resnet50", note="图像单模态基线 Baseline-I"),
    dict(key="mobilenetv2",   paper="MobileNetV2",         family="image_only",
         rel_dir="models/mobilenetv2/output3",
         fold_model="best_model_fold{k}", image_size=IMAGE_SIZES["mobilenetv2"],
         preprocess="mobilenetv2", note="MobileNetV2 图像单模态基线"),
    dict(key="mobilenetv3large", paper="MobileNetV3Large", family="image_only",
         rel_dir="models/train_mobilenetv3/output_mobilenetv3",
         fold_model="best_model_fold{k}", image_size=IMAGE_SIZES["mobilenetv3"],
         preprocess="mobilenetv3", note="图像单模态"),
    dict(key="inceptionv3",   paper="InceptionV3",         family="image_only",
         rel_dir="models/train_inceptionv3/output_inceptionv3",
         fold_model="best_model_fold{k}", image_size=IMAGE_SIZES["inceptionv3"],
         preprocess="inceptionv3", note="图像单模态"),
    dict(key="vgg16",         paper="VGG16",               family="image_only",
         rel_dir="models/train_vgg16/output_vgg16",
         fold_model="best_model_fold{k}", image_size=IMAGE_SIZES["vgg16"],
         preprocess="vgg16", note="图像单模态"),

    # ── 多模态门控融合模型（本文方法 MMGF-Net 及其骨干变体）──
    dict(key="mmgf_resnet50",  paper="MMGF-Net (ResNet50)", family="multimodal",
         rel_dir="models/train_resnet50_multimodal/output_resnet50_multimodal",
         fold_model="best_model_fold{k}", image_size=IMAGE_SIZES["resnet50"],
         preprocess="resnet50", note="本文方法"),
    dict(key="mmgf_mobilenetv2", paper="MMGF-Net (MobileNetV2)", family="multimodal",
         rel_dir="models/mobilenetv2/output3",
         fold_model="best_model_fold{k}", image_size=IMAGE_SIZES["mobilenetv2"],
         preprocess="mobilenetv2", note="多模态骨干变体"),
    dict(key="mmgf_mobilenetv3large", paper="MMGF-Net (MobileNetV3Large)", family="multimodal",
         rel_dir="models/train_mobilenetv3_multimodal/output_mobilenetv3_multimodal",
         fold_model="best_model_fold{k}", image_size=IMAGE_SIZES["mobilenetv3"],
         preprocess="mobilenetv3", note="多模态骨干变体"),
    dict(key="mmgf_inceptionv3", paper="MMGF-Net (InceptionV3)", family="multimodal",
         rel_dir="models/train_inceptionv3_multimodal/output_inceptionv3_multimodal",
         fold_model="best_model_fold{k}", image_size=IMAGE_SIZES["inceptionv3"],
         preprocess="inceptionv3", note="多模态骨干变体"),
    dict(key="mmgf_vgg16",     paper="MMGF-Net (VGG16)",   family="multimodal",
         rel_dir="models/train_vgg16_multimodal/output_vgg16_multimodal",
         fold_model="best_model_fold{k}", image_size=IMAGE_SIZES["vgg16"],
         preprocess="vgg16", note="多模态骨干变体"),

    # ── 表2：复现文献方法（图像单模态复现，与论文一致）──
    dict(key="improved_resnet50",     paper="改进ResNet50[19]",    family="image_only",
         rel_dir="models/model_predictions/Improved_ResNet50",
         rel_dir_alt=("papermodels/model_predictions/Improved_ResNet50",),
         fold_model="best_model", image_size=IMAGE_SIZES["resnet50"],
         preprocess="resnet50", note="复现文献[19] ResNet50+CBAM"),
    dict(key="ccnnet",   paper="CCNNet",   family="image_only",
         rel_dir="models/model_predictions/CCNNet",
         rel_dir_alt=("papermodels/model_predictions/CCNNet",),
         fold_model="best_model", image_size=IMAGE_SIZES["ccnnet"],
         preprocess="ccnnet", note="复现文献[15]"),
    dict(key="convnext", paper="ConvNeXt", family="image_only",
         rel_dir="models/model_predictions/ConvNeXt",
         rel_dir_alt=("papermodels/model_predictions/ConvNeXt",),
         fold_model="best_model.keras", image_size=IMAGE_SIZES["convnext"],
         preprocess="convnext", note="复现文献[16]"),
    dict(key="sfeca",    paper="SFE-CA",   family="image_only",
         rel_dir="models/model_predictions/SFE-CA",
         rel_dir_alt=("papermodels/model_predictions/SFE-CA",),
         fold_model="best_model.keras", image_size=IMAGE_SIZES["efficientnet"],
         preprocess="efficientnet", note="复现文献[18]"),
]

MODEL_INDEX = {m["key"]: m for m in MODELS}


def model_dir(m, project_root=None):
    """返回某模型已训练各折模型所在目录（绝对路径）。

    服务器与本地可能用不同顶层目录保存模型（models vs papermodels）。
    模板 rel_dir 不存在时自动回退到 rel_dir_alt 中的候选，两处布局都可直接用。
    """
    root = project_root or PROJECT_ROOT
    d = m["rel_dir"]
    if not os.path.isabs(d):
        d = os.path.join(root, d)
    if os.path.isdir(d):
        return d
    for alt in m.get("rel_dir_alt", ()):
        a = alt if os.path.isabs(alt) else os.path.join(root, alt)
        if os.path.isdir(a):
            return a
    return d


def fold_model_path(m, fold, project_root=None):
    """返回某模型第 fold 折 SavedModel 目录/文件的绝对路径。

    若按 fold_model 模板找不到，则回退搜索该折目录下的常见命名
    （SavedModel 目录、*.keras），保证不同训练脚本的保存方式都能被找到。
    """
    d = model_dir(m, project_root)
    tmpl = m.get("fold_model") or "best_model_fold{k}"
    name = tmpl.format(k=fold)
    p = os.path.join(d, f"fold_{fold}", name)
    if os.path.exists(p):
        return p

    fd = os.path.join(d, f"fold_{fold}")
    if os.path.isdir(fd):
        import glob
        # 1) SavedModel 目录（含 saved_model.pb）
        for c in sorted(glob.glob(os.path.join(fd, "*"))):
            if (os.path.isdir(c) and
                    os.path.exists(os.path.join(c, "saved_model.pb"))):
                return c
        # 2) Keras 完整模型文件（.keras，排除仅权重的 .weights.h5）
        for c in sorted(glob.glob(os.path.join(fd, "*.keras"))):
            return c
    return p


def resolve_test_set_output(m):
    """每模型结果输出目录。"""
    return os.path.join(OUTPUT_DIR, m["key"])
