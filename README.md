# MMGF-Net

**基于多模态门控融合的中药材饮片识别算法研究** / *Multimodal Gated Fusion Network (MMGF-Net) for Traditional Chinese Medicine Herbal Slice Recognition*

本仓库是论文《基于多模态门控融合的中药材饮片识别算法研究》（投稿《计算机工程》）的官方代码实现与实验结果归档。

## 概述

- **任务**：中药材饮片图像 14 类（类别 0–13）识别。
- **数据**：1119 张图像进行 5 折分层交叉验证；另有 138 张从未参与训练的独立测试集（held-out）。
- **方法**：MMGF-Net 由图像分支与文本分支构成，通过**门控融合**（gated fusion）聚合跨模态信息，以弥补仅依赖视觉信息在相似饮片上的判别不足。
- **主结果**：
  - 5 折交叉验证：**准确率 98.03%**，宏平均 F1 97.75%，宏平均 AUC 99.98%；
  - 独立测试集（138 张）：**准确率 94.93%**，宏平均 F1 94.17%。

## 模型架构

与 `models/train_multimodal_config.py` 中的 `build_multimodal_model()` 一致：

```
图像分支  Visual branch:   backbone → GAP → BN → Dense(512, swish)
文本分支  Text branch:     Embedding → SpatialDropout1D → BiLSTM → GAP1D+GMP1D → LayerNorm → Dense(512, swish)
融合      Fusion:          Concat → Gate(Dense 1024, sigmoid) → Multiply → Dense(512, swish) → BN → Dropout(0.5) → Softmax
```

训练策略（见代码头注释）：

- 单阶段联合训练，骨干网络从初始即部分冻结；
- Adam(lr=1e-4)，30 epochs，EarlyStopping(patience=8) + ReduceLROnPlateau；
- 类别平衡权重处理样本不均衡；BN 层始终冻结；
- 5 折分层交叉验证（shuffle=True, random_state=42），训练/验证 15% 分层划分；
- 文本：jieba 中文分词，逐折重建 Tokenizer（max_words=10000, max_text_len=320）；
- 数据增强：旋转 ±20°、水平翻转、亮度 [0.9, 1.1]。

## 目录结构

```
MMGF-Net/
├── README.md                      本文件
├── requirements.txt               Python 依赖（TensorFlow 2.13）
├── data/                          数据标签（仅标签，不含原始图像）
│   ├── train_labels.csv           1119 行训练池标签
│   └── test_labels.csv            138 行独立测试集标签（102 张自动标注 + 36 张人工标注）
├── models/                        全部模型代码
│   ├── train_multimodal_config.py     MMGF-Net 主模型（图像+文本+门控融合，5 折 CV）
│   ├── train_config.py                图像单模态训练配置
│   ├── train_and_predict_5models.py   文献复现基线（改进ResNet50 / CCNNet / ConvNeXt / SFE-CA）+ 训练/预测
│   ├── multiAblation_resnet50.py      消融实验主脚本
│   ├── ablation_resnet50_final/      消融结果（论文表 8：13 项消融配置 + Full_Model 参考行）
│   ├── train_{resnet50,vgg16,mobilenetv3,inceptionv3}/*.py           表 3 图像骨干入口
│   ├── train_*_multimodal/*.py        表 4 多模态骨干变体入口
│   └── mobilenetv2/                   MobileNetV2（表 3 / 表 4 对应脚本）
│       ├── baseline_model_strict.py   图像单模态（表 3 MobileNetV2 数据源，5 折 CV 94.10）
│       └── MNV2_text.py               多模态门控融合（与 train_multimodal_config.py 一致）
├── test_set_experiment/           138 张独立测试集实验（完整自包含，见其 README）
├── test_set_results/              独立测试集评估结果（14 个模型目录 + 汇总 CSV + 混淆矩阵图）
└── text_slices/                   文本分支用药典式图像描述（代码 + 描述 JSON）
    ├── image_captions.json         训练实际数据源（多模态/主模型/消融均读取，覆盖 1257 张）
    └── generate_herb_captions.py   药典式描述生成脚本（qwen3.6-plus，输出 image_captions.json）
```

## 论文表格 ↔ 代码/结果映射

| 论文位置 | 内容 | 训练/模型代码 | 独立测试集结果 |
|---|---|---|---|
| 表 2 | 文献复现对比（改进ResNet50、CCNNet、ConvNeXt、SFE-CA、MMGF-Net） | `models/train_and_predict_5models.py`（图像单模态复现基线）+ `models/train_multimodal_config.py`（MMGF-Net） | `test_set_results/{improved_resnet50,ccnnet,convnext,sfeca,mmgf_resnet50}/` |
| 表 3 | 图像骨干对比（ResNet50、MobileNetV2、MobileNetV3Large、InceptionV3、VGG16） | `models/train_config.py` + `models/train_*/*.py`、`models/mobilenetv2/baseline_model_strict.py` | `test_set_results/{resnet50,mobilenetv2,mobilenetv3large,inceptionv3,vgg16}/` |
| 表 4 | 多模态骨干变体（同一骨干 × 门控融合） | `models/train_multimodal_config.py` + `models/train_*_multimodal/*.py`、`models/mobilenetv2/MNV2_text.py` | `test_set_results/mmgf_*/` |
| 表 6 | 主实验 5 折结果 | `models/train_multimodal_config.py`（`run_multimodal_cv`） | `test_set_results/mmgf_resnet50/` |
| 表 7 | 逐类别 F1（视觉 vs 多模态） | `models/ablation_resnet50_final/{Image_Only,Full_Model}/fold_*/classification_report.txt`（5 折 CV 逐类 F1 均值） | `test_set_results/test_set_per_class_f1.csv`（138 张测试集逐类 F1，供泛化分析） |
| 表 8 | 消融实验（模态/文本分支/训练/融合） | `models/multiAblation_resnet50.py` | `models/ablation_resnet50_final/ablation_summary_ResNet50.csv`（14 配置权威汇总，与论文逐项一致）及各配置 `fold_results.csv`、`fold_*/classification_report.txt`、`training_curves_fold*.png` |
| 独立测试集 | 独立测试集划分与评估流程 | `test_set_experiment/`（`build_split.py` → `prepare_captions.py` → `evaluate_test_set.py`） | `test_set_results/all_models_test_summary.csv` |

> **表 8 数值**：以 `ablation_summary_ResNet50.csv`（13 项消融配置 + Full_Model 参考行，共 14 行）为权威数据源，各配置 acc 与论文表 8 逐行精确一致（Full_Model 98.03%、Image_Only 96.60%、Image_Only_Joint 97.21%、No_Gate 97.21%、Text_Only 75.16% 等）。仓库包含论文涉及的 13 个消融配置目录 + Full_Model 参考行目录（共 14 个，与表 8 行一一对应）；每配置含 5 折 `fold_results.csv`、逐折 `classification_report.txt` 与训练曲线。

## 复现

### 1. 环境

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install pandas scikit-learn jieba openai        # test_set_experiment 需要
```

训练环境：Python 3.8/3.9 + TensorFlow 2.13 + CUDA。

### 2. 数据布局

脚本内置训练环境的绝对路径（如 `/mnt/data1/spring/multimodal_project/...`、`/home/shify_ug/...`），部署时请按本地实际位置调整。约定布局如下：

```
<PROJECT_ROOT>/
├── labels.csv                  # 1119 行（与 data/train_labels.csv 相同）
├── data/images/                # 1257 张饮片图像（原始图像未随仓库上传，请自行准备）
└── text_slices/image_captions.json   # 药典式图文描述
```

`test_set_experiment/config.py` 已支持环境变量覆盖路径（`HERB_PROJECT_ROOT` / `HERB_IMAGES_DIR` 等），推荐优先使用。

### 3. 训练

```bash
# 图像单模态（表 3）
python models/train_resnet50/train_resnet50.py
python models/train_mobilenetv3/train_mobilenetv3.py
python models/train_inceptionv3/train_inceptionv3.py
python models/train_vgg16/train_vgg16.py
python models/mobilenetv2/baseline_model_strict.py

# 多模态门控融合（表 4 / 表 6 主模型）
python models/train_resnet50_multimodal/train_resnet50_multimodal.py   # MMGF-Net（ResNet50）
python models/train_mobilenetv3_multimodal/train_mobilenetv3_multimodal.py
python models/train_inceptionv3_multimodal/train_inceptionv3_multimodal.py
python models/train_vgg16_multimodal/train_vgg16_multimodal.py
python models/mobilenetv2/MNV2_text.py

# 文献复现基线（表 2）
python models/train_and_predict_5models.py

# 消融实验（表 8）
python models/multiAblation_resnet50.py
```

### 4. 独立测试集评估

```bash
cd test_set_experiment
bash run_all.sh            # 全流程（构建划分 → 生成测试集描述 → 评估 → 汇总）
# 多模态推理需 DASHSCOPE_API_KEY（生成 138 张测试图像的药典式描述，见 README.md）
```

详细说明见 `test_set_experiment/README.md`。

## 许可证

[MIT License](LICENSE)
