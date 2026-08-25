# 独立测试集实验（138 张 held-out / 14 类）

面向论文《基于多模态门控融合的中药材饮片识别算法研究》的独立测试集评估。
该实验的数据即论文正文表 2 / 表 3 / 表 4 / 表 6 中的“独立测试集”列。

数据共 1257 张，其中 1119 张用于分层 5 折交叉验证（训练集与验证集），
其余 **138 张** 固定为**独立测试集**，使全部模型在训练阶段从未接触过的
外部数据上进行评测。

> 科学与方法要点：138 张测试图像从未进入过 1119 张的 5 折交叉验证，
> 各模型**加载已保存的各折模型**（无需重新训练）在测试集上做预测，即可得到
> 真正独立的测试结果。对每个模型同时报告“5 折逐折测试指标”与
> “5 折软投票集成的测试指标”。论文表格中的独立测试集列为
> **5 折模型预测均值**（per-fold mean）；软投票集成结果在
> `all_models_test_summary.csv` 中一并给出，供对照。多模态模型在推理时
> 逐折重建与训练一致的分词器（随机种子 42、分层划分确定，可完全复现）。

---

## 1. 目录结构

```
test_set_experiment/
├── config.py                  共享配置：路径、超参、全部模型注册表（唯一改动点）
├── build_split.py             步骤1：构建 1119 训练 / 138 独立测试 的固定划分
├── train_labels.csv           1119 行（= labels.csv，训练池）
├── test_labels.csv            138 行（独立测试集；102 张自动标注 + 36 张人工标注）
├── prepare_captions.py        步骤2：为 138 张测试图像生成药典式描述（多模态用）
├── evaluate_test_set.py       步骤3：主评估脚本（全部 TF 模型）
├── convnext_loader.py         ConvNeXt 专用：结构化权重加载
├── train_with_holdout_test.py 训练完备性检查 / 按需重新训练
└── run_all.sh                 服务器一键编排
```

## 2. 环境依赖（服务器，与训练环境一致）

```bash
# 训练环境已是 Python3.8/3.9 + TensorFlow 2.13 + CUDA
python3 -m pip install --upgrade pandas scikit-learn jieba openai
```

多模态模型推理需要测试集描述：配置 `DASHSCOPE_API_KEY`（阿里云百炼），
与训练时生成 `image_captions.json` 完全相同的提示词与模型（`qwen3.6-plus`）。

```bash
export DASHSCOPE_API_KEY='sk-xxxxxxxx'
```

## 3. 服务器部署步骤

把 `test_set_experiment/` 整个目录拷到项目根目录下：

```bash
# 服务器项目根目录（labels.csv 所在处）
cd /home/shify_ug/multimodal_project
cp -r <本目录>/test_set_experiment ./
cd test_set_experiment
```

`config.py` 会按以下顺序自动找到数据：

1. 环境变量 `HERB_PROJECT_ROOT` / `HERB_IMAGES_DIR` / `HERB_LABELS_CSV` 等
2. 项目根目录（`/home/shify_ug/multimodal_project` 或 `/mnt/data1/spring/multimodal_project`）
3. 从 `__file__` 反推项目根

`config.py` 中 `IMAGES_DIR` 默认优先取 `<PROJECT_ROOT>/data/images`
（1257 张图像目录）；若图像位于 `<PROJECT_ROOT>/images`，调整
`_first_existing(...)` 中两个候选的先后顺序。建议部署后先运行
`python3 build_split.py --dry-run` 和 `python3 evaluate_test_set.py --check` 验证路径。

## 4. 执行流程

```bash
bash run_all.sh          # 全流程（推荐）
```

或分步执行：

```bash
# 步骤1：构建划分（仓库已提交 train_labels.csv / test_labels.csv，无需重建）
python3 build_split.py --dry-run     # 确认无误后去掉 --dry-run

# 步骤2：多模态模型用 — 为 138 张测试图像生成描述（自动续跑，可中断）
python3 prepare_captions.py

# 步骤3：主评估（含逐折 + 5 折软投票集成 + 混淆矩阵 + 汇总）
python3 evaluate_test_set.py
# 只评估部分模型：python3 evaluate_test_set.py --models=resnet50,mmgf_resnet50
# 控制预测批次（GPU 显存不足时自动减半回退）：python3 evaluate_test_set.py --batch=4
```

> 关于 GPU 显存不足（OOM）：脚本每折预测后 `clear_session()` 释放前折模型，
> 预测批次默认 4，遇到 `ResourceExhaustedError` 时先清会话并重载该折模型、
> 再降为半批次重试（最小 1），批 1 仍失败则自动回退 CPU 计算，保证不中断。
> 若 GPU 被其它进程长期占满，可整程强制 CPU 跑：
> `CUDA_VISIBLE_DEVICES=-1 python3 evaluate_test_set.py --batch=1`

## 5. 数据集与划分来源

- 根目录 `images/`（或 `data/images/`）共 **1257 张**图像，14 个类别（类别 0–13）。
- `labels.csv` 覆盖 **1119 张** —— 训练池（训练脚本读取该表）。
- 剩余 **138 张**从未参与训练/验证 —— 独立测试集。

**测试集标签来源**：
- **102/138** 通过 MD5 内容匹配，在比赛原始数据（`data/train.csv` + `data/test.csv`，
  图像在 `data/data1/images/*.JPG`）中定位同一图像并自动标注类别。
  该方法在 1119 张已标注图像上核验：843 张匹配成功，且 **843/843 标签与
  labels.csv 一致**。
- **36/138** 在比赛数据中不存在，由人工标注（`test_labels.csv` 中
  `source` 列为空即人工标注）。

**类别编码（0–13，重要！）**：模型训练时 pandas 把全数字标签列读成 **int64**，
`LabelEncoder` 按**自然数字序**编码——**模型第 k 类 = 类别 k（0..13）**。
这与字符串字典序（'10' < '2'）不同；评估脚本已按自然序对齐（`encode_labels`）。

## 6. 覆盖的模型（与论文表 2、表 3、表 4 一致）

| 模型 | family | 每折模型目录（相对项目根） |
|---|---|---|
| ResNet50 | image_only | `models/train_resnet50/output_resnet50` |
| MobileNetV2 | image_only | `models/mobilenetv2/output3` |
| MobileNetV3Large | image_only | `models/train_mobilenetv3/output_mobilenetv3` |
| InceptionV3 | image_only | `models/train_inceptionv3/output_inceptionv3` |
| VGG16 | image_only | `models/train_vgg16/output_vgg16` |
| MMGF-Net (ResNet50) 本文 | multimodal | `models/train_resnet50_multimodal/output_resnet50_multimodal` |
| MMGF-Net (MobileNetV2) | multimodal | `models/mobilenetv2/output3` |
| MMGF-Net (MobileNetV3Large) | multimodal | `models/train_mobilenetv3_multimodal/output_mobilenetv3_multimodal` |
| MMGF-Net (InceptionV3) | multimodal | `models/train_inceptionv3_multimodal/output_inceptionv3_multimodal` |
| MMGF-Net (VGG16) | multimodal | `models/train_vgg16_multimodal/output_vgg16_multimodal` |
| 改进ResNet50[19] | image_only | `models/model_predictions/Improved_ResNet50` |
| CCNNet[15] | image_only | `models/model_predictions/CCNNet` |
| ConvNeXt[16] | image_only | `models/model_predictions/ConvNeXt` |
| SFE-CA[18] | image_only | `models/model_predictions/SFE-CA` |

> 对比方法（改进ResNet50 等）为**图像单模态复现**（论文表 2），其 `rel_dir` 指向
> `models/model_predictions/...`（config.py 中同时保留
> `rel_dir_alt=("papermodels/model_predictions/...",)` 作为回退，两处布局都可直接用）。
> 各对比方法的保存格式不同：改进ResNet50 / CCNNet 每折为 SavedModel 目录
> `fold_k/best_model`；SFE-CA 为 `fold_k/best_model.keras`（可直接 `load_model`）；
> ConvNeXt 为 `fold_k/best_model.keras`，但其 config 层名与权重层名不一致、
> 且 TF2.13 无法重建其图，需经 `convnext_loader.py` 结构化加载（见 §9.3）。
> `fold_model_path` 会自动回退搜索该折目录下的常见命名（含 `saved_model.pb` 的目录、
> 或 `*.keras`），通常无需手动改。

评估脚本自动按 `family` 区分输入通道数：`image_only` 单输入（仅图像）；
`multimodal` 双输入（图像 + 文本序列），并逐折用训练集重建分词器。

## 7. 结果输出（`<PROJECT_ROOT>/test_set_results/<model_key>/`）

每模型输出：

| 文件 | 内容 |
|---|---|
| `test_set_predictions.csv` | 138 张逐样本 真实标签 / 预测标签 / 置信度 |
| `test_set_per_fold_metrics.csv` | 5 折逐折 Accuracy / P / R / F1 / AUC |
| `test_set_ensemble_metrics.json` | 5 折软投票集成后的整体指标 |
| `test_set_report.txt` | 每折测试指标 + 集成指标 + 逐类别 F1 |
| `test_set_ensemble_confusion.png` | 集成模型在测试集上的混淆矩阵 |

全局汇总（`<PROJECT_ROOT>/test_set_results/`）：

| 文件 | 内容 |
|---|---|
| `all_models_test_summary.csv` | 全部模型（含每折均值 ± 标准差）测试集指标汇总 |
| `test_set_per_class_f1.csv` | 各模型在 138 张测试集上逐类别的 F1 |

**指标口径**：Accuracy、Macro Precision、Macro Recall、Macro F1、Macro AUC(ovr)。
论文表格的独立测试集列采用 **5 折模型预测均值**（per-fold mean）作为代表指标，
对应 `all_models_test_summary.csv` 中 `test_fold_acc_mean` / `test_fold_f1_mean`；
软投票集成结果列在 `test_ensemble_*` 列中，供对照。

## 8. 按需重新训练（论文结果无需重新训练）

```bash
python3 train_with_holdout_test.py --check        # 检查各模型 5 折是否齐全
python3 train_with_holdout_test.py --train-all    # 按需调用原训练脚本重训
```

**说明**：重新训练会重跑 5 折 CV（训练集 1119，每折内部 15% 验证），与论文协议完全
一致；训练完成后同一测试集 138 张即可再次评估。仓库已包含全部已训练模型的
每折结果，日常评估直接运行 `evaluate_test_set.py` 即可。

## 9. 结果分析

以下为全部 14 个模型在独立测试集上的评估结果
（`test_set_results/all_models_test_summary.csv`），与论文 5 折交叉验证结果对照。
每模型均给出 5 折逐折测试指标与软投票集成指标。

### 9.1 图像单模态骨干（论文表 3）

| 模型 | 每折测试 Acc（均值±std） | 软投票集成 Acc | 集成 Macro-F1 |
|---|---|---|---|
| ResNet50 | 91.16 ± 1.48 | 92.75 | 90.83 |
| MobileNetV2 | 90.58 ± 0.65 | 92.75 | 92.91 |
| MobileNetV3Large | 90.43 ± 1.55 | 92.75 | 92.96 |
| InceptionV3 | 87.97 ± 1.56 | 91.30 | 89.57 |
| VGG16 | 86.96 ± 1.21 | 89.86 | 87.83 |

- 14 类逐类别 F1 全部 ≥ 0.67（`test_set_per_class_f1.csv`），无类别坍缩；
  MobileNet 系与 ResNet50 集成后 Acc 均达 92.75%。
- 所有模型**集成 Acc 高于每折均值**（约 +1~3 个百分点），证明 5 折软投票
  集成的增益在独立测试集上同样成立。
- 独立测试集指标略低于论文 CV 指标（约 2~5 个百分点），属于训练阶段
  “从未见过”的外部数据上的正常泛化回落。

### 9.2 多模态 / 对比方法

- **MMGF-Net 5 个骨干变体**：在 138 张独立测试集上的评估结果见
  `all_models_test_summary.csv`（`test_set_results/`）。测试集描述复用
  训练用 `image_captions.json`，覆盖全部 1257 张。
- **复现对比方法（表 2，图像单模态）**：改进ResNet50 / CCNNet / ConvNeXt / SFE-CA
  每折测试 Acc 均值 = 89.42% / 16.67% / 87.83% / 88.26%（Macro-F1 =
  88.08% / 2.04% / 83.62% / 85.07%）。其中 CCNNet 复现模型的 CV 准确率仅
  ~16%（收敛异常，始终预测类别 7），为忠实复现结果，论文表 2 已如实报告。
- 论文表 2 中 MMGF-Net 每折测试 Acc 均值 94.93%、Macro-F1 94.17%，为对比方法中最高。

### 9.3 ConvNeXt 加载说明

TF2.13 下其 `best_model.keras` 无法经 `load_model` 重建（config 显式层名与
权重层名命名空间不一致）。`convnext_loader.py` 按“类 + 创建顺序 + 形状”从
checkpoint 权重结构化重建（与层名无关），加载后的模型为单输入（仅图像），
与其余对比方法一致。
部署时将 `convnext_loader.py` 与 `evaluate_test_set.py`、`config.py` 一并放入
服务器 `test_set_experiment/`。自检：`python3 convnext_loader.py
models/model_predictions/ConvNeXt/fold_1` → 应打印 `[ok] ConvNeXt 结构化加载成功`。

## 10. 与论文表格的对应

1. 表 2 / 表 3 / 表 4 / 表 6 的“独立测试集”列（Acc / Macro-F1）取自
   `all_models_test_summary.csv` 的 `test_fold_acc_mean` / `test_fold_f1_mean`。
2. 表 7 的逐类别 F1 为 5 折交叉验证结果（见主 README 表 7 映射）；
   `test_set_per_class_f1.csv` 给出 138 张独立测试集上的逐类别 F1，供分析泛化。
3. 论文实验设置中测试集的构建方式：138 张图像从未参与训练/验证，各模型直接
   加载已训练的 5 折模型进行预测，并报告逐折均值与软投票集成指标。

---

如需在本地 Windows 无 TF 环境预演数据管线，可运行 `build_split.py`、
`evaluate_test_set.py --check`（二者不依赖 TensorFlow）。
