# -*- coding: utf-8 -*-
"""
===========================================================================
步骤 3 / 5：在固定独立测试集（138 张）上评估所有已训练模型
Evaluate every trained model on the fixed hold-out test set (138 images)
===========================================================================
核心逻辑
--------
原 5 折分层交叉验证只在 1119 张图像上进行了训练与验证，从未接触过其余
138 张。因此这 138 张对全部已保存的各折模型而言都是“从未见过的外部数据”，
可作为独立测试集。

本脚本：
  1. 对每个模型（图像单模态 / 多模态门控融合 / 对比方法）加载其已保存的
     5 折模型（SavedModel）；
  2. 逐折在 138 张测试集上预测，计算每折测试指标（Acc / 宏P / 宏R / 宏F1 / 宏AUC）；
  3. 计算 5 折软投票集成（soft-voting）结果：集成准确率、宏F1、逐类别F1、混淆矩阵；
  4. 汇总所有模型 → all_models_test_summary.csv，便于直接填入论文表格。

多模态模型说明
--------------
  多模态模型（MMGF-Net 及其骨干变体）需同时输入“图像 + 文本描述”。本脚本
  按训练一致的方式重建每折分词器（同一确定性划分 + 同一分词器超参），
  并用 prepare_captions.py 生成的测试集描述进行编码，保证测试流程与训练一致。

用法
----
  python evaluate_test_set.py                          # 全部模型
  python evaluate_test_set.py --models resnet50,mmgf_resnet50
  python evaluate_test_set.py --no-plot
  python evaluate_test_set.py --check                  # 只做就绪检查
  python evaluate_test_set.py --batch=4                # 预测批次（OOM 时自动减半回退）

显存控制
--------
  每折预测后调用 clear_session() 释放前折模型的图/会话，避免跨折累计占用
  GPU 显存；预测批次遇到 ResourceExhaustedError 时，先 clear_session 并按原
  路径重载模型，再降为半批次重试（最小 1）；批 1 仍失败则回退 CPU 计算，
  保证评估不因 GPU 被占用而中断。默认起始批 4，可用 `--batch=N` 覆盖。
===========================================================================
"""
import os
import sys
import json
import re
import gc
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import config

# 预测批次起点（可用 --batch=N 覆盖；OOM 时自动清会话重载 + 逐次减半回退，
# 批 1 仍失败则回退 CPU）。默认 4：共享 GPU 上过大的批次会触发 OOM 并可能导致
# TF 进程崩溃；138 张图总耗时差异可忽略。
PREDICT_BATCH = 4


# ═══════════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════════════════════════════════
def load_df(csv_path, require_existing=True):
    """读取标签 CSV，拼接图像路径，返回 DataFrame。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["image_name"] = df["image_name"].astype(str).str.strip()
    # 归一化类别标签：pandas 会把含空值的纯数字列推断为 float64，
    # 使 "10" 变成 "10.0"（与训练 LabelEncoder 的 '10' 不匹配），统一去掉 ".0" 尾巴。
    # 例：'10.0' -> '10'；'0.0' -> '0'；空单元格(->'nan')不受影响，由调用方过滤。
    df["Task_Chinese_medicinal_herb"] = (
        df["Task_Chinese_medicinal_herb"].astype(str).str.strip()
        .str.replace(r"\.0+$", "", regex=True))
    df["image_path"] = df["image_name"].apply(
        lambda x: os.path.join(config.IMAGES_DIR, x))
    if require_existing:
        df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return df


def encode_labels(series):
    """
    类别编码：模型训练时 pandas 将全数字标签列读成 int64，
    LabelEncoder 按自然数字序编码（模型第 k 类 = 标签 k，0..13）。
    因此类别索引 = 标签整数值；直接用 int 编码，
    避免按字符串字典序（'0','1','10','11',...）导致类别错位。
    """
    return series.astype(int).values


def build_label_encoder(train_df):
    """
    返回 (le, class_names)。与训练一致的类别顺序 = 自然数字序 0..13。
    le 已不再使用（保留参数占位）。
    """
    labels = encode_labels(train_df["Task_Chinese_medicinal_herb"])
    n_classes = int(labels.max()) + 1
    class_names = [str(i) for i in range(n_classes)]
    return None, class_names


def load_test_df(le, allow_incomplete=False):
    """
    加载独立测试集。默认要求 138 张全部有标签；
    若 allow_incomplete=True，则仅用已标注子集（用于人工标注未完成时的预检）。
    """
    df = load_df(config.TEST_LABELS_CSV)
    missing = df[df["Task_Chinese_medicinal_herb"].isin(["", "nan"])]
    if len(missing) and not allow_incomplete:
        print("=" * 70)
        print(f"[evaluate] 独立测试集 {len(df)} 张中有 {len(missing)} 张缺少类别标签！")
        print("请先补全以下图像的标签（直接在 test_labels.csv 的相应行填写 "
              "Task_Chinese_medicinal_herb 列），再运行：")
        for _, r in missing.iterrows():
            print(f"    {r['image_name']}")
        print("=" * 70)
        sys.exit(1)
    df = df[~df["Task_Chinese_medicinal_herb"].isin(["", "nan"])].reset_index(drop=True)
    df["label_encoded"] = encode_labels(df["Task_Chinese_medicinal_herb"])
    return df


def load_captions(path, required_for):
    """加载描述文件；缺失时返回 None 并打印影响模型。"""
    if not os.path.exists(path):
        print(f"[evaluate] 描述文件不存在：{path}")
        print(f"          （多模态模型 {required_for} 需要它；请先运行 "
              f"python prepare_captions.py）")
        return None
    with open(path, "r", encoding="utf-8") as f:
        cap = json.load(f)
    return cap


# ═══════════════════════════════════════════════════════════════════════════
# 2. 确定性划分重建（用于多模态模型逐折分词器）
# ═══════════════════════════════════════════════════════════════════════════
def build_fold_splits(train_df):
    """
    严格复现训练时的划分：
      StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
      → 每折内 train_test_split(test_size=0.15, stratify, random_state=42)
    返回 [(train_fold_df, val_df), ...] 共 5 折。
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                          random_state=config.SEED)
    folds = []
    for train_val_idx, _ in skf.split(train_df, train_df["label_encoded"]):
        train_val_df = train_df.iloc[train_val_idx]
        train_fold_df, val_df = train_test_split(
            train_val_df, test_size=config.VAL_FRACTION,
            stratify=train_val_df["label_encoded"],
            random_state=config.SEED)
        folds.append((train_fold_df, val_df))
    return folds


def segment_text(text):
    import jieba
    return " ".join(jieba.cut(str(text)))


# ═══════════════════════════════════════════════════════════════════════════
# 3. 批处理推理（不用生成器，直接构造 NumPy 输入，138 张规模很小）
# ═══════════════════════════════════════════════════════════════════════════
def load_image_batch(df, image_size, preprocess_fn):
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    imgs = []
    for _, row in df.iterrows():
        img = load_img(row["image_path"], target_size=image_size)
        arr = img_to_array(img)
        imgs.append(preprocess_fn(arr))
    return np.array(imgs)


def encode_text_batch(df, tokenizer):
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = tokenizer.texts_to_sequences(df["caption_seg"].tolist())
    return pad_sequences(seqs, maxlen=config.MAX_TEXT_LEN)


# ═══════════════════════════════════════════════════════════════════════════
# 4. 指标
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred_probs, n_classes):
    from sklearn.metrics import (precision_recall_fscore_support,
                                 roc_auc_score)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = float(np.mean(y_true == y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_probs,
                            multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    per_class_f1 = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(n_classes)),
        zero_division=0)[2]
    return {
        "acc": acc, "macro_prec": precision, "macro_rec": recall,
        "macro_f1": f1, "macro_auc": auc,
        "y_pred": y_pred, "per_class_f1": per_class_f1,
    }


def fmt_metrics(m):
    return (f"Acc={m['acc']*100:.2f}%  MacroP={m['macro_prec']*100:.2f}%  "
            f"MacroR={m['macro_rec']*100:.2f}%  MacroF1={m['macro_f1']*100:.2f}%  "
            f"MacroAUC={m['macro_auc']*100:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════
# 4.5 TF 运行环境：显存控制与自适应批次
# ═══════════════════════════════════════════════════════════════════════════
def configure_tf():
    """设置 GPU 显存按需增长（不一次性占满），避免与其它进程争抢。"""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass


def predict_adaptive(model, pred_in, max_batch, reload_fn=None):
    """
    预测并自动规避 GPU 显存不足（OOM）：
    按 max_batch 起步，OOM 时先 clear_session() 并重载模型（避免在已损坏的
    会话上重试触发 CPU->GPU Memcpy 崩溃），再降为半批次重试（最小 1）；
    批 1 仍失败则回退 CPU，保证评估不被 GPU 占用情况卡死。
    reload_fn：无参可调用，用于按原路径重载模型（不传则只降批不重载）。
    """
    import tensorflow as tf

    def _reload():
        nonlocal model
        if reload_fn is not None:
            tf.keras.backend.clear_session()
            gc.collect()
            model = reload_fn()

    batch = int(max_batch)
    while batch >= 1:
        try:
            return model.predict(pred_in, batch_size=batch, verbose=0)
        except (tf.errors.ResourceExhaustedError,
                tf.errors.InternalError):
            if batch <= 1:
                break
            print(f"[evaluate] 预测批 {batch} 显存不足（OOM），"
                  f"清会话重载后降为 {max(1, batch // 2)} 重试 ...")
            _reload()
            batch = max(1, batch // 2)

    # 批 1 仍失败：清会话重载后回退 CPU（保证完成）
    print("[evaluate] GPU 显存不足（批 1 仍失败），回退 CPU 计算（较慢）...")
    _reload()
    try:
        with tf.device("/CPU:0"):
            return model.predict(pred_in, batch_size=1, verbose=0)
    except Exception as e:
        raise RuntimeError(f"GPU 与 CPU 预测均失败：{e}") from e


# ═══════════════════════════════════════════════════════════════════════════
# 5. 单模型评估
# ═══════════════════════════════════════════════════════════════════════════
def _skip_result(key, paper, family, reason):
    """构造“被跳过模型”的结果，写入 test_set_skipped.csv 供自诊断。"""
    return {"status": "skipped", "key": key, "paper": paper,
            "family": family, "reason": reason}


def _load_fold_model(m, model_path):
    """加载某模型第 fold 折模型。

    普通模型用 tf.keras.models.load_model()。
    ConvNeXt 例外：服务器 TF2.13 下其 best_model.keras 无法经 load_model 重建
    （TypeError: 'str' object is not callable，TF2.13 顶层图重建 bug），且该
    .keras 的 config（显式层名）与权重（自动层名）命名空间不一致，按名加载必失败。
    因此改用 convnext_loader：按“类 + 创建顺序 + 形状”从 checkpoint 权重结构化
    重建，与层名无关。加载后的模型为单输入（仅图像），与其余对比方法一致。
    """
    if m["key"] == "convnext":
        from convnext_loader import load_convnext_model
        fold_dir = os.path.dirname(model_path)
        model, _ = load_convnext_model(fold_dir, n_classes=14)
        return model
    custom_objects = {}
    try:
        from tensorflow.keras.applications.convnext import LayerScale as _LS
        custom_objects["LayerScale"] = _LS
    except Exception:
        pass
    return tf.keras.models.load_model(model_path, compile=False,
                                      custom_objects=custom_objects)


def evaluate_one_model(m, train_df, test_df, le, class_names, n_classes,
                       train_captions, test_captions, make_plot=True):
    import tensorflow as tf
    configure_tf()          # 显存按需增长，避免与其它进程/前序折模型争抢
    key = m["key"]
    out = config.resolve_test_set_output(m)
    os.makedirs(out, exist_ok=True)
    image_size = m["image_size"]
    preprocess_fn = config.get_preprocess_fn(m["preprocess"])

    if m["family"] == "multimodal" and (
            train_captions is None or "caption_seg" not in train_df.columns):
        reason = f"缺少训练用描述文件 {config.CAPTIONS_JSON}"
        print(f"[evaluate] {key}：{reason}，跳过。")
        return _skip_result(key, m["paper"], m["family"], reason)

    # 逐折重建（多模态需要逐折分词器；图像单模态用不到，仍计算以便通用）
    folds = build_fold_splits(train_df)
    if len(folds) != config.CV_FOLDS:
        print(f"[evaluate] 警告：{key} 折数异常 {len(folds)}")

    # 测试集文本（多模态）
    test_df_multi = None
    if m["family"] == "multimodal":
        if test_captions is None:
            reason = f"缺少测试集描述 {config.TEST_CAPTIONS_JSON}"
            print(f"[evaluate] 跳过 {key}：{reason}")
            return _skip_result(key, m["paper"], m["family"], reason)
        test_df_multi = test_df.copy()
        test_df_multi["caption_raw"] = test_df_multi["image_name"].map(test_captions)
        missing_cap = test_df_multi["caption_raw"].isna() | \
            (test_df_multi["caption_raw"].astype(str).str.strip() == "")
        if missing_cap.any():
            n_miss = int(missing_cap.sum())
            print(f"[evaluate] {key}：{n_miss} 张测试图像无描述，将按缺失处理。")
            # 缺失描述用空白占位（与训练中“描述生成失败”类似的处理）
            test_df_multi.loc[missing_cap, "caption_raw"] = ""
        test_df_multi["caption_seg"] = test_df_multi["caption_raw"].apply(segment_text)

    # 测试图像（一次性加载，避免逐折重复加载的浪费）
    test_imgs = load_image_batch(test_df, image_size, preprocess_fn)

    fold_probs = []
    fold_metrics = []
    used_folds = []

    for fold in range(1, config.CV_FOLDS + 1):
        model_path = config.fold_model_path(m, fold)
        if not os.path.exists(model_path):
            print(f"[evaluate] {key} 折 {fold}：模型不存在，跳过 -> {model_path}")
            continue
        try:
            model = _load_fold_model(m, model_path)
        except Exception as e:
            print(f"[evaluate] {key} 折 {fold}：加载失败，跳过 -> {e}")
            continue

        n_inputs = len(model.inputs)
        if n_inputs == 1:
            pred_in = test_imgs
        elif n_inputs >= 2:
            if test_df_multi is None:
                print(f"[evaluate] {key} 折 {fold}：模型含文本输入但无测试描述，跳过。")
                del model
                continue
            # 重建该折分词器（与训练完全一致：仅在该折训练集上 fit）
            train_fold_df, _ = folds[fold - 1]
            tok = tf.keras.preprocessing.text.Tokenizer(
                num_words=config.MAX_WORDS, oov_token=config.OOV_TOKEN)
            tok.fit_on_texts(train_fold_df["caption_seg"].tolist())
            X_text = encode_text_batch(test_df_multi, tok)
            pred_in = [test_imgs, X_text]
        else:
            print(f"[evaluate] {key} 折 {fold}：不支持 {n_inputs} 个输入的模型，跳过。")
            del model
            continue

        # OOM 时按原路径重载该折模型再降批重试（避免在损坏会话上重试崩溃）
        reload_fn = lambda: _load_fold_model(m, model_path)
        probs = predict_adaptive(model, pred_in, PREDICT_BATCH,
                                 reload_fn=reload_fn)
        del model
        # 释放该折模型的图/会话，防止跨折累计占用 GPU 显存导致后续折 OOM
        tf.keras.backend.clear_session()
        gc.collect()
        fold_probs.append(probs)
        used_folds.append(fold)

        mtr = compute_metrics(test_df["label_encoded"].values, probs, n_classes)
        fold_metrics.append(mtr)
        print(f"[evaluate] {key} 折 {fold}: {fmt_metrics(mtr)}")

    if not fold_probs:
        reason = "无可用折模型（fold_k/ 下未找到 SavedModel）"
        print(f"[evaluate] {key}：{reason}，跳过。")
        return _skip_result(key, m["paper"], m["family"], reason)

    # ── 每折测试指标汇总（均值±标准差）──
    accs = [mtr["acc"] for mtr in fold_metrics]
    f1s = [mtr["macro_f1"] for mtr in fold_metrics]
    aucs = [mtr["macro_auc"] for mtr in fold_metrics]
    precs = [mtr["macro_prec"] for mtr in fold_metrics]
    recs = [mtr["macro_rec"] for mtr in fold_metrics]

    # ── 5 折软投票集成 ──
    stack = np.stack(fold_probs, axis=0)          # [n_folds, n_test, n_classes]
    ens_probs = stack.mean(axis=0)
    ens = compute_metrics(test_df["label_encoded"].values, ens_probs, n_classes)

    print(f"[evaluate] {key} 独立测试集（{len(used_folds)}折）每折均值±标准差: "
          f"Acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}%  "
          f"MacroF1={np.mean(f1s)*100:.2f}±{np.std(f1s)*100:.2f}%")
    print(f"[evaluate] {key} 5折软投票集成: {fmt_metrics(ens)}")

    # ── 保存 ──
    save_model_results(m, key, out, test_df, le, class_names, n_classes,
                       used_folds, fold_probs, fold_metrics, ens, make_plot)
    return {
        "status": "ok", "key": key, "paper": m["paper"], "family": m["family"],
        "n_folds_used": len(used_folds),
        "fold_mean": {"acc": np.mean(accs), "macro_f1": np.mean(f1s),
                      "macro_auc": np.mean(aucs), "macro_prec": np.mean(precs),
                      "macro_rec": np.mean(recs)},
        "fold_std": {"acc": np.std(accs), "macro_f1": np.std(f1s),
                     "macro_auc": np.std(aucs), "macro_prec": np.std(precs),
                     "macro_rec": np.std(recs)},
        "ensemble": {"acc": ens["acc"], "macro_f1": ens["macro_f1"],
                     "macro_auc": ens["macro_auc"], "macro_prec": ens["macro_prec"],
                     "macro_rec": ens["macro_rec"]},
        "per_class_f1": ens["per_class_f1"],
    }


def save_model_results(m, key, out, test_df, le, class_names, n_classes,
                       used_folds, fold_probs, fold_metrics, ens, make_plot):
    y_true = test_df["label_encoded"].values
    stack = np.stack(fold_probs, axis=0)
    ens_probs = stack.mean(axis=0)
    ens_pred = np.argmax(ens_probs, axis=1)

    # 逐折预测表
    pred_cols = {"image_name": test_df["image_name"].values,
                 "true_label": test_df["Task_Chinese_medicinal_herb"].values,
                 "true_enc": y_true}
    for i, fold in enumerate(used_folds):
        pred_cols[f"fold{fold}_pred"] = np.argmax(fold_probs[i], axis=1)
    pred_cols["ensemble_pred"] = ens_pred
    pred_df = pd.DataFrame(pred_cols)
    pred_df.to_csv(os.path.join(out, "test_set_predictions.csv"),
                   index=False, encoding="utf-8-sig")

    # 每折指标表
    fold_rows = []
    for i, fold in enumerate(used_folds):
        mtr = fold_metrics[i]
        fold_rows.append({"fold": fold,
                          "acc": round(mtr["acc"], 4),
                          "macro_precision": round(mtr["macro_prec"], 4),
                          "macro_recall": round(mtr["macro_rec"], 4),
                          "macro_f1": round(mtr["macro_f1"], 4),
                          "macro_auc": round(mtr["macro_auc"], 4)})
    pd.DataFrame(fold_rows).to_csv(
        os.path.join(out, "test_set_per_fold_metrics.csv"),
        index=False, encoding="utf-8-sig")

    # 集成指标
    with open(os.path.join(out, "test_set_ensemble_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({k: float(ens[k]) for k in
                   ["acc", "macro_prec", "macro_rec", "macro_f1", "macro_auc"]},
                  f, indent=2, ensure_ascii=False)

    # 文本报告
    rep = [f"=== {m['paper']} — 独立测试集（{len(used_folds)}折软投票集成） ===",
           f"测试集规模: {len(test_df)} 张 | 类别数: {n_classes}",
           "", "--- 每折测试指标 ---"]
    for i, fold in enumerate(used_folds):
        rep.append(f"  Fold {fold}: {fmt_metrics(fold_metrics[i])}")
    rep.append("")
    rep.append(f"  每折均值: {fmt_metrics({k: float(np.mean([fm[k] for fm in fold_metrics])) for k in ['acc','macro_prec','macro_rec','macro_f1','macro_auc']})}")
    rep.append(f"  每折标准差: Acc={np.std([fm['acc'] for fm in fold_metrics])*100:.2f}%  "
               f"MacroF1={np.std([fm['macro_f1'] for fm in fold_metrics])*100:.2f}%")
    rep.append(f"  软投票集成: {fmt_metrics(ens)}")
    rep.append("")
    rep.append("--- 集成模型逐类别 F1 ---")
    for i, c in enumerate(class_names):
        rep.append(f"  类别 {c}: {ens['per_class_f1'][i]:.4f}")
    rep.append("")
    rep.append(f"模型目录: {config.model_dir(m)}")
    with open(os.path.join(out, "test_set_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))

    # 混淆矩阵
    if make_plot:
        plot_confusion(y_true, ens_pred, class_names,
                       os.path.join(out, "test_set_ensemble_confusion.png"),
                       m["paper"])


def plot_confusion(y_true, y_pred, class_names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{title} — 独立测试集混淆矩阵（{len(y_true)}张）")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] 混淆矩阵已保存 -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. 汇总所有模型
# ═══════════════════════════════════════════════════════════════════════════
def parse_existing_cv_acc(m):
    """从已有各折 classification_report 中读取 CV 准确率（原实验数值）。"""
    accs = []
    for fold in range(1, config.CV_FOLDS + 1):
        for cand in (os.path.join(config.model_dir(m), f"fold_{fold}",
                                  f"classification_report_fold{fold}.txt"),
                     os.path.join(config.model_dir(m), f"fold_{fold}",
                                  "classification_report.txt")):
            if os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        mch = re.search(r"accuracy\s+([\d.]+)", f.read())
                    if mch:
                        accs.append(float(mch.group(1)))
                except Exception:
                    pass
                break
    if not accs:
        return None
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs))}


def _read_csv_dict(path):
    """读取已有 CSV 为 {model_key: row_dict}；文件不存在返回 {}。"""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        return {str(r["model_key"]): r.to_dict() for _, r in df.iterrows()}
    except Exception:
        return {}


def write_global_summary(results, class_names):
    """
    增量合并汇总：只更新本轮评估（results）涉及的模型，其它模型的结果
    保留原样，便于分多次只跑未完成的模型而不会冲掉已完成的行。
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    summary_csv = os.path.join(config.OUTPUT_DIR, "all_models_test_summary.csv")
    pcf_csv = os.path.join(config.OUTPUT_DIR, "test_set_per_class_f1.csv")
    skipped_csv = os.path.join(config.OUTPUT_DIR, "test_set_skipped.csv")

    prev_summary = _read_csv_dict(summary_csv)
    prev_skipped = _read_csv_dict(skipped_csv)

    rows, skipped_rows = [], []
    for r in results:
        if r is None:
            continue
        key = r["key"]
        # 本轮涉及的模型：从历史中移除（本轮的"跳过"或"结果"覆盖历史）
        prev_summary.pop(key, None)
        prev_skipped.pop(key, None)
        if r.get("status") == "skipped":
            skipped_rows.append({
                "model_key": key, "model": r["paper"],
                "family": r["family"], "skip_reason": r["reason"],
            })
            continue
        rows.append({
            "model_key": key,
            "model": r["paper"],
            "family": r["family"],
            "n_folds_used": r["n_folds_used"],
            "test_fold_acc_mean": round(r["fold_mean"]["acc"] * 100, 2),
            "test_fold_acc_std": round(r["fold_std"]["acc"] * 100, 2),
            "test_fold_f1_mean": round(r["fold_mean"]["macro_f1"] * 100, 2),
            "test_fold_f1_std": round(r["fold_std"]["macro_f1"] * 100, 2),
            "test_fold_auc_mean": round(r["fold_mean"]["macro_auc"] * 100, 2),
            "test_ensemble_acc": round(r["ensemble"]["acc"] * 100, 2),
            "test_ensemble_macroP": round(r["ensemble"]["macro_prec"] * 100, 2),
            "test_ensemble_macroR": round(r["ensemble"]["macro_rec"] * 100, 2),
            "test_ensemble_macroF1": round(r["ensemble"]["macro_f1"] * 100, 2),
            "test_ensemble_auc": round(r["ensemble"]["macro_auc"] * 100, 2),
        })

    # 历史行（未在本轮涉及）保持原顺序在前，本轮新结果追加在后
    merged = list(prev_summary.values()) + rows
    pd.DataFrame(merged).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # 逐类别 F1（对应论文逐类别 F1 对比），同样增量合并
    prev_pcf = _read_csv_dict(pcf_csv)
    pcf_rows = []
    for r in results:
        if r is None or r.get("status") == "skipped":
            continue
        key = r["key"]
        prev_pcf.pop(key, None)
        row = {"model_key": key, "model": r["paper"]}
        for i, c in enumerate(class_names):
            row[f"class_{c}"] = round(float(r["per_class_f1"][i]), 4)
        pcf_rows.append(row)
    pd.DataFrame(list(prev_pcf.values()) + pcf_rows).to_csv(
        pcf_csv, index=False, encoding="utf-8-sig")

    # 跳过列表同样增量合并
    skipped = list(prev_skipped.values()) + skipped_rows
    if skipped:
        pd.DataFrame(skipped).to_csv(skipped_csv, index=False, encoding="utf-8-sig")
    elif os.path.exists(skipped_csv):
        os.remove(skipped_csv)

    print("\n" + "=" * 70)
    print("全部模型独立测试集结果汇总（增量合并）")
    print("=" * 70)
    if rows:
        for row in rows:
            print(f"  {row['model']:<24} 集成Acc={row['test_ensemble_acc']:.2f}%  "
                  f"集成F1={row['test_ensemble_macroF1']:.2f}%  "
                  f"(每折Acc均值={row['test_fold_acc_mean']:.2f}%±"
                  f"{row['test_fold_acc_std']:.2f}%)")
    if skipped_rows:
        print(f"\n  以下 {len(skipped_rows)} 个模型本轮未产出结果（跳过）：")
        for s in skipped_rows:
            print(f"  [跳过] {s['model']:<24} 原因: {s['skip_reason']}")
    print(f"\n汇总表已保存 -> {summary_csv}（共 {len(merged)} 个模型）")
    print(f"逐类别F1已保存 -> {pcf_csv}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. 就绪检查
# ═══════════════════════════════════════════════════════════════════════════
def check_readiness(train_df, test_df, test_captions, selected_models):
    print("\n===== 就绪检查 =====")
    print(f"  训练池(train_labels.csv): {len(train_df)} 张")
    print(f"  独立测试集(test_labels.csv): {len(test_df)} 张 "
          f"(其中未标注 {int(test_df['Task_Chinese_medicinal_herb'].isin(['', 'nan']).sum())} 张)")
    multi = [m["key"] for m in selected_models if m["family"] == "multimodal"]
    if multi:
        if test_captions is None:
            print(f"  多模态模型 {multi}：缺 test_captions.json，"
                  f"请运行 python prepare_captions.py")
        else:
            n_cap = sum(1 for _, r in test_df.iterrows()
                        if r["image_name"] in test_captions and
                        str(test_captions[r["image_name"]]).strip())
            print(f"  测试集描述覆盖: {n_cap}/{len(test_df)}（多模态模型需要）")
    print("===========================\n")


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════
def main():
    global PREDICT_BATCH
    args = [a for a in sys.argv[1:]]
    make_plot = "--no-plot" not in args
    check_only = "--check" in args
    for a in args:
        if a.startswith("--batch="):
            PREDICT_BATCH = max(1, int(a.split("=", 1)[1]))

    # 选择模型
    sel_keys = None
    for a in args:
        if a.startswith("--models="):
            sel_keys = [k.strip() for k in a.split("=", 1)[1].split(",") if k.strip()]
    if sel_keys is not None:
        unknown = [k for k in sel_keys if k not in config.MODEL_INDEX]
        if unknown:
            print(f"[evaluate] 未知模型: {unknown}")
            print(f"  可选: {list(config.MODEL_INDEX.keys())}")
            sys.exit(1)
        selected = [config.MODEL_INDEX[k] for k in sel_keys]
    else:
        selected = list(config.MODELS)
        print(f"[evaluate] 默认评估全部 {len(selected)} 个模型")

    # 数据
    train_df = load_df(config.TRAIN_LABELS_CSV)

    # 描述
    train_captions = None
    test_captions = None
    multi_needed = any(m["family"] == "multimodal" for m in selected)
    if multi_needed:
        train_captions = load_captions(config.CAPTIONS_JSON,
                                       [m["key"] for m in selected
                                        if m["family"] == "multimodal"])
        test_captions = load_captions(config.TEST_CAPTIONS_JSON,
                                      [m["key"] for m in selected
                                       if m["family"] == "multimodal"])
        if train_captions is not None:
            # 与训练 pipeline 一致：按 captions.json 过滤，并为每折重建分词器
            train_df["caption_raw"] = train_df["image_name"].map(train_captions)
            train_df = train_df.dropna(subset=["caption_raw"]).reset_index(drop=True)
            train_df["caption_seg"] = train_df["caption_raw"].apply(segment_text)

    le, class_names = build_label_encoder(train_df)
    n_classes = len(class_names)
    train_df["label_encoded"] = encode_labels(train_df["Task_Chinese_medicinal_herb"])

    test_df = load_df(config.TEST_LABELS_CSV, require_existing=True)
    missing = test_df[test_df["Task_Chinese_medicinal_herb"].isin(["", "nan"])]
    test_df_ready = test_df[~test_df["Task_Chinese_medicinal_herb"].isin(["", "nan"])] \
        .copy()
    test_df_ready["label_encoded"] = encode_labels(test_df_ready["Task_Chinese_medicinal_herb"])

    # 训练用描述文件（image_captions.json）覆盖全部 1257 张（含 138 张测试图），
    # 若专门的 test_captions.json 缺失，直接复用训练描述作为测试描述。
    if multi_needed and test_captions is None and train_captions is not None:
        n_test_cov = sum(1 for _, r in test_df_ready.iterrows()
                         if r["image_name"] in train_captions and
                         str(train_captions[r["image_name"]]).strip())
        if n_test_cov == len(test_df_ready):
            print(f"[evaluate] 未找到测试集描述 {config.TEST_CAPTIONS_JSON}；"
                  f"训练用描述 {config.CAPTIONS_JSON} 覆盖全部 {n_test_cov} 张"
                  f"测试图，直接复用。")
            test_captions = train_captions
        else:
            print(f"[evaluate] 训练用描述仅覆盖 {n_test_cov}/{len(test_df_ready)}"
                  f" 张测试图，不足以作为测试描述，多模态模型仍将跳过。")

    # 多模态模型在缺训练/测试描述时给出明确提示（evaluate_one_model 内部会跳过）
    if multi_needed and train_captions is None:
        print("[evaluate] 训练用描述文件缺失，无法重建多模态分词器，"
              "多模态模型将跳过。")

    check_readiness(train_df, test_df, test_captions, selected)
    if check_only:
        return

    if len(missing):
        print(f"[evaluate] 测试集仍有 {len(missing)} 张未标注，"
              f"将仅评估已标注的 {len(test_df_ready)} 张。")
    test_df_eval = test_df_ready

    results = []
    for m in selected:
        print("\n" + "=" * 70)
        print(f"[evaluate] {m['paper']} ({m['key']}) — family={m['family']}")
        print("=" * 70)
        r = evaluate_one_model(m, train_df, test_df_eval, le, class_names,
                               n_classes, train_captions, test_captions,
                               make_plot=make_plot)
        results.append(r)

    # 汇总（逐类别 F1 需基于全部 14 类）
    write_global_summary(results, class_names)


if __name__ == "__main__":
    main()
