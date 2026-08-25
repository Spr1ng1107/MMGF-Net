# -*- coding: utf-8 -*-
"""
===========================================================================
步骤 1 / 5：构建 1119 训练 + 138 独立测试 的固定划分
Build the fixed 1119-train / 138-hold-out-test split
===========================================================================
背景
----
整个数据集共 1257 张图像；其中 1119 张已完成 5 折分层交叉验证（只有训练与验证）。
本脚本将其余 138 张从未参与训练/验证的图像固定为“独立测试集”，
使所有模型都拥有一个训练阶段从未接触过的外部测试集。

输出
----
  train_labels.csv                   1119 行（= 原 labels.csv，训练池）
  test_labels.csv                    138 行（独立测试集；102 张自动回填 + 36 张人工标注）
  控制台报告：各类别数量分布、人工标注图片清单

测试集标签来源
--------------
  102/138 张可通过 MD5 内容匹配，在比赛原始数据（data/train.csv + data/test.csv，
  data/data1/images/*.JPG）中找到同一张图，从而自动回填类别标签。
  该映射已在 1119 张已标注图像上验证：843 张匹配成功，且 843/843 标签与 labels.csv 完全一致。
  其余 36 张在比赛数据中不存在，由人工标注（已录入 test_labels.csv，source 列为空）。

用法
----
  python build_split.py                # 本地或服务器运行
  python build_split.py --force        # 强制重新构建（覆盖已生成文件）
  python build_split.py --dry-run      # 只报告，不写文件
===========================================================================
"""
import os
import sys
import json
import csv
import hashlib
from collections import Counter

import config


# ═══════════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════════
def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_csv(rows, path, columns):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    print(f"[build_split] 已写出 {len(rows)} 行 -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════
def load_labels_csv(path):
    """加载标签 CSV，返回 {image_name(lower) : label}。"""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["image_name"].strip()
            label = row["Task_Chinese_medicinal_herb"].strip()
            out[name.lower()] = label
    return out


def list_root_images(images_dir):
    """列出 images_dir 下全部 jpg 图像文件名（lower）。"""
    files = []
    for fn in os.listdir(images_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            files.append(fn.lower())
    return sorted(files)


# ═══════════════════════════════════════════════════════════════════════════
# 通过比赛原始数据自动回填测试集标签（内容 MD5 匹配）
# ═══════════════════════════════════════════════════════════════════════════
def build_competition_label_map(train_csv, test_csv, images_dir):
    """
    返回 {data1 文件名 : 类别标签} 与 {文件 MD5 : [data1 文件名, ...]}。
    若比赛数据不存在则返回 (None, None)，脚本将退化为全部人工标注。
    """
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)
            and os.path.isdir(images_dir)):
        print("[build_split] 未找到比赛原始数据，无法自动回填测试集标签；"
              "将全部交由人工标注。")
        return None, None

    label_of = {}
    for csv_path in (train_csv, test_csv):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                name = row["image_name"].strip()
                label = row["Task_Chinese_medicinal_herb"].strip()
                label_of[name] = label

    md5_to_files = {}
    for fn in os.listdir(images_dir):
        p = os.path.join(images_dir, fn)
        if not os.path.isfile(p):
            continue
        h = _md5(p)
        md5_to_files.setdefault(h, []).append(fn)
    print(f"[build_split] 已索引比赛图像 {len(md5_to_files)} 个 MD5。")
    return label_of, md5_to_files


def auto_label_image(root_name, root_dir, label_of, md5_to_files):
    """
    对某张根目录图像，尝试在比赛数据中找到内容相同者并返回其标签。
    返回 (标签 or None, 匹配到的比赛文件名 or None)。
    """
    h = _md5(os.path.join(root_dir, root_name))
    cands = md5_to_files.get(h, [])
    for c in cands:
        if c in label_of:
            return label_of[c], c
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════
def main():
    force = "--force" in sys.argv
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("[build_split] --dry-run：仅报告，不写文件。")

    images_dir = config.IMAGES_DIR
    if not os.path.isdir(images_dir):
        print(f"[build_split] 错误：图像目录不存在 {images_dir}")
        print("  请通过环境变量 HERB_IMAGES_DIR 指定 1257 张图像所在目录，")
        print("  或在本脚本所在目录放置 images/ 并重试。")
        sys.exit(1)

    # 1) 已标注训练池（1119）
    train_label_map = load_labels_csv(config.LABELS_CSV)
    print(f"[build_split] labels.csv 训练池：{len(train_label_map)} 张")

    # ── 安全护栏 ─────────────────────────────────────────────────────────
    # labels.csv 现可能已包含独立测试集的人工标签（如 36 张待标注图被写回 labels.csv，
    # 行数 >1119）。此时重新划分会让“独立测试集”缩水（1257 - len(labels)），
    # 破坏论文设定的 138 张 held-out 设计。检测到已存在固定划分文件时，拒绝重建。
    if (len(train_label_map) != 1119 and not force
            and (os.path.exists(config.TRAIN_LABELS_CSV)
                 or os.path.exists(config.TEST_LABELS_CSV))):
        print("=" * 70)
        print(f"[build_split] 注意：labels.csv 现有 {len(train_label_map)} 行（≠1119），")
        print("  可能已将独立测试集的人工标签写入了 labels.csv。")
        print(f"  若按当前 labels.csv 重新划分，测试集将变为 "
              f"1257 - {len(train_label_map)} = {1257 - len(train_label_map)} 张，")
        print("  与论文设定的 138 张独立测试集不一致，且 36 张人工标注测试图会被误归入训练池。")
        print("  请保留现有 train_labels.csv(1119) + test_labels.csv(138)，直接运行评估。")
        print("  如确需按当前 labels.csv 重建划分，请加 --force 后重试。")
        print("=" * 70)
        sys.exit(1)

    # 2) 根目录全部图像（1257）
    all_images = list_root_images(images_dir)
    # 仅保留存在于 images_dir 中的已标注图像
    train_images = sorted(n for n in train_label_map if n in set(all_images))
    unused = sorted(set(all_images) - set(train_images))
    print(f"[build_split] 根目录图像 {len(all_images)} 张 | "
          f"训练池 {len(train_images)} 张 | 独立测试候选 {len(unused)} 张")

    # 3) 比赛数据索引（用于自动回填）
    label_of, md5_to_files = build_competition_label_map(
        config.COMPETITION_TRAIN_CSV, config.COMPETITION_TEST_CSV,
        config.COMPETITION_IMAGES_DIR)

    # 4) 自动回填测试集标签
    test_rows = []
    auto_hits = 0
    for name in unused:
        label, src = None, None
        if label_of and md5_to_files:
            label, src = auto_label_image(name, images_dir, label_of, md5_to_files)
            if label is not None:
                auto_hits += 1
        test_rows.append({
            "image_name": name,
            "Task_Chinese_medicinal_herb": label if label is not None else "",
            "auto_labeled": 1 if label is not None else 0,
            "source": src if src else "",
            "needs_manual_label": 0 if label is not None else 1,
        })

    print(f"[build_split] 自动回填测试集标签：{auto_hits}/{len(unused)} 张。")

    # 5) 报告
    print("\n===== 独立测试集（138）标签分布（自动回填部分） =====")
    dist = Counter(r["Task_Chinese_medicinal_herb"] for r in test_rows
                   if r["Task_Chinese_medicinal_herb"])
    for lbl in sorted(dist, key=lambda x: (len(x), x)):
        print(f"  类别 {lbl}: {dist[lbl]} 张")
    print(f"  合计已标注: {sum(dist.values())} | 需人工标注: "
          f"{sum(1 for r in test_rows if r['needs_manual_label'])}")

    manual = [r for r in test_rows if r["needs_manual_label"]]
    if manual:
        print("\n===== 需人工标注的测试集图像（36 张） =====")
        print("其类别已人工标注并录入 test_labels.csv（source 列为空即人工标注）；")
        print("本脚本重建划分时会从已有 test_labels.csv 保留这些标签。")
        for r in manual:
            print(f"  {r['image_name']}")

    if dry_run:
        return

    # 6) 写出训练池与测试集（重建时从已有 test_labels.csv 保留 36 张人工标签）
    _write_csv(
        [{"image_name": n, "Task_Chinese_medicinal_herb": train_label_map[n]}
         for n in train_images],
        config.TRAIN_LABELS_CSV,
        ["image_name", "Task_Chinese_medicinal_herb"],
    )
    existing_manual = {}
    if os.path.exists(config.TEST_LABELS_CSV):
        with open(config.TEST_LABELS_CSV, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if (r.get("auto_labeled", "").strip() == "0"
                        and r.get("Task_Chinese_medicinal_herb", "").strip()):
                    existing_manual[r["image_name"].strip().lower()] = (
                        r["Task_Chinese_medicinal_herb"].strip())
    for r in test_rows:
        key = r["image_name"].strip().lower()
        if key in existing_manual:
            r["Task_Chinese_medicinal_herb"] = existing_manual[key]
            r["auto_labeled"] = 0
            r["needs_manual_label"] = 0
    _write_csv(
        test_rows,
        config.TEST_LABELS_CSV,
        ["image_name", "Task_Chinese_medicinal_herb",
         "auto_labeled", "source", "needs_manual_label"],
    )

    print("\n[build_split] 完成。下一步：")
    print("  多模态模型还需运行：python prepare_captions.py")


if __name__ == "__main__":
    main()
