#!/usr/bin/env bash
# ===========================================================================
# 独立测试集实验 — 服务器一键编排脚本
# ===========================================================================
# 用法：
#   bash run_all.sh                     # 全流程（含多模态描述生成）
#   bash run_all.sh --no-captions       # 跳过多模态图像描述生成
#   bash run_all.sh --models=resnet50,vgg16   # 只评估部分模型（传给 evaluate_test_set.py）
#   bash run_all.sh --check             # 只做就绪检查，不运行
#
# 前置条件：见 README.md（python3 + tensorflow==2.13 + pandas + scikit-learn +
#           jieba + openai；DASHSCOPE_API_KEY 生成测试集描述用）
# ===========================================================================
set -u

PY=python3
NO_CAPTIONS=0
CHECK_ONLY=0
MODELS_ARG=""

for arg in "$@"; do
  case "$arg" in
    --no-captions) NO_CAPTIONS=1 ;;
    --check)       CHECK_ONLY=1 ;;
    --models=*)    MODELS_ARG="--models=${arg#--models=}" ;;
    *) echo "未知参数: $arg" ; exit 1 ;;
  esac
done

cd "$(dirname "$0")"

echo "================================================================"
echo " 独立测试集实验  (138 张 held-out / 14 类)"
echo "================================================================"

# ── 1. 划分文件 ──────────────────────────────────────────────────────────
if [ ! -f test_labels.csv ]; then
  echo "[1/5] 生成 1119/138 划分 ..."
  $PY build_split.py || exit 1
else
  echo "[1/5] 划分文件已存在，跳过 build_split.py（如需重建加 --force）"
fi

# ── 2. 就绪检查 ─────────────────────────────────────────────────────────
echo "[2/5] 就绪检查 ..."
$PY evaluate_test_set.py --check $MODELS_ARG || exit 1

if [ "$CHECK_ONLY" -eq 1 ]; then
  echo "仅检查，结束。"
  exit 0
fi

# ── 3. 多模态模型：为 138 张测试图像生成药典式描述 ──────────────────────
if [ "$NO_CAPTIONS" -eq 0 ]; then
  echo "[3/5] 生成测试集图像描述（多模态模型需要，自动续跑） ..."
  $PY prepare_captions.py || echo "  [警告] prepare_captions 未完全成功，可稍后重跑"
else
  echo "[3/5] 已跳过图像描述生成（--no-captions）"
fi

# ── 4. 评估 ──────────────────────────────────────────────────────────────
echo "[4/5] 在独立测试集上评估各模型（每折 + 5 折软投票集成） ..."
$PY evaluate_test_set.py $MODELS_ARG || exit 1

# ── 5. 结果完整性汇总 ──────────────────────────────────────────────────────
echo "================================================================"
echo " 结果完整性汇总"
echo "================================================================"
$PY - <<'PYEOF'
import os
import csv
import config

out = config.OUTPUT_DIR
done, missing = [], []
for m in config.MODELS:
    key = m["key"]
    report = os.path.join(out, key, "test_set_report.txt")
    if os.path.exists(report):
        done.append(m["paper"])
    else:
        missing.append(f"{m['paper']} ({key})")

reasons = {}
skipped_csv = os.path.join(out, "test_set_skipped.csv")
if os.path.exists(skipped_csv):
    try:
        with open(skipped_csv, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                reasons[r.get("model_key", "")] = r.get("skip_reason", "")
    except Exception:
        pass

print(f"  已完成 ({len(done)} 个):")
for p in done:
    print(f"    [OK]   {p}")
print(f"  未完成 ({len(missing)} 个):")
for p in missing:
    key = p.rsplit("(", 1)[-1].rstrip(")")
    r = reasons.get(key, "结果目录中缺少报告文件")
    print(f"    [MISS] {p:<28} 原因: {r}")

if missing:
    print("\n  完成建议：")
    print("    · 多模态模型：运行  python3 prepare_captions.py 生成测试集描述，")
    print("      并确认服务器存在训练用 text_slices/image_captions.json；")
    print("      随后重跑  python3 evaluate_test_set.py。")
print("\n================================================================")
PYEOF
echo " 全部完成。结果见 test_set_results/（含汇总 CSV 与控制台报告）"
echo "================================================================"
