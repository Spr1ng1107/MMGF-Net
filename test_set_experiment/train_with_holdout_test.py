# -*- coding: utf-8 -*-
"""
===========================================================================
训练完备性检查 / 重新训练
Verify fold-model completeness and (re)train missing models
===========================================================================
说明
----
独立测试集实验的科学基础：独立测试集 138 张图像从未参与原 5 折交叉验证的
训练与验证，因此“加载原实验已保存的各折模型 → 在其上评估”即可得到真正
独立的测试结果，**无需重新训练**。

本脚本同时提供：
  1. --check        检查各模型 5 折模型是否完整（默认动作）
  2. --train key    对缺失的模型调用其原始训练脚本重新训练（5 折 CV，
                    内部 15% 验证，与论文实验完全一致），训练完成后仍由
                    evaluate_test_set.py 在独立测试集上评估。

用法
----
  python train_with_holdout_test.py --check
  python train_with_holdout_test.py --train resnet50
  python train_with_holdout_test.py --train-all
===========================================================================
"""
import os
import sys
import subprocess

import config

# 各模型对应的原始训练脚本（相对 PROJECT_ROOT）。
# 若服务器上路径不同，请在此修改。
TRAINING_SCRIPTS = {
    "resnet50":   "models/train_resnet50/train_resnet50.py",
    "mobilenetv2": "models/mobilenetv2/baseline_model_strict.py",
    "mobilenetv3large": "models/train_mobilenetv3/train_mobilenetv3.py",
    "inceptionv3": "models/train_inceptionv3/train_inceptionv3.py",
    "vgg16":      "models/train_vgg16/train_vgg16.py",
    # 多模态（本文方法及骨干变体）
    "mmgf_resnet50":  "models/train_resnet50_multimodal/train_resnet50_multimodal.py",
    "mmgf_mobilenetv2": "models/mobilenetv2/MNV2_text.py",
    "mmgf_mobilenetv3large": "models/train_mobilenetv3_multimodal/train_mobilenetv3_multimodal.py",
    "mmgf_inceptionv3": "models/train_inceptionv3_multimodal/train_inceptionv3_multimodal.py",
    "mmgf_vgg16":  "models/train_vgg16_multimodal/train_vgg16_multimodal.py",
    # 对比方法（保存位置在 papermodels/model_predictions；若训练脚本输出路径不同，
    # 训练完成后请把各折 best_model 拷贝到模型注册表对应的 rel_dir/fold_k/）
    "improved_resnet50": "models/train_and_predict_5models.py",
    "ccnnet":   "models/train_and_predict_5models.py",
    "convnext": "models/train_and_predict_5models.py",
    "sfeca":    "models/train_and_predict_5models.py",
}


def missing_folds(m):
    miss = []
    for k in range(1, config.CV_FOLDS + 1):
        p = config.fold_model_path(m, k)
        if not os.path.exists(p):
            miss.append(k)
    return miss


def main():
    args = sys.argv[1:]
    do_check = "--check" in args or not any(a.startswith("--train") for a in args)
    train_keys = []
    if "--train-all" in args:
        train_keys = [m["key"] for m in config.MODELS]
    else:
        for a in args:
            if a.startswith("--train="):
                train_keys += [k.strip() for k in a.split("=", 1)[1].split(",") if k.strip()]

    # ── 检查 ──
    if do_check or train_keys:
        print("\n===== 训练完备性检查（各模型 5 折 SavedModel） =====")
        complete, incomplete = [], []
        for m in config.MODELS:
            miss = missing_folds(m)
            if miss:
                incomplete.append((m, miss))
                print(f"  [缺失] {m['key']:<20} 缺折: {miss}")
            else:
                complete.append(m)
                print(f"  [完整] {m['key']}")
        print(f"\n完整 {len(complete)} 个 | 缺失 {len(incomplete)} 个")
        if not do_check and not train_keys:
            pass

    # ── 训练缺失模型 ──
    if train_keys:
        for key in train_keys:
            if key not in config.MODEL_INDEX:
                print(f"[train] 未知模型: {key}")
                continue
            m = config.MODEL_INDEX[key]
            miss = missing_folds(m)
            if not miss:
                print(f"[train] {key} 5 折模型已完整，跳过训练。")
                continue
            script = TRAINING_SCRIPTS.get(key)
            if not script:
                print(f"[train] {key} 无对应的训练脚本登记（请手动训练）。")
                continue
            script_path = os.path.join(config.PROJECT_ROOT, script)
            if not os.path.exists(script_path):
                print(f"[train] 脚本不存在: {script_path}")
                continue
            print(f"[train] 运行: python {script}")
            ret = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path))
            if ret.returncode != 0:
                print(f"[train] {key} 训练失败（退出码 {ret.returncode}）。")
                continue
            print(f"[train] {key} 训练完成。若模型保存位置与注册表不符，请拷贝后重跑。")

    print("\n完成。下一步运行：python evaluate_test_set.py")


if __name__ == "__main__":
    main()
