# -*- coding: utf-8 -*-
"""
===========================================================================
ConvNeXt 折模型专用加载器 —— 结构化权重指派
Structural checkpoint-weight loader for the ConvNeXt comparison model
===========================================================================
背景
----
服务器 TF2.13 下，ConvNeXt 每折的 best_model.keras 无法用
`tf.keras.models.load_model()` 加载（`TypeError: 'str' object is not callable`，
TF2.13 顶层图重建 bug）。诊断发现该 .keras 是“frankenstein”文件：
  * config.json 用显式层名（convnext_tiny_stage_0_block_0_...）
  * 权重（model.weights.h5 / best_model.weights.h5）用自动层名（dense_0、layer_normalization_2、…）
两个命名空间来自两次不同的模型 build，因此“按名匹配”的 load_weights 必然失败。

但二者是**同一套架构**：config 155 层 = 权重 checkpoint 155 个层分组，拓扑、形状
逐一对应（stem=Sequential[Conv2D(4×4,3→96),LN]、18 个 ConvNeXtBlock、
3 个下采样 Sequential、final LN、GAP、BN、Dense(512,swish)、Dense(14,softmax)）。

方法
----
本模块不依赖层名，按“类 + 创建顺序 + 形状”把权重 checkpoint 的每个层分组指派到
重建模型的对应层：
  * checkpoint 自动名后缀即创建顺序（dense_0 < dense_2 < …；layer_normalization < _2 < …）
  * 重建模型按与训练脚本完全一致的构建方式生成（ConvNeXtTiny + 头），其
    model.layers 顺序同样是拓扑/创建顺序
  * 两者逐类 zip 配对，即可得到一一对应，与层名无关

用法
----
  from convnext_loader import load_convnext_model
  model = load_convnext_model("/path/to/fold_1", n_classes=14)

自检（本地/服务器均可跑）：
  python convnext_loader.py <fold_dir> [--weights <w.h5>]
    -> 打印逐类消费统计；全部 155 组消费、零残留、零缺层即成功。
===========================================================================
"""
import os
import re
import sys

import numpy as np


# 类名 → checkpoint 分组名前缀（含 Sequential 子层内的自动名）
_PREFIX_BY_CLASS = {
    "Dense": "dense",
    "Conv2D": "conv2d",
    "LayerNormalization": "layer_normalization",
    "LayerScale": "layer_scale",
    "BatchNormalization": "batch_normalization",
}


def _class_prefix(layer):
    """返回某层对应的 checkpoint 分组名前缀；无权重层返回 None。"""
    return _PREFIX_BY_CLASS.get(type(layer).__name__)


def _ordered_groups(cd, prefix):
    """checkpoint 分组中属于该前缀的组，按创建顺序（数字后缀升序）排列。"""
    def num(n):
        if n == prefix:
            return 0
        rest = n[len(prefix) + 1:]
        return int(rest) if rest.isdigit() else -1
    names = [n for n in cd.keys()
             if n == prefix or (n.startswith(prefix + "_") and num(n) >= 0)]
    names.sort(key=lambda n: (num(n), n))
    return names


def _read_vars(group):
    """读取一个层分组的 vars（按编号 0,1,2… 顺序），返回 numpy 数组列表。"""
    vs = group["vars"]
    keys = sorted(vs.keys(), key=int)
    return [np.array(vs[k]) for k in keys]


def _assign_seq_sub(seq_layer, group):
    """把 Sequential 容器的嵌套权重分给其子层（按子层类名在嵌套分组里查找）。"""
    subs = group["_layer_checkpoint_dependencies"]
    for sub in seq_layer.layers:
        pref = _class_prefix(sub)
        if pref is None or pref not in subs:
            continue
        sg = subs[pref]
        if "vars" in sg:
            sub.set_weights(_read_vars(sg))


def load_convnext_model(fold_dir, n_classes=14):
    """
    重建 ConvNeXt 模型（与 train_and_predict_5models.py 的 build 完全一致）并从
    checkpoint 权重结构化加载。返回 (model, stats)。
      fold_dir  某折目录（含 best_model.weights.h5 或 best_model.keras）
      n_classes 分类头输出维度（14）
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    # ── 1) 定位权重文件 ──
    wpath = os.path.join(fold_dir, "best_model.weights.h5")
    tmp = None
    if not os.path.exists(wpath):
        kpath = os.path.join(fold_dir, "best_model.keras")
        if not os.path.exists(kpath):
            raise FileNotFoundError(
                f"{fold_dir} 下既无 best_model.weights.h5 也无 best_model.keras")
        import zipfile
        tmp = os.path.join(fold_dir, "_convnext_extracted_weights.h5")
        with zipfile.ZipFile(kpath) as z:
            with open(tmp, "wb") as f:
                f.write(z.read("model.weights.h5"))
        wpath = tmp

    # ── 2) 重建架构（与训练脚本一致）──
    # 训练用 tensorflow.keras.applications.ConvNeXtTiny(weights='imagenet')；
    # 此处 weights=None 即可（权重随后指派），避免联网下载。
    try:
        from tensorflow.keras.applications.convnext import ConvNeXtTiny
    except Exception as e:
        raise RuntimeError(f"服务器上无法导入 ConvNeXtTiny：{e}")
    base = ConvNeXtTiny(weights=None, include_top=False,
                        input_shape=(224, 224, 3))
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="swish")(x)
    x = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(base.input, x)

    # ── 3) 结构化权重指派 ──
    with h5py_open(wpath) as f:
        cd = f["_layer_checkpoint_dependencies"]

        from tensorflow.keras import Sequential
        pools = {pref: _ordered_groups(cd, pref) for pref in _PREFIX_BY_CLASS.values()}
        seq_pool = _ordered_groups(cd, "sequential")

        stats = {pref: 0 for pref in pools}
        stats["sequential"] = 0
        for layer in model.layers:
            if isinstance(layer, Sequential):
                if not seq_pool:
                    raise RuntimeError(
                        f"权重不足：模型 Sequential 层 '{layer.name}' 无对应分组")
                gname = seq_pool.pop(0)
                _assign_seq_sub(layer, cd[gname])
                stats["sequential"] += 1
                continue
            pref = _class_prefix(layer)
            if pref is None:
                continue  # Normalization / GAP / Activation / TFOpLambda / InputLayer
            if not pools[pref]:
                raise RuntimeError(
                    f"权重不足：模型 {type(layer).__name__} 层 '{layer.name}' 无对应分组")
            gname = pools[pref].pop(0)
            layer.set_weights(_read_vars(cd[gname]))
            stats[pref] += 1

        # ── 4) 校验：任何未消费分组即架构不一致 ──
        leftover = {pref: pools[pref] for pref in pools if pools[pref]}
        if leftover:
            raise RuntimeError(
                f"checkpoint 有未消费分组（架构不匹配）: "
                + ", ".join(f"{p}={len(v)}" for p, v in leftover.items()))
        if seq_pool:
            raise RuntimeError(f"Sequential 分组未消费: {seq_pool}")

    if tmp and os.path.exists(tmp):
        os.remove(tmp)

    return model, stats


def h5py_open(path):
    """惰性导入 h5py（仅在有权重文件时使用）。"""
    import h5py
    return h5py.File(path, "r")


# ═══════════════════════════════════════════════════════════════════════════
# 自检入口
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = [a for a in sys.argv[1:]]
    if not args:
        print(__doc__)
        sys.exit(1)
    fold_dir = args[0]
    if not os.path.isdir(fold_dir):
        print(f"目录不存在: {fold_dir}")
        sys.exit(1)
    model, stats = load_convnext_model(fold_dir)
    print("[ok] ConvNeXt 结构化加载成功")
    print(f"    输入: {model.input_shape}  输出: {model.output_shape}")
    print(f"    参数量: {model.count_params():,}")
    print("    逐类消费统计:")
    for pref, n in stats.items():
        print(f"      {pref:22s} {n}")
    # 冒烟测试：2 张随机图前向
    import tensorflow as tf
    from tensorflow.keras.applications.convnext import preprocess_input
    dummy = preprocess_input(np.random.randint(0, 255, (2, 224, 224, 3), dtype=np.uint8))
    pred = model.predict(dummy, verbose=0)
    print(f"    冒烟前向: {pred.shape}, 各行 argmax = {pred.argmax(axis=1).tolist()}")


if __name__ == "__main__":
    main()
