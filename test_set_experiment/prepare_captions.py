# -*- coding: utf-8 -*-
"""
===========================================================================
步骤 2 / 5：为独立测试集（138 张）生成药典式图像描述
Generate Yaodian-style image captions for the hold-out test set
===========================================================================
多模态模型（MMGF-Net 及其骨干变体）在推理时同样需要“图像 + 文本描述”
两个输入。本文方法的文本分支输入为对每张饮片图像生成的《中国药典》式描述
（由阿里云百炼 Qwen 视觉大模型逐图生成）。本脚本沿用 generate_herb_captions.py
的完全相同提示词与调用方式，为 138 张独立测试图像生成描述，保存为
test_captions.json。

依赖
----
  pip install openai
  环境变量 DASHSCOPE_API_KEY（阿里云百炼）

用法
----
  export DASHSCOPE_API_KEY='sk-xxx'
  python prepare_captions.py                 # 逐张生成，可随时中断（自动续跑）
  python prepare_captions.py --check-only    # 仅检查缺漏，不调用 API

输出
----
  test_captions.json   { "1002.jpg": "<原始LLM输出>" , ... } 共 138 键
===========================================================================
"""
import os
import sys
import json
import base64
import time

import config


IMAGES_DIR = config.IMAGES_DIR
OUTPUT_FILE = config.TEST_CAPTIONS_JSON
EXISTING_CAPTIONS = config.CAPTIONS_JSON       # 服务器上训练用的完整描述文件
API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_MODEL = os.getenv("HERB_QWEN_MODEL", "qwen3.6-plus")

# 与 generate_herb_captions.py 完全一致的提示词
CAPTION_PROMPT = (
    "你是一位精通《中国药典》的中药鉴定专家。请观察图像，按"
    "“形态描述+表面特征+断面特征+质地特征”顺序输出描述。字数50-80字，"
    "使用标准药典术语。严格输出JSON格式：{\"文件名\": \"内容\"}，"
    "不要包含任何额外字符。描述语言应为中文，且符合药材鉴定学的术语规范。"
    "仅输出以上最关键的纯汉字特征即可，无需输出多余符号，"
    "最终效果要求精炼、准确，便于编码"
)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_description(client, image_path):
    base64_image = encode_image_to_base64(image_path)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    completion = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }],
    )
    return completion.choices[0].message.content


def main():
    check_only = "--check-only" in sys.argv

    # ── 需要生成描述的图像（138 张测试集）──
    import csv
    if not os.path.exists(config.TEST_LABELS_CSV):
        print(f"[prepare_captions] 未找到 {config.TEST_LABELS_CSV}，"
              f"请先运行 python build_split.py")
        sys.exit(1)
    with open(config.TEST_LABELS_CSV, newline="", encoding="utf-8-sig") as f:
        test_images = [r["image_name"].strip() for r in csv.DictReader(f)]
    print(f"[prepare_captions] 测试集图像：{len(test_images)} 张")

    # ── 已存在描述（服务器完整文件 + 已生成的测试集文件）──
    existing = {}
    if os.path.exists(EXISTING_CAPTIONS):
        try:
            with open(EXISTING_CAPTIONS, "r", encoding="utf-8") as f:
                existing.update(json.load(f))
            print(f"[prepare_captions] 已加载训练用描述文件 {EXISTING_CAPTIONS} "
                  f"({len(existing)} 条)")
        except Exception as e:
            print(f"[prepare_captions] 读取 {EXISTING_CAPTIONS} 失败：{e}")
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing.update(json.load(f))
            print(f"[prepare_captions] 已加载已有测试集描述 {OUTPUT_FILE}")
        except Exception as e:
            print(f"[prepare_captions] 读取 {OUTPUT_FILE} 失败：{e}")

    todo = [n for n in test_images
            if n not in existing or not str(existing[n]).strip()
            or str(existing[n]).strip() == "描述生成失败"]
    print(f"[prepare_captions] 已覆盖 {len(test_images) - len(todo)} 张；"
          f"需生成 {len(todo)} 张")

    if check_only:
        return

    if not API_KEY:
        print("[prepare_captions] 错误：请设置环境变量 DASHSCOPE_API_KEY")
        print("  export DASHSCOPE_API_KEY='sk-xxx'")
        sys.exit(1)

    if not todo:
        # 全部已覆盖（来源通常就是训练用 image_captions.json，其键覆盖 1257 张，
        # 含 138 张测试图）：把 138 张测试图对应的描述写入 test_captions.json，
        # 供 evaluate_test_set.py 使用。
        subset = {n: existing.get(n, "") for n in test_images}
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        covered = [n for n in subset
                   if str(subset[n]).strip()
                   and str(subset[n]).strip() != "描述生成失败"]
        print(f"[prepare_captions] 全部已覆盖，写出测试集描述 "
              f"{len(covered)}/138 → {OUTPUT_FILE}")
        return

    # ── 初始化客户端并逐张生成（自动续跑）──
    try:
        from openai import OpenAI
    except ImportError:
        print("[prepare_captions] 缺少 openai，请 pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    for i, name in enumerate(todo, 1):
        img_path = os.path.join(IMAGES_DIR, name)
        if not os.path.exists(img_path):
            print(f"[prepare_captions] 缺图，跳过：{name} ({img_path})")
            existing[name] = ""
            continue
        try:
            desc = generate_description(client, img_path)
            if not desc:
                desc = "描述生成失败"
            existing[name] = desc
            print(f"[prepare_captions] ({i}/{len(todo)}) OK {name}")
        except Exception as e:
            print(f"[prepare_captions] ({i}/{len(todo)}) 失败 {name}: {e}")
            existing[name] = "描述生成失败"
        # 每 10 张写盘一次
        if i % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    covered = [n for n in test_images
               if n in existing and str(existing[n]).strip()
               and str(existing[n]).strip() != "描述生成失败"]
    print(f"\n[prepare_captions] 完成。测试集描述覆盖 {len(covered)}/138 → {OUTPUT_FILE}")
    if len(covered) < 138:
        print("  请补全后重新运行本脚本（自动续跑缺失项）。")


if __name__ == "__main__":
    main()
