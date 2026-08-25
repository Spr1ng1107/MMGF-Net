# -*- coding: utf-8 -*-
"""
使用阿里云百炼 Qwen 大模型为中药材饮片图片生成《中国药典》式描述。
输出 image_captions.json（覆盖全部 1257 张，多模态模型训练实际数据源）。
模型：qwen3.6-plus；鉴权：环境变量 DASHSCOPE_API_KEY。
"""

import os
import json
import base64
import time
from pathlib import Path
from openai import OpenAI

# 配置
IMAGES_DIR = "/home/shify_ug/multimodal_project/data/images"
OUTPUT_FILE = "image_captions.json"
# API Key 改为从环境变量读取（请先设置环境变量 DASHSCOPE_API_KEY）
API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化客户端
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def encode_image_to_base64(image_path):
    """将图片编码为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_description(image_path, image_name):
    """使用Qwen大模型生成图片描述"""
    try:
        # 将图片编码为base64
        base64_image = encode_image_to_base64(image_path)
        
        # 构建图片URL（使用base64格式）
        image_url = f"data:image/jpeg;base64,{base64_image}"
        
        # 调用API（模型已更新为 qwen3.6-plus）
        completion = client.chat.completions.create(
            model="qwen3.6-plus",   # ← 按新规范修改
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": """"你是一位精通《中国药典》的中药鉴定专家。请观察图像，按“形态描述+表面特征+断面特征+质地特征”顺序输出描述。字数50-80字，使用标准药典术语。严格输出JSON格式：{"文件名": "内容"}，不要包含任何额外字符。描述语言应为中文，且符合药材鉴定学的术语规范。仅输出以上最关键的纯汉字特征即可，无需输出多余符号，最终效果要求精炼、准确，便于编码"""
                    }
                ]
            }]
        )
        
        # 提取描述内容
        description = completion.choices[0].message.content
        
        print(f"✓ 成功生成描述: {image_name}")
        return description
        
    except Exception as e:
        print(f"✗ 生成描述失败 {image_name}: {str(e)}")
        return None

def process_images():
    """处理所有图片并生成描述"""
    print("=" * 60)
    print("开始生成中药材图片描述")
    print("=" * 60)
    
    # 获取所有图片文件
    images_path = Path(IMAGES_DIR)
    if not images_path.exists():
        print(f"错误: 图片目录不存在: {IMAGES_DIR}")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in images_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 加载已有的描述（如果存在）
    existing_captions = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing_captions = json.load(f)
        print(f"已加载 {len(existing_captions)} 条现有描述")
    
    # 处理每张图片
    captions = {}
    total = len(image_files)
    success_count = 0
    skip_count = 0
    
    for idx, image_file in enumerate(sorted(image_files), 1):
        image_name = image_file.name
        
        # 跳过已处理的图片
        if image_name in existing_captions:
            captions[image_name] = existing_captions[image_name]
            skip_count += 1
            print(f"[{idx}/{total}] 跳过已处理: {image_name}")
            continue
        
        print(f"[{idx}/{total}] 正在处理: {image_name}")
        
        # 生成描述
        description = generate_image_description(str(image_file), image_name)
        
        if description:
            captions[image_name] = description
            success_count += 1
            
            # 每处理10张图片保存一次
            if success_count % 10 == 0:
                save_captions(captions)
                print(f"已保存 {len(captions)} 条描述")
        else:
            captions[image_name] = "描述生成失败"
        
        # 添加延迟，避免API调用过快
        time.sleep(1)
    
    # 最终保存
    save_captions(captions)
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"总图片数: {total}")
    print(f"成功生成: {success_count}")
    print(f"跳过已处理: {skip_count}")
    print(f"失败数量: {total - success_count - skip_count}")
    print(f"描述文件: {OUTPUT_FILE}")

def save_captions(captions):
    """保存描述到JSON文件"""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 启动前请确保已设置环境变量 DASHSCOPE_API_KEY
    if not API_KEY:
        print("错误：请先设置环境变量 DASHSCOPE_API_KEY")
        print("例如在终端执行: export DASHSCOPE_API_KEY='sk-xxx'")
        exit(1)
    process_images()