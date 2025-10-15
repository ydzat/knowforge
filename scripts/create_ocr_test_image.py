#!/usr/bin/env python3
"""
创建一个用于OCR测试的图片
"""
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
project_dir = os.path.dirname(script_dir)
# 图片保存路径
save_dir = os.path.join(project_dir, "input", "images")
os.makedirs(save_dir, exist_ok=True)

def create_test_image():
    # 创建一个白色背景图片
    width, height = 800, 400
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    # 尝试加载一个系统字体
    try:
        # 尝试加载不同的系统字体
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 28)
                break
        
        if font is None:
            # 如果没有找到TrueType字体，使用默认字体
            font = ImageFont.load_default()
            print("使用默认字体")
        else:
            print(f"使用字体: {path}")
        
    except Exception as e:
        print(f"无法加载字体: {e}")
        font = ImageFont.load_default()

    # 中英文混合文本
    lines = [
        "KnowForge 项目 OCR 测试图片",
        "This is a test image for OCR functionality.",
        "混合中英文文本识别测试 Mixed Text Recognition Test",
        "数字和特殊符号: 123456 !@#$%^&*()",
        "公式示例: E=mc² and x² + y² = r²"
    ]

    # 在图片上绘制文本
    y_position = 50
    for line in lines:
        draw.text((50, y_position), line, fill="black", font=font)
        y_position += 60

    # 保存图片
    save_path = os.path.join(save_dir, "ocr_test_sample.png")
    image.save(save_path)
    print(f"测试图片已保存到: {save_path}")
    return save_path

if __name__ == "__main__":
    create_test_image()