#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR与LLM集成测试脚本

该脚本用于测试OCR识别与LLM增强的集成效果，使用实际API密钥进行真实测试。
"""
import os
import sys
import time
import cv2
import argparse
import numpy as np

# 将项目根目录添加到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.config_loader import ConfigLoader
from src.utils.locale_manager import LocaleManager

# 在导入InputHandler之前，添加全局的get_text和format_text函数
# 这是为了解决LocaleManager类中方法名与InputHandler期望的方法名不匹配的问题
def get_text(key, default=""):
    """桥接函数，将调用转发到locale_manager.get"""
    if hasattr(get_text, "locale_manager"):
        return get_text.locale_manager.get(key, default)
    return default

def format_text(key, **kwargs):
    """桥接函数，将调用转发到locale_manager.format"""
    if hasattr(format_text, "locale_manager"):
        return format_text.locale_manager.format(key, kwargs)
    return key

# 替换全局模块中的函数
import src.note_generator.input_handler as input_handler_module
input_handler_module.get_text = get_text
input_handler_module.format_text = format_text

# 现在可以安全地导入InputHandler
from src.note_generator.input_handler import InputHandler


def setup_environment():
    """准备测试环境"""
    print("\n=== 准备测试环境 ===")
    
    # 检查API密钥配置
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("警告: 未找到DEEPSEEK_API_KEY环境变量，LLM增强功能将不可用")
        print("请设置环境变量: export DEEPSEEK_API_KEY=your_api_key")
        return False
    
    print(f"API密钥: {'已配置' if api_key else '未配置'}")
    
    # 确保工作目录存在
    workspace_dir = os.path.join(project_root, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    print(f"工作目录: {workspace_dir}")
    
    # 确保图像目录存在
    image_dir = os.path.join(project_root, "input", "images")
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在: {image_dir}")
        return False
    print(f"图像目录: {image_dir}")
    
    return True


def load_configuration():
    """加载配置"""
    print("\n=== 加载配置 ===")
    
    # 加载配置文件
    config_path = os.path.join(project_root, "resources", "config", "config.yaml")
    config_loader = ConfigLoader(config_path)
    config = config_loader._config.copy()
    
    # 设置语言文件路径
    locales_dir = os.path.join(project_root, "resources", "locales")
    config["locale"] = locales_dir
    print(f"使用语言文件目录: {locales_dir}")
    
    return config


def initialize_input_handler(config):
    """初始化输入处理器"""
    print("\n=== 初始化输入处理器 ===")
    
    workspace_dir = os.path.join(project_root, "workspace")
    
    # 初始化LocaleManager并绑定到桥接函数
    locale_manager = LocaleManager(config.get("locale", os.path.join(project_root, "resources", "locales")))
    get_text.locale_manager = locale_manager
    format_text.locale_manager = locale_manager
    
    # 初始化InputHandler
    input_handler = InputHandler(config, workspace_dir)
    
    # 配置OCR参数
    input_handler.ocr_enabled = True
    input_handler.ocr_initial_threshold = 0.1  # 低阈值，增强OCR捕获能力
    input_handler.ocr_confidence_threshold = 0.6  # 高阈值，确保最终结果质量
    input_handler.use_llm_enhancement = True  # 启用LLM增强
    input_handler.advanced_llm_integration = True  # 启用高级LLM集成
    input_handler.knowledge_integration = True  # 启用知识库集成
    
    print("输入处理器初始化完成")
    
    return input_handler


def test_ocr_preprocessing(input_handler, image_path):
    """测试OCR图像预处理功能"""
    print(f"\n=== 测试OCR图像预处理 ({os.path.basename(image_path)}) ===")
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像不存在: {image_path}")
        return None
    
    # 执行图像预处理
    try:
        start_time = time.time()
        processed_img = input_handler._preprocess_image(image_path)
        elapsed = time.time() - start_time
        
        # 验证预处理结果
        if not isinstance(processed_img, np.ndarray) or processed_img.size == 0:
            print("错误: 预处理图像无效")
            return None
        
        # 显示预处理图像信息
        print(f"预处理成功，图像形状: {processed_img.shape}")
        print(f"预处理耗时: {elapsed:.2f}秒")
        
        # 验证预处理后的图像是否保存
        debug_dir = os.path.join(input_handler.workspace_dir, "debug_images")
        debug_path = os.path.join(debug_dir, os.path.basename(image_path))
        if os.path.exists(debug_path):
            print(f"预处理图像已保存到: {debug_path}")
        else:
            print(f"警告: 预处理图像未保存到: {debug_path}")
        
        return processed_img
        
    except Exception as e:
        print(f"图像预处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_standard_ocr(input_handler, image_path):
    """测试标准OCR功能"""
    print(f"\n=== 测试标准OCR功能 ({os.path.basename(image_path)}) ===")
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像不存在: {image_path}")
        return None, 0.0
    
    # 执行标准OCR处理
    try:
        start_time = time.time()
        text, confidence = input_handler._perform_standard_ocr(image_path)
        elapsed = time.time() - start_time
        
        # 打印结果
        print(f"OCR结果 (基础模式):")
        print(f"文本: '{text}'")
        print(f"置信度: {confidence:.4f}")
        print(f"处理耗时: {elapsed:.2f}秒")
        
        return text, confidence
        
    except Exception as e:
        print(f"标准OCR测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0.0


def test_advanced_ocr_llm(input_handler, image_path):
    """测试高级OCR-LLM集成功能"""
    print(f"\n=== 测试高级OCR-LLM集成 ({os.path.basename(image_path)}) ===")
    
    # 检查API密钥是否配置
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("错误: 未配置API密钥，无法测试LLM增强功能")
        return None, 0.0
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像不存在: {image_path}")
        return None, 0.0
    
    # 先获取标准OCR结果作为参考
    standard_text, standard_conf = input_handler._perform_standard_ocr(image_path)
    print(f"标准OCR结果: '{standard_text}' (置信度: {standard_conf:.4f})")
    
    # 执行高级OCR-LLM集成
    try:
        start_time = time.time()
        enhanced_text, enhanced_conf = input_handler.advanced_ocr_llm_integration(image_path)
        elapsed = time.time() - start_time
        
        # 打印结果
        print(f"OCR-LLM集成结果 (耗时: {elapsed:.2f}秒):")
        print(f"文本: '{enhanced_text}'")
        print(f"置信度: {enhanced_conf:.4f}")
        
        # 评估LLM是否提供了增强
        if standard_text:
            print("\n比较增强前后的结果:")
            print(f"原始OCR: '{standard_text}'")
            print(f"LLM增强: '{enhanced_text}'")
            
            # 检查长度变化
            orig_length = len(standard_text)
            enhanced_length = len(enhanced_text)
            length_change = (enhanced_length-orig_length)/max(1, orig_length)*100
            print(f"文本长度变化: {orig_length} -> {enhanced_length} ({length_change:.1f}%)")
            
            # 检查置信度变化
            conf_change = (enhanced_conf-standard_conf)/max(0.001, standard_conf)*100
            print(f"置信度变化: {standard_conf:.4f} -> {enhanced_conf:.4f} ({conf_change:.1f}%)")
            
            # 词语差异分析
            standard_words = set(standard_text.split())
            enhanced_words = set(enhanced_text.split())
            new_words = enhanced_words - standard_words
            lost_words = standard_words - enhanced_words
            
            if new_words:
                print(f"新增词语: {', '.join(new_words)}")
            if lost_words:
                print(f"丢失词语: {', '.join(lost_words)}")
        
        return enhanced_text, enhanced_conf
        
    except Exception as e:
        print(f"OCR-LLM集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0.0


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="OCR与LLM集成测试工具")
    parser.add_argument("--image", help="要测试的图像文件路径", default=None)
    parser.add_argument("--all", action="store_true", help="测试input/images目录下的所有图像")
    args = parser.parse_args()
    
    # 显示欢迎信息
    print("\n========== OCR与LLM集成测试工具 ==========")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置测试环境
    if not setup_environment():
        print("环境设置失败，测试中止")
        return
    
    # 加载配置
    config = load_configuration()
    if not config:
        print("配置加载失败，测试中止")
        return
    
    # 初始化输入处理器
    input_handler = initialize_input_handler(config)
    if not input_handler:
        print("输入处理器初始化失败，测试中止")
        return
    
    # 确定要测试的图像文件
    image_files = []
    if args.image:
        # 使用指定的图像文件
        if os.path.exists(args.image):
            image_files.append(args.image)
        else:
            print(f"错误: 指定的图像文件不存在: {args.image}")
            return
    elif args.all:
        # 测试input/images目录下的所有图像
        images_dir = os.path.join(project_root, "input", "images")
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(images_dir, filename))
    else:
        # 默认测试两个示例图像
        image_files.append(os.path.join(project_root, "input", "images", "ocr_test_sample.png"))
        image_files.append(os.path.join(project_root, "input", "images", "test-note.png"))
    
    # 确保有图像可测试
    if not image_files:
        print("错误: 没有找到可测试的图像文件")
        return
    
    print(f"\n找到{len(image_files)}个图像文件待测试")
    
    # 对每个图像文件执行测试
    results = []
    for image_path in image_files:
        print(f"\n\n===== 测试图像: {os.path.basename(image_path)} =====")
        
        # 1. 测试图像预处理
        processed_img = test_ocr_preprocessing(input_handler, image_path)
        
        # 2. 测试标准OCR功能
        standard_text, standard_conf = test_standard_ocr(input_handler, image_path)
        
        # 3. 测试高级OCR-LLM集成
        enhanced_text, enhanced_conf = test_advanced_ocr_llm(input_handler, image_path)
        
        # 收集结果
        results.append({
            "image": os.path.basename(image_path),
            "standard_text": standard_text,
            "standard_confidence": standard_conf,
            "enhanced_text": enhanced_text,
            "enhanced_confidence": enhanced_conf,
            "improvement": enhanced_text != standard_text
        })
    
    # 显示测试总结
    print("\n\n========== 测试总结 ==========")
    for idx, result in enumerate(results):
        print(f"\n{idx+1}. 图像: {result['image']}")
        print(f"   标准OCR置信度: {result['standard_confidence']:.4f}")
        print(f"   LLM增强置信度: {result['enhanced_confidence']:.4f}")
        print(f"   是否有改进: {'是' if result['improvement'] else '否'}")
    
    print("\n========== 测试完成 ==========")


if __name__ == "__main__":
    main()