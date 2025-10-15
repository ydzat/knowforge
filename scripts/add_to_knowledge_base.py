#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将OCR-LLM处理结果添加到知识库的脚本

该脚本可以将图像的OCR-LLM处理结果添加到知识库，以便后续使用。
"""
import os
import sys
import argparse

# 将项目根目录添加到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.config_loader import ConfigLoader
from src.note_generator.embedding_manager import EmbeddingManager
from src.utils.logger import get_module_logger

logger = get_module_logger("knowledge_base_updater")

def load_config():
    """加载配置"""
    config_path = os.path.join(project_root, "resources", "config", "config.yaml")
    config_loader = ConfigLoader(config_path)
    config = config_loader._config.copy()
    
    # 确保embedding配置正确
    if "embedding" not in config:
        config["embedding"] = {}
    config["embedding"]["model_name"] = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 确保memory配置正确
    if "memory" not in config:
        config["memory"] = {}
    config["memory"]["enabled"] = True
    config["memory"]["top_k"] = 3
    config["memory"]["similarity_threshold"] = 0.6
    
    return config

def add_content_to_knowledge_base(content, metadata=None):
    """添加内容到知识库"""
    if not content:
        logger.warning("内容为空，不添加到知识库")
        return False
    
    config = load_config()
    workspace_dir = os.path.join(project_root, "workspace")
    
    try:
        # 初始化EmbeddingManager
        embedding_manager = EmbeddingManager(workspace_dir, config)
        
        # 准备元数据
        if metadata is None:
            metadata = {"source": "manual_addition", "type": "image_ocr"}
        
        # 添加到知识库
        document_ids = embedding_manager.add_to_knowledge_base([content], [metadata])
        
        if document_ids:
            logger.info(f"成功添加到知识库，文档ID: {document_ids[0]}")
            print(f"成功添加到知识库，文档ID: {document_ids[0]}")
            return True
        else:
            logger.warning("添加到知识库失败")
            print("添加到知识库失败")
            return False
            
    except Exception as e:
        logger.error(f"添加到知识库时出错: {str(e)}")
        return False

def add_sample_content():
    """添加示例内容到知识库"""
    # 示例1：机器学习笔记
    ml_note = """
Advanced Machine Learning
Part 1 – Introduction
B. Leibe
Variables
Optimization Problems
Capacity Constraints
Maximum Capacity
Problem Formulation
Lower Bounds
Machine Learning Models
Transportation Problem
Demand Constraints
Cost Function
Supply Constraints
Service Level
Product Matching
    """
    
    # 示例2：OCR相关信息
    ocr_info = """
OCR (Optical Character Recognition)
图像文本识别技术
预处理技术
- 去噪
- 二值化
- 倾斜校正
- 文本定位
OCR引擎类型
- 基于规则的OCR
- 深度学习OCR
- 混合方法OCR
常见OCR库
- Tesseract
- PaddleOCR
- EasyOCR
识别结果评估
- 字符准确率
- 单词准确率
- 置信度评估
    """
    
    # 添加到知识库
    print("正在添加机器学习笔记到知识库...")
    success1 = add_content_to_knowledge_base(ml_note, {"source": "sample", "topic": "machine_learning"})
    print(f"添加{'成功' if success1 else '失败'}")
    
    print("正在添加OCR信息到知识库...")
    success2 = add_content_to_knowledge_base(ocr_info, {"source": "sample", "topic": "ocr_technology"})
    print(f"添加{'成功' if success2 else '失败'}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="知识库内容更新工具")
    parser.add_argument("--add-sample", action="store_true", help="添加示例内容到知识库")
    parser.add_argument("--add-content", type=str, help="要添加到知识库的内容")
    parser.add_argument("--source", type=str, default="manual", help="内容来源")
    parser.add_argument("--topic", type=str, default="general", help="内容主题")
    args = parser.parse_args()
    
    if args.add_sample:
        add_sample_content()
    elif args.add_content:
        metadata = {"source": args.source, "topic": args.topic}
        success = add_content_to_knowledge_base(args.add_content, metadata)
        print(f"添加内容到知识库{'成功' if success else '失败'}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
