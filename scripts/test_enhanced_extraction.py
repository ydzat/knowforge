#!/usr/bin/env python3
"""
增强型图像提取测试脚本
测试新的图像提取功能并记录警告统计
"""
import os
import sys
import argparse
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 确保可以导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.document_analyzer import DocumentAnalyzer
from src.note_generator.enhanced_extractor import EnhancedImageExtractor
from src.note_generator.warning_monitor import warning_monitor
from src.utils.logger import setup_logger


def test_image_extraction(pdf_path, output_dir=None):
    """
    测试增强型图像提取功能
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 可选的输出目录，用于保存提取的图像
    """
    # 设置日志和输出目录
    logger = setup_logger("ImageExtractionTest")
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 重置警告监控器
    warning_monitor.reset()
    
    # 测试前记录
    logger.info(f"开始测试图像提取功能，文件: {pdf_path}")
    logger.info(f"使用增强型图像提取器")
    
    # 创建文档分析器
    analyzer = DocumentAnalyzer()
    
    # 打开并分析文档
    try:
        result = analyzer.analyze_document(pdf_path)
        
        # 统计图像数量
        image_blocks = [b for b in result["blocks"] if b["type"] == "image"]
        total_images = len(image_blocks)
        
        logger.info(f"文档共 {result['total_pages']} 页，检测到 {total_images} 个图像块")
        
        # 保存图像
        if output_dir:
            for i, img_block in enumerate(image_blocks):
                page = img_block["page"]
                img_data = img_block["image_data"]
                extraction_method = img_block.get("extraction_method", "unknown")
                
                # 保存图像
                img = Image.fromarray(img_data)
                img_path = os.path.join(output_dir, f"image_p{page}_{i+1}_{extraction_method}.png")
                img.save(img_path)
                logger.info(f"保存图像: {img_path}")
        
        # 分析警告统计
        warning_counts = warning_monitor.get_warning_counts()
        logger.info("警告统计:")
        for warning_type, count in warning_counts.items():
            logger.info(f"  - {warning_type}: {count}")
        
        logger.info(f"总警告数: {warning_monitor.get_total_warnings()}")
        
        # 返回分析结果
        return {
            "total_pages": result["total_pages"],
            "total_images": total_images,
            "warning_counts": warning_counts,
            "total_warnings": warning_monitor.get_total_warnings()
        }
        
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试增强型图像提取功能")
    parser.add_argument("pdf_path", help="要分析的PDF文件路径")
    parser.add_argument("--output-dir", "-o", help="图像输出目录")
    args = parser.parse_args()
    
    try:
        # 执行测试
        result = test_image_extraction(args.pdf_path, args.output_dir)
        
        print("\n测试结果摘要:")
        print(f"总页数: {result['total_pages']}")
        print(f"提取图像: {result['total_images']}")
        print(f"警告统计: {result['warning_counts']}")
        print(f"总警告数: {result['total_warnings']}")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        sys.exit(1)
