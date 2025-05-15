#!/usr/bin/env python3
"""
文档综合处理测试脚本
用于测试DocumentAnalyzer、ContentExtractor、ContentProcessor和ContentIntegrator的功能
"""
import os
import sys
import argparse
import json
from PIL import Image
import numpy as np
import cv2

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入所需模块
from src.note_generator.document_analyzer import DocumentAnalyzer
from src.note_generator.content_extractor import ContentExtractor
from src.note_generator.content_processor import ContentProcessor
from src.note_generator.content_integrator import ContentIntegrator
from src.utils.logger import get_module_logger

# 初始化日志
logger = get_module_logger("docprocessing_test")


def test_document_processing(input_file, output_dir=None, config=None):
    """
    测试文档综合处理流程
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        config: 配置选项
    """
    logger.info(f"开始测试文档综合处理: {input_file}")
    
    # 确保输入文件存在
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return False
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_file), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化配置
    if config is None:
        config = {}
    
    try:
        # 步骤1：文档分析
        analyzer = DocumentAnalyzer(config)
        document_blocks = analyzer.analyze_document(input_file)
        logger.info(f"文档分析完成，识别到{len(document_blocks['blocks'])}个内容块")
        
        # 保存分析结果
        analysis_output = os.path.join(output_dir, "analysis_result.json")
        save_json(document_blocks, analysis_output)
        
        # 步骤2：内容提取
        extractor = ContentExtractor(config)
        extracted_blocks = extractor.extract_content(document_blocks['blocks'])
        logger.info(f"内容提取完成，提取了{len(extracted_blocks)}个内容块")
        
        # 保存提取结果
        extraction_output = os.path.join(output_dir, "extraction_result.json")
        save_json(extracted_blocks, extraction_output)
        
        # 步骤3：内容处理
        processor = ContentProcessor(config)
        processed_blocks = processor.process(extracted_blocks)
        logger.info(f"内容处理完成，处理了{len(processed_blocks)}个内容块")
        
        # 保存处理结果
        processing_output = os.path.join(output_dir, "processing_result.json")
        save_json(processed_blocks, processing_output)
        
        # 步骤4：内容整合
        integrator = ContentIntegrator(config)
        integrated_content = integrator.integrate(processed_blocks)
        logger.info(f"内容整合完成，生成了{len(integrated_content)}个内容段落")
        
        # 保存整合结果为文本
        integration_output = os.path.join(output_dir, "integration_result.txt")
        with open(integration_output, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(integrated_content))
        
        # 生成Markdown输出
        markdown_output = os.path.join(output_dir, "output.md")
        markdown_content = integrator.integrate_to_markdown(processed_blocks)
        with open(markdown_output, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # 生成HTML输出
        html_output = os.path.join(output_dir, "output.html")
        html_content = integrator.integrate_to_html(processed_blocks)
        with open(html_output, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"测试完成，输出文件保存在: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"处理文档时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def save_json(data, output_path):
    """
    保存JSON数据到文件
    
    Args:
        data: 要保存的数据
        output_path: 输出文件路径
    """
    # 将numpy数组和其他不可序列化的对象转换为可序列化形式
    serializable_data = make_serializable(data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)


def make_serializable(obj):
    """
    将对象转换为可JSON序列化的形式
    
    Args:
        obj: 要转换的对象
        
    Returns:
        可序列化的对象
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 2:
            return "<2D数组，形状: {}>".format(obj.shape)
        elif obj.ndim == 3 and obj.shape[2] in (1, 3, 4):
            return "<图像数据，形状: {}>".format(obj.shape)
        else:
            return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    elif isinstance(obj, bytes):
        return "<二进制数据，长度: {}>".format(len(obj))
    else:
        return obj


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文档综合处理测试工具")
    parser.add_argument("input_file", help="输入文件路径")
    parser.add_argument("--output-dir", "-o", help="输出目录")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"错误：输入文件不存在: {args.input_file}")
        return 1
    
    # 执行测试
    success = test_document_processing(args.input_file, args.output_dir)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
