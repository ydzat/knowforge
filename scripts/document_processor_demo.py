#!/usr/bin/env python3
"""
文档综合处理示例脚本
"""
import os
import sys
import argparse
import logging
import io

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.note_generator.document_analyzer import DocumentAnalyzer
from src.note_generator.content_extractor import ContentExtractor
from src.note_generator.content_processor import ContentProcessor
from src.note_generator.content_integrator import ContentIntegrator
from src.note_generator.warning_monitor import warning_monitor
from src.utils.logger import get_module_logger

# 创建一个内存流处理器来捕获日志信息
class StringIOHandler(logging.StreamHandler):
    def __init__(self):
        self.string_io = io.StringIO()
        super().__init__(self.string_io)
        
    def get_log_contents(self):
        return self.string_io.getvalue()


def process_document(file_path, output_format="markdown", config=None):
    """
    处理文档的完整流程
    
    Args:
        file_path: 要处理的文档路径
        output_format: 输出格式，可选"markdown"或"html"
        config: 配置选项，可覆盖默认设置
        
    Returns:
        处理后的文档内容和统计信息的元组
    """
    logger = get_module_logger("DocumentProcessor")
    
    # 重置警告监视器
    warning_monitor.reset()
    
    # 初始化统计信息
    stats = {
        "document_type": None,
        "total_pages": 0,
        "content_blocks": {
            "total": 0,
            "text": 0,
            "image": 0,
            "table": 0,
            "formula": 0
        },
        "extraction_methods": {
            "direct": 0,
            "region": 0,
            "structure": 0
        },
        "warnings": {},
        "start_time": None,
        "end_time": None,
        "processing_time": None
    }
    
    # 记录开始时间
    import time
    stats["start_time"] = time.time()
    
    try:
        # 步骤1: 分析文档
        logger.info(f"步骤1: 分析文档 {file_path}")
        analyzer = DocumentAnalyzer(config)
        analysis_result = analyzer.analyze_document(file_path)
        
        # 更新统计信息
        stats["document_type"] = analysis_result["document_type"]
        stats["total_pages"] = analysis_result["total_pages"]
        stats["content_blocks"]["total"] = len(analysis_result["blocks"])
        
        # 统计各类内容块数量
        for block in analysis_result["blocks"]:
            block_type = block.get("type", "unknown")
            stats["content_blocks"][block_type] = stats["content_blocks"].get(block_type, 0) + 1
            
            # 统计图像提取方法
            if block_type == "image" and "extraction_method" in block:
                method = block["extraction_method"]
                stats["extraction_methods"][method] = stats["extraction_methods"].get(method, 0) + 1
        
        # 步骤2: 提取内容
        logger.info(f"步骤2: 提取内容, 共{len(analysis_result['blocks'])}个内容块")
        extractor = ContentExtractor(config)
        extracted_blocks = extractor.extract_content(analysis_result["blocks"])
        
        # 步骤3: 处理内容
        logger.info(f"步骤3: 处理内容, 共{len(extracted_blocks)}个提取块")
        processor = ContentProcessor(config)
        processed_blocks = processor.process(extracted_blocks)
        
        # 步骤4: 整合内容
        logger.info(f"步骤4: 整合内容, 共{len(processed_blocks)}个处理后的块")
        integrator = ContentIntegrator(config)
        
        # 处理完成，记录结束时间
        stats["end_time"] = time.time()
        stats["processing_time"] = stats["end_time"] - stats["start_time"]
        
        # 生成输出内容
        if output_format.lower() == "html":
            content = integrator.integrate_to_html(processed_blocks)
        else:
            content = integrator.integrate_to_markdown(processed_blocks)
            
        # 检查是否存在警告
        import re
        
        # 使用更安全的方法获取警告信息
        warnings_count = {}
        
        # 解析常见警告模式
        common_warnings = [
            "not enough image data",
            "转换图像数据时出错",
            "从区域提取图像时出错"
        ]
        
        # 为常见警告类型预先设置计数
        for warning_type in common_warnings:
            warnings_count[warning_type] = 0
        
        # 从警告监视器获取警告计数
        warning_counts = warning_monitor.get_warning_counts()
        
        # 更新统计信息
        stats["warnings"] = {
            "not enough image data": warning_counts["not_enough_image_data"],
            "转换图像数据时出错": warning_counts["convert_image_error"],
            "从区域提取图像时出错": warning_counts["extract_region_error"],
            "其他警告": warning_counts["other_warnings"]
        }
        
        return content, stats
        
    except Exception as e:
        # 记录错误信息
        stats["error"] = str(e)
        stats["end_time"] = time.time()
        stats["processing_time"] = stats["end_time"] - stats["start_time"]
        logger.error(f"处理文档时出错: {str(e)}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文档综合处理示例")
    parser.add_argument("file_path", help="要处理的文档路径")
    parser.add_argument("--format", "-f", choices=["markdown", "html"], 
                        default="markdown", help="输出格式 (默认: markdown)")
    parser.add_argument("--output", "-o", help="输出文件路径 (默认: 标准输出)")
    parser.add_argument("--stats", "-s", help="统计信息输出文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细处理信息")
    parser.add_argument("--min-image-size", type=int, default=100, 
                        help="最小图像尺寸 (像素，默认: 100)")
    parser.add_argument("--disable-structure-detection", action="store_true",
                       help="禁用文档结构检测")
    parser.add_argument("--alternative-extraction", choices=["always", "auto", "never"],
                       default="auto", help="备用图像提取方法 (默认: auto)")
    
    args = parser.parse_args()
    
    try:
        # 创建配置字典
        config = {
            "min_image_size": args.min_image_size,
            "structure_detection_enabled": not args.disable_structure_detection,
            "alternative_extraction": args.alternative_extraction,
            "verbose": args.verbose
        }
        
        # 处理文档
        processed_content, stats = process_document(args.file_path, args.format, config)
        
        # 输出统计信息
        if args.verbose:
            print("\n===== 处理统计 =====")
            print(f"文档类型: {stats['document_type']}")
            print(f"总页数: {stats['total_pages']}")
            print(f"内容块总数: {stats['content_blocks']['total']}")
            print(f"  - 文本块: {stats['content_blocks'].get('text', 0)}")
            print(f"  - 图像块: {stats['content_blocks'].get('image', 0)}")
            print(f"    - 直接提取: {stats['extraction_methods'].get('direct', 0)}")
            print(f"    - 区域提取: {stats['extraction_methods'].get('region', 0)}")
            print(f"    - 结构检测: {stats['extraction_methods'].get('structure', 0)}")
            print(f"  - 表格块: {stats['content_blocks'].get('table', 0)}")
            print(f"  - 公式块: {stats['content_blocks'].get('formula', 0)}")
            print(f"处理时间: {stats['processing_time']:.2f}秒")
            
            if stats.get("warnings"):
                print("\n警告统计:")
                for warning_type, count in stats["warnings"].items():
                    print(f"  - {warning_type}: {count}次")
        
        # 保存统计信息
        if args.stats:
            import json
            with open(args.stats, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"统计信息已写入: {args.stats}")
        
        # 输出处理结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            print(f"处理完成，结果已写入: {args.output}")
        else:
            print(processed_content)
        
        return 0
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
