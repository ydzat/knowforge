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


def test_document_processing(input_file, output_dir=None, config=None, ocr_processor=None):
    """
    测试文档综合处理流程
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        config: 配置选项
        ocr_processor: OCR处理器实例（可选）
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
        
        # 对于纯文本文件，使用简单内容分析来识别表格和公式
        if input_file.lower().endswith('.txt'):
            logger.info("检测到.txt文件，使用简单内容分析识别表格和公式")
            
            # 读取文本内容
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 将整个内容添加为文本块
            document_blocks = {'blocks': [{'type': 'text', 'content': content, 'page': 1, 'coordinates': [0, 0, 0, 0]}]}
            
            # 尝试识别表格和公式
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # 识别表格 - 查找包含 | 和 --- 的行
                if '|' in line and i+1 < len(lines) and '---' in lines[i+1]:
                    table_start = i
                    table_end = i
                    
                    # 查找表格结束
                    while table_end < len(lines) and '|' in lines[table_end]:
                        table_end += 1
                        
                    # 提取表格内容
                    table_content = '\n'.join(lines[table_start:table_end])
                    # 添加为表格块
                    document_blocks['blocks'].append({
                        'type': 'table',
                        'content': table_content,
                        'page': 1,
                        'coordinates': [0, 0, 0, 0]
                    })
                    i = table_end - 1  # 跳过已处理的表格行
                
                # 识别公式 - 查找包含数学符号的行
                elif any(x in line for x in ['^', '=', 'sum', '\\sum', '\\int', '\\lim', '\\begin{align}']):
                    formula_content = line
                    formula_end = i
                    
                    # 检查是否为多行公式块
                    if '\\begin{align}' in line:
                        while formula_end < len(lines) and '\\end{align}' not in lines[formula_end]:
                            formula_end += 1
                        if formula_end < len(lines):
                            formula_content = '\n'.join(lines[i:formula_end+1])
                            
                    # 添加为公式块
                    document_blocks['blocks'].append({
                        'type': 'formula',
                        'content': formula_content,
                        'page': 1,
                        'coordinates': [0, 0, 0, 0]
                    })
                    
                    if formula_end > i:
                        i = formula_end  # 跳过多行公式
                
                i += 1
        
        # 分析内容块类型统计
        block_types = {}
        for block in document_blocks['blocks']:
            block_type = block.get('type', 'unknown')
            if block_type not in block_types:
                block_types[block_type] = 0
            block_types[block_type] += 1
        
        logger.info(f"文档分析完成，识别到{len(document_blocks['blocks'])}个内容块，类型统计: {block_types}")
        
        # 保存分析结果
        analysis_output = os.path.join(output_dir, "analysis_result.json")
        save_json(document_blocks, analysis_output)
        
        # 步骤2：内容提取
        extractor = ContentExtractor(config)
        
        # 如果是PDF文件，为表格提取添加文档路径信息
        if input_file.lower().endswith('.pdf'):
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(input_file)
                # 添加额外的表格处理信息
                for block in document_blocks['blocks']:
                    if block['type'] == 'table':
                        block['pdf_path'] = input_file
                extracted_blocks = extractor.extract_content_from_document(doc, document_blocks['blocks'])
                doc.close()
            except ImportError:
                logger.warning("未安装PyMuPDF，无法使用高级PDF表格提取功能")
                extracted_blocks = extractor.extract_content(document_blocks['blocks'])
        else:
            extracted_blocks = extractor.extract_content(document_blocks['blocks'])
        
        logger.info(f"内容提取完成，提取了{len(extracted_blocks)}个内容块")
        
        # 保存提取结果
        extraction_output = os.path.join(output_dir, "extraction_result.json")
        save_json(extracted_blocks, extraction_output)
        
        # 步骤3：内容处理
        processor = ContentProcessor(config, ocr_processor)
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
    parser.add_argument("--use-ocr", action="store_true", help="启用OCR处理图像")
    parser.add_argument("--table-processor", choices=["camelot", "tabula", "custom"], default="custom",
                       help="表格处理器类型 (默认: custom)")
    parser.add_argument("--formula-engine", choices=["mathpix", "custom"], default="custom",
                       help="公式处理引擎 (默认: custom)")
    parser.add_argument("--api-key", help="Mathpix API Key (如果使用mathpix引擎)")
    parser.add_argument("--app-id", help="Mathpix App ID (如果使用mathpix引擎)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"错误：输入文件不存在: {args.input_file}")
        return 1
    
    # 配置选项
    config = {
        # 表格处理配置
        "table.processor": args.table_processor,
        "table.clean_empty_rows": True,
        "table.normalize_columns": True,
        "table.enhance_structure": True,
        # 公式处理配置
        "formula.engine": args.formula_engine,
        "formula.detect_formula_type": True,
        "formula.convert_simple_expressions": True,
    }
    
    # 如果提供了API密钥，添加到配置
    if args.api_key and args.formula_engine == "mathpix":
        config["formula.mathpix_api_key"] = args.api_key
    if args.app_id and args.formula_engine == "mathpix":
        config["formula.mathpix_app_id"] = args.app_id
    
    # 初始化OCR处理器
    ocr_processor = None
    if args.use_ocr:
        try:
            # 尝试导入OCR处理器
            from src.note_generator.advanced_ocr_processor import AdvancedOCRProcessor
            ocr_processor = AdvancedOCRProcessor(config)
            logger.info("已初始化OCR处理器")
        except ImportError:
            logger.warning("未能导入AdvancedOCRProcessor，OCR功能已禁用")
    
    # 执行测试
    success = test_document_processing(args.input_file, args.output_dir, config, ocr_processor)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
