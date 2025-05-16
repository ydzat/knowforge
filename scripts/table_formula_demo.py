#!/usr/bin/env python3
"""
表格和公式处理功能演示脚本
专门展示KnowForge的表格和公式专项处理能力
"""
import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import io

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.note_generator.content_processor import TableProcessor, FormulaProcessor
from src.utils.logger import setup_logger, get_module_logger

logger = get_module_logger("TableFormulaDemo")


def demo_table_processor(config=None):
    """表格处理器演示函数"""
    logger.info("开始演示表格处理功能")
    
    # 初始化配置
    if config is None:
        config = {}
    
    # 创建表格处理器
    processor = TableProcessor(config)
    
    # 示例表格数据
    sample_tables = [
        # 简单表格
        [
            ["产品", "单价", "数量", "总价"],
            ["笔记本电脑", "6000", "2", "12000"],
            ["显示器", "1500", "3", "4500"],
            ["鼠标", "80", "5", "400"],
            ["总计", "", "", "16900"]
        ],
        
        # 不规则表格（列数不一致）
        [
            ["季度", "销售额", "成本", "利润", "利润率"],
            ["Q1", "100万", "80万", "20万"],
            ["Q2", "120万", "90万", "30万", "25%"],
            ["Q3", "150万", "110万", "40万", "26.7%", "同比增长33.3%"],
            ["Q4", "200万", "140万", "60万", "30%"]
        ],
        
        # 包含空单元格的表格
        [
            ["姓名", "周一", "周二", "周三", "周四", "周五"],
            ["张三", "上班", "", "上班", "上班", ""],
            ["李四", "", "上班", "上班", "", "上班"],
            ["王五", "上班", "上班", "", "", "上班"]
        ]
    ]
    
    # 处理并展示每个表格
    for i, table_data in enumerate(sample_tables):
        print(f"\n\n==== 示例表格 {i+1} ====\n")
        print("原始表格数据:")
        for row in table_data:
            print(" | ".join(str(cell) for cell in row))
        
        # 创建表格内容块
        table_block = {
            "type": "table",
            "page": 1,
            "coordinates": [0, 0, 500, 300],
            "table_data": table_data
        }
        
        # 处理表格
        processed = processor.process(table_block)
        
        # 显示处理结果
        print("\n处理后的Markdown表格:")
        print(processed["markdown"])
        
        # 如果启用了表格增强，展示结果
        if config.get("table.enhance_structure", False):
            print("\n应用了表格结构增强:")
            # 显示可能的修复（如列数标准化）
            orig_cols = [len(row) for row in table_data]
            proc_cols = [len(row) for row in processed["table_data"]]
            if orig_cols != proc_cols:
                print(f"- 列数标准化: {orig_cols} -> {proc_cols}")
            
            # 显示可能的空值处理
            empty_cells_before = sum(1 for row in table_data for cell in row if cell == "")
            empty_cells_after = sum(1 for row in processed["table_data"] for cell in row if cell == "")
            if empty_cells_before != empty_cells_after:
                print(f"- 空单元格处理: {empty_cells_before}个 -> {empty_cells_after}个")


def demo_formula_processor(config=None):
    """公式处理器演示函数"""
    logger.info("开始演示公式处理功能")
    
    # 初始化配置
    if config is None:
        config = {}
    
    # 创建公式处理器
    processor = FormulaProcessor(config)
    
    # 示例公式
    sample_formulas = [
        # 简单公式
        "E=mc^2",
        
        # 基本数学公式
        "a^2 + b^2 = c^2",
        
        # 含有分数的公式
        "f(x) = x^2 / (x-1)",
        
        # 已包含LaTeX命令的公式
        "\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}",
        
        # 积分公式
        "\\int_{a}^{b} f(x) dx = F(b) - F(a)",
        
        # 多行公式
        """\\begin{align}
        (a+b)^2 &= (a+b)(a+b) \\\\
        &= a^2 + ab + ba + b^2 \\\\
        &= a^2 + 2ab + b^2
        \\end{align}""",
        
        # 包含特殊符号的公式
        "\\lim_{x \\to \\infty} \\frac{1}{x} = 0"
    ]
    
    # 处理并展示每个公式
    for i, formula_text in enumerate(sample_formulas):
        print(f"\n\n==== 示例公式 {i+1} ====\n")
        print("原始公式文本:")
        print(formula_text)
        
        # 创建公式内容块
        formula_block = {
            "type": "formula",
            "page": 1,
            "coordinates": [0, 0, 200, 50],
            "formula_text": formula_text
        }
        
        # 处理公式
        processed = processor.process(formula_block)
        
        # 显示处理结果
        print("\n处理后的LaTeX公式:")
        print(processed["latex"])
        
        # 显示处理器类型和特性
        print(f"\n使用的处理器: {processed.get('processor', '未知')}")
        if processor.detect_formula_type:
            is_inline = processed["latex"].startswith("$") and not processed["latex"].startswith("$$")
            print(f"公式类型: {'内联公式' if is_inline else '块级公式'}")


def main():
    """主函数"""
    # 设置命令行解析器
    parser = argparse.ArgumentParser(description="表格和公式处理功能演示")
    parser.add_argument("--table", action="store_true", help="演示表格处理")
    parser.add_argument("--formula", action="store_true", help="演示公式处理")
    parser.add_argument("--processor", choices=["custom", "camelot", "tabula"], 
                        default="custom", help="表格处理器类型")
    parser.add_argument("--engine", choices=["custom", "mathpix"], 
                        default="custom", help="公式处理引擎")
    parser.add_argument("--api-key", help="Mathpix API密钥")
    parser.add_argument("--app-id", help="Mathpix App ID")
    parser.add_argument("--output", "-o", help="输出结果到文件")
    
    args = parser.parse_args()
    
    # 如果没有指定，则两种功能都演示
    if not args.table and not args.formula:
        args.table = True
        args.formula = True
    
    # 设置配置
    config = {
        # 表格处理配置
        "table.processor": args.processor,
        "table.clean_empty_rows": True,
        "table.normalize_columns": True,
        "table.enhance_structure": True,
        # 公式处理配置
        "formula.engine": args.engine,
        "formula.detect_formula_type": True,
        "formula.convert_simple_expressions": True
    }
    
    # 如果提供了API密钥，加入配置
    if args.api_key:
        config["formula.mathpix_api_key"] = args.api_key
    if args.app_id:
        config["formula.mathpix_app_id"] = args.app_id
    
    # 如果指定了输出文件，将输出重定向到文件
    if args.output:
        original_stdout = sys.stdout
        with open(args.output, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            if args.table:
                demo_table_processor(config)
            if args.formula:
                demo_formula_processor(config)
                
            # 恢复原来的标准输出
            sys.stdout = original_stdout
        print(f"演示结果已写入: {args.output}")
    else:
        # 直接输出到控制台
        if args.table:
            demo_table_processor(config)
        if args.formula:
            demo_formula_processor(config)
    
    return 0


if __name__ == "__main__":
    # 设置日志
    setup_logger()
    sys.exit(main())
