#!/usr/bin/env python3
"""
创建一个测试PDF文档，包含文本、表格和公式
用于测试KnowForge的文档处理功能
"""
import os
import sys
import argparse
from fpdf import FPDF  # 使用fpdf2库

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logger import get_module_logger

# 初始化日志
logger = get_module_logger("create_test_doc")

def create_test_document(output_path):
    """
    创建一个测试PDF文档，包含文本、表格和公式
    
    Args:
        output_path: 输出文件路径
    """
    logger.info(f"开始创建测试文档: {output_path}")
    
    # 创建PDF对象
    pdf = FPDF()
    pdf.add_page()
    
    # 设置字体 - 使用标准字体
    pdf.set_font('Arial', '', 12)
    
    # 添加标题
    pdf.set_font('Arial', '', 16)
    pdf.cell(0, 10, 'Test Document: Tables and Formulas', 0, 1, 'C')
    pdf.ln(10)
    
    # 添加文本
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, 'This is a test document for testing KnowForge document processing capabilities, especially table and formula processing.')
    pdf.ln(5)
    
    # 添加表格
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'Table Example:', 0, 1)
    
    # 表格1：产品销售表
    col_width = 40
    row_height = 10
    
    # 表头
    pdf.cell(col_width, row_height, 'Product', 1, 0, 'C')
    pdf.cell(col_width, row_height, 'Price', 1, 0, 'C')
    pdf.cell(col_width, row_height, 'Qty', 1, 0, 'C')
    pdf.cell(col_width, row_height, 'Total', 1, 1, 'C')
    
    # 行1
    pdf.cell(col_width, row_height, 'Laptop', 1, 0)
    pdf.cell(col_width, row_height, '6000', 1, 0)
    pdf.cell(col_width, row_height, '2', 1, 0)
    pdf.cell(col_width, row_height, '12000', 1, 1)
    
    # 行2
    pdf.cell(col_width, row_height, 'Monitor', 1, 0)
    pdf.cell(col_width, row_height, '1500', 1, 0)
    pdf.cell(col_width, row_height, '3', 1, 0)
    pdf.cell(col_width, row_height, '4500', 1, 1)
    
    # 行3
    pdf.cell(col_width, row_height, 'Mouse', 1, 0)
    pdf.cell(col_width, row_height, '80', 1, 0)
    pdf.cell(col_width, row_height, '5', 1, 0)
    pdf.cell(col_width, row_height, '400', 1, 1)
    
    # 行4（总计）
    pdf.cell(col_width, row_height, 'Total', 1, 0)
    pdf.cell(col_width, row_height, '', 1, 0)
    pdf.cell(col_width, row_height, '', 1, 0)
    pdf.cell(col_width, row_height, '16900', 1, 1)
    
    pdf.ln(10)
    
    # 添加公式
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'Formula Example:', 0, 1)
    
    # 公式1：E=mc²（爱因斯坦质能方程）
    pdf.multi_cell(0, 10, 'E=mc^2')
    
    # 公式2：勾股定理
    pdf.multi_cell(0, 10, 'a^2 + b^2 = c^2')
    
    # 公式3：函数定义
    pdf.multi_cell(0, 10, 'f(x) = x^2 / (x-1)')
    
    # 公式4：求和公式
    pdf.multi_cell(0, 10, 'Sum formula: ∑(i=1 to n) i = n(n+1)/2')
    
    # 公式5：积分公式
    pdf.multi_cell(0, 10, 'Integration formula: ∫(a to b) f(x) dx = F(b) - F(a)')
    
    # 保存PDF
    pdf.output(output_path)
    logger.info(f"测试文档创建完成: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="创建测试PDF文档")
    parser.add_argument("--output", "-o", default="test_document.pdf", help="输出文件路径")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建文档
    create_test_document(args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
