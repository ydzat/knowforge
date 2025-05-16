#!/usr/bin/env python3
'''
 * @Author: @ydzat
 * @Date: 2025-06-01 10:00:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-06-01 10:00:00
 * @Description: 测试KnowForge 0.1.7新增的输出功能，包括HTML格式生成、增强的PDF生成和Jupyter Notebook输出
'''
import os
import sys
import time
import argparse
import logging
from datetime import datetime

# 添加src到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.output_writer import OutputWriter
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger, get_logger

# 配置日志
setup_logger()
logger = get_logger('KnowForge-OutputTest')

def generate_test_content():
    """生成包含各种元素的测试内容，用于验证输出格式处理能力"""
    
    content = []
    
    # 添加标题和基本文本
    content.append("# KnowForge 输出功能测试")
    content.append("\n这是一个测试文档，用于验证KnowForge 0.1.7版本的新增输出功能。\n")
    
    # 添加二级标题和段落
    content.append("## 1. 基本格式支持")
    content.append("这里我们测试基本的Markdown格式支持，包括：\n")
    content.append("- **粗体文本** 和 *斜体文本*")
    content.append("- `行内代码` 和 代码块")
    content.append("- [超链接](https://github.com/ydzat/knowforge)")
    content.append("- > 引用文本块\n")
    
    # 添加代码块示例
    content.append("### 1.1 代码块示例")
    content.append("下面是一个Python代码块：\n")
    content.append("```python\ndef hello_world():\n    print('Hello, KnowForge!')\n    return True\n```\n")
    
    # 添加表格示例
    content.append("## 2. 表格支持")
    content.append("下面是一个表格示例：\n")
    content.append("| 名称 | 类型 | 说明 |")
    content.append("| ---- | ---- | ---- |")
    content.append("| 输出格式 | 字符串 | 输出文件的格式类型 |")
    content.append("| 模板文件 | 文件路径 | 用于生成输出的模板文件 |")
    content.append("| 目标路径 | 文件路径 | 生成文件的保存位置 |\n")
    
    # 添加LaTeX公式示例
    content.append("## 3. 数学公式支持")
    content.append("测试行内数学公式：$E = mc^2$")
    content.append("\n测试块级数学公式：\n")
    content.append("$$")
    content.append("\\frac{\\partial f}{\\partial x} = 2x")
    content.append("$$\n")
    
    # 添加嵌套列表
    content.append("## 4. 复杂列表结构")
    content.append("测试嵌套列表结构：\n")
    content.append("1. 第一级列表项")
    content.append("   - 第二级列表项A")
    content.append("   - 第二级列表项B")
    content.append("     - 第三级列表项I")
    content.append("     - 第三级列表项II")
    content.append("2. 另一个第一级列表项")
    content.append("   - 嵌套列表项\n")
    
    # 添加一张表情符号和特殊字符
    content.append("## 5. 特殊字符支持")
    content.append("测试表情符号: 😊 🚀 📚")
    content.append("测试特殊字符: ©️ ®️ ™️ § ¥ € £")
    content.append("测试汉字: 知识锻造 人工智能 机器学习\n")
    
    return content

def test_markdown_output(output_writer, test_content, filename="test_output"):
    """测试Markdown输出功能"""
    logger.info("=== 测试Markdown输出功能 ===")
    
    try:
        start_time = time.time()
        output_path = output_writer.generate_markdown(test_content, filename, "KnowForge输出测试")
        duration = time.time() - start_time
        
        logger.info(f"Markdown生成成功，耗时: {duration:.4f}秒")
        logger.info(f"输出文件路径: {output_path}")
        
        # 验证文件是否存在
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"文件大小: {file_size} 字节")
            return True
        else:
            logger.error("Markdown文件未成功生成")
            return False
    except Exception as e:
        logger.error(f"Markdown生成失败: {str(e)}")
        return False

def test_html_output(output_writer, test_content, filename="test_output"):
    """测试HTML输出功能"""
    logger.info("=== 测试HTML输出功能 ===")
    
    try:
        start_time = time.time()
        output_path = output_writer.generate_html(test_content, filename, "KnowForge输出测试")
        duration = time.time() - start_time
        
        logger.info(f"HTML生成成功，耗时: {duration:.4f}秒")
        logger.info(f"输出文件路径: {output_path}")
        
        # 验证文件是否存在和内容
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"文件大小: {file_size} 字节")
            
            # 检查HTML内容中是否包含Bootstrap和表格
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                has_bootstrap = "bootstrap" in content.lower()
                has_responsive_table = "table-responsive" in content.lower() or "table-striped" in content.lower()
                has_math = "mathjax" in content.lower() or "<div class=\"math\">" in content.lower()
                
            logger.info(f"Bootstrap支持: {has_bootstrap}")
            logger.info(f"响应式表格: {has_responsive_table}")
            logger.info(f"数学公式支持: {has_math}")
            
            return True
        else:
            logger.error("HTML文件未成功生成")
            return False
    except Exception as e:
        logger.error(f"HTML生成失败: {str(e)}")
        return False

def test_pdf_output(output_writer, test_content, filename="test_output"):
    """测试PDF输出功能"""
    logger.info("=== 测试PDF输出功能 ===")
    
    try:
        start_time = time.time()
        output_path = output_writer.generate_pdf(test_content, filename, "KnowForge输出测试")
        duration = time.time() - start_time
        
        logger.info(f"PDF生成成功，耗时: {duration:.4f}秒")
        logger.info(f"输出文件路径: {output_path}")
        
        # 验证文件是否存在
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"文件大小: {file_size} 字节")
            
            # 检查PDF扩展名
            is_pdf = output_path.lower().endswith('.pdf')
            logger.info(f"是否为真正的PDF文件: {is_pdf}")
            
            return True
        else:
            logger.error("PDF文件未成功生成")
            return False
    except Exception as e:
        logger.error(f"PDF生成失败: {str(e)}")
        return False

def test_notebook_output(output_writer, test_content, filename="test_output"):
    """测试Jupyter Notebook输出功能"""
    logger.info("=== 测试Jupyter Notebook输出功能 ===")
    
    try:
        start_time = time.time()
        output_path = output_writer.generate_notebook(test_content, filename, "KnowForge输出测试")
        duration = time.time() - start_time
        
        logger.info(f"Notebook生成成功，耗时: {duration:.4f}秒")
        logger.info(f"输出文件路径: {output_path}")
        
        # 验证文件是否存在和内容
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"文件大小: {file_size} 字节")
            
            # 导入nbformat检查笔记本结构
            import nbformat
            
            nb = nbformat.read(output_path, as_version=4)
            cell_count = len(nb.cells)
            
            # 检查单元格是否正确分隔
            logger.info(f"笔记本包含 {cell_count} 个单元格")
            
            # 检查特殊元素是否被单独分离为单元格
            has_table_cell = False
            has_code_cell = False
            has_math_cell = False
            
            for cell in nb.cells:
                content = cell.get('source', '')
                if content and '|' in content and '----' in content:
                    has_table_cell = True
                if content and '```python' in content:
                    has_code_cell = True
                if content and ('$$' in content or '$E = mc^2$' in content):
                    has_math_cell = True
            
            logger.info(f"表格单独分隔为单元格: {has_table_cell}")
            logger.info(f"代码块单独分隔为单元格: {has_code_cell}")
            logger.info(f"数学公式单独分隔为单元格: {has_math_cell}")
            
            return True
        else:
            logger.error("Notebook文件未成功生成")
            return False
    except Exception as e:
        logger.error(f"Notebook生成失败: {str(e)}")
        return False

def main():
    """主函数，运行测试"""
    parser = argparse.ArgumentParser(description='KnowForge输出功能测试工具')
    parser.add_argument('--workspace', default='workspace', help='工作空间目录')
    parser.add_argument('--output', default='output', help='输出目录')
    parser.add_argument('--format', choices=['all', 'markdown', 'html', 'pdf', 'notebook'], 
                      default='all', help='要测试的输出格式')
    parser.add_argument('--filename', default='output_test', help='输出文件名(不含扩展名)')
    args = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(args.workspace, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    
    logger.info("===== KnowForge 0.1.7 输出功能测试开始 =====")
    start_time = time.time()
    
    # 加载配置
    config = ConfigLoader("resources/config/config.yaml")
    
    # 初始化输出写入器
    output_writer = OutputWriter(args.workspace, args.output, config)
    
    # 生成测试内容
    test_content = generate_test_content()
    logger.info(f"已生成测试内容，包含 {len(test_content)} 个片段")
    
    # 根据参数决定测试哪些格式
    success_count = 0
    total_tests = 0
    
    if args.format in ['all', 'markdown']:
        total_tests += 1
        if test_markdown_output(output_writer, test_content, args.filename):
            success_count += 1
    
    if args.format in ['all', 'html']:
        total_tests += 1
        if test_html_output(output_writer, test_content, args.filename):
            success_count += 1
    
    if args.format in ['all', 'pdf']:
        total_tests += 1
        if test_pdf_output(output_writer, test_content, args.filename):
            success_count += 1
    
    if args.format in ['all', 'notebook']:
        total_tests += 1
        if test_notebook_output(output_writer, test_content, args.filename):
            success_count += 1
    
    # 输出测试结果统计
    duration = time.time() - start_time
    logger.info(f"\n===== 测试完成 ({success_count}/{total_tests} 通过) =====")
    logger.info(f"总耗时: {duration:.2f}秒")
    
    if success_count == total_tests:
        logger.info("全部测试通过！")
        return 0
    else:
        logger.warning(f"有 {total_tests - success_count} 项测试未通过，请检查日志")
        return 1

if __name__ == "__main__":
    sys.exit(main())
