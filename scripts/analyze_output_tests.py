#!/usr/bin/env python3
'''
 * @Author: @ydzat
 * @Date: 2025-06-02 16:00:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-06-02 16:00:00
 * @Description: KnowForge 0.1.7 输出测试结果分析与可视化工具
'''
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import setup_logger, get_logger

# 配置日志
setup_logger()
logger = get_logger('KnowForge-OutputAnalysis')

def generate_text_report(result_files):
    """生成纯文本格式的报告"""
    report = []
    report.append("=" * 80)
    report.append("KnowForge 0.1.7 输出测试结果报告")
    report.append("生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report.append("=" * 80)
    report.append("")
    
    # 处理PDF性能结果
    pdf_results = find_latest_results(result_files, "pdf_perf_results.json")
    if pdf_results:
        report.append("-" * 40)
        report.append("PDF输出性能测试结果")
        report.append("-" * 40)
        
        for complexity, data in pdf_results.get("results", {}).items():
            report.append(f"\n[{complexity.upper()}复杂度文档]")
            report.append(f"  - 生成时间: {data.get('time', 'N/A'):.2f}秒")
            report.append(f"  - 文件大小: {data.get('size', 0)/1024:.2f}KB")
            
        report.append("")
    
    # 处理Notebook测试结果
    notebook_results = find_latest_results(result_files, "notebook_test_results.json")
    if notebook_results:
        report.append("-" * 40)
        report.append("Notebook单元格分割结果")
        report.append("-" * 40)
        
        cells_data = notebook_results.get("cells_analysis", {})
        report.append(f"\n总单元格数量: {cells_data.get('total', 0)}")
        report.append(f"单元格类型分布:")
        for cell_type, count in cells_data.get("types", {}).items():
            report.append(f"  - {cell_type}: {count}")
        
        report.append("")
    
    # 处理表格和公式渲染结果
    formula_table_results = find_latest_results(result_files, "formula_table_results.json")
    if formula_table_results:
        report.append("-" * 40)
        report.append("表格和公式渲染测试结果")
        report.append("-" * 40)
        
        for format_name, data in formula_table_results.get("formats", {}).items():
            report.append(f"\n[{format_name.upper()}格式]")
            report.append(f"  - 表格渲染: {'✅ 成功' if data.get('table_success') else '❌ 失败'}")
            report.append(f"  - 公式渲染: {'✅ 成功' if data.get('formula_success') else '❌ 失败'}")
        
        report.append("")
    
    # 总结和建议
    report.append("=" * 40)
    report.append("总结和建议")
    report.append("=" * 40)
    
    # 自动生成建议
    suggestions = generate_suggestions(pdf_results, notebook_results, formula_table_results)
    for suggestion in suggestions:
        report.append(f"- {suggestion}")
    
    return "\n".join(report)

def find_latest_results(result_files, filename_pattern):
    """查找最新的指定类型结果文件"""
    matching_files = [f for f in result_files if filename_pattern in os.path.basename(f)]
    
    if not matching_files:
        return None
    
    # 按修改时间排序，取最新的
    latest_file = max(matching_files, key=os.path.getmtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"无法读取结果文件 {latest_file}: {str(e)}")
        return None

def generate_suggestions(pdf_results, notebook_results, formula_table_results):
    """基于测试结果生成改进建议"""
    suggestions = []
    
    # PDF相关建议
    if pdf_results:
        pdf_times = [data.get('time', 0) for _, data in pdf_results.get("results", {}).items()]
        if pdf_times and max(pdf_times) > 4.0:
            suggestions.append("PDF生成速度较慢，建议实施性能优化措施")
            
        if "complex" in pdf_results.get("results", {}) and "basic" in pdf_results.get("results", {}):
            complex_time = pdf_results["results"]["complex"].get("time", 0)
            basic_time = pdf_results["results"]["basic"].get("time", 0)
            if complex_time > basic_time * 4:
                suggestions.append("复杂文档处理效率显著低于简单文档，建议优化复杂元素渲染器")
    
    # Notebook相关建议
    if notebook_results:
        cells_data = notebook_results.get("cells_analysis", {})
        if cells_data.get("mixed_content_cells", 0) > 0:
            suggestions.append("检测到混合内容单元格，可能需要改进单元格分割算法")
            
        if cells_data.get("empty_cells", 0) > 1:
            suggestions.append("生成了多个空单元格，建议优化空行处理")
    
    # 表格和公式渲染建议
    if formula_table_results:
        formats = formula_table_results.get("formats", {})
        if any(not data.get("table_success") for _, data in formats.items()):
            suggestions.append("某些格式存在表格渲染问题，建议检查表格处理代码")
            
        if any(not data.get("formula_success") for _, data in formats.items()):
            suggestions.append("某些格式存在公式渲染问题，建议改进数学公式支持")
    
    # 通用建议
    suggestions.append("为持续测试创建自动化测试用例，确保功能正常")
    suggestions.append("考虑添加更多用户配置选项，以支持不同的输出偏好")
    
    return suggestions

def save_markdown_report(report, output_dir):
    """将报告保存为Markdown文件"""
    report_md = report.replace("=" * 80, "#")
    report_md = report_md.replace("=" * 40, "##")
    report_md = report_md.replace("-" * 40, "###")
    
    # 替换简单的文本格式为Markdown格式
    report_md = report_md.replace("✅ 成功", "`✅ 成功`")
    report_md = report_md.replace("❌ 失败", "`❌ 失败`")
    
    filename = f"output_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    return filepath

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowForge输出测试结果分析工具')
    parser.add_argument('--output-dir', type=str, default='output', 
                       help='输出目录，默认为项目根目录下的output文件夹')
    parser.add_argument('--format', choices=['text', 'markdown'], default='text',
                       help='报告格式，默认为文本格式')
    args = parser.parse_args()
    
    # 设置工作目录
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(workspace_dir, args.output_dir)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有结果文件
    result_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.json') and ('results' in file or 'test' in file):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        logger.error(f"在 {output_dir} 中未找到任何测试结果文件")
        return
    
    logger.info(f"找到 {len(result_files)} 个测试结果文件")
    
    # 生成报告
    report = generate_text_report(result_files)
    
    if args.format == 'text':
        # 打印到控制台
        print(report)
        
        # 也保存到文件
        filename = f"output_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"文本报告已保存到: {filepath}")
    else:
        # 生成并保存Markdown报告
        filepath = save_markdown_report(report, output_dir)
        logger.info(f"Markdown报告已保存到: {filepath}")
        
        # 也打印到控制台
        print(report)

if __name__ == "__main__":
    main()
