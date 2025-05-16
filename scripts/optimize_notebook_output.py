#!/usr/bin/env python3
'''
 * @Author: @ydzat
 * @Date: 2025-06-01 18:00:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-06-01 18:00:00
 * @Description: KnowForge 0.1.7 Jupyter Notebook输出优化测试脚本
'''
import os
import sys
import time
import argparse
import logging
from datetime import datetime
import json
import nbformat
from pprint import pprint

# 添加src到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.output_writer import OutputWriter
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger, get_logger

# 配置日志
setup_logger()
logger = get_logger('KnowForge-NotebookOptimizer')

def generate_test_content_with_complex_elements():
    """生成包含更多复杂元素的测试内容，用于测试Notebook单元格分割算法"""
    
    content = []
    
    content.append("# KnowForge Notebook 单元格优化测试")
    content.append("\n这是一个测试文档，用于测试和优化KnowForge的Jupyter Notebook输出功能。\n")
    
    # 添加嵌套标题结构
    content.append("## 1. 嵌套标题结构测试")
    content.append("\n这是一级子标题下的内容。\n")
    
    content.append("### 1.1 二级子标题测试")
    content.append("\n这是二级子标题下的内容。\n")
    
    content.append("#### 1.1.1 三级子标题测试")
    content.append("\n这是更深层次的标题结构测试，用于验证标题层次如何影响单元格分割。\n")
    
    # 添加多个类型混合的复杂内容
    content.append("## 2. 混合内容测试")
    
    # 文本 + 表格
    content.append("### 2.1 文本和表格的混合")
    content.append("\n以下是一个表格，后面紧跟着一段文字描述：\n")
    content.append("| 组件 | 功能描述 | 优化建议 |")
    content.append("|------|---------|---------|")
    content.append("| 单元格分割 | 将内容分成多个单元格 | 增强边缘情况处理 |")
    content.append("| 公式渲染 | 支持LaTeX数学公式 | 改进内联公式识别 |")
    content.append("\n这个表格展示了Notebook组件与相应的功能说明和优化建议。\n")
    
    # 代码块 + 文本
    content.append("### 2.2 代码和文本的混合")
    content.append("\n代码块前的说明文字。接下来是代码示例：\n")
    content.append("```python\ndef optimize_notebook_cells(content):\n    \"\"\"优化Notebook单元格分割\"\"\"\n    # 分析内容结构\n    cells = []\n    # 实现优化逻辑\n    return cells\n```")
    content.append("\n代码块后的解释性文字，说明代码实现的功能。这段文本应该与代码块保持在同一单元格中。\n")
    
    # 多个公式块和内联公式
    content.append("### 2.3 公式和文本的混合")
    content.append("\n以下是一个内联公式 $E=mc^2$ 和一个块级公式：\n")
    content.append("$$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$")
    content.append("\n另一个复杂的块级公式和解释：")
    content.append("$$\\frac{d}{dx}\\left( \\int_{a(x)}^{b(x)} f(t,x) \\, dt \\right) = f(b(x),x) \\frac{d}{dx}b(x) - f(a(x),x)\\frac{d}{dx}a(x) + \\int_{a(x)}^{b(x)} \\frac{\\partial f}{\\partial x}(t,x) \\, dt$$")
    content.append("\n这是莱布尼茨积分规则的一个变形。\n")
    
    # 表格 + 代码 + 公式的复杂混合
    content.append("## 3. 多元素组合测试")
    content.append("\n这部分测试多种元素的组合场景，包括表格、代码块和公式共存的情况。\n")
    
    content.append("### 3.1 表格后紧跟代码")
    content.append("\n| 序号 | 调用函数 | 返回值 |")
    content.append("|-----|---------|-------|")
    content.append("| 1 | optimize() | True |")
    content.append("| 2 | split_cells() | [cells] |")
    content.append("\n相应的实现代码：")
    content.append("```python\ndef optimize():\n    return True\n\ndef split_cells():\n    return ['cell1', 'cell2']\n```")
    
    content.append("### 3.2 代码后紧跟公式")
    content.append("\n代码实现：")
    content.append("```python\ndef calculate_area(radius):\n    return math.pi * radius**2\n```")
    content.append("\n计算公式为：")
    content.append("$$A = \\pi r^2$$")
    
    # 测试边缘情况
    content.append("## 4. 边缘情况测试")
    
    # 非常短的内容
    content.append("### 4.1 极短内容")
    content.append("\n一行文本。\n")
    
    # 多个连续空行
    content.append("### 4.2 连续空行")
    content.append("\n这前面有空行。\n\n\n\n这后面有多个空行。\n")
    
    # 连续多个代码块
    content.append("### 4.3 连续代码块")
    content.append("\n第一个代码块：")
    content.append("```python\nprint('First block')\n```")
    content.append("第二个代码块紧随其后：")
    content.append("```javascript\nconsole.log('Second block');\n```")
    
    # 连续多个公式块
    content.append("### 4.4 连续公式块")
    content.append("\n第一个公式：")
    content.append("$$E = mc^2$$")
    content.append("第二个公式：")
    content.append("$$F = ma$$")
    
    return "\n".join(content)

def analyze_notebook_cell_splitting(output_path):
    """分析Notebook单元格分割情况"""
    with open(output_path, "r", encoding="utf-8") as f:
        notebook_data = json.load(f)
    
    cells = notebook_data.get("cells", [])
    
    print(f"\n共生成 {len(cells)} 个单元格")
    
    # 分析各类型单元格
    cell_types = {}
    for i, cell in enumerate(cells):
        # 获取单元格内容
        source = "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else cell.get("source", "")
        
        # 确定单元格类型
        cell_type = "未知"
        if source.lstrip().startswith("#"):
            header_level = len(source.lstrip().split(" ")[0].strip("#"))
            cell_type = f"标题-L{header_level}"
        elif source.lstrip().startswith("|") and "---" in source:
            cell_type = "表格"
        elif "```" in source:
            cell_type = "代码块"
        elif source.lstrip().startswith("$$"):
            cell_type = "公式块"
        elif "$" in source:
            cell_type = "含内联公式"
        else:
            cell_type = "普通文本"
        
        # 统计各类型数量
        cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
        
        # 打印简短预览
        preview = source.split("\n")[0][:40] + ("..." if len(source.split("\n")[0]) > 40 else "")
        print(f"单元格 {i+1}: {cell_type} - {preview}")
    
    # 打印统计结果
    print("\n单元格类型统计:")
    for cell_type, count in cell_types.items():
        print(f"  {cell_type}: {count}个")
    
    return cells

def measure_generation_performance(writer, content, iterations=5):
    """测量Notebook生成性能"""
    print("\n性能测试中...")
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        output_path = writer.generate_notebook([content], f"perf_test_{i}")
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"  迭代 {i+1}: {elapsed:.3f}秒")
    
    avg_time = sum(times) / len(times)
    print(f"\n平均生成时间: {avg_time:.3f}秒")
    print(f"最短时间: {min(times):.3f}秒")
    print(f"最长时间: {max(times):.3f}秒")
    
    return avg_time, min(times), max(times)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowForge Notebook输出优化测试')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--analyze-only', action='store_true', help='只分析已生成的Notebook')
    parser.add_argument('--perf-test', action='store_true', help='执行性能测试')
    args = parser.parse_args()
    
    # 设置工作目录和输出目录
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.output_dir))
    
    # 加载配置
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'config', 'config.yaml'))
    config = ConfigLoader(config_path)
    
    # 创建输出写入器
    writer = OutputWriter(workspace_dir, output_dir, config)
    
    # 生成测试内容
    print("生成测试内容...")
    content = generate_test_content_with_complex_elements()
    
    # 如果不是只分析模式，则生成新的Notebook
    if not args.analyze_only:
        print("生成Notebook...")
        output_path = writer.generate_notebook([content], "notebook_optimization_test")
        print(f"Notebook已保存至: {output_path}")
    else:
        # 使用最新的测试文件
        output_path = os.path.join(output_dir, "notebook", "notebook_optimization_test.ipynb")
        print(f"分析现有Notebook: {output_path}")
    
    # 分析单元格分割
    cells = analyze_notebook_cell_splitting(output_path)
    
    # 性能测试
    if args.perf_test:
        print("\n执行性能基准测试...")
        avg_time, min_time, max_time = measure_generation_performance(writer, content)
        
        # 保存结果
        perf_results = {
            "timestamp": datetime.now().isoformat(),
            "notebook_size": len(content),
            "cell_count": len(cells),
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time
        }
        
        result_path = os.path.join(output_dir, "notebook_perf_results.json")
        
        # 加载现有结果或创建新结果文件
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = {"results": []}
        
        history["results"].append(perf_results)
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        print(f"性能测试结果已保存至: {result_path}")
    
    print("\n优化建议:")
    print("1. 考虑根据内容长度和复杂度调整单元格分割策略")
    print("2. 添加启发式规则，将相关内容保持在同一单元格中")
    print("3. 为用户提供单元格分割配置选项")
    print("4. 在单元格之间添加更好的过渡元素")
    print("5. 优化代码以提高性能，特别是对大型文档")

if __name__ == "__main__":
    main()
