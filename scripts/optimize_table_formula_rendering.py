#!/usr/bin/env python3
'''
 * @Author: @ydzat
 * @Date: 2025-06-01 19:00:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-06-01 19:00:00
 * @Description: KnowForge 0.1.7 表格和公式渲染优化测试脚本
'''
import os
import sys
import time
import argparse
import logging
import json
from datetime import datetime

# 添加src到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.output_writer import OutputWriter
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger, get_logger

# 配置日志
setup_logger()
logger = get_logger('KnowForge-TableFormulaOptimizer')

def generate_table_test_content():
    """生成各种复杂度的表格测试内容"""
    
    content = []
    
    content.append("# 表格渲染优化测试")
    content.append("\n本文档测试KnowForge中表格渲染功能的各种场景和边缘情况。\n")
    
    # 1. 基本表格
    content.append("## 1. 基本表格")
    content.append("\n最简单的表格，只有标题和内容行：\n")
    content.append("| 列1 | 列2 | 列3 |")
    content.append("|-----|-----|-----|")
    content.append("| 数据1 | 数据2 | 数据3 |")
    content.append("| 数据4 | 数据5 | 数据6 |")
    
    # 2. 复杂表格 - 列宽不等
    content.append("\n## 2. 不等列宽表格")
    content.append("\n列宽不等的表格，测试排版效果：\n")
    content.append("| 短列 | 中等长度的列 | 这是一个非常非常非常非常非常非常长的列标题 |")
    content.append("|------|------------|--------------------------------------|")
    content.append("| 短 | 中等长度的数据 | 这是一个非常非常非常非常非常非常长的数据项 |")
    content.append("| 短 | 中 | 长长长长长长长长长长长长长长长长长长长长长长长长长长长 |")
    
    # 3. 对齐方式
    content.append("\n## 3. 表格对齐方式")
    content.append("\n测试不同的列对齐方式：\n")
    content.append("| 左对齐 | 居中对齐 | 右对齐 |")
    content.append("|:------|:------:|------:|")
    content.append("| 文本 | 文本 | 文本 |")
    content.append("| 较长的文本 | 较长的文本 | 较长的文本 |")
    
    # 4. 大型表格
    content.append("\n## 4. 大型表格")
    content.append("\n测试大型表格的渲染性能：\n")
    content.append("| 序号 | 名称 | 描述 | 状态 | 优先级 | 完成度 | 备注 |")
    content.append("|------|------|------|------|--------|--------|------|")
    
    # 生成20行数据
    for i in range(1, 21):
        content.append(f"| {i} | 项目{i} | 这是项目{i}的描述 | {'完成' if i % 3 == 0 else '进行中' if i % 3 == 1 else '未开始'} | {'高' if i % 5 == 0 else '中' if i % 5 == 1 else '低'} | {i*5}% | 备注{i} |")
    
    # 5. 表格中的格式化文本
    content.append("\n## 5. 表格中的格式化文本")
    content.append("\n测试表格单元格中的各种格式化文本：\n")
    content.append("| 格式 | 示例 | 描述 |")
    content.append("|------|------|------|")
    content.append("| 粗体 | **粗体文本** | 使用双星号 |")
    content.append("| 斜体 | *斜体文本* | 使用单星号 |")
    content.append("| 代码 | `行内代码` | 使用反引号 |")
    content.append("| 链接 | [链接文本](http://example.com) | 方括号加圆括号 |")
    content.append("| 图片引用 | ![图片Alt](image.jpg) | 感叹号加方括号和圆括号 |")
    
    # 6. 嵌套结构
    content.append("\n## 6. 嵌套结构")
    content.append("\n测试表格与其他元素的嵌套：\n")
    content.append("| 类型 | 内容 |")
    content.append("|------|------|")
    content.append("| 列表 | - 项目1<br>- 项目2<br>- 项目3 |")
    content.append("| 公式 | $E=mc^2$ |")
    content.append("| 多行文本 | 第一行<br>第二行<br>第三行 |")
    
    # 7. 边缘情况
    content.append("\n## 7. 边缘情况")
    content.append("\n测试表格渲染的边缘情况：\n")
    
    # 7.1 空单元格
    content.append("\n### 7.1 空单元格")
    content.append("\n包含空单元格的表格：\n")
    content.append("| A | B | C |")
    content.append("|---|---|---|")
    content.append("| 1 |  | 3 |")
    content.append("|  | 2 |  |")
    content.append("| 4 | 5 | 6 |")
    
    # 7.2 不一致列数
    content.append("\n### 7.2 不一致列数")
    content.append("\n列数不一致的表格：\n")
    content.append("| A | B | C |")
    content.append("|---|---|---|")
    content.append("| 1 | 2 |")
    content.append("| 3 | 4 | 5 | 6 |")
    content.append("| 7 |")
    
    # 7.3 特殊字符
    content.append("\n### 7.3 特殊字符")
    content.append("\n包含特殊字符的表格：\n")
    content.append("| 符号 | 显示 |")
    content.append("|------|------|")
    content.append("| 竖线 | \\| |")
    content.append("| 反斜杠 | \\\\ |")
    content.append("| 星号 | \\* |")
    content.append("| 下划线 | \\_ |")
    content.append("| 大于号 | > |")
    content.append("| HTML | <div>标签</div> |")
    
    return "\n".join(content)

def generate_formula_test_content():
    """生成各种复杂度的数学公式测试内容"""
    
    content = []
    
    content.append("# LaTeX公式渲染优化测试")
    content.append("\n本文档测试KnowForge中LaTeX公式渲染功能的各种场景和边缘情况。\n")
    
    # 1. 基础公式
    content.append("## 1. 基础行内公式")
    content.append("\n行内公式用单个美元符号包围，嵌入在文本中：\n")
    content.append("- 二次方程求根公式: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$")
    content.append("- 欧拉公式: $e^{i\\pi} + 1 = 0$")
    content.append("- 普通微分方程: $\\frac{dy}{dx} + P(x)y = Q(x)$")
    
    # 2. 块级公式
    content.append("\n## 2. 块级公式")
    content.append("\n块级公式用双美元符号包围，独占整行：\n")
    
    content.append("\n### 2.1 简单积分")
    content.append("\n$$\\int_{a}^{b} f(x) \\, dx = F(b) - F(a)$$")
    
    content.append("\n### 2.2 泰勒展开式")
    content.append("\n$$f(x) = f(a) + \\frac{f'(a)}{1!}(x-a) + \\frac{f''(a)}{2!}(x-a)^2 + \\cdots$$")
    
    content.append("\n### 2.3 矩阵表示")
    content.append("\n$$A = \\begin{bmatrix} a_{11} & a_{12} & a_{13} \\\\ a_{21} & a_{22} & a_{23} \\\\ a_{31} & a_{32} & a_{33} \\end{bmatrix}$$")
    
    # 3. 复杂数学结构
    content.append("\n## 3. 复杂数学结构")
    
    content.append("\n### 3.1 分段函数")
    content.append("\n$$f(x) = \\begin{cases} x^2, & \\text{if } x \\geq 0 \\\\ -x^2, & \\text{if } x < 0 \\end{cases}$$")
    
    content.append("\n### 3.2 多重积分")
    content.append("\n$$\\iiint_V \\nabla \\cdot \\mathbf{F} \\, dV = \\oiint_S \\mathbf{F} \\cdot \\mathbf{n} \\, dS$$")
    
    content.append("\n### 3.3 复杂方程组")
    content.append("\n$$\\begin{aligned} a_1 x + b_1 y + c_1 z &= d_1 \\\\ a_2 x + b_2 y + c_2 z &= d_2 \\\\ a_3 x + b_3 y + c_3 z &= d_3 \\end{aligned}$$")
    
    # 4. 专业领域公式
    content.append("\n## 4. 专业领域公式")
    
    content.append("\n### 4.1 物理学公式")
    content.append("\n狭义相对论质能方程：")
    content.append("\n$$E = \\gamma m c^2, \\text{ where } \\gamma = \\frac{1}{\\sqrt{1 - \\frac{v^2}{c^2}}}$$")
    
    content.append("\n麦克斯韦方程组：")
    content.append("\n$$\\begin{aligned} \\nabla \\cdot \\mathbf{E} &= \\frac{\\rho}{\\varepsilon_0} \\\\ \\nabla \\cdot \\mathbf{B} &= 0 \\\\ \\nabla \\times \\mathbf{E} &= -\\frac{\\partial \\mathbf{B}}{\\partial t} \\\\ \\nabla \\times \\mathbf{B} &= \\mu_0 \\mathbf{J} + \\mu_0 \\varepsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t} \\end{aligned}$$")
    
    content.append("\n### 4.2 量子力学公式")
    content.append("\n薛定谔方程：")
    content.append("\n$$i\\hbar\\frac{\\partial}{\\partial t}\\Psi(\\mathbf{r},t) = \\left [ -\\frac{\\hbar^2}{2m}\\nabla^2 + V(\\mathbf{r},t)\\right ] \\Psi(\\mathbf{r},t)$$")
    
    # 5. 公式与文本混合
    content.append("\n## 5. 公式与文本混合")
    content.append("\n本节测试公式与文本混合的情况。")
    content.append("\n首先，考虑公式 $f(x) = \\sin(x)$ 在区间 $[0, \\pi]$ 上的积分。我们可以计算：")
    content.append("\n$$\\int_{0}^{\\pi} \\sin(x) \\, dx = \\left. -\\cos(x) \\right|_{0}^{\\pi} = -\\cos(\\pi) - (-\\cos(0)) = -(-1) - (-1) = 1 + 1 = 2$$")
    content.append("\n接着，对于函数 $g(x) = e^{-x^2}$，我们知道 $\\int_{-\\infty}^{\\infty} e^{-x^2} \\, dx = \\sqrt{\\pi}$，这是一个重要的高斯积分。")

    # 6. 边缘情况
    content.append("\n## 6. 边缘情况")
    
    content.append("\n### 6.1 极长公式")
    content.append("\n$$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2} = \\frac{n^2 + n}{2} = \\frac{1}{2}n^2 + \\frac{1}{2}n = \\frac{1}{2}(n^2 + n) = \\frac{1}{2}n(n + 1) = \\frac{n(n+1)}{2}$$")
    
    content.append("\n### 6.2 嵌套结构")
    content.append("\n$$\\left( \\left[ \\left\\{ \\left| \\frac{\\left( \\frac{a}{b} \\right)}{\\left( \\frac{c}{d} \\right)} \\right| \\right\\} \\right] \\right)$$")
    
    content.append("\n### 6.3 特殊字符")
    content.append("\n$$\\text{特殊字符: }\\&, \\%, \\$, \\#, \\{, \\}, \\_$$")
    
    content.append("\n### 6.4 错误的公式")
    content.append("\n部分语法不正确的公式，测试渲染鲁棒性：")
    content.append("\n$\\frac{1}{2$$")
    content.append("\n$$\\sqrt{x + y}$$")
    
    return "\n".join(content)

def test_rendering_performance(writer, table_content, formula_content):
    """测试表格和公式的渲染性能"""
    results = {}
    
    # 定义输出格式
    formats = ["markdown", "html", "pdf", "notebook"]
    
    # 测试表格
    print("\n测试表格渲染性能...")
    table_results = {}
    
    for fmt in formats:
        print(f"  使用 {fmt} 格式...")
        try:
            start_time = time.time()
            if fmt == "markdown":
                output_path = writer.generate_markdown([table_content], f"table_test_{fmt}")
            elif fmt == "html":
                output_path = writer.generate_html([table_content], f"table_test_{fmt}")
            elif fmt == "pdf":
                output_path = writer.generate_pdf([table_content], f"table_test_{fmt}")
            else:  # notebook
                output_path = writer.generate_notebook([table_content], f"table_test_{fmt}")
            
            end_time = time.time()
            
            table_results[fmt] = {
                "time": end_time - start_time,
                "path": output_path,
                "success": True
            }
            print(f"    时间: {end_time - start_time:.3f}秒")
            
        except Exception as e:
            table_results[fmt] = {
                "time": None,
                "path": None,
                "success": False,
                "error": str(e)
            }
            print(f"    失败: {str(e)}")
    
    # 测试公式
    print("\n测试公式渲染性能...")
    formula_results = {}
    
    for fmt in formats:
        print(f"  使用 {fmt} 格式...")
        try:
            start_time = time.time()
            if fmt == "markdown":
                output_path = writer.generate_markdown([formula_content], f"formula_test_{fmt}")
            elif fmt == "html":
                output_path = writer.generate_html([formula_content], f"formula_test_{fmt}")
            elif fmt == "pdf":
                output_path = writer.generate_pdf([formula_content], f"formula_test_{fmt}")
            else:  # notebook
                output_path = writer.generate_notebook([formula_content], f"formula_test_{fmt}")
            
            end_time = time.time()
            
            formula_results[fmt] = {
                "time": end_time - start_time,
                "path": output_path,
                "success": True
            }
            print(f"    时间: {end_time - start_time:.3f}秒")
            
        except Exception as e:
            formula_results[fmt] = {
                "time": None,
                "path": None,
                "success": False,
                "error": str(e)
            }
            print(f"    失败: {str(e)}")
    
    # 返回结果
    results["table"] = table_results
    results["formula"] = formula_results
    return results

def analyze_results(results):
    """分析测试结果"""
    print("\n渲染性能分析:")
    
    # 对于表格和公式计算平均时间
    for content_type, content_results in results.items():
        print(f"\n{content_type.capitalize()}渲染:")
        
        format_times = []
        for fmt, fmt_result in content_results.items():
            if fmt_result["success"] and fmt_result["time"] is not None:
                format_times.append((fmt, fmt_result["time"]))
        
        # 按时间排序
        format_times.sort(key=lambda x: x[1])
        
        # 输出排序后的时间
        for fmt, time_taken in format_times:
            print(f"  {fmt}: {time_taken:.3f}秒")
        
        # 找出最快的格式
        if format_times:
            fastest_format, fastest_time = format_times[0]
            print(f"  最快的格式: {fastest_format} ({fastest_time:.3f}秒)")
        
        # 计算成功率
        success_count = sum(1 for r in content_results.values() if r["success"])
        success_rate = success_count / len(content_results) * 100
        print(f"  成功率: {success_rate:.1f}% ({success_count}/{len(content_results)})")
            
def suggest_optimizations(results):
    """基于测试结果提出优化建议"""
    print("\n优化建议:")
    
    # 检查表格渲染
    table_results = results.get("table", {})
    if any(not r["success"] for r in table_results.values()):
        print("\n表格渲染优化:")
        print("1. 解决表格渲染失败的问题")
        for fmt, result in table_results.items():
            if not result["success"]:
                print(f"   - {fmt} 格式渲染失败: {result.get('error', '未知错误')}")
    
    table_times = [(fmt, result["time"]) for fmt, result in table_results.items() if result["success"] and result["time"] is not None]
    if table_times:
        avg_table_time = sum(t for _, t in table_times) / len(table_times)
        if avg_table_time > 1.0:
            print("\n表格渲染性能优化:")
            print(f"2. 表格渲染平均时间较长 ({avg_table_time:.3f}秒)，建议:")
            print("   - 优化表格解析算法")
            print("   - 对大型表格实现分块渲染")
            print("   - 缓存已渲染的表格结构")
            print("   - 简化复杂表格的HTML结构")
    
    # 检查公式渲染
    formula_results = results.get("formula", {})
    if any(not r["success"] for r in formula_results.values()):
        print("\n公式渲染优化:")
        print("3. 解决公式渲染失败的问题")
        for fmt, result in formula_results.items():
            if not result["success"]:
                print(f"   - {fmt} 格式渲染失败: {result.get('error', '未知错误')}")
    
    formula_times = [(fmt, result["time"]) for fmt, result in formula_results.items() if result["success"] and result["time"] is not None]
    if formula_times:
        avg_formula_time = sum(t for _, t in formula_times) / len(formula_times)
        if avg_formula_time > 1.0:
            print("\n公式渲染性能优化:")
            print(f"4. 公式渲染平均时间较长 ({avg_formula_time:.3f}秒)，建议:")
            print("   - 使用更高效的公式解析库")
            print("   - 添加公式缓存机制")
            print("   - 考虑使用预编译的MathJax配置")
            print("   - PDF生成时可使用SVG而非PNG格式渲染公式")
    
    # 通用建议
    print("\n通用优化建议:")
    print("5. 改进渲染引擎选择策略:")
    print("   - 根据内容复杂度自动选择最合适的渲染方法")
    print("   - 添加用户可配置的渲染选项")
    print("6. 增强错误处理和回退机制:")
    print("   - 为复杂表格和公式添加专门的错误处理")
    print("   - 实现渲染失败后的自动回退到更简单格式")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowForge表格与公式渲染优化测试')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--content-type', choices=['all', 'table', 'formula'], 
                       default='all', help='测试内容类型')
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
    table_content = generate_table_test_content() if args.content_type in ['all', 'table'] else None
    formula_content = generate_formula_test_content() if args.content_type in ['all', 'formula'] else None
    
    # 测试渲染性能
    print("\n开始渲染测试...")
    results = test_rendering_performance(writer, table_content, formula_content)
    
    # 分析结果
    analyze_results(results)
    
    # 提出优化建议
    suggest_optimizations(results)
    
    # 保存测试结果
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    result_path = os.path.join(output_dir, "table_formula_perf_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n测试结果已保存至: {result_path}")

if __name__ == "__main__":
    main()
