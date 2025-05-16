#!/usr/bin/env python3
'''
 * @Author: @ydzat
 * @Date: 2025-06-02 17:00:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-06-02 17:00:00
 * @Description: KnowForge 0.1.7 增强的LaTeX公式渲染测试
'''
import os
import sys
import argparse
import logging
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.output_writer import OutputWriter
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger, get_logger

# 配置日志
setup_logger()
logger = get_logger('KnowForge-LaTeXTest')

def generate_advanced_latex_content():
    """生成各种复杂的LaTeX公式测试内容"""
    content = []
    
    content.append("# KnowForge 高级LaTeX公式渲染测试")
    content.append("\n本文档用于测试KnowForge对复杂LaTeX公式的渲染支持。\n")
    
    # 1. 多行公式与对齐
    content.append("## 1. 多行公式与对齐")
    content.append("\n多行公式通常需要特殊处理，尤其是对齐时：\n")
    content.append("$$")
    content.append("\\begin{align}")
    content.append("f(x) &= \\int_{-\\infty}^{\\infty} \\hat{f}(\\xi) e^{2\\pi i \\xi x} d\\xi \\\\")
    content.append("&= \\int_{-\\infty}^{\\infty} \\hat{f}(\\xi) \\cos(2\\pi \\xi x) d\\xi + i\\int_{-\\infty}^{\\infty} \\hat{f}(\\xi) \\sin(2\\pi \\xi x) d\\xi \\\\")
    content.append("&= f_1(x) + i f_2(x)")
    content.append("\\end{align}")
    content.append("$$")
    
    # 2. 矩阵与行列式
    content.append("\n## 2. 矩阵与行列式")
    content.append("\nLaTeX矩阵的不同表示方式：\n")
    
    # 基本矩阵
    content.append("### 2.1 基本矩阵")
    content.append("$$")
    content.append("\\begin{pmatrix}")
    content.append("a & b & c \\\\")
    content.append("d & e & f \\\\")
    content.append("g & h & i")
    content.append("\\end{pmatrix}")
    content.append("$$")
    
    # 带方括号的矩阵
    content.append("\n### 2.2 方括号矩阵")
    content.append("$$")
    content.append("\\begin{bmatrix}")
    content.append("a & b & c \\\\")
    content.append("d & e & f \\\\")
    content.append("g & h & i")
    content.append("\\end{bmatrix}")
    content.append("$$")
    
    # 带花括号的矩阵
    content.append("\n### 2.3 花括号矩阵")
    content.append("$$")
    content.append("\\begin{Bmatrix}")
    content.append("a & b & c \\\\")
    content.append("d & e & f \\\\")
    content.append("g & h & i")
    content.append("\\end{Bmatrix}")
    content.append("$$")
    
    # 行列式
    content.append("\n### 2.4 行列式")
    content.append("$$")
    content.append("\\begin{vmatrix}")
    content.append("a & b & c \\\\")
    content.append("d & e & f \\\\")
    content.append("g & h & i")
    content.append("\\end{vmatrix}")
    content.append("$$")
    
    # 3. 分段函数
    content.append("\n## 3. 分段函数")
    content.append("$$")
    content.append("f(x) = \\begin{cases}")
    content.append("x^2, & \\text{if } x \\geq 0 \\\\")
    content.append("\\sin(x), & \\text{if } -\\pi \\leq x < 0 \\\\")
    content.append("-x, & \\text{otherwise}")
    content.append("\\end{cases}")
    content.append("$$")
    
    # 4. 复杂的积分和极限
    content.append("\n## 4. 复杂积分与极限")
    
    content.append("### 4.1 多重积分")
    content.append("$$")
    content.append("\\iiint_{V} \\nabla \\cdot \\vec{F} dV = \\oiint_{S} \\vec{F} \\cdot \\hat{n} dS")
    content.append("$$")
    
    content.append("\n### 4.2 带有条件的极限")
    content.append("$$")
    content.append("\\lim_{x \\to 0^{+}} \\frac{\\sin(x)}{x} = 1")
    content.append("$$")
    
    # 5. 复杂的数学表达式与符号
    content.append("\n## 5. 复杂数学表达式")
    
    content.append("### 5.1 求和与乘积")
    content.append("$$")
    content.append("\\sum_{i=1}^{n} \\prod_{j=1}^{m} a_{ij} = \\prod_{j=1}^{m} \\sum_{i=1}^{n} a_{ij}")
    content.append("$$")
    
    content.append("\n### 5.2 分数嵌套")
    content.append("$$")
    content.append("\\frac{1}{1 + \\frac{1}{1 + \\frac{1}{1 + \\frac{1}{1 + a}}}} = \\cfrac{1}{1 + \\cfrac{1}{1 + \\cfrac{1}{1 + \\cfrac{1}{1 + a}}}}")
    content.append("$$")
    
    content.append("\n### 5.3 数学物理方程")
    content.append("$$")
    content.append("\\nabla \\times \\vec{\\mathbf{B}} -\\frac{1}{c}\\frac{\\partial\\vec{\\mathbf{E}}}{\\partial t} = \\frac{4\\pi}{c}\\vec{\\mathbf{j}}")
    content.append("$$")
    
    # 6. 行内公式测试
    content.append("\n## 6. 行内公式测试")
    content.append("\n在科学论文中，我们经常需要在文本中插入行内公式，例如爱因斯坦的著名方程 $E = mc^2$ 表示能量与质量的等价性。")
    content.append("薛定谔方程 $i\\hbar\\frac{\\partial}{\\partial t}\\Psi(\\mathbf{r},t) = \\hat H\\Psi(\\mathbf{r},t)$ 是量子力学的基本方程。")
    content.append("也可以在一行中使用多个行内公式，如 $a^2 + b^2 = c^2$ 和 $e^{i\\pi} + 1 = 0$ 都是著名的数学公式。")
    
    # 7. 特殊符号和命令
    content.append("\n## 7. 特殊符号和命令")
    content.append("\n以下是一些特殊的数学符号和命令：\n")
    content.append("- 无穷大符号: $\\infty$")
    content.append("- 集合符号: $A \\cup B \\cap C \\setminus D$")
    content.append("- 箭头符号: $\\rightarrow, \\Rightarrow, \\leftrightarrow, \\Leftrightarrow, \\mapsto$")
    content.append("- 希腊字母: $\\alpha, \\beta, \\gamma, \\delta, \\epsilon, \\varepsilon, \\zeta, \\eta, \\theta, \\vartheta, \\iota, \\kappa, \\lambda, \\mu, \\nu, \\xi, \\pi, \\varpi, \\rho, \\varrho, \\sigma, \\varsigma, \\tau, \\upsilon, \\phi, \\varphi, \\chi, \\psi, \\omega$")
    content.append("- 数学符号: $\\partial, \\nabla, \\propto, \\degree, \\angle, \\prime, \\pm, \\mp, \\times, \\div, \\cdot, \\equiv, \\approx, \\cong, \\simeq, \\sim, \\doteq, \\neq$")
    
    return content

def test_latex_rendering_across_formats():
    """测试所有输出格式中的LaTeX公式渲染"""
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(workspace_dir, "output")
    config = ConfigLoader(os.path.join(workspace_dir, "resources", "config"))
    
    output_writer = OutputWriter(workspace_dir, output_dir, config)
    
    # 生成测试内容
    content = generate_advanced_latex_content()
    
    logger.info("开始测试各格式的LaTeX公式渲染...")
    
    # 生成所有格式的输出
    results = {}
    
    # 1. Markdown格式 (基准，主要是为了生成内容)
    try:
        md_path = output_writer.generate_markdown(content, "advanced_latex_test")
        logger.info(f"Markdown输出已生成: {md_path}")
        results["markdown"] = {"path": md_path, "success": True}
    except Exception as e:
        logger.error(f"Markdown生成失败: {str(e)}")
        results["markdown"] = {"error": str(e), "success": False}
    
    # 2. HTML格式
    try:
        html_path = output_writer.generate_html(content, "advanced_latex_test")
        logger.info(f"HTML输出已生成: {html_path}")
        results["html"] = {"path": html_path, "success": True}
    except Exception as e:
        logger.error(f"HTML生成失败: {str(e)}")
        results["html"] = {"error": str(e), "success": False}
    
    # 3. PDF格式
    try:
        pdf_path = output_writer.generate_pdf(content, "advanced_latex_test")
        logger.info(f"PDF输出已生成: {pdf_path}")
        results["pdf"] = {"path": pdf_path, "success": True}
    except Exception as e:
        logger.error(f"PDF生成失败: {str(e)}")
        results["pdf"] = {"error": str(e), "success": False}
    
    # 4. Jupyter Notebook格式
    try:
        nb_path = output_writer.generate_notebook(content, "advanced_latex_test")
        logger.info(f"Notebook输出已生成: {nb_path}")
        results["notebook"] = {"path": nb_path, "success": True}
    except Exception as e:
        logger.error(f"Notebook生成失败: {str(e)}")
        results["notebook"] = {"error": str(e), "success": False}
    
    return results

def check_dependencies():
    """检查LaTeX公式渲染所需的依赖"""
    dependencies = {
        "markdown": False,
        "mdx_math": False,
        "weasyprint": False,
        "nbformat": False
    }
    
    try:
        import markdown
        dependencies["markdown"] = True
    except ImportError:
        logger.warning("未找到markdown包，这会影响HTML和PDF中的公式渲染")
    
    try:
        import mdx_math
        dependencies["mdx_math"] = True
    except ImportError:
        logger.warning("未找到mdx_math包，这会影响HTML和PDF中的LaTeX公式支持")
    
    try:
        import weasyprint
        dependencies["weasyprint"] = True
    except ImportError:
        logger.warning("未找到weasyprint包，这会影响PDF生成")
    
    try:
        import nbformat
        dependencies["nbformat"] = True
    except ImportError:
        logger.warning("未找到nbformat包，这会影响Notebook生成")
    
    return dependencies

def print_recommendations(results, dependencies):
    """打印建议改进方案"""
    logger.info("\nLaTeX公式渲染测试结果分析:")
    
    # 检查各格式的成功状态
    all_success = all(format_data.get("success", False) for format_data in results.values())
    
    if all_success:
        logger.info("✅ 所有格式的LaTeX公式渲染测试通过！")
    else:
        logger.warning("⚠️ 某些格式的LaTeX公式渲染测试失败")
    
    # 根据依赖情况提出建议
    logger.info("\n依赖状态及建议:")
    
    if not dependencies["markdown"]:
        logger.info("- 安装markdown包以支持基本的Markdown转HTML功能")
        logger.info("  运行: pip install markdown")
    
    if not dependencies["mdx_math"]:
        logger.info("- 安装mdx_math包以支持LaTeX公式渲染")
        logger.info("  运行: pip install python-markdown-math")
    
    if not dependencies["weasyprint"]:
        logger.info("- 安装weasyprint以获得更好的PDF生成能力")
        logger.info("  运行: pip install weasyprint")
    
    # 进一步的改进建议
    logger.info("\n进一步改进建议:")
    logger.info("1. 在OutputWriter中添加对不同LaTeX环境的更好支持")
    logger.info("2. 考虑增加本地MathJax支持，减少对在线资源的依赖")
    logger.info("3. 为PDF添加专门的LaTeX公式处理器以提高渲染质量")
    logger.info("4. 在Notebook输出中提供执行LaTeX公式的选项")
    logger.info("5. 添加更多用户自定义配置，如公式尺寸、字体等")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KnowForge高级LaTeX公式渲染测试')
    parser.add_argument('--check-only', action='store_true', help='仅检查依赖而不运行测试')
    args = parser.parse_args()
    
    # 首先检查依赖
    logger.info("检查LaTeX公式渲染依赖...")
    dependencies = check_dependencies()
    
    if args.check_only:
        print_recommendations({}, dependencies)
    else:
        # 运行渲染测试
        logger.info("开始LaTeX公式渲染测试...")
        results = test_latex_rendering_across_formats()
        
        # 打印建议
        print_recommendations(results, dependencies)
        logger.info("LaTeX公式渲染测试完成!")
