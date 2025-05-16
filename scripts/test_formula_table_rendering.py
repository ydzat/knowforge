#!/usr/bin/env python3
'''
 * @Author: @ydzat
 * @Date: 2025-06-01 15:00:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-06-01 15:00:00
 * @Description: 测试KnowForge 0.1.7中表格和数学公式的增强渲染能力
'''
import os
import sys
import argparse
import logging

# 添加src到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.output_writer import OutputWriter
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger, get_logger

# 配置日志
setup_logger()
logger = get_logger('KnowForge-FormulaTableTest')

def generate_formula_examples():
    """生成各种复杂度的LaTeX公式示例"""
    content = []
    
    content.append("# LaTeX数学公式渲染测试")
    content.append("\n本文档测试KnowForge 0.1.7在各种输出格式中对LaTeX公式的渲染能力。\n")
    
    # 简单行内公式
    content.append("## 1. 行内公式")
    content.append("行内公式是嵌入在文本中的公式，使用单美元符号$...$包围。\n")
    content.append("- 简单算术: $a + b = c$")
    content.append("- 二次方程: $ax^2 + bx + c = 0$")
    content.append("- 分数: $\\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}$")
    content.append("- 平方根: $\\sqrt{x^2 + y^2} = r$")
    content.append("- 求和: $\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$")
    content.append("- 无穷级数: $\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}$\n")
    
    # 基础块级公式
    content.append("## 2. 基础块级公式")
    content.append("块级公式使用双美元符号$$...$$包围，独占一行或多行。\n")
    
    # 二次方程求根公式
    content.append("### 2.1 二次方程求根公式")
    content.append("$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$")
    
    # 泰勒展开式
    content.append("### 2.2 泰勒展开式")
    content.append("$$f(x) = f(a) + \\frac{f'(a)}{1!}(x-a) + \\frac{f''(a)}{2!}(x-a)^2 + \\cdots$$")
    
    # 欧拉恒等式
    content.append("### 2.3 欧拉恒等式")
    content.append("$$e^{i\\pi} + 1 = 0$$")
    
    # 高级数学公式
    content.append("## 3. 高级数学公式")
    
    # 定积分
    content.append("### 3.1 定积分")
    content.append("$$\\int_{a}^{b} f(x) \\, dx = F(b) - F(a)$$")
    
    # 多重积分
    content.append("### 3.2 多重积分")
    content.append("$$\\iiint_V \\rho(x,y,z) \\, dV = \\iiint_V \\rho(x,y,z) \\, dx\\,dy\\,dz$$")
    
    # 向量微积分
    content.append("### 3.3 向量微积分")
    content.append("$$\\nabla \\cdot \\vec{F} = \\frac{\\partial F_x}{\\partial x} + \\frac{\\partial F_y}{\\partial y} + \\frac{\\partial F_z}{\\partial z}$$")
    content.append("$$\\nabla \\times \\vec{F} = \\begin{pmatrix} \\frac{\\partial F_z}{\\partial y} - \\frac{\\partial F_y}{\\partial z} \\\\ \\frac{\\partial F_x}{\\partial z} - \\frac{\\partial F_z}{\\partial x} \\\\ \\frac{\\partial F_y}{\\partial x} - \\frac{\\partial F_x}{\\partial y} \\end{pmatrix}$$")
    
    # 复杂数学结构
    content.append("## 4. 复杂数学结构")
    
    # 矩阵
    content.append("### 4.1 矩阵")
    content.append("$$A = \\begin{pmatrix} a_{11} & a_{12} & \\cdots & a_{1n} \\\\ a_{21} & a_{22} & \\cdots & a_{2n} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ a_{m1} & a_{m2} & \\cdots & a_{mn} \\end{pmatrix}$$")
    
    # 行列式
    content.append("### 4.2 行列式")
    content.append("$$\\det(A) = |A| = \\begin{vmatrix} a_{11} & a_{12} & \\cdots & a_{1n} \\\\ a_{21} & a_{22} & \\cdots & a_{2n} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ a_{n1} & a_{n2} & \\cdots & a_{nn} \\end{vmatrix}$$")
    
    # 分段函数
    content.append("### 4.3 分段函数")
    content.append("$$f(x) = \\begin{cases} x^2, & \\text{if } x \\geq 0 \\\\ -x^2, & \\text{if } x < 0 \\end{cases}$$")
    
    # 概率与统计
    content.append("## 5. 概率与统计公式")
    
    # 正态分布
    content.append("### 5.1 正态分布")
    content.append("$$f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2}$$")
    
    # 贝叶斯定理
    content.append("### 5.2 贝叶斯定理")
    content.append("$$P(A|B) = \\frac{P(B|A) \\, P(A)}{P(B)}$$")
    
    # 物理学公式
    content.append("## 6. 物理学公式")
    
    # 薛定谔方程
    content.append("### 6.1 薛定谔方程")
    content.append("$$i\\hbar\\frac{\\partial}{\\partial t}\\Psi(\\mathbf{r},t) = \\hat H\\Psi(\\mathbf{r},t)$$")
    
    # 爱因斯坦场方程
    content.append("### 6.2 爱因斯坦场方程")
    content.append("$$R_{\\mu\\nu} - \\frac{1}{2}Rg_{\\mu\\nu} + \\Lambda g_{\\mu\\nu} = \\frac{8\\pi G}{c^4}T_{\\mu\\nu}$$")
    
    # 麦克斯韦方程组
    content.append("### 6.3 麦克斯韦方程组")
    content.append("$$\\begin{aligned} \\nabla \\cdot \\vec{E} &= \\frac{\\rho}{\\epsilon_0} \\\\ \\nabla \\cdot \\vec{B} &= 0 \\\\ \\nabla \\times \\vec{E} &= -\\frac{\\partial \\vec{B}}{\\partial t} \\\\ \\nabla \\times \\vec{B} &= \\mu_0 \\vec{J} + \\mu_0 \\epsilon_0 \\frac{\\partial \\vec{E}}{\\partial t} \\end{aligned}$$")
    
    return content

def generate_table_examples():
    """生成各种复杂度的表格示例"""
    content = []
    
    content.append("# 表格渲染测试")
    content.append("\n本文档测试KnowForge 0.1.7在各种输出格式中对表格的渲染能力。\n")
    
    # 简单表格
    content.append("## 1. 基础表格")
    content.append("最基本的Markdown表格包含标题行和数据行：\n")
    content.append("| 姓名 | 年龄 | 职业 |")
    content.append("| ---- | ---- | ---- |")
    content.append("| 张三 | 28 | 工程师 |")
    content.append("| 李四 | 32 | 设计师 |")
    content.append("| 王五 | 45 | 经理 |\n")
    
    # 对齐方式
    content.append("## 2. 表格对齐方式")
    content.append("表格可以设置不同的对齐方式：左对齐、居中和右对齐：\n")
    content.append("| 左对齐 | 居中对齐 | 右对齐 |")
    content.append("| :---- | :----: | ----: |")
    content.append("| 靠左 | 居中 | 靠右 |")
    content.append("| 文本 | 文本 | 文本 |")
    content.append("| 数据 | 数据 | 数据 |\n")
    
    # 复杂表格
    content.append("## 3. 复杂表格")
    
    # 多列多行
    content.append("### 3.1 多列多行表格")
    content.append("| 产品 | Q1 销量 | Q2 销量 | Q3 销量 | Q4 销量 | 总计 |")
    content.append("| ---- | ---: | ---: | ---: | ---: | ---: |")
    content.append("| 产品A | 100 | 120 | 130 | 140 | 490 |")
    content.append("| 产品B | 85 | 90 | 95 | 105 | 375 |")
    content.append("| 产品C | 50 | 55 | 60 | 65 | 230 |")
    content.append("| **总计** | **235** | **265** | **285** | **310** | **1095** |\n")
    
    # 嵌套格式
    content.append("### 3.2 表格中的格式化文本")
    content.append("表格中可以包含各种Markdown格式的文本：\n")
    content.append("| 特性 | 描述 | 示例 |")
    content.append("| ---- | ---- | ---- |")
    content.append("| **粗体** | 使用双星号 | **重要信息** |")
    content.append("| *斜体* | 使用单星号 | *强调文本* |")
    content.append("| `代码` | 使用反引号 | `var x = 10;` |")
    content.append("| [链接](https://example.com) | 使用方括号和圆括号 | [查看详情](https://example.com) |")
    content.append("| 组合格式 | 多种格式组合 | **粗体中的*斜体*和`代码`** |\n")
    
    # 数据分析表格
    content.append("## 4. 数据分析表格")
    
    # 统计数据
    content.append("### 4.1 数据统计表格")
    content.append("| 指标 | 最小值 | 第一四分位数 | 中位数 | 第三四分位数 | 最大值 | 平均值 | 标准差 |")
    content.append("| ---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    content.append("| 身高(cm) | 155 | 165 | 170 | 175 | 190 | 169.5 | 7.8 |")
    content.append("| 体重(kg) | 45 | 55 | 62 | 70 | 95 | 63.2 | 10.3 |")
    content.append("| 年龄(岁) | 18 | 25 | 30 | 42 | 65 | 34.7 | 12.6 |")
    content.append("| 收入(万元/年) | 10 | 15 | 25 | 40 | 120 | 32.5 | 22.4 |\n")
    
    # 分类比较
    content.append("### 4.2 多类别比较表格")
    content.append("| 算法 | 准确率 | 召回率 | F1 分数 | 训练时间(s) | 预测时间(ms) | 内存占用(MB) |")
    content.append("| ---- | ---: | ---: | ---: | ---: | ---: | ---: |")
    content.append("| 随机森林 | 0.92 | 0.89 | 0.90 | 120 | 5.2 | 450 |")
    content.append("| 梯度提升树 | 0.94 | 0.91 | 0.92 | 350 | 8.7 | 680 |")
    content.append("| 支持向量机 | 0.88 | 0.84 | 0.86 | 540 | 3.4 | 320 |")
    content.append("| 深度神经网络 | 0.96 | 0.95 | 0.95 | 1250 | 12.3 | 1500 |\n")
    
    # 表格和公式混合
    content.append("## 5. 表格与公式混合")
    content.append("下面的表格包含数学公式：\n")
    content.append("| 分布名称 | 概率密度函数 | 期望 | 方差 |")
    content.append("| ---- | ---- | ---- | ---- |")
    content.append("| 正态分布 | $f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2}$ | $\\mu$ | $\\sigma^2$ |")
    content.append("| 指数分布 | $f(x) = \\lambda e^{-\\lambda x}$ | $\\frac{1}{\\lambda}$ | $\\frac{1}{\\lambda^2}$ |")
    content.append("| 泊松分布 | $P(X=k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}$ | $\\lambda$ | $\\lambda$ |")
    content.append("| 二项分布 | $P(X=k) = \\binom{n}{k} p^k (1-p)^{n-k}$ | $np$ | $np(1-p)$ |\n")
    
    return content

def test_formulas_and_tables(output_writer, filename_prefix="formula_table_test"):
    """生成测试文件并输出为各种格式"""
    
    # 测试LaTeX公式
    formula_content = generate_formula_examples()
    formula_title = "KnowForge 0.1.7 - LaTeX公式渲染测试"
    
    # 输出公式测试文档
    formula_md = output_writer.generate_markdown(formula_content, f"{filename_prefix}_formulas", formula_title)
    formula_html = output_writer.generate_html(formula_content, f"{filename_prefix}_formulas", formula_title)
    formula_pdf = output_writer.generate_pdf(formula_content, f"{filename_prefix}_formulas", formula_title)
    formula_nb = output_writer.generate_notebook(formula_content, f"{filename_prefix}_formulas", formula_title)
    
    logger.info(f"公式测试文档已生成:")
    logger.info(f"  Markdown: {formula_md}")
    logger.info(f"  HTML: {formula_html}")
    logger.info(f"  PDF: {formula_pdf}")
    logger.info(f"  Jupyter Notebook: {formula_nb}")
    
    # 测试表格
    table_content = generate_table_examples()
    table_title = "KnowForge 0.1.7 - 表格渲染测试"
    
    # 输出表格测试文档
    table_md = output_writer.generate_markdown(table_content, f"{filename_prefix}_tables", table_title)
    table_html = output_writer.generate_html(table_content, f"{filename_prefix}_tables", table_title)
    table_pdf = output_writer.generate_pdf(table_content, f"{filename_prefix}_tables", table_title)
    table_nb = output_writer.generate_notebook(table_content, f"{filename_prefix}_tables", table_title)
    
    logger.info(f"表格测试文档已生成:")
    logger.info(f"  Markdown: {table_md}")
    logger.info(f"  HTML: {table_html}")
    logger.info(f"  PDF: {table_pdf}")
    logger.info(f"  Jupyter Notebook: {table_nb}")
    
    # 生成综合测试文档（表格+公式）
    combined_content = []
    combined_content.append("# KnowForge 0.1.7 - 表格与公式综合测试")
    combined_content.append("\n本文档测试KnowForge 0.1.7在各种输出格式中同时处理表格和公式的能力。\n")
    
    # 添加一些表格内容
    combined_content.extend(table_content[5:15])
    combined_content.append("\n## 高级数学公式示例\n")
    combined_content.extend(formula_content[20:30])
    combined_content.append("\n## 表格与公式结合\n")
    combined_content.extend(table_content[-10:])
    
    combined_title = "KnowForge 0.1.7 - 表格与公式综合测试"
    
    # 输出综合测试文档
    combined_md = output_writer.generate_markdown(combined_content, f"{filename_prefix}_combined", combined_title)
    combined_html = output_writer.generate_html(combined_content, f"{filename_prefix}_combined", combined_title)
    combined_pdf = output_writer.generate_pdf(combined_content, f"{filename_prefix}_combined", combined_title)
    combined_nb = output_writer.generate_notebook(combined_content, f"{filename_prefix}_combined", combined_title)
    
    logger.info(f"综合测试文档已生成:")
    logger.info(f"  Markdown: {combined_md}")
    logger.info(f"  HTML: {combined_html}")
    logger.info(f"  PDF: {combined_pdf}")
    logger.info(f"  Jupyter Notebook: {combined_nb}")
    
    return {
        "formula": {
            "md": formula_md,
            "html": formula_html,
            "pdf": formula_pdf,
            "notebook": formula_nb
        },
        "table": {
            "md": table_md,
            "html": table_html,
            "pdf": table_pdf,
            "notebook": table_nb
        },
        "combined": {
            "md": combined_md,
            "html": combined_html,
            "pdf": combined_pdf,
            "notebook": combined_nb
        }
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowForge 0.1.7 表格与公式渲染测试工具')
    parser.add_argument('--output-dir', default='output', help='输出目录')
    parser.add_argument('--filename', default='formula_table_test', help='输出文件名前缀(不含扩展名)')
    parser.add_argument('--open-html', action='store_true', help='生成后尝试打开HTML文件(如果可用)')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("===== KnowForge 0.1.7 表格与公式渲染测试开始 =====")
    
    # 加载配置
    config = ConfigLoader("resources/config/config.yaml")
    
    # 初始化输出写入器
    workspace_dir = "workspace"
    output_writer = OutputWriter(workspace_dir, args.output_dir, config)
    
    # 运行测试
    results = test_formulas_and_tables(output_writer, args.filename)
    
    # 尝试打开HTML文件(如果--open-html选项被设置)
    if args.open_html:
        try:
            import webbrowser
            
            logger.info("尝试在浏览器中打开生成的HTML文件...")
            
            # 打开公式测试HTML
            formula_url = f"file://{os.path.abspath(results['formula']['html'])}"
            webbrowser.open(formula_url)
            
            # 打开表格测试HTML
            import time
            time.sleep(1)  # 等待1秒，避免浏览器同时打开多个标签页时的问题
            table_url = f"file://{os.path.abspath(results['table']['html'])}"
            webbrowser.open(table_url)
            
            # 打开综合测试HTML
            time.sleep(1)
            combined_url = f"file://{os.path.abspath(results['combined']['html'])}"
            webbrowser.open(combined_url)
            
        except Exception as e:
            logger.warning(f"无法打开HTML文件: {str(e)}")
    
    logger.info("===== 测试完成 =====")
    return 0

if __name__ == "__main__":
    sys.exit(main())
