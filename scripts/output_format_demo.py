#!/usr/bin/env python3
'''
 * @Author: @ydzat
 * @Date: 2025-06-01 12:00:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-06-01 12:00:00
 * @Description: KnowForge 0.1.7 输出格式演示，包括HTML、PDF和Jupyter Notebook输出的增强功能
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
logger = get_logger('KnowForge-OutputDemo')

def generate_complex_demo():
    """
    生成包含复杂元素的演示内容，用于测试各种输出格式
    
    包括:
    1. 复杂表格 - 嵌套表头、多行合并
    2. 复杂数学公式 - 积分、矩阵等
    3. 代码块 - 多种语言
    4. 复杂列表结构
    5. 引用块和特殊格式
    """
    content = []
    
    # 标题和简介
    content.append("# KnowForge 0.1.7 输出格式演示")
    content.append("\n本文档展示了KnowForge 0.1.7版本的各种输出格式能力，包括增强的HTML、PDF和Jupyter Notebook输出。\n")
    
    # 介绍
    content.append("## 1. 概述")
    content.append("KnowForge 0.1.7版本增加了多种输出格式支持，改进了表格和数学公式渲染。主要特性包括：")
    content.append("- HTML输出：支持响应式设计，集成Bootstrap和MathJax")
    content.append("- PDF输出：支持表格和LaTeX公式，多种渲染方式")
    content.append("- Jupyter Notebook：优化的单元格分割和展示")
    content.append("\n这些功能使知识内容可以以更丰富、更美观的方式展现。\n")
    
    # 表格部分
    content.append("## 2. 表格支持")
    content.append("### 2.1 简单表格")
    content.append("以下是一个基本的Markdown表格：\n")
    content.append("| 功能 | 描述 | 版本支持 |")
    content.append("| ---- | ---- | ---- |")
    content.append("| HTML输出 | 带响应式设计的HTML生成 | 0.1.7+ |")
    content.append("| PDF输出 | 支持表格和公式的PDF生成 | 0.1.7+ |")
    content.append("| Notebook输出 | 优化单元格分割的Notebook | 0.1.7+ |\n")
    
    content.append("### 2.2 复杂表格")
    content.append("KnowForge支持更复杂的表格结构，在HTML和PDF输出中有更好的展示效果：\n")
    content.append("| 功能分类 | 子功能 | 描述 | 依赖库 |")
    content.append("| ---- | ---- | ---- | ---- |")
    content.append("| **输出格式** | HTML | 响应式网页设计 | markdown, Bootstrap |")
    content.append("| | PDF | 高质量文档输出 | weasyprint, fpdf |")
    content.append("| | Jupyter Notebook | 交互式笔记本 | nbformat |")
    content.append("| **渲染特性** | 数学公式 | LaTeX数学表达式 | mdx_math, MathJax |")
    content.append("| | 代码高亮 | 语法高亮显示 | pygments |")
    content.append("| | 自动目录 | 根据标题生成目录 | 内置功能 |\n")
    
    # 数学公式部分
    content.append("## 3. 数学公式支持")
    content.append("KnowForge支持LaTeX数学公式渲染。在HTML输出中使用MathJax，在PDF中使用weasyprint或备用渲染。\n")
    
    content.append("### 3.1 行内公式")
    content.append("以下是一些行内公式示例：")
    content.append("- 质能方程：$E = mc^2$")
    content.append("- 欧拉公式：$e^{i\\pi} + 1 = 0$")
    content.append("- 平方和公式：$\\sum_{i=1}^{n} i^2 = \\frac{n(n+1)(2n+1)}{6}$\n")
    
    content.append("### 3.2 块级公式")
    content.append("更复杂的公式可以使用块级模式：\n")
    
    # 微积分公式
    content.append("#### 微积分公式")
    content.append("$$\\int_{a}^{b} f(x) \\, dx = F(b) - F(a)$$")
    
    # 矩阵公式
    content.append("#### 矩阵表示")
    content.append("$$A = \\begin{pmatrix} a_{11} & a_{12} & a_{13} \\\\ a_{21} & a_{22} & a_{23} \\\\ a_{31} & a_{32} & a_{33} \\end{pmatrix}$$")
    
    # 复杂公式
    content.append("#### 麦克斯韦方程组")
    content.append("$$\\begin{aligned} \\nabla \\cdot \\vec{E} &= \\frac{\\rho}{\\epsilon_0} \\\\ \\nabla \\cdot \\vec{B} &= 0 \\\\ \\nabla \\times \\vec{E} &= -\\frac{\\partial \\vec{B}}{\\partial t} \\\\ \\nabla \\times \\vec{B} &= \\mu_0 \\vec{J} + \\mu_0 \\epsilon_0 \\frac{\\partial \\vec{E}}{\\partial t} \\end{aligned}$$")
    
    # 代码块部分
    content.append("## 4. 代码块支持")
    content.append("KnowForge支持多种编程语言的代码高亮显示。以下是几个示例：\n")
    
    # Python示例
    content.append("### 4.1 Python代码")
    content.append("```python\nimport numpy as np\n\ndef calculate_covariance(X):\n    \"\"\"计算协方差矩阵\"\"\"\n    # 减去均值\n    X_centered = X - np.mean(X, axis=0)\n    # 计算协方差矩阵\n    cov_matrix = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)\n    return cov_matrix\n\n# 示例使用\ndata = np.random.randn(100, 3)\ncov = calculate_covariance(data)\nprint(f\"协方差矩阵:\\n{cov}\")\n```\n")
    
    # JavaScript示例
    content.append("### 4.2 JavaScript代码")
    content.append("```javascript\n// 一个简单的Promise示例\nfunction fetchUserData(userId) {\n    return new Promise((resolve, reject) => {\n        setTimeout(() => {\n            if (userId > 0) {\n                const userData = {\n                    id: userId,\n                    name: `User${userId}`,\n                    role: 'member'\n                };\n                resolve(userData);\n            } else {\n                reject(new Error('Invalid user ID'));\n            }\n        }, 1000);\n    });\n}\n\n// 使用async/await调用\nasync function displayUserInfo(userId) {\n    try {\n        const user = await fetchUserData(userId);\n        console.log(`User info: ${JSON.stringify(user)}`);\n    } catch (error) {\n        console.error(`Failed to fetch user: ${error.message}`);\n    }\n}\n\ndisplayUserInfo(42);\n```\n")
    
    # SQL示例
    content.append("### 4.3 SQL代码")
    content.append("```sql\n-- 创建用户表\nCREATE TABLE users (\n    user_id SERIAL PRIMARY KEY,\n    username VARCHAR(50) UNIQUE NOT NULL,\n    email VARCHAR(100) UNIQUE NOT NULL,\n    password_hash VARCHAR(255) NOT NULL,\n    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,\n    last_login TIMESTAMP WITH TIME ZONE\n);\n\n-- 查询活跃用户\nSELECT \n    u.username, \n    u.email, \n    COUNT(p.post_id) AS post_count\nFROM \n    users u\nJOIN \n    posts p ON u.user_id = p.user_id\nWHERE \n    p.created_at > CURRENT_DATE - INTERVAL '30 days'\nGROUP BY \n    u.user_id, u.username, u.email\nHAVING \n    COUNT(p.post_id) > 5\nORDER BY \n    post_count DESC;\n```\n")
    
    # 复杂列表结构
    content.append("## 5. 复杂列表结构")
    content.append("KnowForge支持复杂的列表结构和嵌套格式：\n")
    
    content.append("### 5.1 嵌套列表")
    content.append("1. **输出格式**")
    content.append("   - Markdown")
    content.append("     - 基本文本格式")
    content.append("     - 链接和图片")
    content.append("     - 表格支持")
    content.append("   - HTML")
    content.append("     - 响应式设计")
    content.append("     - Bootstrap集成")
    content.append("     - 数学公式渲染")
    content.append("   - PDF")
    content.append("     - 通过weasyprint渲染")
    content.append("     - 备用fpdf渲染")
    content.append("2. **处理功能**")
    content.append("   - 自动目录生成")
    content.append("     - 基于标题结构")
    content.append("     - 自动锚点链接")
    content.append("   - 代码高亮")
    content.append("     - 多语言支持")
    content.append("   - 数学公式")
    content.append("     - 行内公式")
    content.append("     - 块级公式")

    # 混合内容部分
    content.append("## 6. 混合内容示例")
    content.append("以下是同时包含表格、代码和数学公式的混合内容示例：\n")
    
    content.append("### 6.1 机器学习算法比较")
    content.append("下表比较了几种常见的机器学习算法：\n")
    
    content.append("| 算法 | 类型 | 优势 | 劣势 | 典型使用场景 |")
    content.append("| --- | --- | --- | --- | --- |")
    content.append("| 线性回归 | 监督学习 | 简单、可解释性强 | 只能建模线性关系 | 预测连续值，如房价 |")
    content.append("| 逻辑回归 | 监督学习 | 概率输出、计算高效 | 只能线性分类 | 二分类问题，如垃圾邮件检测 |")
    content.append("| 决策树 | 监督学习 | 易于理解、可处理分类和连续特征 | 容易过拟合 | 特征重要性分析 |")
    content.append("| 随机森林 | 集成学习 | 准确率高、不易过拟合 | 计算量大、黑盒模型 | 高维特征的分类和回归 |")
    content.append("| K-均值 | 无监督学习 | 简单、可扩展 | 需要预先确定K值 | 客户细分、图像压缩 |\n")
    
    content.append("决策树的分裂条件可以用以下公式表示：\n")
    content.append("$$Gain(D, a) = Ent(D) - \\sum_{v=1}^{V}\\frac{|D^v|}{|D|}Ent(D^v)$$")
    
    content.append("下面是一个简单的随机森林实现：\n")
    
    content.append("```python\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.datasets import make_classification\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\n\n# 生成示例数据\nX, y = make_classification(\n    n_samples=1000, n_features=20, n_informative=15,\n    n_redundant=5, random_state=42\n)\n\n# 分割训练集和测试集\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\n# 训练随机森林模型\nrf = RandomForestClassifier(\n    n_estimators=100, max_depth=None,\n    min_samples_split=2, random_state=42\n)\nrf.fit(X_train, y_train)\n\n# 预测并评估\ny_pred = rf.predict(X_test)\naccuracy = accuracy_score(y_test, y_pred)\nprint(f\"随机森林准确率: {accuracy:.4f}\")\n```\n")
    
    # 引用和特殊格式
    content.append("## 7. 引用和特殊格式")
    content.append("### 7.1 块引用")
    content.append("> \"知识就是力量\"——培根\n")
    content.append("> 多行引用示例：\n> 数据可以转化为信息，\n> 信息可以转化为知识，\n> 而知识则可以转化为智慧。\n")
    
    content.append("### 7.2 强调与高亮")
    content.append("- **重要信息**可以用粗体标记")
    content.append("- *斜体文本*用于强调")
    content.append("- ~~删除线~~用于标记废弃内容")
    content.append("- `行内代码`用于表示代码片段")
    content.append("- ==高亮文本==在某些Markdown解释器中可用\n")
    
    content.append("### 7.3 特殊符号和表情符号")
    content.append("- 特殊符号: © ® ™ ° ± ≠ ≤ ≥ ÷ × ∑ ∏ √ ∞")
    content.append("- 表情符号: 😊 🚀 📚 💡 🔍 ⚠️ ✅ ❌\n")
    
    # 结论
    content.append("## 8. 结论")
    content.append("KnowForge 0.1.7的输出增强功能显著提升了知识展示的质量和多样性。多种格式支持使内容呈现更加灵活，可以根据不同场景选择合适的展示方式。")
    content.append("\n特别是对于包含复杂表格、数学公式和代码的技术文档，新版本提供了卓越的渲染效果。随着未来版本的迭代，我们期待KnowForge提供更多创新功能和更优质的用户体验。")
    
    return content

def run_output_demo(output_writer, content, formats=None, filename="output_demo"):
    """
    运行输出格式演示
    
    Args:
        output_writer: OutputWriter实例
        content: 要输出的内容
        formats: 要生成的输出格式列表
        filename: 输出文件名（不含扩展名）
    """
    if formats is None:
        formats = ["markdown", "html", "pdf", "notebook"]
    
    title = "KnowForge 0.1.7 输出格式演示"
    output_paths = {}
    
    # 开始计时
    total_start_time = time.time()
    
    # 按格式生成输出
    if "markdown" in formats:
        logger.info("生成Markdown输出...")
        start_time = time.time()
        md_path = output_writer.generate_markdown(content, filename, title)
        duration = time.time() - start_time
        output_paths["markdown"] = md_path
        logger.info(f"Markdown生成完成，耗时: {duration:.2f}秒")
    
    if "html" in formats:
        logger.info("生成HTML输出...")
        start_time = time.time()
        html_path = output_writer.generate_html(content, filename, title)
        duration = time.time() - start_time
        output_paths["html"] = html_path
        logger.info(f"HTML生成完成，耗时: {duration:.2f}秒")
    
    if "pdf" in formats:
        logger.info("生成PDF输出...")
        start_time = time.time()
        pdf_path = output_writer.generate_pdf(content, filename, title)
        duration = time.time() - start_time
        output_paths["pdf"] = pdf_path
        logger.info(f"PDF生成完成，耗时: {duration:.2f}秒")
    
    if "notebook" in formats:
        logger.info("生成Jupyter Notebook输出...")
        start_time = time.time()
        nb_path = output_writer.generate_notebook(content, filename, title)
        duration = time.time() - start_time
        output_paths["notebook"] = nb_path
        logger.info(f"Notebook生成完成，耗时: {duration:.2f}秒")
    
    # 总耗时
    total_duration = time.time() - total_start_time
    logger.info(f"所有输出格式生成完成，总耗时: {total_duration:.2f}秒")
    
    return output_paths

def main():
    """主函数，运行演示"""
    parser = argparse.ArgumentParser(description='KnowForge 0.1.7 输出格式演示工具')
    parser.add_argument('--output-dir', default='output', help='输出目录')
    parser.add_argument('--formats', choices=['all', 'markdown', 'html', 'pdf', 'notebook'],
                        default='all', help='要生成的输出格式')
    parser.add_argument('--filename', default='output_demo', help='输出文件名(不含扩展名)')
    parser.add_argument('--open', action='store_true', help='生成后尝试打开HTML文件(如果可用)')
    args = parser.parse_args()
    
    # 设置格式列表
    formats = ["markdown", "html", "pdf", "notebook"] if args.formats == 'all' else [args.formats]
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("===== KnowForge 0.1.7 输出格式演示开始 =====")
    
    # 加载配置
    config = ConfigLoader("resources/config/config.yaml")
    
    # 初始化输出写入器
    workspace_dir = "workspace"
    output_writer = OutputWriter(workspace_dir, args.output_dir, config)
    
    # 生成演示内容
    logger.info("生成演示内容...")
    demo_content = generate_complex_demo()
    logger.info(f"演示内容已生成，包含 {len(demo_content)} 个片段")
    
    # 运行演示
    output_paths = run_output_demo(output_writer, demo_content, formats, args.filename)
    
    # 输出结果路径
    logger.info("\n===== 输出文件路径 =====")
    for format_name, path in output_paths.items():
        logger.info(f"{format_name.upper()}: {path}")
    
    # 尝试打开HTML文件(如果--open选项被设置且生成了HTML)
    if args.open and 'html' in output_paths:
        try:
            import webbrowser
            html_path = output_paths['html']
            file_url = f"file://{os.path.abspath(html_path)}"
            logger.info(f"尝试在浏览器中打开HTML文件: {file_url}")
            webbrowser.open(file_url)
        except Exception as e:
            logger.warning(f"无法打开HTML文件: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
