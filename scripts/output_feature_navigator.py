#!/usr/bin/env python3
'''
 * @Author: @ydzat
 * @Date: 2025-06-01 16:30:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-06-01 16:30:00
 * @Description: KnowForge 0.1.7版本功能导航和批量测试工具
'''
import os
import sys
import time
import argparse
import logging
import subprocess
from datetime import datetime

# 添加src到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import setup_logger, get_logger

# 配置日志
setup_logger()
logger = get_logger('KnowForge-Navigator')

# 定义测试脚本列表
TEST_SCRIPTS = [
    {
        "name": "基础输出功能测试",
        "script": "test_output_features.py",
        "description": "测试基础输出功能，包括Markdown、HTML、PDF和Jupyter Notebook输出",
        "command": "python scripts/test_output_features.py"
    },
    {
        "name": "表格和公式渲染测试",
        "script": "test_formula_table_rendering.py",
        "description": "测试表格和LaTeX公式在各种输出格式中的渲染效果",
        "command": "python scripts/test_formula_table_rendering.py"
    },
    {
        "name": "完整输出演示",
        "script": "output_format_demo.py",
        "description": "生成包含各种复杂元素的演示文档，展示各种输出格式的完整能力",
        "command": "python scripts/output_format_demo.py"
    },
    {
        "name": "PDF输出优化测试",
        "script": "optimize_pdf_output.py",
        "description": "测试和优化PDF输出性能，包括不同内容复杂度的对比",
        "command": "python scripts/optimize_pdf_output.py"
    },
    {
        "name": "Notebook单元格优化",
        "script": "optimize_notebook_output.py",
        "description": "测试和优化Jupyter Notebook单元格分割算法",
        "command": "python scripts/optimize_notebook_output.py"
    },
    {
        "name": "高级LaTeX公式渲染",
        "script": "test_advanced_latex_rendering.py",
        "description": "测试复杂LaTeX公式在各种输出格式中的渲染能力",
        "command": "python scripts/test_advanced_latex_rendering.py"
    },
    {
        "name": "运行所有测试",
        "script": "run_output_tests.py",
        "description": "运行所有输出测试并生成综合报告",
        "command": "python scripts/run_output_tests.py --analyze"
    }
]

def check_requirements():
    """检查依赖库是否已安装"""
    logger.info("检查依赖库...")
    
    required_packages = [
        "markdown", 
        "python-markdown-math",
        "weasyprint", 
        "fpdf", 
        "Pygments", 
        "tabulate",
        "nbformat"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.split('-')[0])  # 处理包名中的连字符
            logger.info(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} 未安装")
    
    if missing_packages:
        logger.warning(f"\n缺少以下依赖库: {', '.join(missing_packages)}")
        logger.info("可以通过以下命令安装所有依赖:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("所有依赖库已安装 ✅")
    return True

def prepare_test_environment():
    """准备测试环境，确保必要的目录存在"""
    logger.info("准备测试环境...")
    
    # 创建必要的目录
    dirs = [
        "output",
        "output/markdown",
        "output/html",
        "output/pdf",
        "output/notebook",
        "workspace"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✅ 确保目录存在: {directory}")
    
    logger.info("测试环境准备完成 ✅")
    return True

def run_script(script_info):
    """运行指定的测试脚本"""
    name = script_info["name"]
    command = script_info["command"]
    
    logger.info(f"\n===== 运行 {name} =====")
    logger.info(f"执行命令: {command}")
    
    try:
        # 运行脚本并捕获输出
        start_time = time.time()
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 实时显示输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 获取退出码和错误输出
        exit_code = process.poll()
        errors = process.stderr.read()
        duration = time.time() - start_time
        
        if exit_code == 0:
            logger.info(f"✅ {name} 执行成功! 耗时: {duration:.2f}秒")
            return True
        else:
            logger.error(f"❌ {name} 执行失败，退出码: {exit_code}")
            if errors:
                logger.error(f"错误信息:\n{errors}")
            return False
    
    except Exception as e:
        logger.error(f"❌ 运行 {name} 时发生错误: {str(e)}")
        return False

def run_all_tests():
    """运行所有测试脚本"""
    logger.info("===== 开始运行所有测试 =====")
    
    total = len(TEST_SCRIPTS)
    success = 0
    failed = []
    
    start_time = time.time()
    
    for script_info in TEST_SCRIPTS:
        if run_script(script_info):
            success += 1
        else:
            failed.append(script_info["name"])
    
    duration = time.time() - start_time
    
    logger.info("\n===== 测试运行完成 =====")
    logger.info(f"总计: {total} 个测试")
    logger.info(f"成功: {success} 个")
    logger.info(f"失败: {len(failed)} 个")
    if failed:
        logger.info(f"失败的测试: {', '.join(failed)}")
    logger.info(f"总耗时: {duration:.2f}秒")
    
    return success == total

def show_menu():
    """显示交互式菜单"""
    while True:
        print("\n===== KnowForge 0.1.7 输出功能测试导航 =====")
        print("请选择要运行的测试:")
        
        for i, script in enumerate(TEST_SCRIPTS, 1):
            print(f"{i}. {script['name']} - {script['description']}")
        
        print("A. 运行所有测试")
        print("C. 检查依赖库")
        print("Q. 退出")
        
        choice = input("\n请输入选项: ").strip().upper()
        
        if choice == 'Q':
            break
        elif choice == 'A':
            run_all_tests()
        elif choice == 'C':
            check_requirements()
        elif choice.isdigit() and 1 <= int(choice) <= len(TEST_SCRIPTS):
            script_index = int(choice) - 1
            run_script(TEST_SCRIPTS[script_index])
        else:
            print("无效的选项，请重新选择")

def show_results():
    """显示测试结果和生成的文件"""
    # 列出输出目录中的文件
    logger.info("\n===== 生成的输出文件 =====")
    
    output_formats = {
        "HTML": "output/html",
        "PDF": "output/pdf",
        "Markdown": "output/markdown",
        "Jupyter Notebook": "output/notebook"
    }
    
    for format_name, directory in output_formats.items():
        if os.path.exists(directory):
            files = os.listdir(directory)
            if files:
                logger.info(f"\n{format_name} 文件:")
                for file in files:
                    file_path = os.path.join(directory, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    logger.info(f"  - {file} (大小: {file_size:.1f}KB, 时间: {file_time})")
            else:
                logger.info(f"\n{format_name} 文件: 无")

def open_output_files():
    """打开生成的输出文件(HTML)"""
    try:
        import webbrowser
        
        html_dir = "output/html"
        if not os.path.exists(html_dir):
            logger.warning(f"HTML输出目录不存在: {html_dir}")
            return False
        
        html_files = [f for f in os.listdir(html_dir) if f.endswith(".html")]
        if not html_files:
            logger.warning("没有找到HTML输出文件")
            return False
        
        logger.info("\n===== 在浏览器中打开HTML文件 =====")
        for i, html_file in enumerate(html_files):
            file_path = os.path.join(html_dir, html_file)
            file_url = f"file://{os.path.abspath(file_path)}"
            logger.info(f"打开: {html_file}")
            webbrowser.open(file_url)
            
            # 等待一会，避免浏览器同时打开太多标签页
            if i < len(html_files) - 1:
                time.sleep(1)
        
        return True
    
    except Exception as e:
        logger.error(f"打开HTML文件时出错: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowForge 0.1.7 输出功能测试导航工具')
    parser.add_argument('--check', action='store_true', help='检查依赖库')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    parser.add_argument('--interactive', '-i', action='store_true', help='启动交互式菜单')
    parser.add_argument('--test', type=int, help=f'运行指定的测试(1-{len(TEST_SCRIPTS)})')
    parser.add_argument('--list', action='store_true', help='列出所有可用的测试')
    parser.add_argument('--show-results', action='store_true', help='显示测试结果和生成的文件')
    parser.add_argument('--open-html', action='store_true', help='在浏览器中打开生成的HTML文件')
    args = parser.parse_args()
    
    logger.info("===== KnowForge 0.1.7 输出功能测试导航工具 =====")
    
    # 准备测试环境
    prepare_test_environment()
    
    # 处理命令行参数
    if args.list:
        print("\n===== 可用的测试 =====")
        for i, script in enumerate(TEST_SCRIPTS, 1):
            print(f"{i}. {script['name']} - {script['description']}")
        return
    
    if args.check:
        check_requirements()
    
    if args.all:
        run_all_tests()
    
    if args.test and 1 <= args.test <= len(TEST_SCRIPTS):
        run_script(TEST_SCRIPTS[args.test - 1])
    
    if args.show_results:
        show_results()
    
    if args.open_html:
        open_output_files()
    
    # 如果没有指定任何操作，或者指定了交互模式，显示交互式菜单
    if args.interactive or not (args.check or args.all or args.test or args.show_results or args.open_html):
        show_menu()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
