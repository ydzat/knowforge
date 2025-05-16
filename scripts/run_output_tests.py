#!/usr/bin/env python3
'''
 * @Author: @ydzat
 * @Date: 2025-06-02 18:00:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-06-02 18:00:00
 * @Description: KnowForge 0.1.7 输出功能测试启动器，整合所有测试脚本
'''
import os
import sys
import argparse
import logging
import subprocess
import time
from datetime import datetime

# 添加src到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import setup_logger, get_logger

# 配置日志
setup_logger()
logger = get_logger('KnowForge-TestLauncher')

# 定义可用的测试脚本
TEST_SCRIPTS = {
    "basic": "test_output_features.py",
    "formula": "test_formula_table_rendering.py",
    "pdf": "optimize_pdf_output.py",
    "notebook": "optimize_notebook_output.py",
    "table": "optimize_table_formula_rendering.py",
    "latex": "test_advanced_latex_rendering.py",
    "demo": "output_format_demo.py"
}

def run_script(script_name, args=None):
    """运行指定的测试脚本"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        logger.error(f"脚本不存在: {script_path}")
        return False
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    logger.info(f"执行: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        process = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        
        logger.info(f"脚本 {script_name} 执行完成 (用时: {duration:.2f}秒)")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"脚本执行失败 ({script_name}): {e}")
        return False

def run_all_tests(skip=None, output_dir=None):
    """运行所有测试"""
    skip = skip or []
    results = {}
    total_start_time = time.time()
    
    for test_key, script_name in TEST_SCRIPTS.items():
        if test_key in skip:
            logger.info(f"跳过测试: {test_key}")
            results[test_key] = {"skipped": True}
            continue
        
        logger.info(f"运行测试: {test_key} ({script_name})")
        start_time = time.time()
        
        args = []
        if output_dir:
            # 只添加--output-dir参数给那些接受这个参数的脚本
            if test_key in ["pdf", "notebook", "table", "demo"]:
                args.extend(["--output-dir", output_dir])
        
        success = run_script(script_name, args)
        duration = time.time() - start_time
        
        results[test_key] = {
            "success": success,
            "duration": duration,
            "script": script_name
        }
    
    total_duration = time.time() - total_start_time
    logger.info(f"所有测试完成，总用时: {total_duration:.2f}秒")
    
    return results

def analyze_test_results(results):
    """分析测试结果并打印摘要"""
    success_count = sum(1 for r in results.values() if r.get("success", False))
    skipped_count = sum(1 for r in results.values() if r.get("skipped", False))
    failed_count = len(results) - success_count - skipped_count
    
    print("\n" + "="*50)
    print(f"KnowForge 0.1.7 输出功能测试摘要")
    print("="*50)
    
    print(f"\n总测试数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"跳过: {skipped_count}")
    
    if failed_count > 0:
        print("\n失败的测试:")
        for test_key, result in results.items():
            if not result.get("skipped", False) and not result.get("success", False):
                print(f"  - {test_key} ({result['script']})")
    
    print("\n各测试用时:")
    for test_key, result in sorted(results.items(), key=lambda x: x[1].get("duration", 0), reverse=True):
        if "duration" in result:
            print(f"  - {test_key}: {result['duration']:.2f}秒")
    
    print("\n" + "="*50)

def run_analysis(output_dir):
    """运行结果分析脚本"""
    analysis_script = "analyze_output_tests.py"
    logger.info("运行测试结果分析...")
    
    args = []
    if output_dir:
        args.extend(["--output-dir", output_dir])
    args.extend(["--format", "markdown"])
    
    return run_script(analysis_script, args)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowForge输出功能测试启动器')
    parser.add_argument('tests', nargs='*', 
                       choices=['all', 'basic', 'formula', 'pdf', 'notebook', 'table', 'latex', 'demo'],
                       default=['all'], help='要运行的测试')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录')
    parser.add_argument('--skip', nargs='+', choices=['basic', 'formula', 'pdf', 'notebook', 'table', 'latex', 'demo'],
                       default=[], help='要跳过的测试')
    parser.add_argument('--analyze', action='store_true',
                       help='测试完成后运行分析')
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定要运行的测试
    if 'all' in args.tests:
        tests_to_run = []  # 空列表表示全部运行
    else:
        tests_to_run = args.tests
    
    # 确定要跳过的测试
    skip_tests = args.skip
    
    # 如果指定了特定测试，则只运行这些测试
    if tests_to_run:
        results = {}
        for test in tests_to_run:
            if test in skip_tests:
                logger.info(f"跳过测试: {test}")
                results[test] = {"skipped": True}
                continue
                
            script_name = TEST_SCRIPTS.get(test)
            if not script_name:
                logger.error(f"未知测试: {test}")
                continue
                
            logger.info(f"运行测试: {test} ({script_name})")
            start_time = time.time()
            
            test_args = []
            if args.output_dir:
                # 只添加--output-dir参数给那些接受这个参数的脚本
                if test in ["pdf", "notebook", "table", "demo"]:
                    test_args.extend(["--output-dir", args.output_dir])
            
            success = run_script(script_name, test_args)
            duration = time.time() - start_time
            
            results[test] = {
                "success": success,
                "duration": duration,
                "script": script_name
            }
    else:
        # 运行所有测试，除了应该跳过的
        results = run_all_tests(skip=skip_tests, output_dir=args.output_dir)
    
    # 分析测试结果
    analyze_test_results(results)
    
    # 如果指定了，运行分析脚本
    if args.analyze:
        run_analysis(args.output_dir)

if __name__ == "__main__":
    main()
