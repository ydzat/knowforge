#!/usr/bin/env python3
'''
 * @Author: @ydzat
 * @Date: 2025-06-01 18:30:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-06-01 18:30:00
 * @Description: KnowForge 0.1.7 PDF输出生成优化脚本
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
logger = get_logger('KnowForge-PDFOptimizer')

def generate_test_content_varying_complexity():
    """生成不同复杂度的测试内容，用于测试PDF生成性能"""
    
    # 基础内容 - 简单文本
    basic_content = """# 简单文档测试
    
## 简介

这是一个简单的文档，用于测试PDF生成性能。只包含基本的文本和简单格式。

## 基本格式

这里包含**粗体**和*斜体*文本，以及`行内代码`。

## 总结

这是一个简单文档的结尾。
"""
    
    # 中等内容 - 添加表格和简单公式
    medium_content = """# 中等复杂度文档测试
    
## 简介

这是一个中等复杂度的文档，包含表格和简单数学公式。

## 表格示例

| 功能 | 状态 | 完成度 |
|-----|------|-------|
| Markdown生成 | ✅ | 100% |
| HTML输出 | ✅ | 100% |
| PDF生成 | 🔄 | 90% |
| Notebook输出 | ✅ | 95% |

## 公式示例

简单公式: $E=mc^2$

稍复杂公式:

$$F = G \\frac{m_1 m_2}{r^2}$$

## 代码示例

```python
def generate_pdf(content):
    # 处理内容
    result = process(content)
    # 生成PDF
    return create_pdf(result)
```

## 总结

这是中等复杂度文档的结尾。
"""
    
    # 高复杂度内容 - 大量表格、公式和复杂格式
    complex_content = """# 高复杂度文档测试
    
## 简介

这是一个高复杂度的文档，包��多个表格、复杂公式和嵌套结构。

## 数据表格

### 性能测试结果

| 测试项 | 原始时间(秒) | 优化后(秒) | 提升比例 |
|-------|------------|-----------|---------|
| 小文档PDF | 3.45 | 1.27 | 63.2% |
| 中型文档PDF | 8.72 | 3.91 | 55.2% |
| 大型文档PDF | 25.31 | 10.54 | 58.4% |
| 包含复杂公式 | 12.48 | 5.86 | 53.0% |
| 包含大量表格 | 18.92 | 7.35 | 61.2% |

### 方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| WeasyPrint | 高质量排版 | 速度较慢 | 需要精确排版 |
| FPDF | 速度快 | 功能有限 | 简单文档 |
| 自定义渲染 | 可高度定制 | 开发复杂 | 特殊需求 |
| 混合模式 | 平衡性能和质量 | 实现复杂 | 大多数情况 |

## 复杂公式示例

行内公式示例: $\\int_{a}^{b} f(x) \\, dx = F(b) - F(a)$

复杂公式1:

$$\\frac{\\partial}{\\partial t} \\int_{\\Omega(t)} \\rho \\, d V = \\int_{\\Omega(t)} \\frac{\\partial \\rho}{\\partial t} \\, d V + \\int_{\\partial \\Omega(t)} \\rho (\\mathbf{v} \\cdot \\mathbf{n}) \\, d S$$

复杂公式2:

$$\\begin{aligned}
(\\nabla \\times \\mathbf{B}) \\times \\mathbf{B} &= (\\mathbf{B} \\cdot \\nabla)\\mathbf{B} - \\nabla\\left(\\frac{\\mathbf{B}^2}{2}\\right) \\\\
&= \\nabla \\cdot (\\mathbf{B}\\mathbf{B}) - \\nabla\\left(\\frac{\\mathbf{B}^2}{2}\\right)
\\end{aligned}$$

## 多层级标题结构

### 三级标题A
内容文本A

#### 四级标题A1
内容文本A1

#### 四级标题A2
内容文本A2

### 三级标题B
内容文本B

#### 四级标题B1
内容文本B1

##### 五级标题B1a
非常深的嵌套内容

## 代码块示例

```python
class PDFOptimizer:
    def __init__(self, config):
        self.config = config
        self.strategies = {
            'fast': self._fast_generation,
            'quality': self._quality_generation,
            'balanced': self._balanced_generation
        }
    
    def optimize(self, content, strategy='balanced'):
        if strategy not in self.strategies:
            strategy = 'balanced'
        
        return self.strategies[strategy](content)
    
    def _fast_generation(self, content):
        # 实现快速生成策略
        pass
    
    def _quality_generation(self, content):
        # 实现高质量生成策略
        pass
    
    def _balanced_generation(self, content):
        # 平衡速度和质量
        pass
```

## 图表描述

这里本应有图表，但为了测试不包含实际图像，仅描述图表内容：

1. 第一张图表展示了PDF生成时间与文档大小的关系曲线
2. 第二张图表展示了不同渲染方法的性能对比柱状图
3. 第三张图表展示了优化前后的内存占用饼图

## 总结与建议

经过多种方法的测试和对比，推荐采用混合渲染策略来平衡性能和渲染质量。对于大型文档，建议采用分块渲染并行处理的方式提高性能。

复杂图表和大型表格应按需渲染，并可考虑使用矢量格式提高质量。
"""

    return {
        "basic": basic_content,
        "medium": medium_content,
        "complex": complex_content
    }

def test_pdf_generation_methods(writer, contents):
    """测试不同方法生成PDF的性能"""
    results = {}
    
    # 对每种复杂度内容测试
    for complexity, content in contents.items():
        method_results = {}
        print(f"\n测试 {complexity} 复杂度内容...")
        
        # 1. 标准方法 (通过Markdown -> HTML -> PDF)
        try:
            print("使用标准方法生成PDF...")
            start_time = time.time()
            md_path = writer.generate_markdown([content], f"pdf_test_{complexity}_standard")
            pdf_path = writer.generate_pdf([content], f"pdf_test_{complexity}_standard")
            end_time = time.time()
            
            method_results["standard"] = {
                "time": end_time - start_time,
                "path": pdf_path,
                "success": True,
                "method": "通过Markdown -> HTML -> PDF"
            }
            print(f"  完成时间: {end_time - start_time:.3f}秒")
            
        except Exception as e:
            method_results["standard"] = {
                "time": None,
                "path": None,
                "success": False,
                "error": str(e),
                "method": "通过Markdown -> HTML -> PDF"
            }
            print(f"  失败: {str(e)}")
        
        # 收集结果
        results[complexity] = method_results
    
    return results

def analyze_pdf_files(results):
    """分析生成的PDF文件"""
    print("\n生成的PDF文件分析:")
    
    for complexity, methods in results.items():
        print(f"\n{complexity}复杂度文档:")
        
        for method_name, result in methods.items():
            if result["success"]:
                pdf_path = result["path"]
                file_size = os.path.getsize(pdf_path) / 1024  # KB
                
                print(f"  方法: {result['method']}")
                print(f"  文件大小: {file_size:.2f} KB")
                print(f"  生成时间: {result['time']:.3f} 秒")
            else:
                print(f"  方法: {result['method']} - 生成失败: {result.get('error', '未知错误')}")

def suggest_optimizations(results):
    """基于测试结果提出优化建议"""
    print("\nPDF生成优化建议:")
    
    # 检查是否所有方法都成功
    all_successful = all(
        result["success"] 
        for complexity_results in results.values() 
        for result in complexity_results.values()
    )
    
    if not all_successful:
        print("1. 解决PDF生成失败问题，确保所有生成方法可靠工作")
    
    # 分析生成时间
    generation_times = []
    for complexity_results in results.values():
        for result in complexity_results.values():
            if result["success"] and result["time"] is not None:
                generation_times.append(result["time"])
    
    if generation_times:
        avg_time = sum(generation_times) / len(generation_times)
        
        if avg_time > 5.0:
            print("2. PDF生成时间较长，建议实现以下优化措施:")
            print("   - 使用缓存机制避免重复渲染")
            print("   - 将CSS样式表预编译而不是内联")
            print("   - 实现并行渲染页面")
            print("   - 考虑使用PyPDF2或reportlab库进行直接PDF生成而不经过HTML")
        elif avg_time > 2.0:
            print("2. PDF生成时间适中，可考虑以下优化:")
            print("   - 简化CSS提高渲染速度")
            print("   - 对大文档实现分段渲染")
            print("   - 添加进度反馈机制")
        else:
            print("2. PDF生成时间表现良好!")
            
    # 检查不同复杂度之间的差异
    if "complex" in results and "basic" in results:
        complex_time = next((r["time"] for r in results["complex"].values() if r["success"]), None)
        basic_time = next((r["time"] for r in results["basic"].values() if r["success"]), None)
        
        if complex_time and basic_time and complex_time > basic_time * 5:
            print("3. 复杂文档处理效率明显低于简单文档，建议:")
            print("   - 针对不同复杂度内容采用不同的渲染策略")
            print("   - 为复杂元素（如表格、公式）实现专门的渲染优化")
            print("   - 考虑实现惰性加载或渲染")
    
    print("4. 通用优化建议:")
    print("   - 添加PDF生成进度反馈")
    print("   - 增加对渲染方法的自动选择")
    print("   - 实现CSS优化，减少不必要的样式计算")
    print("   - 添加用户可配置的PDF样式选项")
    print("   - 增加对大型表格的分页处理")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowForge PDF输出优化测试')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--complexity', choices=['all', 'basic', 'medium', 'complex'], 
                       default='all', help='测试内容复杂度')
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
    all_contents = generate_test_content_varying_complexity()
    
    # 根据用户选择过滤内容
    if args.complexity != 'all':
        contents = {args.complexity: all_contents[args.complexity]}
    else:
        contents = all_contents
    
    # 测试PDF生成
    print("\n开始PDF生成测试...")
    results = test_pdf_generation_methods(writer, contents)
    
    # 分析生成的PDF文件
    analyze_pdf_files(results)
    
    # 提出优化建议
    suggest_optimizations(results)
    
    # 保存测试结果
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    result_path = os.path.join(output_dir, "pdf_perf_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n测试结果已保存至: {result_path}")

if __name__ == "__main__":
    main()
