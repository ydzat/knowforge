#!/usr/bin/env python3
"""
测试表格处理器和公式处理器的单元测试

此测试验证TableProcessor和FormulaProcessor的核心功能
"""
import os
import sys
import pytest

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.content_processor import TableProcessor, FormulaProcessor


@pytest.fixture
def table_processor():
    """创建表格处理器实例"""
    config = {
        "table.processor": "custom",
        "table.clean_empty_rows": True,
        "table.normalize_columns": True,
        "table.enhance_structure": True,
    }
    return TableProcessor(config)


@pytest.fixture
def formula_processor():
    """创建公式处理器实例"""
    config = {
        "formula.engine": "custom",
        "formula.detect_formula_type": True,
        "formula.convert_simple_expressions": True,
    }
    return FormulaProcessor(config)


class TestTableProcessor:
    """测试表格处理器功能"""
    
    def test_init(self, table_processor):
        """测试初始化"""
        assert table_processor.processor_type == "custom"
        assert table_processor.clean_empty_rows is True
        assert table_processor.normalize_columns is True
        assert table_processor.enhance_structure is True
    
    def test_simple_table_processing(self, table_processor):
        """测试简单表格处理"""
        # 测试简单表格
        table_data = """
        | Product | Price | Qty | Total |
        | --- | --- | --- | --- |
        | Laptop | 6000 | 2 | 12000 |
        | Monitor | 1500 | 3 | 4500 |
        | Mouse | 80 | 5 | 400 |
        | Total | | | 16900 |
        """
        
        result = table_processor.process(table_data)
        
        # 验证结果包含markdown格式的表格
        assert "markdown" in result
        assert "| Product | Price | Qty | Total |" in result["markdown"]
        assert "| Laptop | 6000 | 2 | 12000 |" in result["markdown"]
    
    def test_irregular_table_processing(self, table_processor):
        """测试不规则表格处理"""
        # 测试列数不一致的表格
        table_data = """
        | Quarter | Sales | Cost | Profit | Margin |
        | --- | --- | --- | --- | --- |
        | Q1 | 100 | 80 | 20 |
        | Q2 | 120 | 90 | 30 | 25% |
        | Q3 | 150 | 110 | 40 | 26.7% | Growth: 33.3% |
        | Q4 | 200 | 140 | 60 | 30% |
        """
        
        result = table_processor.process(table_data)
        
        # 验证结果
        assert "markdown" in result
        # 验证列数标准化
        markdown_lines = result["markdown"].strip().split("\n")
        assert all(row.count("|") == markdown_lines[0].count("|") for row in markdown_lines)


class TestFormulaProcessor:
    """测试公式处理器功能"""
    
    def test_init(self, formula_processor):
        """测试初始化"""
        assert formula_processor.engine == "custom"
        assert formula_processor.detect_formula_type is True
        assert formula_processor.convert_simple_expressions is True
    
    def test_simple_formula(self, formula_processor):
        """测试简单公式处理"""
        formula = "E=mc^2"
        
        result = formula_processor.process(formula)
        
        # 验证结果
        assert result["processor"] == "custom"
        assert "latex" in result
        assert result["latex"] == "$E=mc^{2}$"
    
    def test_formula_with_fractions(self, formula_processor):
        """测试包含分数的公式处理"""
        formula = "f(x) = 1/x + x^2/2"
        
        result = formula_processor.process(formula)
        
        # 验证结果
        assert result["processor"] == "custom"
        assert "latex" in result
        # 检查指数转换
        assert "x^{2}" in result["latex"]
    
    def test_block_formula_detection(self, formula_processor):
        """测试块级公式检测"""
        formula = "\\begin{align} (a+b)^2 &= a^2 + 2ab + b^2 \\end{align}"
        
        result = formula_processor.process(formula)
        
        # 验证结果
        assert result["processor"] == "custom"
        assert "latex" in result
        # 检查是否作为块级公式处理（以$$包围）
        assert result["latex"].startswith("$$")
        assert result["latex"].endswith("$$")
