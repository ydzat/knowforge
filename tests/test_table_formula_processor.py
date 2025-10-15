#!/usr/bin/env python3
"""
测试表格处理器和公式处理器的单元测试

此测试验证TableProcessor和FormulaProcessor的核心功能
"""
import os
import sys
import unittest
import pytest

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.content_processor import TableProcessor, FormulaProcessor


class TestTableProcessor(unittest.TestCase):
    """测试表格处理器功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            "table.processor": "custom",
            "table.clean_empty_rows": True,
            "table.normalize_columns": True,
            "table.enhance_structure": True,
        }
        self.table_processor = TableProcessor(self.config)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.table_processor.processor_type, "custom")
        self.assertTrue(self.table_processor.clean_empty_rows)
        self.assertTrue(self.table_processor.normalize_columns)
        self.assertTrue(self.table_processor.enhance_structure)
    
    def test_simple_table_processing(self):
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
        
        result = self.table_processor.process(table_data)
        
        # 验证结果包含markdown格式的表格
        self.assertIn("markdown", result)
        self.assertIn("| Product | Price | Qty | Total |", result["markdown"])
        self.assertIn("| Laptop | 6000 | 2 | 12000 |", result["markdown"])
    
    def test_irregular_table_processing(self):
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
        
        result = self.table_processor.process(table_data)
        
        # 验证结果
        self.assertIn("markdown", result)
        # 验证列数标准化
        markdown_lines = result["markdown"].strip().split("\n")
        self.assertTrue(all(row.count("|") == markdown_lines[0].count("|") for row in markdown_lines))


class TestFormulaProcessor(unittest.TestCase):
    """测试公式处理器功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            "formula.engine": "custom",
            "formula.detect_formula_type": True,
            "formula.convert_simple_expressions": True,
        }
        self.formula_processor = FormulaProcessor(self.config)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.formula_processor.engine, "custom")
        self.assertTrue(self.formula_processor.detect_formula_type)
        self.assertTrue(self.formula_processor.convert_simple_expressions)
    
    def test_simple_formula(self):
        """测试简单公式处理"""
        formula = "E=mc^2"
        
        result = self.formula_processor.process(formula)
        
        # 验证结果
        self.assertEqual(result["processor"], "custom")
        self.assertIn("latex", result)
        self.assertEqual(result["latex"], "$E=mc^{2}$")
    
    def test_formula_with_fractions(self):
        """测试包含分数的公式处理"""
        formula = "f(x) = 1/x + x^2/2"
        
        result = self.formula_processor.process(formula)
        
        # 验证结果
        self.assertEqual(result["processor"], "custom")
        self.assertIn("latex", result)
        # 检查指数转换
        self.assertIn("x^{2}", result["latex"])
    
    def test_block_formula_detection(self):
        """测试块级公式检测"""
        formula = "\\begin{align} (a+b)^2 &= a^2 + 2ab + b^2 \\end{align}"
        
        result = self.formula_processor.process(formula)
        
        # 验证结果
        self.assertEqual(result["processor"], "custom")
        self.assertIn("latex", result)
        # 检查是否作为块级公式处理（以$$包围）
        self.assertTrue(result["latex"].startswith("$$"))
        self.assertTrue(result["latex"].endswith("$$"))


if __name__ == "__main__":
    unittest.main()
