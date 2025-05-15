#!/usr/bin/env python3
"""
内容提取器单元测试
"""
import os
import sys
import unittest
import numpy as np
from PIL import Image
import io

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.content_extractor import ContentExtractor


class TestContentExtractor(unittest.TestCase):
    """ContentExtractor单元测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.config = {}
        self.extractor = ContentExtractor(self.config)
        
        # 创建测试数据
        self.text_block = {
            "type": "text",
            "page": 1,
            "coordinates": [0, 0, 100, 20],
            "content": "这是一个测试文本\n包含多行内容",
            "confidence": 0.95
        }
        
        # 创建测试图像数据
        img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        img_array = np.array(img)
        self.image_block = {
            "type": "image",
            "page": 1,
            "coordinates": [0, 0, 100, 50],
            "image_data": img_array,
            "caption": "测试图像"
        }
        
        # 创建测试表格数据
        self.table_block = {
            "type": "table",
            "page": 1,
            "coordinates": [0, 0, 200, 100],
            "rows": 3,
            "columns": 2,
            "confidence": 0.8
        }
        
        # 创建测试公式数据
        self.formula_block = {
            "type": "formula",
            "page": 1,
            "coordinates": [0, 0, 100, 20],
            "content": "E=mc^2",
            "confidence": 0.7
        }
    
    def test_extract_text(self):
        """测试文本内容提取功能"""
        extracted = self.extractor._extract_text(self.text_block)
        
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["type"], "text")
        self.assertTrue("extracted" in extracted)
        self.assertTrue(extracted["extracted"])
        self.assertEqual(extracted["content"], self.text_block["content"])
    
    def test_extract_image(self):
        """测试图像内容提取功能"""
        extracted = self.extractor._extract_image(self.image_block)
        
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["type"], "image")
        self.assertTrue("extracted" in extracted)
        self.assertTrue(extracted["extracted"])
        self.assertTrue("image_data" in extracted)
        self.assertEqual(extracted["caption"], self.image_block["caption"])
    
    def test_extract_table(self):
        """测试表格内容提取功能"""
        extracted = self.extractor._extract_table(self.table_block)
        
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["type"], "table")
        self.assertTrue("extracted" in extracted)
        self.assertTrue(extracted["extracted"])
        self.assertTrue("table_data" in extracted)
        
        # 检查表格数据结构
        self.assertEqual(len(extracted["table_data"]), self.table_block["rows"])
        self.assertEqual(len(extracted["table_data"][0]), self.table_block["columns"])
    
    def test_extract_formula(self):
        """测试公式内容提取功能"""
        extracted = self.extractor._extract_formula(self.formula_block)
        
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["type"], "formula")
        self.assertTrue("extracted" in extracted)
        self.assertTrue(extracted["extracted"])
        self.assertTrue("formula_text" in extracted)
        self.assertEqual(extracted["formula_text"], self.formula_block["content"])
    
    def test_extract_content(self):
        """测试一次提取多个内容块的功能"""
        blocks = [
            self.text_block,
            self.image_block,
            self.table_block,
            self.formula_block
        ]
        
        extracted_blocks = self.extractor.extract_content(blocks)
        
        self.assertEqual(len(extracted_blocks), 4)
        self.assertTrue(all("extracted" in block for block in extracted_blocks))
    
    def test_extract_unknown_block_type(self):
        """测试提取未知类型的内容块"""
        unknown_block = {
            "type": "unknown",
            "page": 1,
            "content": "未知内容"
        }
        
        # 应该会跳过未知类型的块
        extracted_blocks = self.extractor.extract_content([unknown_block])
        self.assertEqual(len(extracted_blocks), 0)


if __name__ == "__main__":
    unittest.main()
