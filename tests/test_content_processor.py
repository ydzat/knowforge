#!/usr/bin/env python3
"""
内容处理器单元测试
"""
import os
import sys
import unittest
import numpy as np
from PIL import Image
import io
from unittest.mock import MagicMock

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.content_processor import ContentProcessor, TextProcessor, ImageProcessor, TableProcessor, FormulaProcessor


class TestContentProcessor(unittest.TestCase):
    """ContentProcessor单元测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.config = {}
        
        # 创建OCR-LLM处理器的mock对象
        self.mock_ocr_llm_processor = MagicMock()
        self.mock_ocr_llm_processor.process_image.return_value = "识别的图像文本"
        
        self.processor = ContentProcessor(config=self.config, ocr_llm_processor=self.mock_ocr_llm_processor)
        
        # 创建测试数据
        self.text_block = {
            "type": "text",
            "page": 1,
            "extracted": True,
            "content": "这是一个测试文本\n有多余的   空格\n和重复的行\n和重复的行",
            "confidence": 0.95
        }
        
        # 创建测试图像数据
        img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        img_array = np.array(img)
        self.image_block = {
            "type": "image",
            "page": 1,
            "extracted": True,
            "image_data": img_array,
            "caption": "测试图像"
        }
        
        # 创建测试表格数据
        self.table_block = {
            "type": "table",
            "page": 1,
            "extracted": True,
            "table_data": [
                ["标题1", "标题2"],
                ["数据1", "数据2"],
                ["数据3", "数据4"]
            ],
            "confidence": 0.8
        }
        
        # 创建测试公式数据
        self.formula_block = {
            "type": "formula",
            "page": 1,
            "extracted": True,
            "formula_text": "E=mc^2",
            "confidence": 0.7
        }
    
    def test_text_processor(self):
        """测试文本处理器功能"""
        text_processor = TextProcessor(self.config)
        processed = text_processor.process(self.text_block)
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed["type"], "text")
        self.assertTrue("processed" in processed)
        self.assertTrue(processed["processed"])
        
        # 检查文本处理结果
        self.assertFalse("和重复的行\n和重复的行" in processed["content"])
        self.assertFalse("多余的   空格" in processed["content"])
    
    def test_image_processor(self):
        """测试图像处理器功能"""
        image_processor = ImageProcessor(self.config, self.mock_ocr_llm_processor)
        processed = image_processor.process(self.image_block)
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed["type"], "image")
        self.assertTrue("processed" in processed)
        self.assertTrue(processed["processed"])
        
        # 检查是否调用了OCR-LLM处理器
        self.mock_ocr_llm_processor.process_image.assert_called_once()
        self.assertEqual(processed["extracted_text"], "识别的图像文本")
    
    def test_table_processor(self):
        """测试表格处理器功能"""
        table_processor = TableProcessor(self.config)
        processed = table_processor.process(self.table_block)
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed["type"], "table")
        self.assertTrue("processed" in processed)
        self.assertTrue(processed["processed"])
        
        # 检查Markdown表格
        self.assertTrue("markdown" in processed)
        markdown = processed["markdown"]
        self.assertTrue("| 标题1 | 标题2 |" in markdown)
        self.assertTrue("| --- | --- |" in markdown)
        self.assertTrue("| 数据1 | 数据2 |" in markdown)
    
    def test_formula_processor(self):
        """测试公式处理器功能"""
        formula_processor = FormulaProcessor(self.config)
        processed = formula_processor.process(self.formula_block)
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed["type"], "formula")
        self.assertTrue("processed" in processed)
        self.assertTrue(processed["processed"])
        
        # 检查LaTeX公式
        self.assertTrue("latex" in processed)
        self.assertEqual(processed["latex"], "$$E=mc^2$$")
    
    def test_process_all_types(self):
        """测试处理所有类型的内容块"""
        blocks = [
            self.text_block,
            self.image_block,
            self.table_block,
            self.formula_block
        ]
        
        processed_blocks = self.processor.process(blocks)
        
        self.assertEqual(len(processed_blocks), 4)
        self.assertTrue(all("processed" in block for block in processed_blocks))
    
    def test_handle_processing_error(self):
        """测试处理出错时的行为"""
        # 创建一个会导致错误的内容块
        bad_image_block = {
            "type": "image",
            "page": 1,
            "extracted": True,
            # image_data缺失
            "caption": "测试图像"
        }
        
        # 处理应该不会崩溃，而是返回原始块
        processed_blocks = self.processor.process([bad_image_block])
        self.assertEqual(len(processed_blocks), 1)
        self.assertEqual(processed_blocks[0], bad_image_block)


if __name__ == "__main__":
    unittest.main()
