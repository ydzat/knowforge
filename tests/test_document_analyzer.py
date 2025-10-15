#!/usr/bin/env python3
"""
文档分析器单元测试
"""
import os
import sys
import unittest
import numpy as np
from PIL import Image
import io
import cv2
import fitz  # PyMuPDF

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.document_analyzer import DocumentAnalyzer
from src.utils.exceptions import InputError


class TestDocumentAnalyzer(unittest.TestCase):
    """DocumentAnalyzer单元测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.config = {}
        self.analyzer = DocumentAnalyzer(self.config)
        
        # 测试数据目录
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # 创建一个简单的文本文件用于测试
        self.text_file = os.path.join(self.test_data_dir, "test.txt")
        with open(self.text_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试文件\n用于测试文档分析器功能")
        
        # 创建一个简单的图像文件用于测试
        self.image_file = os.path.join(self.test_data_dir, "test.png")
        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        img.save(self.image_file)
        
        # PDF文件路径（假设已有PDF文件可用于测试）
        self.pdf_file = None
        pdf_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input", "pdf")
        if os.path.exists(pdf_dir):
            pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
            if pdf_files:
                self.pdf_file = os.path.join(pdf_dir, pdf_files[0])
    
    def tearDown(self):
        """测试后的清理工作"""
        # 删除测试文件
        if os.path.exists(self.text_file):
            os.remove(self.text_file)
        if os.path.exists(self.image_file):
            os.remove(self.image_file)
    
    def test_analyze_text_file(self):
        """测试分析文本文件功能"""
        result = self.analyzer.analyze_document(self.text_file)
        
        self.assertEqual(result["document_type"], "text")
        self.assertEqual(result["total_pages"], 1)
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["blocks"][0]["type"], "text")
    
    def test_analyze_image_file(self):
        """测试分析图像文件功能"""
        result = self.analyzer.analyze_document(self.image_file)
        
        self.assertEqual(result["document_type"], "image")
        self.assertEqual(result["total_pages"], 1)
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["blocks"][0]["type"], "image")
    
    def test_analyze_nonexistent_file(self):
        """测试分析不存在的文件时抛出异常"""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze_document("nonexistent_file.txt")
    
    def test_analyze_unsupported_file_type(self):
        """测试分析不支持的文件类型时抛出异常"""
        test_file = os.path.join(self.test_data_dir, "test.xyz")
        with open(test_file, "w") as f:
            f.write("test")
        
        try:
            with self.assertRaises(InputError):
                self.analyzer.analyze_document(test_file)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_analyze_pdf_file(self):
        """测试分析PDF文件功能"""
        if not self.pdf_file or not os.path.exists(self.pdf_file):
            self.skipTest("未找到可用的PDF测试文件")
        
        result = self.analyzer.analyze_document(self.pdf_file)
        
        self.assertEqual(result["document_type"], "pdf")
        self.assertGreater(result["total_pages"], 0)
        self.assertGreater(len(result["blocks"]), 0)


if __name__ == "__main__":
    unittest.main()
