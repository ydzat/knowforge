#!/usr/bin/env python3
"""
内容整合器单元测试
"""
import os
import sys
import unittest

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.content_integrator import ContentIntegrator


class TestContentIntegrator(unittest.TestCase):
    """ContentIntegrator单元测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.config = {}
        self.integrator = ContentIntegrator(self.config)
        
        # 创建测试数据 - 处理后的内容块
        self.processed_blocks = [
            {
                "type": "text",
                "page": 1,
                "coordinates": [0, 0, 100, 50],
                "content": "这是第一页的第一段文本",
                "processed": True
            },
            {
                "type": "image",
                "page": 1,
                "coordinates": [0, 60, 100, 100],
                "caption": "第一页图片",
                "extracted_text": "图片中的文本内容",
                "processed": True
            },
            {
                "type": "text",
                "page": 1,
                "coordinates": [0, 110, 100, 150],
                "content": "这是第一页的第二段文本",
                "processed": True
            },
            {
                "type": "table",
                "page": 2,
                "coordinates": [0, 0, 200, 100],
                "markdown": "| 标题1 | 标题2 |\n| --- | --- |\n| 数据1 | 数据2 |",
                "processed": True
            },
            {
                "type": "formula",
                "page": 2,
                "coordinates": [0, 110, 100, 130],
                "latex": "$$E=mc^2$$",
                "processed": True
            },
            {
                "type": "text",
                "page": 2,
                "coordinates": [0, 140, 100, 180],
                "content": "这是第二页的文本",
                "processed": True
            }
        ]
    
    def test_sort_blocks(self):
        """测试内容块排序功能"""
        # 打乱顺序
        shuffled = [
            self.processed_blocks[3],  # 第二页表格
            self.processed_blocks[0],  # 第一页第一段文本
            self.processed_blocks[5],  # 第二页文本
            self.processed_blocks[2],  # 第一页第二段文本
            self.processed_blocks[1],  # 第一页图片
            self.processed_blocks[4],  # 第二页公式
        ]
        
        sorted_blocks = self.integrator._sort_blocks(shuffled)
        
        # 检查排序结果
        # 应该按页码排序，然后在同一页内按y坐标（从上到下）排序
        self.assertEqual(sorted_blocks[0]["type"], "text")  # 第一页第一段文本
        self.assertEqual(sorted_blocks[1]["type"], "image")  # 第一页图片
        self.assertEqual(sorted_blocks[2]["type"], "text")  # 第一页第二段文本
        self.assertEqual(sorted_blocks[3]["type"], "table")  # 第二页表格
        self.assertEqual(sorted_blocks[4]["type"], "formula")  # 第二页公式
        self.assertEqual(sorted_blocks[5]["type"], "text")  # 第二页文本
    
    def test_group_by_page(self):
        """测试按页面分组功能"""
        page_groups = self.integrator._group_by_page(self.processed_blocks)
        
        self.assertEqual(len(page_groups), 2)  # 应该有2个页面组
        self.assertEqual(len(page_groups[1]), 3)  # 第一页有3个内容块
        self.assertEqual(len(page_groups[2]), 3)  # 第二页有3个内容块
    
    def test_integrate(self):
        """测试内容整合功能"""
        integrated = self.integrator.integrate(self.processed_blocks)
        
        # 检查整合结果
        self.assertGreater(len(integrated), 0)
        
        # 应该有一个页面分隔符
        separators = [p for p in integrated if p == "---"]
        self.assertEqual(len(separators), 1)
        
        # 第一个内容应该是第一页第一段文本
        self.assertEqual(integrated[0], "这是第一页的第一段文本")
        
        # 检查图片格式
        image_parts = [p for p in integrated if p.startswith(self.integrator.image_placeholder)]
        self.assertEqual(len(image_parts), 1)
        self.assertTrue("图片中的文本内容" in image_parts[0])
        
        # 检查表格格式
        table_parts = [p for p in integrated if "| 标题1 | 标题2 |" in p]
        self.assertEqual(len(table_parts), 1)
        
        # 检查公式格式
        formula_parts = [p for p in integrated if p.startswith("$$")]
        self.assertEqual(len(formula_parts), 1)
        self.assertTrue("E=mc^2" in formula_parts[0])
    
    def test_integrate_to_markdown(self):
        """测试生成Markdown格式输出"""
        markdown = self.integrator.integrate_to_markdown(self.processed_blocks)
        
        self.assertIsNotNone(markdown)
        self.assertGreater(len(markdown), 0)
        
        # 检查表格格式
        self.assertTrue("| 标题1 | 标题2 |" in markdown)
        self.assertTrue("| --- | --- |" in markdown)
        
        # 检查公式格式
        self.assertTrue("$$E=mc^2$$" in markdown)
        
        # 检查图片格式
        self.assertTrue("![" in markdown and "](image_placeholder)" in markdown)
    
    def test_integrate_to_html(self):
        """测试生成HTML格式输出"""
        html = self.integrator.integrate_to_html(self.processed_blocks)
        
        self.assertIsNotNone(html)
        self.assertGreater(len(html), 0)
        
        # 检查HTML结构
        self.assertTrue("<div class=\"document-content\">" in html)
        self.assertTrue("</div>" in html)
        
        # 检查表格格式
        self.assertTrue("<table>" in html)
        self.assertTrue("<th>标题1</th>" in html)
        self.assertTrue("</table>" in html)
        
        # 检查公式格式
        self.assertTrue("<div class=\"formula\">E=mc^2</div>" in html)
        
        # 检查图片格式
        self.assertTrue("<img src=\"image_placeholder\"" in html)
        
        # 检查页面分隔符
        self.assertTrue("<hr class=\"page-break\">" in html)


if __name__ == "__main__":
    unittest.main()
