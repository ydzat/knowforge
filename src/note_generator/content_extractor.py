"""
内容提取器模块
从文档中提取不同类型的内容：文本、图像、表格和公式
"""
import os
import io
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple, Union, Optional

from src.utils.logger import get_module_logger
from src.utils.exceptions import InputError


class ContentExtractor:
    """
    内容提取器
    负责从文档分析器识别的内容块中提取实际内容
    """
    
    def __init__(self, config=None):
        """
        初始化内容提取器
        
        Args:
            config: 配置选项
        """
        self.config = config or {}
        self.logger = get_module_logger("ContentExtractor")
        
        # 设置默认参数
        self.image_format = self.config.get("image_format", "png")
        self.image_quality = self.config.get("image_quality", 95)
        self.temp_dir = self.config.get("temp_dir", "/tmp")
        
        self.logger.info("ContentExtractor初始化完成")
    
    def extract_content(self, document_blocks):
        """
        从文档块中提取内容
        
        Args:
            document_blocks: 文档块列表，通常由DocumentAnalyzer生成
            
        Returns:
            提取后的内容块列表
        """
        self.logger.info(f"开始提取内容，共{len(document_blocks)}个内容块")
        
        extracted_blocks = []
        
        for block in document_blocks:
            try:
                extracted = self._extract_block(block)
                if extracted:
                    extracted_blocks.append(extracted)
            except Exception as e:
                self.logger.warning(f"提取内容块时出错: {str(e)}")
        
        self.logger.info(f"内容提取完成，共提取{len(extracted_blocks)}个内容块")
        return extracted_blocks
    
    def _extract_block(self, block):
        """
        根据块类型提取内容
        
        Args:
            block: 内容块
            
        Returns:
            提取后的内容块
        """
        block_type = block.get("type")
        
        if block_type == "text":
            return self._extract_text(block)
        elif block_type == "image":
            return self._extract_image(block)
        elif block_type == "table":
            return self._extract_table(block)
        elif block_type == "formula":
            return self._extract_formula(block)
        else:
            self.logger.warning(f"未知的内容块类型: {block_type}")
            return None
    
    def _extract_text(self, block):
        """
        提取文本内容
        
        Args:
            block: 文本内容块
            
        Returns:
            提取后的内容块
        """
        # 对于文本块，通常不需要额外处理，直接返回原内容
        # 但可以进行一些清理和格式化操作
        content = block.get("content", "")
        
        # 清理文本：删除多余空白字符，修复排版问题等
        content = self._clean_text(content)
        
        # 构建提取后的块
        extracted = block.copy()
        extracted["content"] = content
        extracted["extracted"] = True
        
        return extracted
    
    def _clean_text(self, text):
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 分行处理以保留换行符
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 替换每行中的多个连续空格为单个空格
            line = ' '.join(line.split())
            # 修复常见的排版问题
            line = line.replace('- ', '')  # 移除断字符
            cleaned_lines.append(line)
        
        # 重新用换行符连接
        return '\n'.join(cleaned_lines)
    
    def _extract_image(self, block):
        """
        提取图像内容
        
        Args:
            block: 图像内容块
            
        Returns:
            提取后的内容块
        """
        # 获取图像数据
        image_data = block.get("image_data")
        
        if image_data is None:
            self.logger.warning("图像数据为空")
            return None
        
        try:
            # 创建提取后的块
            extracted = block.copy()
            
            # 确保图像数据是numpy数组
            if not isinstance(image_data, np.ndarray):
                self.logger.warning("图像数据不是numpy数组，尝试转换")
                # 尝试转换为numpy数组
                if isinstance(image_data, bytes):
                    # 从二进制数据加载图像
                    image = Image.open(io.BytesIO(image_data))
                    image_data = np.array(image)
                else:
                    raise ValueError("无法处理的图像数据类型")
            
            # 保存图像到临时文件或内存中
            image = Image.fromarray(image_data)
            img_buffer = io.BytesIO()
            image.save(img_buffer, format=self.image_format, quality=self.image_quality)
            img_binary = img_buffer.getvalue()
            
            # 替换原始图像数据为处理后的二进制数据
            extracted["image_data"] = img_binary
            extracted["image_format"] = self.image_format
            extracted["extracted"] = True
            
            return extracted
            
        except Exception as e:
            self.logger.error(f"处理图像时出错: {str(e)}")
            return None
    
    def _extract_table(self, block):
        """
        提取表格内容
        
        Args:
            block: 表格内容块
            
        Returns:
            提取后的内容块，包含表格结构数据
        """
        # 基础表格提取实现，未来版本将增强
        self.logger.info("尝试提取表格内容")
        
        # 获取表格坐标
        coordinates = block.get("coordinates", [0, 0, 0, 0])
        
        try:
            # 创建初始表格数据结构
            rows = block.get("rows", 0)
            columns = block.get("columns", 0)
            
            # 构建空表格数据
            table_data = []
            for r in range(rows):
                row_data = ["" for c in range(columns)]
                table_data.append(row_data)
            
            # 创建提取后的块
            extracted = block.copy()
            extracted["table_data"] = table_data
            extracted["extracted"] = True
            
            return extracted
            
        except Exception as e:
            self.logger.error(f"提取表格时出错: {str(e)}")
            return None
    
    def _extract_formula(self, block):
        """
        提取公式内容
        
        Args:
            block: 公式内容块
            
        Returns:
            提取后的内容块
        """
        # 基础公式提取实现，未来版本将增强
        self.logger.info("尝试提取公式内容")
        
        content = block.get("content", "")
        
        try:
            # 将公式文本复制到提取结果
            extracted = block.copy()
            extracted["formula_text"] = content
            extracted["extracted"] = True
            
            return extracted
            
        except Exception as e:
            self.logger.error(f"提取公式时出错: {str(e)}")
            return None
            
    def extract_content_from_document(self, doc, content_blocks):
        """
        直接从文档对象提取内容（适用于已经打开的文档）
        
        Args:
            doc: 文档对象（如fitz.Document）
            content_blocks: 内容块列表
            
        Returns:
            提取后的内容块列表
        """
        self.logger.info(f"从文档对象中提取内容，共{len(content_blocks)}个内容块")
        
        extracted_blocks = []
        
        for block in content_blocks:
            try:
                block_type = block.get("type")
                page_num = block.get("page", 1) - 1  # 转换为从0开始的索引
                
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    
                    if block_type == "text":
                        extracted = self._extract_text_from_doc(doc, page, block)
                    elif block_type == "image":
                        extracted = self._extract_image_from_doc(doc, page, block)
                    elif block_type == "table":
                        extracted = self._extract_table_from_doc(doc, page, block)
                    elif block_type == "formula":
                        extracted = self._extract_formula_from_doc(doc, page, block)
                    else:
                        self.logger.warning(f"未知的内容块类型: {block_type}")
                        continue
                    
                    if extracted:
                        extracted_blocks.append(extracted)
                        
            except Exception as e:
                self.logger.warning(f"从文档提取内容块时出错: {str(e)}")
        
        return extracted_blocks
    
    def _extract_text_from_doc(self, doc, page, block):
        """
        从文档页面提取文本
        
        Args:
            doc: 文档对象
            page: 页面对象
            block: 文本内容块
            
        Returns:
            提取后的内容块
        """
        # 获取文本区域坐标
        coords = block.get("coordinates", [0, 0, 0, 0])
        x0, y0, x1, y1 = coords
        
        # 创建矩形区域
        rect = fitz.Rect(x0, y0, x1, y1)
        
        # 从区域中提取文本
        text = page.get_text("text", clip=rect)
        
        # 清理文本
        text = self._clean_text(text)
        
        # 构建提取后的块
        extracted = block.copy()
        extracted["content"] = text
        extracted["extracted"] = True
        
        return extracted
    
    def _extract_image_from_doc(self, doc, page, block):
        """
        从文档页面提取图像
        
        Args:
            doc: 文档对象
            page: 页面对象
            block: 图像内容块
            
        Returns:
            提取后的内容块
        """
        # 对于已经有图像数据的块，直接处理
        if "image_data" in block and block["image_data"] is not None:
            return self._extract_image(block)
        
        # 否则，从坐标中提取图像
        coords = block.get("coordinates", [0, 0, 0, 0])
        x0, y0, x1, y1 = coords
        
        # 创建矩形区域
        rect = fitz.Rect(x0, y0, x1, y1)
        
        # 将区域渲染为图像
        pix = page.get_pixmap(clip=rect, alpha=False)
        img_data = pix.tobytes()
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        img_array = np.array(img)
        
        # 创建新的块
        new_block = block.copy()
        new_block["image_data"] = img_array
        
        # 提取图像
        return self._extract_image(new_block)
    
    def _extract_table_from_doc(self, doc, page, block):
        """
        从文档页面提取表格
        
        Args:
            doc: 文档对象
            page: 页面对象
            block: 表格内容块
            
        Returns:
            提取后的内容块
        """
        # 基础实现，未来版本将增强
        return self._extract_table(block)
    
    def _extract_formula_from_doc(self, doc, page, block):
        """
        从文档页面提取公式
        
        Args:
            doc: 文档对象
            page: 页面对象
            block: 公式内容块
            
        Returns:
            提取后的内容块
        """
        # 基础实现，未来版本将增强
        return self._extract_formula(block)
