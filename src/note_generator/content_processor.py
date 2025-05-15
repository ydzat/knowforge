"""
内容处理器模块
处理不同类型的内容：文本、图像、表格和公式，应用专门的处理策略
"""
import os
import io
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Any, Tuple, Union, Optional

from src.utils.logger import get_module_logger
from src.utils.exceptions import InputError


class ContentProcessor:
    """
    内容处理器
    负责处理不同类型的内容，应用专门的处理策略
    """
    
    def __init__(self, config=None, ocr_llm_processor=None):
        """
        初始化内容处理器
        
        Args:
            config: 配置选项
            ocr_llm_processor: OCR-LLM处理器实例，用于处理图像内容
        """
        self.config = config or {}
        self.logger = get_module_logger("ContentProcessor")
        
        # 保存OCR-LLM处理器引用
        self.ocr_llm_processor = ocr_llm_processor
        
        # 初始化各类型处理器
        self.text_processor = TextProcessor(config)
        self.image_processor = ImageProcessor(config, ocr_llm_processor)
        self.table_processor = TableProcessor(config)
        self.formula_processor = FormulaProcessor(config)
        
        self.logger.info("ContentProcessor初始化完成")
    
    def process(self, content_blocks):
        """
        处理所有内容块
        
        Args:
            content_blocks: 内容块列表，通常由ContentExtractor提取
            
        Returns:
            处理后的内容块列表
        """
        self.logger.info(f"开始处理内容块，共{len(content_blocks)}个")
        
        processed_blocks = []
        
        for block in content_blocks:
            try:
                block_type = block.get("type")
                
                if block_type == "text":
                    processed = self.text_processor.process(block)
                elif block_type == "image":
                    processed = self.image_processor.process(block)
                elif block_type == "table":
                    processed = self.table_processor.process(block)
                elif block_type == "formula":
                    processed = self.formula_processor.process(block)
                else:
                    self.logger.warning(f"未知的内容块类型: {block_type}")
                    processed = block  # 原样返回
                
                if processed:
                    processed_blocks.append(processed)
                    
            except Exception as e:
                self.logger.warning(f"处理内容块时出错: {str(e)}")
                # 如果处理失败，保留原始块
                processed_blocks.append(block)
        
        self.logger.info(f"内容块处理完成，共{len(processed_blocks)}个")
        return processed_blocks


class TextProcessor:
    """文本内容处理器"""
    
    def __init__(self, config=None):
        """
        初始化文本处理器
        
        Args:
            config: 配置选项
        """
        self.config = config or {}
        self.logger = get_module_logger("TextProcessor")
        
        # 文本处理配置
        self.remove_duplicates = self.config.get("text.remove_duplicates", True)
        self.normalize_whitespace = self.config.get("text.normalize_whitespace", True)
        
        self.logger.info("TextProcessor初始化完成")
    
    def process(self, block):
        """
        处理文本内容
        
        Args:
            block: 文本内容块
            
        Returns:
            处理后的内容块
        """
        content = block.get("content", "")
        
        # 应用文本处理策略
        if self.normalize_whitespace:
            content = self._normalize_whitespace(content)
        
        if self.remove_duplicates:
            content = self._remove_duplicate_lines(content)
        
        # 构建处理后的块
        processed = block.copy()
        processed["content"] = content
        processed["processed"] = True
        
        return processed
    
    def _normalize_whitespace(self, text):
        """
        规范化空白字符
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        # 替换多个连续空格为单个空格
        text = ' '.join(text.split())
        # 确保段落间有空行
        text = text.replace('\n\n', '\n').replace('\n', '\n\n')
        return text
    
    def _remove_duplicate_lines(self, text):
        """
        移除重复行
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        lines = text.split('\n')
        unique_lines = []
        for line in lines:
            if line not in unique_lines:
                unique_lines.append(line)
        return '\n'.join(unique_lines)


class ImageProcessor:
    """图像内容处理器"""
    
    def __init__(self, config=None, ocr_llm_processor=None):
        """
        初始化图像处理器
        
        Args:
            config: 配置选项
            ocr_llm_processor: OCR-LLM处理器实例
        """
        self.config = config or {}
        self.logger = get_module_logger("ImageProcessor")
        
        # 保存OCR-LLM处理器引用
        self.ocr_llm_processor = ocr_llm_processor
        
        # 图像处理配置
        self.extract_text = self.config.get("image.extract_text", True)
        self.enhance_image = self.config.get("image.enhance_image", False)
        
        self.logger.info("ImageProcessor初始化完成")
    
    def process(self, block):
        """
        处理图像内容
        
        Args:
            block: 图像内容块
            
        Returns:
            处理后的内容块
        """
        # 获取图像数据
        image_data = block.get("image_data")
        if image_data is None:
            self.logger.warning("图像数据为空")
            return block
        
        processed = block.copy()
        
        try:
            # 如果是二进制数据，转换为numpy数组
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
                image_data = np.array(img)
            
            # 如果配置了图像增强，应用增强
            if self.enhance_image:
                image_data = self._enhance_image(image_data)
                processed["image_data"] = image_data
            
            # 如果配置了文本提取且有OCR-LLM处理器，提取文本
            if self.extract_text and self.ocr_llm_processor:
                text = self._extract_text_from_image(image_data)
                if text:
                    processed["extracted_text"] = text
            
            processed["processed"] = True
            return processed
            
        except Exception as e:
            self.logger.error(f"处理图像时出错: {str(e)}")
            return block
    
    def _enhance_image(self, image):
        """
        增强图像质量
        
        Args:
            image: 原始图像数据
            
        Returns:
            增强后的图像
        """
        # 简单的图像增强：调整对比度和亮度
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 应用CLAHE（限制对比度自适应直方图均衡化）到L通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # 转回RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _extract_text_from_image(self, image):
        """
        从图像中提取文本
        
        Args:
            image: 图像数据
            
        Returns:
            提取的文本
        """
        # 使用OCR-LLM处理器提取文本
        if self.ocr_llm_processor:
            try:
                self.logger.info("使用OCR-LLM处理器提取图像文本")
                return self.ocr_llm_processor.process_image(image)
            except Exception as e:
                self.logger.error(f"OCR处理器提取文本时出错: {str(e)}")
                return None
        else:
            self.logger.warning("OCR-LLM处理器未配置，无法提取图像文本")
            return None


class TableProcessor:
    """表格内容处理器"""
    
    def __init__(self, config=None):
        """
        初始化表格处理器
        
        Args:
            config: 配置选项
        """
        self.config = config or {}
        self.logger = get_module_logger("TableProcessor")
        
        # 表格处理配置
        self.processor_type = self.config.get("table.processor", "camelot")
        self.max_rows = self.config.get("table.max_rows", 100)
        self.max_cols = self.config.get("table.max_cols", 20)
        
        self.logger.info(f"TableProcessor初始化完成，使用处理器: {self.processor_type}")
    
    def process(self, block):
        """
        处理表格内容
        
        Args:
            block: 表格内容块
            
        Returns:
            处理后的内容块
        """
        # 获取表格数据
        table_data = block.get("table_data")
        if table_data is None:
            self.logger.warning("表格数据为空")
            return block
        
        try:
            # 根据处理器类型选择不同的处理方法
            if self.processor_type == "camelot":
                return self._process_with_camelot(block)
            elif self.processor_type == "tabula":
                return self._process_with_tabula(block)
            else:
                return self._process_with_custom(block)
                
        except Exception as e:
            self.logger.error(f"处理表格时出错: {str(e)}")
            return block
    
    def _process_with_camelot(self, block):
        """
        使用Camelot处理表格
        
        Args:
            block: 表格内容块
            
        Returns:
            处理后的内容块
        """
        # 这里是一个简单的示例，实际应用中需要真正集成Camelot
        self.logger.info("使用Camelot处理表格")
        
        # 获取表格数据
        table_data = block.get("table_data", [])
        
        # 创建处理后的块
        processed = block.copy()
        processed["processed"] = True
        processed["processor"] = "camelot"
        
        # 转换为Markdown格式
        markdown_table = self._convert_to_markdown(table_data)
        processed["markdown"] = markdown_table
        
        return processed
    
    def _process_with_tabula(self, block):
        """
        使用Tabula处理表格
        
        Args:
            block: 表格内容块
            
        Returns:
            处理后的内容块
        """
        # 这里是一个简单的示例，实际应用中需要真正集成Tabula
        self.logger.info("使用Tabula处理表格")
        
        # 获取表格数据
        table_data = block.get("table_data", [])
        
        # 创建处理后的块
        processed = block.copy()
        processed["processed"] = True
        processed["processor"] = "tabula"
        
        # 转换为Markdown格式
        markdown_table = self._convert_to_markdown(table_data)
        processed["markdown"] = markdown_table
        
        return processed
    
    def _process_with_custom(self, block):
        """
        使用自定义方法处理表格
        
        Args:
            block: 表格内容块
            
        Returns:
            处理后的内容块
        """
        self.logger.info("使用自定义处理器处理表格")
        
        # 获取表格数据
        table_data = block.get("table_data", [])
        
        # 创建处理后的块
        processed = block.copy()
        processed["processed"] = True
        processed["processor"] = "custom"
        
        # 转换为Markdown格式
        markdown_table = self._convert_to_markdown(table_data)
        processed["markdown"] = markdown_table
        
        return processed
    
    def _convert_to_markdown(self, table_data):
        """
        将表格数据转换为Markdown格式
        
        Args:
            table_data: 表格数据
            
        Returns:
            Markdown格式的表格
        """
        if not table_data:
            return ""
        
        # 创建Markdown表格
        markdown = []
        
        # 添加表头（使用第一行作为表头）
        header = "| " + " | ".join(str(cell) for cell in table_data[0]) + " |"
        markdown.append(header)
        
        # 添加分隔行
        separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
        markdown.append(separator)
        
        # 添加数据行
        for row in table_data[1:]:
            data_row = "| " + " | ".join(str(cell) for cell in row) + " |"
            markdown.append(data_row)
        
        return "\n".join(markdown)


class FormulaProcessor:
    """公式内容处理器"""
    
    def __init__(self, config=None):
        """
        初始化公式处理器
        
        Args:
            config: 配置选项
        """
        self.config = config or {}
        self.logger = get_module_logger("FormulaProcessor")
        
        # 公式处理配置
        self.engine = self.config.get("formula.engine", "mathpix")
        self.api_key = self.config.get("formula.mathpix_api_key", "")
        
        self.logger.info(f"FormulaProcessor初始化完成，使用引擎: {self.engine}")
    
    def process(self, block):
        """
        处理公式内容
        
        Args:
            block: 公式内容块
            
        Returns:
            处理后的内容块，包含LaTeX格式
        """
        # 获取公式文本
        formula_text = block.get("formula_text")
        if formula_text is None:
            self.logger.warning("公式文本为空")
            return block
        
        try:
            # 根据引擎选择不同的处理方法
            if self.engine == "mathpix":
                return self._process_with_mathpix(block)
            else:
                return self._process_with_custom(block)
                
        except Exception as e:
            self.logger.error(f"处理公式时出错: {str(e)}")
            return block
    
    def _process_with_mathpix(self, block):
        """
        使用Mathpix处理公式
        
        Args:
            block: 公式内容块
            
        Returns:
            处理后的内容块
        """
        # 这里是一个简单的示例，实际应用中需要真正集成Mathpix API
        self.logger.info("使用Mathpix处理公式")
        
        # 获取公式文本或图像
        formula_text = block.get("formula_text", "")
        
        # 假设这是已经转换为LaTeX的公式
        # 实际应用中需要调用Mathpix API进行转换
        latex = "$$" + formula_text + "$$"
        
        # 创建处理后的块
        processed = block.copy()
        processed["processed"] = True
        processed["processor"] = "mathpix"
        processed["latex"] = latex
        
        return processed
    
    def _process_with_custom(self, block):
        """
        使用自定义方法处理公式
        
        Args:
            block: 公式内容块
            
        Returns:
            处理后的内容块
        """
        self.logger.info("使用自定义处理器处理公式")
        
        # 获取公式文本
        formula_text = block.get("formula_text", "")
        
        # 简单的LaTeX格式化
        latex = "$$" + formula_text + "$$"
        
        # 创建处理后的块
        processed = block.copy()
        processed["processed"] = True
        processed["processor"] = "custom"
        processed["latex"] = latex
        
        return processed
