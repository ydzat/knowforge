"""
文档分析器模块
负责识别和分析文档中的不同内容区域：文本、图像、表格和公式
"""
import os
import re
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import io
from typing import List, Dict, Any, Tuple, Union, Optional

from src.utils.logger import get_module_logger
from src.utils.exceptions import InputError
from src.note_generator.warning_monitor import warning_monitor


class DocumentAnalyzer:
    """
    文档结构分析器
    负责识别文档结构，划分内容区域类型
    """
    
    def __init__(self, config=None):
        """
        初始化文档分析器
        
        Args:
            config: 配置选项
        """
        # 初始化配置
        self.config = config or {}
        self.logger = get_module_logger("DocumentAnalyzer")
        
        # 设置默认参数
        self.min_image_size = self.config.get("min_image_size", 100)  # 最小图像尺寸(像素)
        self.table_detection_threshold = self.config.get("table_detection_threshold", 0.7)  # 表格检测阈值
        self.formula_detection_enabled = self.config.get("formula_detection_enabled", True)  # 公式检测开关
        
        self.logger.info("DocumentAnalyzer初始化完成")
        
    def analyze_document(self, document_path):
        """
        分析文档，识别不同内容区域
        
        Args:
            document_path: 文档文件路径
            
        Returns:
            解析结果字典，包含文档类型、总页数和内容块列表
        """
        # 检查文件是否存在
        if not os.path.exists(document_path):
            self.logger.error(f"文件不存在: {document_path}")
            raise FileNotFoundError(f"文件不存在: {document_path}")
        
        # 根据文件类型选择不同的分析方法
        ext = os.path.splitext(document_path)[1].lower()
        if ext == '.pdf':
            return self._analyze_pdf(document_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return self._analyze_image(document_path)
        elif ext in ['.txt', '.md', '.py', '.java', '.js', '.c', '.cpp', '.h']:
            return self._analyze_text_file(document_path)
        else:
            raise InputError(f"不支持的文档类型: {ext}")
            
    def _analyze_pdf(self, pdf_path):
        """
        分析PDF文档，识别其中的内容区域
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            解析结果字典
        """
        self.logger.info(f"开始分析PDF文档: {pdf_path}")
        
        try:
            # 打开PDF文档
            doc = fitz.open(pdf_path)
            
            result = {
                "document_type": "pdf",
                "total_pages": len(doc),
                "blocks": []
            }
            
            # 遍历每一页
            for page_index in range(len(doc)):
                page = doc[page_index]
                
                # 提取文本块
                self._extract_text_blocks(page, page_index, result["blocks"])
                
                # 提取图像块
                self._extract_image_blocks(page, page_index, result["blocks"])
                
                # 识别表格区域
                self._detect_tables(page, page_index, result["blocks"])
                
                # 如果启用了公式检测，识别公式区域
                if self.formula_detection_enabled:
                    self._detect_formulas(page, page_index, result["blocks"])
            
            self.logger.info(f"PDF文档分析完成: {pdf_path}, 共{len(doc)}页, {len(result['blocks'])}个内容块")
            return result
            
        except Exception as e:
            self.logger.error(f"分析PDF时出错: {str(e)}")
            raise InputError(f"无法分析PDF文档: {str(e)}")
    
    def _extract_text_blocks(self, page, page_index, blocks):
        """
        从PDF页面中提取文本块
        
        Args:
            page: PDF页面对象
            page_index: 页码（从0开始）
            blocks: 内容块列表，将追加提取的文本块
        """
        # 获取页面上的文本块
        text_blocks = page.get_text("blocks")
        
        for block in text_blocks:
            # block格式: (x0, y0, x1, y1, "文本内容", block_no, block_type)
            x0, y0, x1, y1, content, block_no, block_type = block
            
            # 过滤空块
            if not content.strip():
                continue
            
            # 添加到结果中
            blocks.append({
                "type": "text",
                "page": page_index + 1,  # 页码从1开始计算
                "coordinates": [x0, y0, x1, y1],
                "content": content,
                "confidence": 1.0  # PDF提取的文本置信度默认为1.0
            })
    
    def _extract_image_blocks(self, page, page_index, blocks):
        """
        从PDF页面中提取图像块
        
        Args:
            page: PDF页面对象
            page_index: 页码（从0开始）
            blocks: 内容块列表，将追加提取的图像块
        """
        # 记录找到的所有图像区域，用于后续可能的备用提取
        all_image_regions = []
        extracted_images_count = 0
        
        # 提取页面上的图像对象
        try:
            image_list = page.get_images(full=True)
        except Exception as e:
            self.logger.warning(f"获取页面图像列表时出错: {str(e)}")
            image_list = []

        # 尝试标准图像提取
        for img_index, img_info in enumerate(image_list):
            if not img_info:
                continue
                
            try:
                xref = img_info[0]  # 图像引用号
                if not xref:
                    continue
                
                # 确定图像在页面上的位置（先获取位置，即使提取失败也记录位置）
                image_rect = self._get_image_rect(page, xref)
                
                if image_rect is None:
                    # 如果无法确定位置，使用默认值或继续下一个
                    continue
                
                # 记录图像区域信息
                all_image_regions.append({
                    "rect": image_rect,
                    "index": img_index
                })
                
                # 提取图像
                base_image = self._extract_image(page.parent, xref)
                
                if base_image is None:
                    continue
                
                # 将图像添加到结果中
                blocks.append({
                    "type": "image",
                    "page": page_index + 1,
                    "coordinates": list(image_rect),
                    "image_data": base_image,
                    "caption": f"图像 {page_index+1}-{img_index+1}",
                    "extraction_method": "direct"  # 标记为直接提取
                })
                extracted_images_count += 1
                
            except Exception as e:
                self.logger.warning(f"直接提取图像时出错: {str(e)}")
        
        # 如果直接提取的图像数量太少，尝试使用备用方法
        if extracted_images_count < len(image_list) * 0.5 and len(all_image_regions) > 0:
            self.logger.info(f"页面 {page_index+1} 使用备用方法提取图像，找到 {len(all_image_regions)} 个区域")
            for region_info in all_image_regions:
                try:
                    # 使用备用方法提取区域图像
                    rect = region_info["rect"]
                    img_index = region_info["index"]
                    
                    # 检查是否已经处理过此区域
                    already_extracted = any(
                        b.get("type") == "image" and 
                        b.get("page") == page_index + 1 and
                        self._rects_overlap(b.get("coordinates"), list(rect))
                        for b in blocks
                    )
                    
                    if already_extracted:
                        continue
                    
                    # 使用备用方法提取图像
                    base_image = self._extract_image_from_region(page, rect)
                    
                    if base_image is not None:
                        blocks.append({
                            "type": "image",
                            "page": page_index + 1,
                            "coordinates": list(rect),
                            "image_data": base_image,
                            "caption": f"图像 {page_index+1}-{img_index+1}",
                            "extraction_method": "region"  # 标记为区域提取
                        })
                        
                except Exception as e:
                    self.logger.warning(f"备用方法提取图像时出错: {str(e)}")
        
        # 尝试从文档结构中检测可能的图像块
        self._detect_additional_images(page, page_index, blocks)
    
    def _extract_image(self, doc, xref):
        """
        从PDF中提取图像
        
        Args:
            doc: PDF文档对象
            xref: 图像引用号
            
        Returns:
            提取的图像数据
        """
        try:
            # 检查xref是否有效
            if xref <= 0 or not doc.xref_object(xref):
                return None
                
            try:
                # 获取图像数据
                pix = fitz.Pixmap(doc, xref)
            except Exception as e:
                # 特别处理"not enough image data"错误
                if "not enough image data" in str(e).lower():
                    # 记录警告到监视器
                    warning_monitor.add_not_enough_image_data_warning()
                    # 记录到日志
                    self.logger.warning(f"转换图像数据时出错: not enough image data")
                    # 尝试跳过这个图像
                    return None
                # 其他错误重新抛出
                raise
            
            # 检查图像数据是否有效
            if pix is None or pix.width <= 0 or pix.height <= 0:
                return None
                
            # 检查图像尺寸
            if pix.width < self.min_image_size or pix.height < self.min_image_size:
                return None
            
            # 转换为RGB格式（如果是CMYK）
            if pix.n - pix.alpha > 3:
                try:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                except Exception as e:
                    self.logger.warning(f"转换图像颜色空间时出错: {str(e)}")
                    return None
            
            # 检查像素数据是否足够
            if not pix.samples or len(pix.samples) < pix.width * pix.height:
                self.logger.warning("图像数据不完整")
                return None
                
            try:
                # 转换为PIL图像对象
                img_data = pix.tobytes()
                img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                
                # 转换为numpy数组
                img_array = np.array(img)
                
                return img_array
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"转换图像数据时出错: {error_msg}")
                
                # 记录到警告监视器
                if "not enough image data" in error_msg.lower():
                    warning_monitor.add_not_enough_image_data_warning(error_msg)
                else:
                    warning_monitor.add_convert_image_error(error_msg)
                    
                return None
            
        except Exception as e:
            self.logger.warning(f"提取图像数据时出错: {str(e)}")
            return None
    
    def _get_image_rect(self, page, xref):
        """
        获取图像在页面上的位置
        
        Args:
            page: PDF页面对象
            xref: 图像引用号
            
        Returns:
            图像矩形坐标 (x0, y0, x1, y1)
        """
        try:
            # 遍历页面上的所有对象，查找指定的图像
            for obj in page.get_images(full=True):
                if obj and obj[0] == xref:
                    # 使用图像矩阵确定位置
                    try:
                        for drawing in page.get_drawings():
                            if not isinstance(drawing, dict):
                                continue
                                
                            items = drawing.get("items", [])
                            if not items:
                                continue
                                
                            for item in items:
                                if not isinstance(item, dict):
                                    continue
                                    
                                if item.get("type") == "i" and item.get("xref") == xref:
                                    # 获取图像变换矩阵
                                    ctm = item.get("ctm")
                                    if ctm:
                                        # 计算矩形位置
                                        try:
                                            rect = fitz.Rect(0, 0, 1, 1).transform(ctm)
                                            return (rect.x0, rect.y0, rect.x1, rect.y1)
                                        except Exception as e:
                                            self.logger.warning(f"转换图像矩形时出错: {str(e)}")
                    except Exception as e:
                        self.logger.warning(f"查找图像位置时出错: {str(e)}")
            
            # 如果没有找到位置，尝试从结构中提取
            try:
                for item in page.get_text("dict")["blocks"]:
                    if item.get("type") == 1:  # 图像类型
                        # 这里只是一个近似，因为我们不知道确切的图像ID
                        return (item["bbox"][0], item["bbox"][1], 
                                item["bbox"][2], item["bbox"][3])
            except Exception:
                pass
                
            return None
            
        except Exception as e:
            self.logger.warning(f"获取图像位置时出错: {str(e)}")
            return None
    
    def _detect_tables(self, page, page_index, blocks):
        """
        在PDF页面中检测表格区域
        
        基于规则的简单表格检测:
        1. 查找矩形区域内的水平和垂直线条
        2. 分析线条间的交叉点
        
        Args:
            page: PDF页面对象
            page_index: 页码（从0开始）
            blocks: 内容块列表，将追加检测到的表格
        """
        # 基础实现，将在未来版本中增强
        # 查找页面上的所有线条（可能是表格的一部分）
        horizontal_lines = []
        vertical_lines = []
        
        # 从页面绘图对象中查找线条
        try:
            # 检查页面对象是否有效
            if not hasattr(page, "get_drawings"):
                self.logger.warning(f"页面对象不支持get_drawings方法，跳过表格检测")
                return
                
            # 获取页面上的所有绘图元素
            drawings = page.get_drawings()
            if not drawings:
                return
                
            for drawing in drawings:
                if not isinstance(drawing, dict):
                    continue
                    
                items = drawing.get("items", [])
                if not items:
                    continue
                    
                for item in items:
                    if not isinstance(item, dict):
                        continue
                        
                    if item.get("type") == "l":  # 线条
                        rect = item.get("rect")
                        if not rect:
                            continue
                            
                        # 确保有p1和p2属性
                        if not hasattr(rect, "p1") or not hasattr(rect, "p2"):
                            continue
                            
                        p1, p2 = rect.p1, rect.p2
                        
                        # 判断是水平线还是垂直线
                        if abs(p1.y - p2.y) < 2:  # 水平线
                            horizontal_lines.append((p1.x, p1.y, p2.x, p2.y))
                        elif abs(p1.x - p2.x) < 2:  # 垂直线
                            vertical_lines.append((p1.x, p1.y, p2.x, p2.y))
                            
        except Exception as e:
            self.logger.warning(f"检测表格时出错: {str(e)}")
            return
        
        # 如果有足够多的水平和垂直线，可能是表格
        if len(horizontal_lines) >= 3 and len(vertical_lines) >= 3:
            try:
                # 查找表格边界
                x_min = min([min(line[0], line[2]) for line in vertical_lines])
                y_min = min([min(line[1], line[3]) for line in horizontal_lines])
                x_max = max([max(line[0], line[2]) for line in vertical_lines])
                y_max = max([max(line[1], line[3]) for line in horizontal_lines])
                
                # 计算表格的行数和列数
                rows = len(set([round(line[1], 1) for line in horizontal_lines]))  # 四舍五入到小数点后1位，减少浮点误差
                cols = len(set([round(line[0], 1) for line in vertical_lines]))
                
                # 添加表格块
                blocks.append({
                    "type": "table",
                    "page": page_index + 1,
                    "coordinates": [x_min, y_min, x_max, y_max],
                    "rows": max(1, rows - 1),  # 线条数量比单元格行数多1，确保至少有1行
                    "columns": max(1, cols - 1),  # 线条数量比单元格列数多1，确保至少有1列
                    "confidence": self.table_detection_threshold
                })
            except Exception as e:
                self.logger.warning(f"计算表格边界时出错: {str(e)}")
                return
    
    def _detect_formulas(self, page, page_index, blocks):
        """
        在PDF页面中检测数学公式区域
        
        基础公式检测策略:
        1. 识别常见的数学符号集中区域
        2. 分析文本布局和特殊字符分布
        
        Args:
            page: PDF页面对象
            page_index: 页码（从0开始）
            blocks: 内容块列表，将追加检测到的公式
        """
        try:
            # 获取页面文本
            text = page.get_text("text")
            
            # 如果文本为空，直接返回
            if not text:
                return
            
            # 定义可能表示公式的符号模式
            formula_patterns = [
                r'\$\$.+?\$\$',  # LaTeX公式块: $$公式$$
                r'\$.+?\$',      # LaTeX行内公式: $公式$
                r'\\begin\{equation\}.+?\\end\{equation\}',  # LaTeX方程环境
                r'\\begin\{align\}.+?\\end\{align\}',        # LaTeX对齐环境
                r'[=><∑∫∏√∞±≤≥≈≠∂∆∇∀∃∈∉⊂⊃∪∩]'  # 常见数学符号
            ]
            
            # 使用正则表达式查找可能的公式
            potential_formulas = []
            for pattern in formula_patterns:
                try:
                    matches = re.finditer(pattern, text, re.DOTALL)
                    for match in matches:
                        # 添加匹配到的公式文本和位置信息
                        formula_text = match.group(0)
                        # 过滤掉可能的误识别（例如单个=号）
                        if len(formula_text) > 1 or formula_text not in '=><':
                            potential_formulas.append({
                                "text": formula_text,
                                "start": match.start(),
                                "end": match.end()
                            })
                except Exception as e:
                    self.logger.warning(f"公式模式匹配出错 {pattern}: {str(e)}")
            
            # 如果没有找到任何公式，返回
            if not potential_formulas:
                return
                
            # 合并相近的公式区域
            merged_formulas = self._merge_formula_regions(potential_formulas)
            
            # 将检测到的公式添加到结果中
            for formula in merged_formulas:
                self.logger.info("尝试提取公式内容")
                # 尝试确定公式在页面上的精确位置(这是一个近似值)
                rect = self._estimate_formula_rect(page, formula["text"])
                
                if rect:
                    blocks.append({
                        "type": "formula",
                        "page": page_index + 1,
                        "coordinates": list(rect),
                        "content": formula["text"],
                        "confidence": 0.7  # 公式检测的置信度
                    })
        except Exception as e:
            self.logger.warning(f"检测公式时出错: {str(e)}")
    
    def _merge_formula_regions(self, formulas):
        """
        合并相近的公式区域
        
        Args:
            formulas: 检测到的公式列表
            
        Returns:
            合并后的公式列表
        """
        if not formulas:
            return []
            
        try:
            # 按开始位置排序
            sorted_formulas = sorted(formulas, key=lambda x: x["start"])
            
            merged = [sorted_formulas[0]]
            
            for current in sorted_formulas[1:]:
                previous = merged[-1]
                
                # 如果当前公式与上一个公式足够近，合并它们
                if current["start"] - previous["end"] < 20:  # 20个字符内视为相近
                    previous["end"] = max(previous["end"], current["end"])
                    previous["text"] = previous["text"] + " " + current["text"]
                else:
                    merged.append(current)
                    
            return merged
        except Exception as e:
            self.logger.warning(f"合并公式区域时出错: {str(e)}")
            # 如果出错，返回原始列表中的有效项
            return [f for f in formulas if isinstance(f, dict) and "text" in f]
    
    def _estimate_formula_rect(self, page, formula_text):
        """
        估算公式在页面上的矩形区域
        
        Args:
            page: PDF页面对象
            formula_text: 公式文本
            
        Returns:
            估计的矩形坐标 (x0, y0, x1, y1)
        """
        try:
            # 处理极短的公式文本
            if len(formula_text) <= 1:
                return None
                
            # 检查页面对象是否有效
            if not hasattr(page, "get_text"):
                return None
                
            # 搜索页面上的文本块，查找包含公式文本的块
            try:
                text_blocks = page.get_text("blocks")
                if not text_blocks:
                    return None
                    
                # 首先尝试精确匹配
                for block in text_blocks:
                    if len(block) >= 5 and formula_text in block[4]:
                        # 返回文本块的坐标
                        return (block[0], block[1], block[2], block[3])
                        
                # 如果找不到精确匹配，尝试部分匹配
                # 对于较长的公式，可能分布在多个文本块中，找包含最多公式部分的块
                best_match = None
                max_overlap = 0
                for block in text_blocks:
                    if len(block) >= 5:
                        block_text = block[4]
                        # 计算重叠部分
                        overlap_length = sum(1 for c in formula_text if c in block_text)
                        if overlap_length > max_overlap and overlap_length > len(formula_text) * 0.5:
                            max_overlap = overlap_length
                            best_match = (block[0], block[1], block[2], block[3])
                
                if best_match:
                    return best_match
            except Exception as e:
                self.logger.warning(f"通过文本块查找公式位置时出错: {str(e)}")
            
            # 如果仍未找到，返回None
            return None
            
        except Exception as e:
            self.logger.warning(f"估算公式区域时出错: {str(e)}")
            return None
    
    def _analyze_image(self, image_path):
        """
        分析图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            解析结果字典
        """
        self.logger.info(f"开始分析图像文件: {image_path}")
        
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise InputError(f"无法读取图像文件: {image_path}")
                
            # 将图像转换为RGB格式
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 创建结果字典
            result = {
                "document_type": "image",
                "total_pages": 1,
                "blocks": [
                    {
                        "type": "image",
                        "page": 1,
                        "coordinates": [0, 0, img.shape[1], img.shape[0]],
                        "image_data": img_rgb,
                        "caption": os.path.basename(image_path)
                    }
                ]
            }
            
            self.logger.info(f"图像文件分析完成: {image_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"分析图像时出错: {str(e)}")
            raise InputError(f"无法分析图像文件: {str(e)}")
    
    def _analyze_text_file(self, text_path):
        """
        分析文本文件
        
        Args:
            text_path: 文本文件路径
            
        Returns:
            解析结果字典
        """
        self.logger.info(f"开始分析文本文件: {text_path}")
        
        try:
            # 读取文本文件内容
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 创建结果字典
            result = {
                "document_type": "text",
                "total_pages": 1,
                "blocks": [
                    {
                        "type": "text",
                        "page": 1,
                        "coordinates": [0, 0, 0, 0],  # 文本文件没有明确的坐标
                        "content": content,
                        "confidence": 1.0  # 文本文件的内容置信度为1.0
                    }
                ]
            }
            
            self.logger.info(f"文本文件分析完成: {text_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"分析文本文件时出错: {str(e)}")
            raise InputError(f"无法分析文本文件: {str(e)}")
    
    def _rects_overlap(self, rect1, rect2, threshold=0.5):
        """
        检查两个矩形是否重叠
        
        Args:
            rect1: 第一个矩形 [x0, y0, x1, y1]
            rect2: 第二个矩形 [x0, y0, x1, y1]
            threshold: 重叠阈值（0到1之间）
            
        Returns:
            是否重叠
        """
        if not rect1 or not rect2 or len(rect1) != 4 or len(rect2) != 4:
            return False
            
        # 计算交叉区域
        x0 = max(rect1[0], rect2[0])
        y0 = max(rect1[1], rect2[1])
        x1 = min(rect1[2], rect2[2])
        y1 = min(rect1[3], rect2[3])
        
        # 如果矩形不相交
        if x0 >= x1 or y0 >= y1:
            return False
            
        # 计算交叉区域面积
        intersection = (x1 - x0) * (y1 - y0)
        
        # 计算两个矩形面积
        area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        
        # 如果任一矩形面积为0
        if area1 <= 0 or area2 <= 0:
            return False
            
        # 计算重叠率
        overlap_ratio = intersection / min(area1, area2)
        
        return overlap_ratio >= threshold
    
    def _extract_image_from_region(self, page, rect):
        """
        从页面指定区域提取图像
        
        Args:
            page: PDF页面对象
            rect: 图像区域矩形 (x0, y0, x1, y1)
            
        Returns:
            提取的图像数据（NumPy数组）或None
        """
        try:
            # 确保矩形区域有效
            if rect[2] - rect[0] < self.min_image_size or rect[3] - rect[1] < self.min_image_size:
                return None
                
            # 创建裁剪区域
            clip_rect = fitz.Rect(rect)
            
            # 计算适当的缩放因子，确保图像质量
            zoom = 2.0  # 默认缩放因子，提高清晰度
            
            # 从页面指定区域渲染图像
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_rect)
            
            # 转换为NumPy数组
            img_data = pix.tobytes()
            img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
            img_array = np.array(img)
            
            return img_array
            
        except Exception as e:
            error_msg = str(e)
            self.logger.warning(f"从区域提取图像时出错: {error_msg}")
            
            # 记录到警告监视器
            if "not enough image data" in error_msg.lower():
                warning_monitor.add_not_enough_image_data_warning(error_msg)
            else:
                warning_monitor.add_extract_region_error(error_msg)
                
            return None
    
    def _detect_additional_images(self, page, page_index, blocks):
        """
        从文档结构中检测可能的额外图像
        
        Args:
            page: PDF页面对象
            page_index: 页码（从0开始）
            blocks: 内容块列表，将追加检测到的图像
        """
        try:
            # 尝试从页面字典中获取潜在图像块
            page_dict = page.get_text("dict")
            
            # 检查页面字典是否有效
            if not isinstance(page_dict, dict) or "blocks" not in page_dict:
                return
                
            for block in page_dict["blocks"]:
                # 检查块类型（1 = 图像, 0 = 文本）
                if block.get("type") == 1:
                    # 获取坐标
                    rect = block.get("bbox", [0, 0, 0, 0])
                    
                    # 检查矩形是否有效
                    if rect[2] - rect[0] < self.min_image_size or rect[3] - rect[1] < self.min_image_size:
                        continue
                    
                    # 检查是否与已处理的图像重叠
                    already_extracted = any(
                        b.get("type") == "image" and 
                        b.get("page") == page_index + 1 and
                        self._rects_overlap(b.get("coordinates"), rect)
                        for b in blocks
                    )
                    
                    if already_extracted:
                        continue
                    
                    # 使用区域提取方法获取图像
                    base_image = self._extract_image_from_region(page, rect)
                    
                    if base_image is not None:
                        blocks.append({
                            "type": "image",
                            "page": page_index + 1,
                            "coordinates": rect,
                            "image_data": base_image,
                            "caption": f"图像 {page_index+1}-结构检测",
                            "extraction_method": "structure"  # 标记为结构检测
                        })
                        
        except Exception as e:
            self.logger.warning(f"检测额外图像时出错: {str(e)}")
