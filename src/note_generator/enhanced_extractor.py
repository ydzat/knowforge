#!/usr/bin/env python3
"""
增强型图像提取器模块
提供多种高级图像提取和处理方法，以提高PDF图像提取的成功率
"""
import io
import numpy as np
import cv2
from PIL import Image
import fitz  # PyMuPDF

from src.utils.logger import get_module_logger
from src.note_generator.warning_monitor import warning_monitor


class EnhancedImageExtractor:
    """
    增强型图像提取器类
    提供多种图像提取和修复方法，特别针对"not enough image data"等常见错误
    """
    
    def __init__(self, min_image_size=100, config=None):
        """
        初始化图像提取器
        
        Args:
            min_image_size: 最小图像尺寸 (像素)
            config: 其他配置选项
        """
        self.config = config or {}
        self.min_image_size = min_image_size
        self.logger = get_module_logger("EnhancedExtractor")
        
        # 图像增强参数
        self.enable_enhancement = self.config.get("enable_enhancement", True)
        self.denoising_strength = self.config.get("denoising_strength", 0.5)
        self.sharpening_strength = self.config.get("sharpening_strength", 1.2)
        
    def extract_from_xref(self, doc, xref):
        """
        使用多种方法从PDF引用号提取图像
        
        Args:
            doc: PDF文档对象
            xref: 图像引用号
            
        Returns:
            提取的图像数组或None
        """
        # 尝试多种提取方法
        extraction_methods = [
            self._extract_standard,
            self._extract_buffer_method,
            self._extract_raw_stream,
            self._extract_with_recovery
        ]
        
        # 逐一尝试不同方法
        for i, method in enumerate(extraction_methods):
            try:
                img_array = method(doc, xref)
                if img_array is not None and self._is_valid_image(img_array):
                    # 如果启用了增强，进行图像增强
                    if self.enable_enhancement:
                        img_array = self._enhance_image(img_array)
                    
                    return img_array
            except Exception as e:
                self.logger.debug(f"提取方法{i+1}失败: {str(e)}")
                continue
        
        # 记录所有方法都失败
        warning_monitor.add_not_enough_image_data_warning("所有提取方法都失败")
        return None
    
    def extract_from_region(self, page, rect):
        """
        使用多种方法从页面区域提取图像
        
        Args:
            page: PDF页面对象
            rect: 图像区域矩形 [x0, y0, x1, y1]
            
        Returns:
            提取的图像数组或None
        """
        try:
            # 确保区域有效
            if not rect or len(rect) != 4:
                return None
                
            if rect[2] - rect[0] < self.min_image_size or rect[3] - rect[1] < self.min_image_size:
                return None
                
            # 创建裁剪区域
            clip_rect = fitz.Rect(rect)
            
            # 尝试多个缩放级别
            zoom_levels = [3.0, 2.0, 1.5, 1.0]
            
            for zoom in zoom_levels:
                try:
                    # 尝试提取图像
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_rect, alpha=False)
                    
                    # 确保pixmap有效
                    if not pix or pix.width < self.min_image_size or pix.height < self.min_image_size:
                        continue
                    
                    # 转换为PIL图像
                    img_data = pix.tobytes()
                    img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                    img_array = np.array(img)
                    
                    # 检查图像质量
                    if self._is_valid_image(img_array):
                        # 应用图像增强
                        if self.enable_enhancement:
                            img_array = self._enhance_image(img_array)
                        return img_array
                except Exception as e:
                    self.logger.debug(f"区域提取(缩放={zoom})失败: {str(e)}")
            
            # 尝试备用提取方法 - 渲染整个页面然后裁剪
            try:
                # 渲染整个页面
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img_data = pix.tobytes()
                img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                
                # 计算裁剪坐标
                zoom_factor = 2.0  # 与渲染时相同
                crop_x0 = int(rect[0] * zoom_factor)
                crop_y0 = int(rect[1] * zoom_factor)
                crop_x1 = int(rect[2] * zoom_factor)
                crop_y1 = int(rect[3] * zoom_factor)
                
                # 裁剪图像
                cropped_img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                img_array = np.array(cropped_img)
                
                if self._is_valid_image(img_array):
                    if self.enable_enhancement:
                        img_array = self._enhance_image(img_array)
                    return img_array
            except Exception as e:
                self.logger.debug(f"整页渲染裁剪失败: {str(e)}")
            
            return None
            
        except Exception as e:
            error_msg = str(e)
            self.logger.warning(f"区域提取方法失败: {error_msg}")
            
            # 记录到警告监视器
            if "not enough image data" in error_msg.lower():
                warning_monitor.add_not_enough_image_data_warning(error_msg)
            else:
                warning_monitor.add_extract_region_error(error_msg)
            
            return None
    
    def _extract_standard(self, doc, xref):
        """标准提取方法"""
        pix = fitz.Pixmap(doc, xref)
        if not pix or pix.width < self.min_image_size or pix.height < self.min_image_size:
            return None
            
        # 转换为RGB格式（如果是CMYK）
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            
        img_data = pix.tobytes()
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        return np.array(img)
    
    def _extract_buffer_method(self, doc, xref):
        """使用buffer方法提取"""
        pix = fitz.Pixmap(doc, xref)
        if not pix or pix.width < self.min_image_size or pix.height < self.min_image_size:
            return None
            
        # 转换为RGB格式（如果是CMYK）
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombuffer(mode, [pix.width, pix.height], pix.samples, "raw", mode, 0, 1)
        return np.array(img)
    
    def _extract_raw_stream(self, doc, xref):
        """从原始流数据提取"""
        stream = doc.xref_stream(xref)
        if not stream:
            return None
            
        # 尝试识别图像格式
        if stream.startswith(b'\xff\xd8'):  # JPEG
            img = Image.open(io.BytesIO(stream))
        elif stream.startswith(b'\x89PNG'):  # PNG
            img = Image.open(io.BytesIO(stream))
        else:
            # 尝试其他常见格式
            try:
                img = Image.open(io.BytesIO(stream))
            except Exception:
                return None
                
        if img.width < self.min_image_size or img.height < self.min_image_size:
            return None
            
        return np.array(img)
    
    def _extract_with_recovery(self, doc, xref):
        """使用恢复模式提取"""
        try:
            # 获取图像信息
            img_info = doc.extract_image(xref)
            if not img_info:
                return None
                
            img_bytes = img_info.get("image")
            if not img_bytes:
                return None
                
            # 尝试解码图像
            img = Image.open(io.BytesIO(img_bytes))
            if img.width < self.min_image_size or img.height < self.min_image_size:
                return None
                
            return np.array(img)
        except Exception:
            return None
    
    def _is_valid_image(self, img_array):
        """
        检查图像是否有效
        
        Args:
            img_array: 图像数组
            
        Returns:
            是否有效
        """
        if img_array is None:
            return False
            
        # 检查尺寸
        if img_array.shape[0] < self.min_image_size or img_array.shape[1] < self.min_image_size:
            return False
            
        # 检查是否是空白或纯色图像
        if np.std(img_array) < 5:
            return False
            
        # 检查有效像素比例
        non_white_pixels = np.sum(np.mean(img_array, axis=2) < 240)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        if non_white_pixels / total_pixels < 0.05:  # 小于5%的非白色像素
            return False
            
        return True
    
    def _enhance_image(self, img_array):
        """
        增强图像质量
        
        Args:
            img_array: 图像数组
            
        Returns:
            增强后的图像数组
        """
        try:
            # 检查是否是彩色图像
            if len(img_array.shape) < 3 or img_array.shape[2] < 3:
                return img_array
                
            # 转换为OpenCV格式
            cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # 降噪
            if self.denoising_strength > 0:
                cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 
                                                     h=10 * self.denoising_strength, 
                                                     hColor=10 * self.denoising_strength, 
                                                     templateWindowSize=7, searchWindowSize=21)
            
            # 增强对比度
            lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            cv_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 锐化
            if self.sharpening_strength > 1.0:
                kernel = np.array([[-1, -1, -1], 
                                  [-1, 9 + (self.sharpening_strength - 1) * 4, -1], 
                                  [-1, -1, -1]])
                cv_img = cv2.filter2D(cv_img, -1, kernel)
            
            # 转回RGB
            return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.warning(f"图像增强失败: {str(e)}")
            return img_array  # 如果增强失败，返回原图像
