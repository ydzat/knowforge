#!/usr/bin/env python3
"""
文档分析器示例实现 - 用于演示DocumentAnalyzer的基本功能
"""
import os
import sys
import argparse
import json
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import io

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 模拟logger，实际应用中应使用项目的日志系统
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("document_analyzer_demo")

class DocumentAnalyzer:
    """文档结构分析器示例实现"""
    
    def __init__(self, config=None):
        """初始化文档分析器"""
        self.config = config or {}
        self.logger = logger
        
        # 设置默认参数
        self.min_image_size = self.config.get("min_image_size", 100)  # 最小图像尺寸(像素)
        self.table_detection_threshold = self.config.get("table_detection_threshold", 0.7)  # 表格检测阈值
        
        self.logger.info("DocumentAnalyzer初始化完成")
        
    def analyze_document(self, document_path):
        """
        分析文档，识别不同内容区域
        
        Args:
            document_path: 文档文件路径
            
        Returns:
            解析结果字典
        """
        # 检查文件是否存在
        if not os.path.exists(document_path):
            self.logger.error(f"文件不存在: {document_path}")
            raise FileNotFoundError(f"文件不存在: {document_path}")
            
        # 根据文件类型选择不同的分析方法
        ext = os.path.splitext(document_path)[1].lower()
        if ext == '.pdf':
            return self._analyze_pdf(document_path)
        elif ext in ['.png', '.jpg', '.jpeg']:
            return self._analyze_image(document_path)
        else:
            self.logger.error(f"不支持的文档类型: {ext}")
            raise ValueError(f"不支持的文档类型: {ext}")
    
    def _analyze_pdf(self, pdf_path):
        """
        分析PDF文档
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            解析结果字典
        """
        self.logger.info(f"开始分析PDF文档: {pdf_path}")
        
        try:
            # 打开PDF文档
            doc = fitz.open(pdf_path)
            
            # 初始化结果
            result = {
                "document_type": "pdf",
                "total_pages": len(doc),
                "metadata": {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "keywords": doc.metadata.get("keywords", "")
                },
                "blocks": []
            }
            
            # 按页分析内容
            for page_idx in range(len(doc)):
                self.logger.info(f"分析第 {page_idx+1}/{len(doc)} 页")
                page = doc[page_idx]
                
                # 提取文本块
                text_blocks = self._extract_text_blocks(page, page_idx)
                result["blocks"].extend(text_blocks)
                
                # 提取图像块
                image_blocks = self._extract_image_blocks(page, page_idx)
                result["blocks"].extend(image_blocks)
                
                # 提取表格块 (简单实现)
                table_blocks = self._detect_table_blocks(page, page_idx)
                result["blocks"].extend(table_blocks)
                
            self.logger.info(f"PDF分析完成，共识别 {len(result['blocks'])} 个内容块")
            return result
            
        except Exception as e:
            self.logger.error(f"PDF分析失败: {str(e)}")
            raise
    
    def _extract_text_blocks(self, page, page_idx):
        """提取页面中的文本块"""
        blocks = []
        
        # 获取页面中的文本块
        text_blocks = page.get_text("blocks")
        
        for i, block in enumerate(text_blocks):
            # block格式: (x0, y0, x1, y1, text, block_no, block_type)
            if not block[4].strip():  # 跳过空文本块
                continue
                
            blocks.append({
                "type": "text",
                "page": page_idx + 1,
                "block_id": f"p{page_idx+1}_t{i+1}",
                "coordinates": block[:4],  # [x0, y0, x1, y1]
                "content": block[4],
                "confidence": 1.0  # 原始PDF文本默认置信度为1
            })
            
        return blocks
    
    def _extract_image_blocks(self, page, page_idx):
        """提取页面中的图像块"""
        blocks = []
        
        # 获取页面上的图像列表
        image_list = page.get_images(full=True)
        
        for i, img in enumerate(image_list):
            # 获取图像信息
            xref = img[0]  # 图像在PDF中的引用号
            
            try:
                # 获取图像基本信息
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # 将图像转换为PIL格式，计算尺寸
                pil_image = Image.open(io.BytesIO(image_bytes))
                width, height = pil_image.size
                
                # 跳过过小的图像 (可能是背景或图标)
                if width < self.min_image_size or height < self.min_image_size:
                    continue
                
                # 获取图像在页面上的位置 (简化处理)
                # 实际应用中需要更准确地计算图像在页面上的坐标
                bbox = page.get_image_bbox(img)
                if bbox:
                    coordinates = [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
                else:
                    # 如果无法获取准确位置，使用估计值
                    coordinates = [0, 0, width, height]
                
                blocks.append({
                    "type": "image",
                    "page": page_idx + 1,
                    "block_id": f"p{page_idx+1}_i{i+1}",
                    "coordinates": coordinates,
                    "image_data": image_bytes,
                    "format": base_image["ext"],
                    "width": width,
                    "height": height,
                    "caption": self._extract_nearby_caption(page, coordinates)
                })
                
            except Exception as e:
                self.logger.warning(f"提取图像失败: {str(e)}")
                continue
                
        return blocks
    
    def _detect_table_blocks(self, page, page_idx):
        """检测页面中的表格 (简单实现)"""
        blocks = []
        
        # 简单的表格检测: 查找有多个水平和垂直线的区域
        # 实际应用中需要使用更复杂的表格检测算法
        
        # 提取页面上的所有线条
        paths = page.get_drawings()
        horizontal_lines = []
        vertical_lines = []
        
        for path in paths:
            # 检查是否为直线
            if path.get("type") == "l" and len(path.get("items", [])) == 1:
                item = path["items"][0]
                x0, y0 = item[0], item[1]
                x1, y1 = item[2], item[3]
                
                # 判断是水平线还是垂直线
                if abs(y1 - y0) < 2:  # 水平线
                    horizontal_lines.append((x0, y0, x1, y1))
                elif abs(x1 - x0) < 2:  # 垂直线
                    vertical_lines.append((x0, y0, x1, y1))
        
        # 如果水平线和垂直线数量超过阈值，可能存在表格
        if len(horizontal_lines) > 3 and len(vertical_lines) > 2:
            # 简单估算表格范围
            x_coords = [x for line in horizontal_lines + vertical_lines for x in [line[0], line[2]]]
            y_coords = [y for line in horizontal_lines + vertical_lines for y in [line[1], line[3]]]
            
            if x_coords and y_coords:
                x0, y0 = min(x_coords), min(y_coords)
                x1, y1 = max(x_coords), max(y_coords)
                
                blocks.append({
                    "type": "table",
                    "page": page_idx + 1,
                    "block_id": f"p{page_idx+1}_tab1",
                    "coordinates": [x0, y0, x1, y1],
                    "rows": len(set(y for line in horizontal_lines for y in [line[1], line[3]])),
                    "columns": len(set(x for line in vertical_lines for x in [line[0], line[2]])),
                    "confidence": self.table_detection_threshold
                })
                
        return blocks
    
    def _extract_nearby_caption(self, page, img_coords):
        """
        提取图像附近可能的标题文本
        
        简单实现: 查找图像下方或上方靠近的文本，可能是图片标题
        """
        x0, y0, x1, y1 = img_coords
        img_width = x1 - x0
        img_center_x = (x0 + x1) / 2
        
        # 获取页面所有文本块
        text_blocks = page.get_text("blocks")
        
        closest_caption = None
        min_distance = float('inf')
        
        for block in text_blocks:
            block_x0, block_y0, block_x1, block_y1 = block[:4]
            block_text = block[4].strip()
            
            if not block_text:
                continue
                
            # 检查文本块是否在图像下方或上方
            block_center_x = (block_x0 + block_x1) / 2
            x_overlap_ratio = min(x1, block_x1) - max(x0, block_x0)
            
            # 如果文本块水平位置与图像重叠，且在图像附近
            if (x_overlap_ratio > 0.3 * img_width or 
                abs(block_center_x - img_center_x) < 0.3 * img_width):
                
                # 计算垂直距离
                if block_y0 >= y1:  # 文本在图像下方
                    distance = block_y0 - y1
                elif block_y1 <= y0:  # 文本在图像上方
                    distance = y0 - block_y1
                else:
                    continue  # 文本与图像重叠，不考虑作为标题
                
                # 如果距离合适且比之前找到的更近
                if distance < min(100, min_distance) and len(block_text) < 200:
                    min_distance = distance
                    closest_caption = block_text
        
        return closest_caption or ""
    
    def _analyze_image(self, image_path):
        """
        分析图像文档
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            解析结果字典
        """
        self.logger.info(f"开始分析图像文档: {image_path}")
        
        try:
            # 读取图像
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            # 创建结果
            result = {
                "document_type": "image",
                "total_pages": 1,
                "metadata": {
                    "filename": os.path.basename(image_path),
                    "width": width,
                    "height": height
                },
                "blocks": [
                    {
                        "type": "image",
                        "page": 1,
                        "block_id": "p1_i1",
                        "coordinates": [0, 0, width, height],
                        "image_path": image_path,
                        "width": width,
                        "height": height,
                        "caption": ""
                    }
                ]
            }
            
            self.logger.info("图像分析完成")
            return result
            
        except Exception as e:
            self.logger.error(f"图像分析失败: {str(e)}")
            raise


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="文档结构分析器示例")
    parser.add_argument("document_path", help="要分析的文档路径 (PDF或图片)")
    parser.add_argument("--output", "-o", help="结果输出文件路径 (JSON格式)")
    parser.add_argument("--min-image-size", type=int, default=100, help="最小图像尺寸 (像素)")
    args = parser.parse_args()
    
    # 配置
    config = {
        "min_image_size": args.min_image_size,
        "table_detection_threshold": 0.7
    }
    
    try:
        # 创建文档分析器
        analyzer = DocumentAnalyzer(config)
        
        # 分析文档
        result = analyzer.analyze_document(args.document_path)
        
        # 输出结果
        if args.output:
            # 需要处理图像数据，将二进制数据转换为路径引用
            processed_result = result.copy()
            for block in processed_result["blocks"]:
                if block["type"] == "image" and "image_data" in block:
                    # 将二进制图像数据替换为占位符
                    block["image_data"] = "[binary_data]"
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(processed_result, f, indent=2, ensure_ascii=False)
            print(f"分析结果已保存到: {args.output}")
        else:
            # 输出摘要信息
            print("\n--- 文档分析结果摘要 ---")
            print(f"文档类型: {result['document_type']}")
            print(f"总页数: {result['total_pages']}")
            print(f"内容块数量: {len(result['blocks'])}")
            
            # 统计不同类型的块
            block_types = {}
            for block in result["blocks"]:
                block_type = block["type"]
                block_types[block_type] = block_types.get(block_type, 0) + 1
                
            print("\n块类型统计:")
            for block_type, count in block_types.items():
                print(f"  - {block_type}: {count}个")
                
            if "text" in block_types:
                print("\n文本块示例:")
                for block in result["blocks"]:
                    if block["type"] == "text":
                        text = block["content"]
                        if len(text) > 100:
                            text = text[:100] + "..."
                        print(f"  页码: {block['page']}, 内容: {text}")
                        break
                        
            if "image" in block_types:
                print("\n图像块示例:")
                for block in result["blocks"]:
                    if block["type"] == "image":
                        print(f"  页码: {block['page']}, 尺寸: {block['width']}x{block['height']}")
                        if block.get("caption"):
                            print(f"  标题: {block['caption']}")
                        break
                        
            if "table" in block_types:
                print("\n表格块示例:")
                for block in result["blocks"]:
                    if block["type"] == "table":
                        print(f"  页码: {block['page']}, 行数: {block.get('rows', '未知')}, 列数: {block.get('columns', '未知')}")
                        break
    
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
