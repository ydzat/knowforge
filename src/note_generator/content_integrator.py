"""
内容整合器模块
整合处理后的内容块，保持原始文档结构
"""
import os
from typing import List, Dict, Any, Tuple, Union, Optional

from src.utils.logger import get_module_logger
from src.utils.exceptions import OutputError


class ContentIntegrator:
    """
    内容整合器
    负责整合处理后的内容块，保持原始文档结构
    """
    
    def __init__(self, config=None):
        """
        初始化内容整合器
        
        Args:
            config: 配置选项
        """
        self.config = config or {}
        self.logger = get_module_logger("ContentIntegrator")
        
        # 整合配置
        self.preserve_structure = self.config.get("integrate.preserve_structure", True)
        self.image_placeholder = self.config.get("integrate.image_placeholder", "[图片]")
        self.table_placeholder = self.config.get("integrate.table_placeholder", "[表格]")
        self.formula_placeholder = self.config.get("integrate.formula_placeholder", "[公式]")
        
        self.logger.info("ContentIntegrator初始化完成")
    
    def integrate(self, processed_blocks):
        """
        整合处理后的内容块
        
        Args:
            processed_blocks: 处理后的内容块列表
            
        Returns:
            整合后的内容段落
        """
        self.logger.info(f"开始整合内容块，共{len(processed_blocks)}个")
        
        # 如果没有内容块，返回空列表
        if not processed_blocks:
            return []
        
        # 对块进行排序，保持原始顺序
        if self.preserve_structure:
            sorted_blocks = self._sort_blocks(processed_blocks)
        else:
            sorted_blocks = processed_blocks
        
        # 合并处理后的内容
        integrated_content = []
        
        # 按页面分组
        page_groups = self._group_by_page(sorted_blocks)
        
        # 处理每个页面的内容
        for page_num, page_blocks in page_groups.items():
            # 整合页面内容
            page_content = self._integrate_page(page_blocks)
            integrated_content.extend(page_content)
            
            # 如果不是最后一页，添加页面分隔符
            if page_num != max(page_groups.keys()):
                integrated_content.append("---")
        
        self.logger.info(f"内容整合完成，共{len(integrated_content)}个段落")
        return integrated_content
    
    def _sort_blocks(self, blocks):
        """
        对内容块进行排序，保持原始文档顺序
        
        Args:
            blocks: 内容块列表
            
        Returns:
            排序后的内容块列表
        """
        # 首先按页码排序
        blocks_by_page = sorted(blocks, key=lambda b: b.get("page", 1))
        
        # 然后在每一页内按位置排序（从上到下，从左到右）
        return sorted(blocks_by_page, key=lambda b: (
            b.get("page", 1),
            b.get("coordinates", [0, 0, 0, 0])[1],  # y0 坐标
            b.get("coordinates", [0, 0, 0, 0])[0]   # x0 坐标
        ))
    
    def _group_by_page(self, blocks):
        """
        按页码分组内容块
        
        Args:
            blocks: 内容块列表
            
        Returns:
            按页码分组的内容块字典
        """
        page_groups = {}
        
        for block in blocks:
            page_num = block.get("page", 1)
            if page_num not in page_groups:
                page_groups[page_num] = []
            page_groups[page_num].append(block)
        
        return page_groups
    
    def _integrate_page(self, page_blocks):
        """
        整合单个页面的内容块
        
        Args:
            page_blocks: 页面内容块列表
            
        Returns:
            整合后的内容段落列表
        """
        page_content = []
        
        # 处理每个内容块
        for block in page_blocks:
            block_type = block.get("type")
            
            if block_type == "text":
                # 处理文本块
                text_content = block.get("content", "").strip()
                if text_content:
                    page_content.append(text_content)
                    
            elif block_type == "image":
                # 处理图像块
                page_content.append(self._format_image(block))
                
            elif block_type == "table":
                # 处理表格块
                page_content.append(self._format_table(block))
                
            elif block_type == "formula":
                # 处理公式块
                page_content.append(self._format_formula(block))
        
        return page_content
    
    def _format_image(self, block):
        """
        格式化图像块
        
        Args:
            block: 图像内容块
            
        Returns:
            格式化后的图像内容
        """
        # 获取图像标题
        caption = block.get("caption", "图片")
        
        # 检查是否有提取的文本
        extracted_text = block.get("extracted_text")
        
        # 生成图像格式
        if extracted_text:
            return f"{self.image_placeholder}: {caption}\n{extracted_text}"
        else:
            return f"{self.image_placeholder}: {caption}"
    
    def _format_table(self, block):
        """
        格式化表格块
        
        Args:
            block: 表格内容块
            
        Returns:
            格式化后的表格内容
        """
        # 如果有Markdown表格，直接使用
        markdown = block.get("markdown")
        if markdown:
            return markdown
        
        # 否则使用占位符
        return self.table_placeholder
    
    def _format_formula(self, block):
        """
        格式化公式块
        
        Args:
            block: 公式内容块
            
        Returns:
            格式化后的公式内容
        """
        # 如果有LaTeX公式，直接使用
        latex = block.get("latex")
        if latex:
            return latex
        
        # 否则使用公式文本或占位符
        formula_text = block.get("formula_text", self.formula_placeholder)
        return formula_text
    
    def _maintain_structure(self, processed_blocks):
        """
        保持原始文档结构
        
        Args:
            processed_blocks: 处理后的内容块
            
        Returns:
            保持原始结构的内容段落
        """
        # 基础实现，未来版本将增强
        paragraphs = []
        
        # 所有文本块
        text_blocks = [b for b in processed_blocks if b.get("type") == "text"]
        
        # 提取段落和结构信息
        for block in text_blocks:
            content = block.get("content", "")
            paragraphs.extend(content.split("\n\n"))
        
        return paragraphs
    
    def integrate_to_markdown(self, processed_blocks):
        """
        将处理后的内容块整合为Markdown格式
        
        Args:
            processed_blocks: 处理后的内容块列表
            
        Returns:
            Markdown格式的整合内容
        """
        self.logger.info("生成Markdown格式输出")
        
        # 首先整合内容
        integrated = self.integrate(processed_blocks)
        
        # 转换为Markdown
        markdown_parts = []
        
        for part in integrated:
            # 处理表格（已经是Markdown格式）
            if part.startswith("|") and " | " in part:
                markdown_parts.append(part)
                markdown_parts.append("")  # 空行
            
            # 处理LaTeX公式
            elif part.startswith("$$") and part.endswith("$$"):
                markdown_parts.append(part)
                markdown_parts.append("")  # 空行
            
            # 处理图片
            elif part.startswith(self.image_placeholder):
                # 提取标题和文本
                try:
                    caption = part.split(":", 1)[1].strip()
                    text = part.split("\n", 1)[1] if "\n" in part else ""
                    
                    markdown_parts.append(f"![{caption}](image_placeholder)")
                    if text:
                        markdown_parts.append(f"> {caption}：{text}")
                    markdown_parts.append("")  # 空行
                except:
                    markdown_parts.append(part)
                    markdown_parts.append("")  # 空行
            
            # 处理普通文本
            else:
                markdown_parts.append(part)
                markdown_parts.append("")  # 空行
        
        return "\n".join(markdown_parts)
    
    def integrate_to_html(self, processed_blocks):
        """
        将处理后的内容块整合为HTML格式
        
        Args:
            processed_blocks: 处理后的内容块列表
            
        Returns:
            HTML格式的整合内容
        """
        self.logger.info("生成HTML格式输出")
        
        # 首先整合内容
        integrated = self.integrate(processed_blocks)
        
        # 转换为HTML
        html_parts = ["<div class=\"document-content\">"]
        
        for part in integrated:
            # 处理表格
            if part.startswith("|") and " | " in part:
                # 将Markdown表格转换为HTML
                html_table = self._markdown_table_to_html(part)
                html_parts.append(html_table)
            
            # 处理LaTeX公式
            elif part.startswith("$$") and part.endswith("$$"):
                formula = part[2:-2]  # 移除$$
                html_parts.append(f"<div class=\"formula\">{formula}</div>")
            
            # 处理图片
            elif part.startswith(self.image_placeholder):
                # 提取标题和文本
                try:
                    caption = part.split(":", 1)[1].strip()
                    text = part.split("\n", 1)[1] if "\n" in part else ""
                    
                    html_parts.append(f"<figure>")
                    html_parts.append(f"<img src=\"image_placeholder\" alt=\"{caption}\">")
                    html_parts.append(f"<figcaption>{caption}</figcaption>")
                    if text:
                        html_parts.append(f"<p class=\"image-text\">{text}</p>")
                    html_parts.append(f"</figure>")
                except:
                    html_parts.append(f"<p>{part}</p>")
            
            # 处理页面分隔符
            elif part == "---":
                html_parts.append("<hr class=\"page-break\">")
            
            # 处理普通文本
            else:
                html_parts.append(f"<p>{part}</p>")
        
        html_parts.append("</div>")
        return "\n".join(html_parts)
    
    def _markdown_table_to_html(self, markdown_table):
        """
        将Markdown表格转换为HTML表格
        
        Args:
            markdown_table: Markdown格式的表格
            
        Returns:
            HTML格式的表格
        """
        lines = markdown_table.split("\n")
        
        # 确保这是一个Markdown表格
        if len(lines) < 3 or not all("|" in line for line in lines):
            return f"<p>{markdown_table}</p>"
        
        html = ["<table>"]
        
        # 处理表头
        header = lines[0].strip()
        cells = [cell.strip() for cell in header.split("|")[1:-1]]
        html.append("<thead>")
        html.append("<tr>")
        for cell in cells:
            html.append(f"<th>{cell}</th>")
        html.append("</tr>")
        html.append("</thead>")
        
        # 处理表格内容
        html.append("<tbody>")
        for i, line in enumerate(lines[2:]):  # 跳过表头和分隔行
            if not line.strip():
                continue
            
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            html.append("<tr>")
            for cell in cells:
                html.append(f"<td>{cell}</td>")
            html.append("</tr>")
        html.append("</tbody>")
        
        html.append("</table>")
        return "\n".join(html)
