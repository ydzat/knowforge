"""
输出写入器模块，负责生成不同格式的输出文件
"""
import os
import re
import time
from datetime import datetime
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell
from typing import List, Dict, Optional, Any, Tuple
from src.utils.logger import get_module_logger
from src.utils.exceptions import OutputError
from src.utils.config_loader import ConfigLoader
from src.utils.locale_manager import LocaleManager
from src import __version__

logger = get_module_logger("output_writer")

class OutputWriter:
    """输出写入器类，负责生成不同格式的输出文件"""
    
    def __init__(self, workspace_dir: str, output_dir: str, config: ConfigLoader, locale_manager: LocaleManager = None):
        """
        初始化输出写入器
        
        Args:
            workspace_dir: 工作空间目录
            output_dir: 输出目录
            config: 配置加载器
            locale_manager: 语言资源管理器
        """
        self.workspace_dir = workspace_dir
        self.output_dir = output_dir
        self.config = config
        self.locale = locale_manager
        
        # 初始化输出目录
        self.md_output_dir = os.path.join(output_dir, "markdown")
        self.nb_output_dir = os.path.join(output_dir, "notebook")
        self.pdf_output_dir = os.path.join(output_dir, "pdf")
        
        os.makedirs(self.md_output_dir, exist_ok=True)
        os.makedirs(self.nb_output_dir, exist_ok=True)
        os.makedirs(self.pdf_output_dir, exist_ok=True)
        
        # 加载模板
        self.template_path = self.config.get(
            "output.template_path", 
            "resources/templates/note_template.md"
        )
        
        self.template = self._load_template(self.template_path)
        
        if self.locale:
            logger.info(self.locale.get("output.initialized").format(output_dir=output_dir))
        else:
            logger.info("OutputWriter initialized: {0}".format(output_dir))
    
    def generate_markdown(self, segments: List[str], filename: str, 
                         title: str = None) -> str:
        """
        生成Markdown格式输出
        
        Args:
            segments: 文本片段列表
            filename: 输出文件名（不含扩展名）
            title: 文档标题，若为None则使用filename
            
        Returns:
            生成的Markdown文件路径
        """
        if self.locale:
            logger.info(self.locale.get("output.markdown.generating").format(filename=filename))
        else:
            logger.info("Generating Markdown document: {0}".format(filename))
        
        if not title:
            title = filename.replace('_', ' ').title()
        
        # 合并所有文本片段
        content = self._merge_segments(segments)
        
        # 生成目录
        toc = self._generate_toc(content)
        
        # 应用模板
        markdown_text = self._apply_template(title, content, toc)
        
        # 保存到文件
        output_path = os.path.join(self.md_output_dir, f"{filename}.md")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            if self.locale:
                logger.info(self.locale.get("output.markdown.saved").format(path=output_path))
            else:
                logger.info("Markdown document saved: {0}".format(output_path))
            return output_path
            
        except Exception as e:
            if self.locale:
                logger.error(self.locale.get("output.markdown.error.save").format(error=str(e)))
                raise OutputError(self.locale.get("output.markdown.error.generate").format(error=str(e)))
            else:
                logger.error("Failed to save Markdown file: {0}".format(str(e)))
                raise OutputError("Failed to generate Markdown file: {0}".format(str(e)))
    
    def generate_notebook(self, segments: List[str], filename: str, 
                         title: str = None) -> str:
        """
        生成Jupyter Notebook格式输出
        
        Args:
            segments: 文本片段列表
            filename: 输出文件名（不含扩展名）
            title: 文档标题，若为None则使用filename
            
        Returns:
            生成的Notebook文件路径
        """
        if self.locale:
            logger.info(self.locale.get("output.notebook.generating").format(filename=filename))
        else:
            logger.info("Generating Jupyter Notebook: {0}".format(filename))
        
        try:
            if not title:
                title = filename.replace('_', ' ').title()
            
            # 创建一个新的notebook
            nb = new_notebook()
            
            # 添加标题单元格
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if self.locale:
                header_timestamp = self.locale.get("output.timestamp").format(timestamp=timestamp)
                header = f"# {title}\n\n**{header_timestamp}**\n\n---\n"
            else:
                header = f"# {title}\n\n**Generated at**: {timestamp}\n\n---\n"
                
            nb.cells.append(new_markdown_cell(header))
            
            # 生成目录单元格
            content = self._merge_segments(segments)
            toc = self._generate_toc(content)
            if toc:
                if self.locale:
                    toc_title = self.locale.get("output.toc")
                    nb.cells.append(new_markdown_cell(f"## {toc_title}\n" + toc))
                else:
                    nb.cells.append(new_markdown_cell("## Table of Contents\n" + toc))
            
            # 添加内容单元格
            # 将内容按章节或段落拆分成多个单元格，提高可读性
            chapter_pattern = r'^#+\s+(.+)$'  # Markdown标题格式
            current_section = []
            
            for line in content.splitlines():
                if re.match(chapter_pattern, line) and current_section:
                    # 发现新章节，保存当前部分
                    nb.cells.append(new_markdown_cell('\n'.join(current_section)))
                    current_section = [line]
                else:
                    current_section.append(line)
            
            # 添加最后一部分
            if current_section:
                nb.cells.append(new_markdown_cell('\n'.join(current_section)))
            
            # 添加footer
            if self.locale:
                footer_text = self.locale.get("output.footer").format(version=str(__version__))
                footer = f"---\n\n{footer_text}"
            else:
                footer = f"---\n\n*This note was automatically generated by KnowForge*\n*Version: {__version__}*"
                
            nb.cells.append(new_markdown_cell(footer))
            
            # 保存notebook
            output_path = os.path.join(self.nb_output_dir, f"{filename}.ipynb")
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            
            if self.locale:
                logger.info(self.locale.get("output.notebook.saved").format(path=output_path))
            else:
                logger.info("Jupyter Notebook saved: {0}".format(output_path))
            return output_path
            
        except Exception as e:
            if self.locale:
                logger.error(self.locale.get("output.notebook.error.generate").format(error=str(e)))
                raise OutputError(self.locale.get("output.notebook.error.generate").format(error=str(e)))
            else:
                logger.error("Failed to generate Notebook: {0}".format(str(e)))
                raise OutputError("Failed to generate Jupyter Notebook: {0}".format(str(e)))
    
    def generate_pdf(self, markdown_path: str, filename: str) -> str:
        """
        从Markdown生成PDF格式输出(使用weasyprint)
        
        Args:
            markdown_path: Markdown文件路径
            filename: 输出文件名（不含扩展名）
            
        Returns:
            生成的PDF文件路径
        """
        if self.locale:
            logger.info(self.locale.get("output.pdf.generating").format(filename=filename))
        else:
            logger.info("Generating PDF from Markdown: {0}".format(filename))
        
        try:
            # 由于完整的PDF生成需要引入较多依赖，在此我们仅创建一个PDF占位符
            # 实际项目中，此处可以使用weasyprint或其他库将markdown转换为PDF
            
            # 导入weasyprint可能需要额外安装依赖，在迭代2中我们先使用简化实现
            # from weasyprint import HTML, CSS
            # html = markdown_to_html(markdown_content)
            # pdf_bytes = HTML(string=html).write_pdf()
            
            # 简化实现：直接复制markdown文件并改扩展名为pdf
            output_path = os.path.join(self.pdf_output_dir, f"{filename}.pdf")
            
            # 读取markdown内容
            with open(markdown_path, 'r', encoding='utf-8') as md_file:
                md_content = md_file.read()
                
            # 写入简单的PDF占位符
            with open(output_path, 'w', encoding='utf-8') as pdf_file:
                if self.locale:
                    header = self.locale.get("output.pdf.placeholder.header")
                    pdf_file.write(f"{header}\n\n{md_content}")
                else:
                    pdf_file.write(f"PDF VERSION OF:\n\n{md_content}")
            
            if self.locale:
                logger.info(self.locale.get("output.pdf.saved").format(path=output_path))
                logger.warning(self.locale.get("output.pdf.placeholder.warning"))
            else:
                logger.info("PDF document saved: {0}".format(output_path))
                logger.warning("Note: Only a PDF placeholder has been generated. Actual PDF formatting will be implemented in future iterations")
            
            return output_path
            
        except Exception as e:
            if self.locale:
                logger.error(self.locale.get("output.pdf.error.generate").format(error=str(e)))
                raise OutputError(self.locale.get("output.pdf.error.generate").format(error=str(e)))
            else:
                logger.error("Failed to generate PDF: {0}".format(str(e)))
                raise OutputError("Failed to generate PDF file: {0}".format(str(e)))
    
    def _load_template(self, template_path: str) -> str:
        """
        加载Markdown模板
        
        Args:
            template_path: 模板文件路径
            
        Returns:
            模板内容
        """
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            if self.locale:
                logger.warning(self.locale.get("output.template.missing").format(path=template_path))
                
                # 默认模板也应该使用语言资源
                default_template = (
                    "# {{ title }}\n\n"
                    "**{timestamp_label}**: {{ timestamp }}\n"
                    "**{source_label}**: {{ source }}\n\n"
                    "---\n\n"
                    "## {toc_label}\n"
                    "{{ toc }}\n\n"
                    "---\n\n"
                    "{{ content }}\n\n"
                    "---\n\n"
                    "{footer_text}"
                )
                
                # 替换模板中的标签
                default_template = default_template.format(
                    timestamp_label=self.locale.get("output.timestamp_label"),
                    source_label=self.locale.get("output.source_label"),
                    toc_label=self.locale.get("output.toc"),
                    footer_text=self.locale.get("output.footer").format(version="{{ version }}")
                )
                
                return default_template
            else:
                logger.warning("Template file does not exist: {0}, using default template".format(template_path))
                return "# {{ title }}\n\n**Generated at**: {{ timestamp }}\n**Source**: {{ source }}\n\n---\n\n## Table of Contents\n{{ toc }}\n\n---\n\n{{ content }}\n\n---\n\n*This note was automatically generated by KnowForge*\n*Version: {{ version }}*"
        except Exception as e:
            if self.locale:
                logger.error(self.locale.get("output.template.error.load").format(error=str(e)))
                raise OutputError(self.locale.get("output.template.error.load").format(error=str(e)))
            else:
                logger.error("Failed to load template: {0}".format(str(e)))
                raise OutputError("Failed to load note template: {0}".format(str(e)))
    
    def _apply_template(self, title: str, content: str, toc: str) -> str:
        """
        应用模板，替换占位符
        
        Args:
            title: 文档标题
            content: 文档内容
            toc: 目录
            
        Returns:
            应用模板后的文本
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取来源说明
        if self.locale:
            source = self.locale.get("output.source.multiple")
        else:
            source = "Multiple sources"
        
        # 替换模板占位符
        text = self.template
        text = text.replace("{{ title }}", title)
        text = text.replace("{{ timestamp }}", timestamp)
        text = text.replace("{{ source }}", source)
        text = text.replace("{{ toc }}", toc)
        text = text.replace("{{ content }}", content)
        text = text.replace("{{ version }}", str(__version__))
        
        return text
    
    def _merge_segments(self, segments: List[str]) -> str:
        """
        合并多个文本片段为一个文档
        
        Args:
            segments: 文本片段列表
            
        Returns:
            合并后的文档文本
        """
        if not segments:
            return ""
        
        # 如果只有一个片段，直接返回
        if len(segments) == 1:
            return segments[0]
        
        # 合并多个片段，适当添加分隔符
        merged = ""
        prev_segment = ""
        
        for i, segment in enumerate(segments):
            # 如果当前片段是前一个片段的完整重复，跳过
            if segment in prev_segment:
                continue
                
            # 检查重叠内容
            overlap = self._find_overlap(prev_segment, segment, 50)
            
            if i == 0:
                # 第一个片段直接添加
                merged += segment
            elif overlap:
                # 如果有重叠，去除重叠部分再添加
                merged += segment[len(overlap):]
            else:
                # 无重叠，添加分隔线和片段
                merged += "\n\n---\n\n" + segment
            
            prev_segment = segment
            
        return merged
    
    def _find_overlap(self, text1: str, text2: str, max_length: int = 100) -> str:
        """
        查找两段文本之间的重叠部分
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            max_length: 最大检查长度
            
        Returns:
            重叠的文本片段
        """
        if not text1 or not text2:
            return ""
            
        # 取text1的末尾和text2的开头进行匹配
        end_of_text1 = text1[-min(max_length, len(text1)):]
        start_of_text2 = text2[:min(max_length, len(text2))]
        
        # 寻找最长的重叠
        overlap = ""
        for i in range(1, min(len(end_of_text1), len(start_of_text2)) + 1):
            if end_of_text1[-i:] == start_of_text2[:i]:
                overlap = start_of_text2[:i]
        
        return overlap
    
    def _generate_toc(self, content: str) -> str:
        """
        从文档内容生成目录
        
        Args:
            content: 文档内容
            
        Returns:
            目录文本
        """
        toc = []
        headers = []
        
        # 查找所有标题行
        for line in content.splitlines():
            if line.startswith('#'):
                # 计算标题级别和文本
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                if level > 0 and level <= 6:  # Markdown支持1-6级标题
                    header_text = line[level:].strip()
                    headers.append((level, header_text))
        
        # 生成目录
        for level, text in headers:
            indent = '  ' * (level - 1)
            # 创建anchor链接
            anchor = text.lower().replace(' ', '-')
            anchor = re.sub(r'[^\w\-]', '', anchor)
            toc_entry = f"{indent}- [{text}](#{anchor})"
            toc.append(toc_entry)
        
        return '\n'.join(toc)