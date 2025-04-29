"""
主处理器模块，协调所有处理流程
"""
import os
import time
from typing import List, Optional, Dict, Any
from src.utils.config_loader import ConfigLoader
from src.utils.locale_manager import LocaleManager
from src.utils.logger import setup_logger, get_module_logger
from src.utils.exceptions import NoteGenError
from src.note_generator.input_handler import InputHandler
from src.note_generator.splitter import Splitter
from src.note_generator.output_writer import OutputWriter

logger = get_module_logger("processor")

class Processor:
    """笔记生成主处理器"""
    
    def __init__(self, 
                 input_dir: str = "input/", 
                 output_dir: str = "output/", 
                 config_path: str = "resources/config/config.yaml"):
        """
        初始化处理器
        
        Args:
            input_dir: 输入文件目录
            output_dir: 输出文件目录
            config_path: 配置文件路径
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 设置日志
        self.logger = setup_logger(
            log_dir=os.path.join(output_dir, "logs"),
            log_name="note_gen.log"
        )
        
        try:
            # 加载配置
            self.config = ConfigLoader(config_path)
            
            # 初始化语言资源
            language = self.config.get("system.language", "zh")
            self.locale = LocaleManager(f"resources/locales/{language}.yaml", language)
            
            self.logger.info(self.locale.get("processor.initialized"))
            self.logger.info(self.locale.get("processor.config_loaded").format(path=config_path))
            
            # 确保工作目录存在
            self.workspace_dir = self.config.get(
                "system.workspace_dir", 
                "workspace/"
            )
            os.makedirs(self.workspace_dir, exist_ok=True)
            
            # 初始化输入/输出目录
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # 初始化核心组件
            self.input_handler = InputHandler(input_dir, self.workspace_dir, self.config)
            self.splitter = Splitter(self.config)
            self.output_writer = OutputWriter(self.workspace_dir, output_dir, self.config, self.locale)
            
        except Exception as e:
            # 如果配置加载失败，尝试获取默认的locale错误消息，否则使用英文
            error_msg = "Processor initialization failed: {0}"
            if hasattr(self, 'locale') and self.locale:
                self.logger.error(self.locale.get("processor.init_failed").format(error=str(e)))
            else:
                self.logger.error(error_msg.format(str(e)))
            raise NoteGenError(error_msg.format(str(e)))
    
    def run_full_pipeline(self, output_formats: List[str] = None) -> Dict[str, str]:
        """
        运行完整的处理流程
        
        Args:
            output_formats: 输出格式列表，如["markdown", "ipynb", "pdf"]
                           为None时使用配置中的默认值
        
        Returns:
            输出文件路径的字典，格式为 {"markdown": "path/to/file.md", ...}
        """
        start_time = time.time()
        self.logger.info(self.locale.get("processor.pipeline_start"))
        
        if output_formats is None:
            # 使用配置中的默认输出格式
            output_formats = self.config.get("output.formats", ["markdown"])
        
        try:
            # 1. 处理输入文件
            self.logger.info(self.locale.get("processor.processing_input"))
            segments = self.input_handler.extract_texts()
            self.logger.info(self.locale.get("processor.segments_extracted").format(count=len(segments)))
            
            if not segments:
                self.logger.warning(self.locale.get("processor.no_valid_input"))
                return {}
            
            # 2. 拆分文本
            self.logger.info(self.locale.get("processor.splitting_text"))
            split_segments = self.splitter.split_text(segments)
            self.logger.info(self.locale.get("processor.split_completed").format(count=len(split_segments)))
            
            # 3. 生成输出
            self.logger.info(self.locale.get("processor.generating_output").format(formats=", ".join(output_formats)))
            output_paths = {}
            
            for fmt in output_formats:
                if fmt == "markdown":
                    path = self.output_writer.generate_markdown(split_segments, "notes")
                    output_paths["markdown"] = path
                elif fmt == "ipynb":
                    path = self.output_writer.generate_notebook(split_segments, "notes")
                    output_paths["ipynb"] = path
                elif fmt == "pdf":
                    md_path = output_paths.get("markdown")
                    if not md_path:
                        md_path = self.output_writer.generate_markdown(split_segments, "notes")
                        output_paths["markdown"] = md_path
                    path = self.output_writer.generate_pdf(md_path, "notes")
                    output_paths["pdf"] = path
            
            elapsed = time.time() - start_time
            self.logger.info(self.locale.get("processor.pipeline_completed").format(elapsed=elapsed))
            
            return output_paths
            
        except Exception as e:
            self.logger.error(self.locale.get("processor.pipeline_failed").format(error=str(e)))
            raise NoteGenError(self.locale.get("processor.note_generation_failed").format(error=str(e)))
    
    def generate_note(self, 
                     input_file: str, 
                     output_format: str = "markdown") -> str:
        """
        从单个文件生成笔记
        
        Args:
            input_file: 输入文件路径
            output_format: 输出格式，默认为markdown
        
        Returns:
            输出文件路径
        """
        self.logger.info(self.locale.get("processor.processing_single_file").format(file_path=input_file))
        
        try:
            # 1. 处理单个输入文件
            text = self.input_handler.process_file(input_file)
            self.logger.info(self.locale.get("processor.text_extracted").format(char_count=len(text)))
            
            # 2. 拆分文本
            segments = self.splitter.split_text([text])
            self.logger.info(self.locale.get("processor.split_completed").format(count=len(segments)))
            
            # 3. 生成输出
            filename = os.path.splitext(os.path.basename(input_file))[0]
            self.logger.info(self.locale.get("processor.generating_format_output").format(format=output_format, filename=filename))
            
            if output_format == "markdown":
                return self.output_writer.generate_markdown(segments, filename)
            elif output_format == "ipynb":
                return self.output_writer.generate_notebook(segments, filename)
            elif output_format == "pdf":
                md_path = self.output_writer.generate_markdown(segments, filename)
                return self.output_writer.generate_pdf(md_path, filename)
            else:
                raise NoteGenError(self.locale.get("processor.unsupported_format").format(format=output_format))
            
        except Exception as e:
            self.logger.error(self.locale.get("processor.file_processing_failed").format(error=str(e)))
            raise NoteGenError(self.locale.get("processor.file_note_generation_failed").format(error=str(e)))