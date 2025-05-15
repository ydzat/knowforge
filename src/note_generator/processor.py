"""
主处理器模块，协调所有处理流程
"""
import os
import time
from typing import List, Optional, Dict, Any
from src.utils.config_loader import ConfigLoader
from src.utils.locale_manager import LocaleManager
from src.utils.logger import setup_logger, get_module_logger
from src.utils.locale_manager import safe_get_text, safe_format_text
from src.utils.exceptions import NoteGenError
from src.note_generator.input_handler import InputHandler
from src.note_generator.splitter import Splitter
from src.note_generator.output_writer import OutputWriter
from src.note_generator.embedder import Embedder
from src.note_generator.memory_manager import MemoryManager

# 使用安全版本的文本获取函数，避免循环依赖
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
            
            # 使用安全版本的格式化函数记录日志
            logger.info(safe_get_text("processor.initialized"))
            logger.info(safe_format_text("processor.config_loaded", {"path": config_path}))
            
            # 确保工作目录存在
            self.workspace_dir = self.config.get(
                "system.workspace_dir", 
                "workspace/"
            )
            os.makedirs(self.workspace_dir, exist_ok=True)
            
            # 初始化输入/输出目录
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # 确保config包含input_dir路径配置
            if not self.config.get("paths"):
                self.config.set("paths", {})
            self.config.set("paths.input_dir", input_dir)
            
            # 初始化核心组件
            self.input_handler = InputHandler(self.config, self.workspace_dir)
            self.splitter = Splitter(self.config)
            
            # 检查是否启用记忆功能
            memory_enabled = self.config.get("memory.enabled", True)
            
            if memory_enabled:
                # 初始化向量化组件
                embedding_model = self.config.get(
                    "embedding.model_name", 
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                embeddings_dir = self.config.get(
                    "embedding.cache_dir",
                    os.path.join(self.workspace_dir, "embeddings")
                )
                os.makedirs(embeddings_dir, exist_ok=True)
                self.embedder = Embedder(embedding_model, cache_dir=embeddings_dir)
                
                # 初始化记忆管理组件
                chroma_db_path = self.config.get(
                    "memory.chroma_db_path", 
                    os.path.join(self.workspace_dir, "memory_db")
                )
                collection_name = self.config.get(
                    "memory.collection_name", 
                    "knowforge_memory"
                )
                
                # 使用配置初始化记忆管理器
                memory_config = self.config.get_section("memory")
                self.memory_manager = MemoryManager(
                    chroma_db_path=chroma_db_path,
                    embedder=self.embedder,
                    collection_name=collection_name,
                    config=memory_config
                )
                
                logger.info(safe_format_text("processor.memory_initialized", {
                    "retrieval_mode": memory_config.get("retrieval_strategy", {}).get("mode", "simple"),
                    "collection": collection_name,
                    "count": self.memory_manager.collection.count()
                }))
            else:
                self.embedder = None
                self.memory_manager = None
                logger.info(safe_get_text("processor.memory_disabled"))
            
            # 初始化输出组件
            self.output_writer = OutputWriter(self.workspace_dir, output_dir, self.config, self.locale)
            
        except Exception as e:
            # 使用安全版本记录错误日志
            logger.error(safe_format_text("processor.init_failed", {"error": str(e)}))
            raise NoteGenError(safe_format_text("processor.init_failed", {"error": str(e)}))
    
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
        logger.info(safe_get_text("processor.pipeline_start"))
        
        if output_formats is None:
            # 使用配置中的默认输出格式
            output_formats = self.config.get("output.formats", ["markdown"])
        
        try:
            # 1. 处理输入文件
            logger.info(safe_get_text("processor.processing_input"))
            segments = self.input_handler.extract_texts()
            logger.info(safe_format_text("processor.segments_extracted", {"count": len(segments)}))
            
            if not segments:
                logger.warning(safe_get_text("processor.no_valid_input"))
                return {}
            
            # 2. 拆分文本
            logger.info(safe_get_text("processor.splitting_text"))
            split_segments = self.splitter.split_text(segments)
            logger.info(safe_format_text("processor.split_completed", {"count": len(split_segments)}))
            
            # 3. 向量化并存储到记忆库 - 如果启用了记忆功能
            if self.memory_manager is not None:
                self._process_memory(split_segments)
                
                # 使用记忆增强生成内容
                if self.config.get("memory.augmentation.enabled", True):
                    split_segments = self._enhance_with_memory(split_segments)
            
            # 4. 生成输出
            logger.info(safe_format_text("processor.generating_output", {"formats": ", ".join(output_formats)}))
            output_paths = {}
            
            for fmt in output_formats:
                if fmt == "markdown":
                    path = self.output_writer.generate_markdown(split_segments, "notes")
                    output_paths["markdown"] = path
                elif fmt == "notebook" or fmt == "ipynb":
                    path = self.output_writer.generate_notebook(split_segments, "notes")
                    # 确保使用客户端请求的键名
                    output_paths[fmt] = path
                elif fmt == "pdf":
                    md_path = output_paths.get("markdown")
                    if not md_path:
                        md_path = self.output_writer.generate_markdown(split_segments, "notes")
                        output_paths["markdown"] = md_path
                    path = self.output_writer.generate_pdf(md_path, "notes")
                    output_paths["pdf"] = path
            
            elapsed = time.time() - start_time
            logger.info(safe_format_text("processor.pipeline_completed", {"elapsed": elapsed}))
            
            return output_paths
            
        except Exception as e:
            logger.error(safe_format_text("processor.pipeline_failed", {"error": str(e)}))
            raise NoteGenError(safe_format_text("processor.note_generation_failed", {"error": str(e)}))
    
    def _process_memory(self, segments: List[str]):
        """处理文本片段并更新记忆库"""
        logger.info(safe_get_text("processor.vectorizing_segments"))
        
        # 保存向量到文件系统（可选）
        if self.config.get("embedding.save_embeddings", False):
            embeddings_dir = self.config.get("embedding.cache_dir", os.path.join(self.workspace_dir, "embeddings"))
            self.embedder.save_embeddings(segments, embeddings_dir)
            logger.info(safe_format_text("processor.embeddings_saved", 
                                      {"count": len(segments), "dir": embeddings_dir}))
        
        # 添加到记忆库
        metadata = [{"source": "input_processing", 
                    "timestamp": str(time.time()),
                    "access_count": "0",
                    "index": i} for i in range(len(segments))]
        self.memory_manager.add_segments(segments, metadata)
        
        # 获取记忆库统计信息并记录
        stats = self.memory_manager.get_collection_stats()
        logger.info(safe_format_text("processor.memory_stats", 
                                  {"count": stats["count"], "avg_length": stats.get("avg_text_length", 0)}))
    
    def _enhance_with_memory(self, segments: List[str]) -> List[str]:
        """使用记忆库增强文本片段"""
        if not self.memory_manager:
            return segments
            
        logger.info(safe_get_text("processor.enhancing_with_memory"))
        
        # 获取增强配置
        augmentation_config = self.config.get_section("memory.augmentation")
        max_refs = augmentation_config.get("max_references", 3)
        min_similarity = augmentation_config.get("min_similarity", 0.75)
        ref_format = augmentation_config.get("reference_format", "markdown")
        
        enhanced_segments = []
        total_refs = 0
        
        # 遍历所有片段，使用记忆增强
        for i, segment in enumerate(segments):
            # 上下文感知检索时的上下文文本
            context_texts = []
            if i > 0:
                context_texts.append(segments[i-1])
            if i < len(segments) - 1:
                context_texts.append(segments[i+1])
                
            # 查询相似内容
            similar_results = self.memory_manager.query_similar(
                query_text=segment,
                # 请求更多结果，后续会进行过滤
                top_k=max_refs * 2,
                threshold=min_similarity * 0.8,  # 稍微降低阈值以获取更多候选项
                context_texts=context_texts
            )
            
            # 过滤掉自身及相似度低于阈值的结果
            filtered_results = [
                item for item in similar_results
                if item["text"] != segment and item["similarity"] >= min_similarity
            ]
            
            # 如果找到了相关记忆
            if filtered_results:
                # 限制引用数量
                refs_to_use = filtered_results[:max_refs]
                total_refs += len(refs_to_use)
                
                # 根据配置的格式添加引用
                if ref_format == "markdown":
                    # Markdown引用格式
                    references = "\n\n### 相关内容:\n\n" + "\n\n".join([
                        f"> {item['text']} (相似度: {item['similarity']:.2f})"
                        for item in refs_to_use
                    ])
                else:
                    # 纯文本格式
                    references = "\n\n相关内容:\n\n" + "\n\n".join([
                        f"{item['text']} (相似度: {item['similarity']:.2f})"
                        for item in refs_to_use
                    ])
                
                # 合并原始文本和引用
                enhanced_text = segment + references
                enhanced_segments.append(enhanced_text)
            else:
                # 没有找到相关记忆，保持原样
                enhanced_segments.append(segment)
        
        logger.info(safe_format_text("processor.memory_enhancement_completed", 
                                  {"total_refs": total_refs}))
        return enhanced_segments
    
    def generate_note(self, 
                     input_file: str, 
                     output_format: str = "markdown",
                     use_memory: bool = None) -> str:
        """
        从单个文件生成笔记
        
        Args:
            input_file: 输入文件路径
            output_format: 输出格式，默认为markdown
            use_memory: 是否使用记忆库检索相关内容，默认使用配置中的设置
        
        Returns:
            输出文件路径
        """
        logger.info(safe_format_text("processor.processing_single_file", {"file_path": input_file}))
        
        # 从配置中获取是否启用记忆
        if use_memory is None:
            use_memory = self.config.get("memory.enabled", True) and self.memory_manager is not None
        
        try:
            # 1. 处理单个输入文件
            text = self.input_handler.process_file(input_file)
            logger.info(safe_format_text("processor.text_extracted", {"char_count": len(text)}))
            
            # 2. 拆分文本
            segments = self.splitter.split_text([text])
            logger.info(safe_format_text("processor.split_completed", {"count": len(segments)}))
            
            # 3. 向量化并利用记忆库 - 如果启用了记忆功能
            if use_memory and self.memory_manager is not None:
                # 添加新内容到记忆库
                metadata = [{
                    "source": "single_file", 
                    "file": os.path.basename(input_file),
                    "timestamp": str(time.time()),
                    "access_count": "0",
                    "index": i
                } for i in range(len(segments))]
                self.memory_manager.add_segments(segments, metadata)
                
                # 使用记忆增强生成
                segments = self._enhance_with_memory(segments)
            
            # 4. 生成输出
            filename = os.path.splitext(os.path.basename(input_file))[0]
            logger.info(safe_format_text("processor.generating_format_output", 
                                    {"format": output_format, "filename": filename}))
            
            if output_format == "markdown":
                return self.output_writer.generate_markdown(segments, filename)
            elif output_format == "ipynb":
                return self.output_writer.generate_notebook(segments, filename)
            elif output_format == "pdf":
                md_path = self.output_writer.generate_markdown(segments, filename)
                return self.output_writer.generate_pdf(md_path, filename)
            else:
                raise NoteGenError(safe_format_text("processor.unsupported_format", {"format": output_format}))
            
        except Exception as e:
            logger.error(safe_format_text("processor.file_processing_failed", {"error": str(e)}))
            raise NoteGenError(safe_format_text("processor.file_note_generation_failed", {"error": str(e)}))
    
    def query_memory(
        self, 
        query_text: str, 
        top_k: int = None,
        threshold: float = None,
        retrieval_mode: str = None,
        with_stats: bool = False
    ) -> Dict[str, Any]:
        """
        查询记忆库中与输入文本相似的片段
        
        Args:
            query_text: 查询文本
            top_k: 返回的最大结果数量，默认使用配置
            threshold: 相似度阈值，默认使用配置
            retrieval_mode: 检索模式，默认使用配置
            with_stats: 是否返回记忆库统计信息
            
        Returns:
            查询结果及相关信息的字典
            
        Raises:
            NoteGenError: 记忆功能未启用或查询失败
        """
        if not self.memory_manager:
            logger.warning(safe_get_text("processor.memory_not_enabled"))
            raise NoteGenError(safe_get_text("processor.memory_not_enabled"))
            
        logger.info(safe_format_text("processor.querying_memory", {"query": query_text[:50] + "..."}))
        
        try:
            # 查询记忆
            results = self.memory_manager.query_similar(
                query_text=query_text,
                top_k=top_k,
                threshold=threshold,
                retrieval_mode=retrieval_mode
            )
            
            response = {
                "query": query_text,
                "results": results,
                "count": len(results)
            }
            
            # 添加统计信息
            if with_stats:
                response["stats"] = self.memory_manager.get_collection_stats()
                
            logger.info(safe_format_text("processor.memory_query_results", {"count": len(results)}))
            return response
            
        except Exception as e:
            logger.error(safe_format_text("processor.memory_query_failed", {"error": str(e)}))
            raise NoteGenError(safe_format_text("processor.memory_query_failed", {"error": str(e)}))
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆库统计信息
        
        Returns:
            记忆库统计信息字典
            
        Raises:
            NoteGenError: 记忆功能未启用或获取失败
        """
        if not self.memory_manager:
            logger.warning(safe_get_text("processor.memory_not_enabled"))
            raise NoteGenError(safe_get_text("processor.memory_not_enabled"))
            
        try:
            stats = self.memory_manager.get_collection_stats()
            logger.info(safe_get_text("processor.memory_stats_retrieved"))
            return stats
        except Exception as e:
            logger.error(safe_format_text("processor.get_memory_stats_failed", {"error": str(e)}))
            raise NoteGenError(safe_format_text("processor.get_memory_stats_failed", {"error": str(e)}))
