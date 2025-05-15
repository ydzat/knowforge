"""
向量嵌入管理模块，负责知识检索和相似文档查找
"""
import os
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import time
from src.utils.logger import get_module_logger
from src.utils.exceptions import NoteGenError
from src.utils.config_loader import ConfigLoader
from src.note_generator.embedder import Embedder, EmbeddingError

logger = get_module_logger("embedding_manager")

class Document(NamedTuple):
    """表示检索到的文档"""
    id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]

class EmbeddingManagerError(NoteGenError):
    """向量嵌入管理过程中的异常"""
    pass

class EmbeddingManager:
    """向量嵌入管理器，用于知识检索和相似文档查找"""
    
    def __init__(self, workspace_dir: str, config: Dict[str, Any] = None):
        """
        初始化向量嵌入管理器
        
        Args:
            workspace_dir: 工作空间目录
            config: 应用配置
        """
        self.workspace_dir = workspace_dir
        self.config = config or {}
        
        # 加载向量化配置
        embedding_config = self.config.get("embedding", {})
        model_name = embedding_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        cache_dir = embedding_config.get("cache_dir", os.path.join(workspace_dir, "embeddings"))
        
        # 初始化向量化模块
        try:
            logger.info(f"初始化向量嵌入管理器，使用模型: {model_name}")
            self.embedder = Embedder(model_name=model_name, cache_dir=cache_dir)
        except EmbeddingError as e:
            logger.error(f"初始化向量嵌入管理器失败: {str(e)}")
            raise EmbeddingManagerError(f"初始化向量嵌入管理器失败: {str(e)}")
        
        # 加载记忆配置
        memory_config = self.config.get("memory", {})
        self.memory_enabled = memory_config.get("enabled", True)
        self.chroma_db_path = memory_config.get(
            "chroma_db_path", os.path.join(workspace_dir, "memory_db"))
        self.collection_name = memory_config.get("collection_name", "knowforge_memory")
        
        # 最大检索数量
        self.default_top_k = memory_config.get("top_k", 5)
        self.similarity_threshold = memory_config.get("similarity_threshold", 0.6)
        
        # 懒加载MemoryManager实例，避免不必要的初始化
        self._memory_manager = None
    
    @property
    def memory_manager(self):
        """懒加载MemoryManager实例"""
        if self._memory_manager is None and self.memory_enabled:
            try:
                from src.note_generator.memory_manager import MemoryManager
                self._memory_manager = MemoryManager(
                    chroma_db_path=self.chroma_db_path,
                    embedder=self.embedder,
                    collection_name=self.collection_name,
                    config=self.config.get("memory", {})
                )
                logger.info("成功初始化记忆管理器")
            except Exception as e:
                logger.error(f"初始化记忆管理器失败: {str(e)}")
                self._memory_manager = None
        return self._memory_manager
    
    def search_similar_content(self, query_text: str, top_k: int = None) -> List[Document]:
        """
        搜索与查询文本相似的内容
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量，默认使用配置中的值
            
        Returns:
            包含相似文档信息的Document对象列表
        """
        if not query_text or query_text.strip() == "":
            logger.warning("查询文本为空")
            return []
        
        # 确定返回数量
        if top_k is None:
            top_k = self.default_top_k
        
        # 使用记忆管理器执行查询
        if self.memory_manager is not None:
            try:
                logger.info(f"从记忆库中查询相似内容: top_k={top_k}")
                results = self.memory_manager.query_similar(
                    query_text=query_text,
                    top_k=top_k,
                    threshold=self.similarity_threshold,
                    include_embeddings=False
                )
                
                # 转换结果为Document对象列表
                documents = [
                    Document(
                        id=result.get("id", f"unknown_{i}"),
                        content=result.get("text", ""),
                        similarity=result.get("similarity", 0.0),
                        metadata=result.get("metadata", {})
                    )
                    for i, result in enumerate(results)
                ]
                
                logger.info(f"找到 {len(documents)} 个相关文档")
                return documents
                
            except Exception as e:
                logger.error(f"记忆库查询失败: {str(e)}")
        
        # 如果没有记忆管理器或查询失败，使用本地向量相似度计算
        try:
            logger.info("使用本地向量相似度计算")
            # 这里可以实现一个备用的本地相似度计算逻辑
            # 但需要有本地存储的文档才能比较，目前先返回空列表
            logger.warning("本地向量相似度计算暂未实现，返回空结果")
            return []
        except Exception as e:
            logger.error(f"本地向量相似度计算失败: {str(e)}")
            return []
    
    def add_to_knowledge_base(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        将文档添加到知识库
        
        Args:
            documents: 文档内容列表
            metadata: 与文档对应的元数据列表
            
        Returns:
            添加的文档ID列表
        """
        if not documents:
            logger.warning("没有文档需要添加到知识库")
            return []
            
        if self.memory_manager is not None:
            try:
                logger.info(f"向知识库添加 {len(documents)} 个文档")
                # 添加当前时间戳到元数据
                if metadata is None:
                    metadata = [{"source": "embedding_manager", "timestamp": str(time.time())} for _ in documents]
                
                # 使用记忆管理器添加文档
                ids = self.memory_manager.add_segments(documents, metadata)
                logger.info(f"成功添加 {len(ids)} 个文档到知识库")
                return ids
                
            except Exception as e:
                logger.error(f"添加文档到知识库失败: {str(e)}")
                
        return []
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息
        
        Returns:
            包含统计信息的字典
        """
        if self.memory_manager is not None:
            try:
                stats = self.memory_manager.get_collection_stats()
                return stats
            except Exception as e:
                logger.error(f"获取知识库统计信息失败: {str(e)}")
        
        return {
            "count": 0,
            "status": "not_available",
            "error": "记忆管理器未初始化或不可用"
        }
