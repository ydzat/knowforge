'''
 * @Author: @ydzat, GitHub Copilot
 * @Date: 2025-05-17 11:15:00
 * @LastEditors: @ydzat, GitHub Copilot
 * @LastEditTime: 2025-05-17 11:15:00
 * @Description: AdvancedMemoryManager增强集成方案 - 集成VectorIndex和QueryCache
'''
import os
import time
import tempfile
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# 导入现有组件
from src.note_generator.advanced_memory_manager import AdvancedMemoryManager
from src.note_generator.vector_index import VectorIndex
from src.note_generator.query_cache import QueryCache
from src.note_generator.embedder import Embedder

class AdvancedMemoryManagerIntegration:
    """
    高级记忆管理系统集成方案
    
    此类演示了如何将VectorIndex和QueryCache集成到AdvancedMemoryManager中
    """
    
    def __init__(self, memory_manager: AdvancedMemoryManager = None):
        """
        初始化集成管理器
        
        Args:
            memory_manager: 现有的高级记忆管理器，如果为None则创建新实例
        """
        self.memory_manager = memory_manager or self._create_default_manager()
        
        # 初始化向量索引
        self._init_vector_index()
        
        # 初始化查询缓存
        self._init_query_cache()
        
        # 集成功能到原始方法中
        self._integrate_methods()
    
    def _create_default_manager(self) -> AdvancedMemoryManager:
        """创建默认的高级记忆管理器"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            chroma_db_path = os.path.join(tmp_dir, "default_memory")
            
            # 默认配置
            config = {
                "memory": {
                    "vector_index": {
                        "enabled": True,
                        "type": "hybrid",
                        "threshold": 1000
                    },
                    "query_cache": {
                        "enabled": True,
                        "capacity": 500,
                        "ttl": 3600
                    }
                }
            }
            
            # 创建嵌入器和记忆管理器
            embedder = Embedder()
            return AdvancedMemoryManager(chroma_db_path=chroma_db_path, embedder=embedder, config=config)
    
    def _init_vector_index(self) -> None:
        """初始化向量索引"""
        # 获取配置
        config = self.memory_manager.config.get("memory", {}).get("vector_index", {})
        enabled = config.get("enabled", True)
        
        if not enabled:
            return
            
        # 获取索引参数
        index_type = config.get("type", "hybrid")
        vector_dim = self.memory_manager.embedder.vector_size
        max_elements = config.get("max_elements", 100000)
        
        # 创建索引路径
        if hasattr(self.memory_manager.long_term_memory, 'db_path'):
            index_path = os.path.join(os.path.dirname(self.memory_manager.long_term_memory.db_path), "vector_index.pkl")
        else:
            index_path = None
            
        # 创建向量索引
        self.memory_manager.vector_index = VectorIndex(
            index_type=index_type,
            vector_dim=vector_dim,
            max_elements=max_elements,
            index_path=index_path,
            config=config
        )
        
        # 同步现有记忆到索引
        self._sync_memory_to_index()
    
    def _init_query_cache(self) -> None:
        """初始化查询缓存"""
        # 获取配置
        config = self.memory_manager.config.get("memory", {}).get("query_cache", {})
        enabled = config.get("enabled", True)
        
        if not enabled:
            return
            
        # 获取缓存参数
        capacity = config.get("capacity", 1000)
        ttl = config.get("ttl", 3600)  # 默认1小时
        similarity_threshold = config.get("similarity_threshold", 0.95)
        
        # 创建查询缓存
        self.memory_manager.query_cache = QueryCache(
            capacity=capacity,
            ttl=ttl,
            similarity_threshold=similarity_threshold,
            enable_stats=True
        )
    
    def _sync_memory_to_index(self) -> None:
        """同步现有记忆数据到向量索引"""
        # 如果存在长期记忆管理器，从中加载数据
        if hasattr(self.memory_manager, 'ltm') and hasattr(self.memory_manager.ltm, 'collection'):
            try:
                # 获取所有记忆项
                collection = self.memory_manager.ltm.collection
                result = collection.get(include=["documents", "embeddings", "ids"])
                
                if result["ids"]:
                    # 同步到向量索引
                    ids = result["ids"]
                    
                    # 如果有嵌入向量，直接使用
                    if "embeddings" in result and result["embeddings"]:
                        vectors = result["embeddings"]
                    # 否则重新计算嵌入
                    elif "documents" in result and result["documents"]:
                        vectors = [self.memory_manager.embedder.embed_single(doc) 
                                  for doc in result["documents"]]
                    else:
                        return
                        
                    # 批量添加到索引
                    self.memory_manager.vector_index.batch_add(ids, vectors)
            except Exception as e:
                print(f"同步记忆到向量索引失败: {str(e)}")
    
    def _integrate_methods(self) -> None:
        """集成增强方法到原始记忆管理器中"""
        # 保存原始方法引用
        self.memory_manager._original_retrieve = self.memory_manager.retrieve
        self.memory_manager._original_add_knowledge = self.memory_manager.add_knowledge
        
        # 替换为增强方法
        self.memory_manager.retrieve = self._enhanced_retrieve
        self.memory_manager.add_knowledge = self._enhanced_add_knowledge
        
        # 添加新方法
        self.memory_manager.batch_retrieve = self._batch_retrieve
        self.memory_manager.get_cache_stats = self._get_cache_stats
    
    def _enhanced_retrieve(self, query_text: str, top_k: int = None, threshold: float = None,
                          retrieval_mode: str = None) -> List[Dict[str, Any]]:
        """
        增强版检索方法，集成了查询缓存和向量索引
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值
            retrieval_mode: 检索模式
            
        Returns:
            检索结果列表
        """
        # 生成缓存键
        cache_key = f"{query_text}|{top_k}|{threshold}|{retrieval_mode}"
        
        # 如果启用缓存，尝试从缓存获取
        if hasattr(self.memory_manager, 'query_cache'):
            # 计算查询向量用于相似匹配
            query_vector = self.memory_manager.embedder.embed_single(query_text)
            
            # 尝试从缓存获取
            cached_result = self.memory_manager.query_cache.get(cache_key, query_vector)
            if cached_result is not None:
                return cached_result
        
        # 如果缓存未命中或未启用缓存，执行检索
        result = self._perform_retrieval(query_text, top_k, threshold, retrieval_mode)
        
        # 如果启用缓存，添加到缓存
        if hasattr(self.memory_manager, 'query_cache') and result:
            query_vector = self.memory_manager.embedder.embed_single(query_text)
            self.memory_manager.query_cache.put(cache_key, result, query_vector)
        
        return result
    
    def _perform_retrieval(self, query_text: str, top_k: int = None, 
                          threshold: float = None, retrieval_mode: str = None) -> List[Dict[str, Any]]:
        """
        执行实际的检索操作
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值
            retrieval_mode: 检索模式
            
        Returns:
            检索结果列表
        """
        # 如果启用向量索引，使用索引检索
        if hasattr(self.memory_manager, 'vector_index') and len(self.memory_manager.vector_index) > 0:
            # 计算查询向量
            query_vector = self.memory_manager.embedder.embed_single(query_text)
            
            # 使用向量索引查询
            top_k_actual = top_k or 5
            results = self.memory_manager.vector_index.query(query_vector, top_k=top_k_actual * 2)
            
            # 获取ID列表
            result_ids = [id for id, _ in results]
            
            # 从长期记忆获取完整信息
            if hasattr(self.memory_manager, 'ltm') and hasattr(self.memory_manager.ltm, 'collection'):
                collection_results = self.memory_manager.ltm.collection.get(
                    ids=result_ids,
                    include=["documents", "metadatas"]
                )
                
                # 格式化结果
                formatted_results = []
                for i, (id, similarity) in enumerate(results):
                    if i < len(collection_results["ids"]):
                        idx = collection_results["ids"].index(id)
                        formatted_results.append({
                            "id": id,
                            "text": collection_results["documents"][idx],
                            "metadata": collection_results["metadatas"][idx],
                            "similarity": similarity
                        })
                
                # 应用阈值过滤
                if threshold:
                    formatted_results = [r for r in formatted_results if r["similarity"] >= threshold]
                
                # 限制结果数量
                return formatted_results[:top_k_actual]
            
            # 如果无法获取完整信息，回退到原始方法
        
        # 使用原始检索方法
        return self.memory_manager._original_retrieve(query_text, top_k, threshold, retrieval_mode)
    
    def _enhanced_add_knowledge(self, segments: List[str], metadata: List[Dict[str, Any]] = None) -> List[str]:
        """
        增强版添加知识方法，同步到向量索引
        
        Args:
            segments: 文本片段列表
            metadata: 元数据列表
            
        Returns:
            添加的知识ID列表
        """
        # 使用原始方法添加知识
        ids = self.memory_manager._original_add_knowledge(segments, metadata)
        
        # 如果启用了向量索引，同步到索引
        if hasattr(self.memory_manager, 'vector_index') and ids:
            # 计算向量
            vectors = [self.memory_manager.embedder.embed_single(segment) for segment in segments]
            
            # 批量添加到索引
            self.memory_manager.vector_index.batch_add(ids, vectors)
        
        # 添加新知识后清除相关缓存
        if hasattr(self.memory_manager, 'query_cache'):
            self.memory_manager.query_cache.clear()
        
        return ids
    
    def _batch_retrieve(self, queries: List[str], top_k: int = None, 
                       threshold: float = None) -> List[List[Dict[str, Any]]]:
        """
        批量检索知识
        
        Args:
            queries: 查询文本列表
            top_k: 每个查询返回的结果数量
            threshold: 相似度阈值
            
        Returns:
            每个查询的检索结果列表
        """
        results = []
        for query in queries:
            result = self.memory_manager.retrieve(query, top_k, threshold)
            results.append(result)
        return results
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        if hasattr(self.memory_manager, 'query_cache'):
            return self.memory_manager.query_cache.get_stats()
        return {"enabled": False}


# 示例使用方法
def integrate_advanced_features(memory_manager: AdvancedMemoryManager) -> AdvancedMemoryManager:
    """
    集成高级功能到现有记忆管理器
    
    Args:
        memory_manager: 现有的高级记忆管理器
        
    Returns:
        增强后的记忆管理器
    """
    integration = AdvancedMemoryManagerIntegration(memory_manager)
    return memory_manager  # 已经在原位增强
