'''
 * @Author: @ydzat
 * @Date: 2025-05-14 10:35:22
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-05-14 14:10:22
 * @Description: 向量记忆管理模块 - 负责文本向量的存储与检索
'''
import os
import uuid
import time
import json
import math
import shutil
import datetime
from typing import List, Dict, Any, Union, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import EmbeddingFunction  # 导入ChromaDB的嵌入函数类型

from src.utils.logger import setup_logger
from src.utils.exceptions import NoteGenError
from src.note_generator.embedder import Embedder

class MemoryError(NoteGenError):
    """记忆管理过程中的异常"""
    pass

# 自定义嵌入函数，封装Embedder类，符合ChromaDB的接口要求
class KnowForgeEmbeddingFunction(EmbeddingFunction):
    """
    自定义嵌入函数类，符合ChromaDB的接口要求
    """
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        符合ChromaDB EmbeddingFunction接口的调用方法
        
        Args:
            input: 文本列表
            
        Returns:
            向量列表
        """
        return self.embedder.embed_texts(input)

class MemoryManager:
    """
    向量记忆管理器，负责文本向量的存储与检索
    
    基于ChromaDB实现向量存储和语义检索功能
    """
    
    DEFAULT_COLLECTION = "knowforge_memory"
    
    # 支持的检索模式
    RETRIEVAL_MODES = {
        "simple": "仅相似度检索",
        "time_weighted": "时间加权检索",
        "context_aware": "上下文感知检索",
        "hybrid": "混合检索策略"
    }
    
    # 支持的清理策略
    CLEANUP_STRATEGIES = {
        "oldest": "删除最早添加的记忆",
        "least_used": "删除最少使用的记忆",
        "relevance": "删除最不相关的记忆"
    }
    
    def __init__(
        self, 
        chroma_db_path: str, 
        embedder: Optional[Embedder] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
        collection_name: str = DEFAULT_COLLECTION,
        config: Dict[str, Any] = None
    ):
        """
        初始化MemoryManager
        
        Args:
            chroma_db_path: ChromaDB存储路径
            embedder: 向量化工具实例，如果为None则内部创建
            embedding_model: 使用的嵌入模型名称（当embedder为None时使用）
            collection_name: 集合名称，默认为"knowforge_memory"
            config: 配置字典，包含记忆管理相关配置
        """
        self.logger = setup_logger()
        self.chroma_db_path = os.path.abspath(chroma_db_path)
        self.collection_name = collection_name
        
        # 初始化配置
        self.config = config or {}
        self.top_k = self.config.get("top_k", 5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.6)
        
        # 获取记忆检索策略配置
        retrieval_config = self.config.get("retrieval_strategy", {})
        self.retrieval_mode = retrieval_config.get("mode", "simple")
        self.time_decay_factor = retrieval_config.get("time_decay_factor", 0.1)
        self.context_window = retrieval_config.get("context_window", 1)
        self.enforce_topic_consistency = retrieval_config.get("enforce_topic_consistency", True)
        
        # 获取记忆管理策略配置
        management_config = self.config.get("management", {})
        self.max_memory_size = management_config.get("max_memory_size", 10000)
        self.cleanup_strategy = management_config.get("cleanup_strategy", "relevance")
        self.auto_backup = management_config.get("auto_backup", True)
        self.backup_interval_days = management_config.get("backup_interval_days", 7)
        
        # 获取记忆增强生成配置
        augmentation_config = self.config.get("augmentation", {})
        self.augmentation_enabled = augmentation_config.get("enabled", True)
        self.max_references = augmentation_config.get("max_references", 3)
        self.min_reference_similarity = augmentation_config.get("min_similarity", 0.75)
        self.reference_format = augmentation_config.get("reference_format", "markdown")
        
        # 确保数据库目录存在
        os.makedirs(self.chroma_db_path, exist_ok=True)
        
        # 如果没有提供embedder实例，则创建一个
        self.embedder = embedder if embedder else Embedder(embedding_model)
        
        # 记录最后一次备份时间
        self.last_backup_time = time.time()
        
        # 记录使用频率，用于least_used清理策略
        self.usage_counter = {}
        
        try:
            self.logger.info(f"初始化向量记忆管理模块，数据库路径: {self.chroma_db_path}")
            
            # 初始化ChromaDB客户端
            self.client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,  # 禁用遥测
                    allow_reset=True  # 允许重置数据库
                )
            )
            
            # 创建符合ChromaDB接口的嵌入函数
            embedding_function = KnowForgeEmbeddingFunction(self.embedder)
            
            # 获取或创建集合
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function  # 使用符合接口的嵌入函数
                )
                self.logger.info(f"已连接到现有集合: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function,  # 使用符合接口的嵌入函数
                    metadata={"description": "KnowForge记忆管理集合", "created_at": time.time()}
                )
                self.logger.info(f"已创建新的集合: {self.collection_name}")
            
            # 检查是否需要清理记忆库
            self._check_and_cleanup_memory()
            
            # 检查是否需要备份记忆库
            self._check_and_backup_memory()
            
        except Exception as e:
            self.logger.error(f"初始化ChromaDB失败: {str(e)}")
            raise MemoryError(f"初始化记忆管理器失败: {str(e)}")
    
    def add_segments(self, segments: List[str], metadata: Optional[List[Dict[str, str]]] = None) -> List[str]:
        """
        将文本片段向量化并添加到记忆库
        
        Args:
            segments: 文本片段列表
            metadata: 元数据列表，每个元素对应一个文本片段的元数据
            
        Returns:
            添加的条目IDs
            
        Raises:
            MemoryError: 添加过程中出现异常
        """
        if not segments:
            self.logger.warning("接收到空片段列表，无法添加到记忆库")
            return []
        
        # 为每个片段生成唯一ID
        ids = [str(uuid.uuid4()) for _ in range(len(segments))]
        
        # 确保metadata列表长度与segments一致
        current_time = str(time.time())
        if metadata is None:
            metadata = [{"source": "unknown", "timestamp": current_time, "access_count": "0"} for _ in range(len(segments))]
        elif len(metadata) != len(segments):
            self.logger.warning("元数据列表长度与片段列表长度不匹配，将使用默认元数据")
            metadata = [{"source": "unknown", "timestamp": current_time, "access_count": "0"} for _ in range(len(segments))]
        else:
            # 确保每个元数据都有timestamp和access_count字段
            for i, meta in enumerate(metadata):
                if "timestamp" not in meta:
                    meta["timestamp"] = current_time
                if "access_count" not in meta:
                    meta["access_count"] = "0"
        
        try:
            self.logger.info(f"开始向记忆库添加 {len(segments)} 个文本片段")
            self.collection.add(
                ids=ids,
                documents=segments,
                metadatas=metadata
            )
            
            # 如果超出最大容量，执行清理
            if self.collection.count() > self.max_memory_size:
                self._cleanup_memory()
                
            self.logger.info(f"成功添加 {len(segments)} 个文本片段到记忆库")
            return ids
        except Exception as e:
            self.logger.error(f"向记忆库添加片段失败: {str(e)}")
            raise MemoryError(f"添加片段到记忆库失败: {str(e)}")
    
    def query_similar(
        self, 
        query_text: str, 
        top_k: int = None, 
        threshold: float = None,
        include_embeddings: bool = False,
        context_texts: List[str] = None,
        retrieval_mode: str = None
    ) -> List[Dict[str, Any]]:
        """
        检索与查询文本最相似的片段
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量，默认使用配置中的值
            threshold: 相似度阈值，低于此值的结果将被过滤，默认使用配置中的值
            include_embeddings: 是否在结果中包含向量表示
            context_texts: 上下文文本列表，用于上下文感知检索
            retrieval_mode: 检索模式，默认使用配置中的值
            
        Returns:
            相似片段列表，每个元素包含文本内容、相似度分数和元数据
            
        Raises:
            MemoryError: 检索过程中出现异常
        """
        if not query_text:
            self.logger.warning("接收到空查询文本，无法检索")
            return []
        
        # 使用配置中的默认值
        if top_k is None:
            top_k = self.top_k
        if threshold is None:
            threshold = self.similarity_threshold
        if retrieval_mode is None:
            retrieval_mode = self.retrieval_mode
            
        try:
            self.logger.info(f"开始检索与查询文本相似的前 {top_k} 个片段，使用 {retrieval_mode} 检索模式")
            
            # 根据检索模式选择不同的检索策略
            if retrieval_mode == "context_aware" and context_texts:
                return self._context_aware_retrieval(query_text, context_texts, top_k, threshold, include_embeddings)
            elif retrieval_mode == "time_weighted":
                return self._time_weighted_retrieval(top_k, self.time_decay_factor, threshold, include_embeddings)
            elif retrieval_mode == "hybrid":
                return self._hybrid_retrieval(query_text, top_k, threshold, include_embeddings, context_texts)
            else:
                # 默认使用简单相似度检索
                return self._simple_similarity_retrieval(query_text, top_k, threshold, include_embeddings)
                
        except Exception as e:
            self.logger.error(f"检索相似片段失败: {str(e)}")
            raise MemoryError(f"检索相似片段失败: {str(e)}")
    
    def _simple_similarity_retrieval(
        self, 
        query_text: str, 
        top_k: int = 5, 
        threshold: float = 0.0,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        简单相似度检索 - 仅通过语义相似度检索
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数
            threshold: 相似度阈值
            include_embeddings: 是否包含向量表示
            
        Returns:
            相似片段列表
        """
        # 生成查询的嵌入向量
        query_embedding = self.embedder.embed_single(query_text)
        
        # 查询结果
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        final_results = []
        
        # 处理结果
        if result["ids"] and result["ids"][0]:
            for i, doc_id in enumerate(result["ids"][0]):
                distance = result["distances"][0][i] if "distances" in result else 0
                similarity_score = 1.0 - min(max(0.0, distance), 1.0)
                
                if similarity_score < threshold:
                    continue
                
                metadata = result["metadatas"][0][i] if "metadatas" in result and result["metadatas"][0] else {}
                
                item = {
                    "id": doc_id,
                    "text": result["documents"][0][i],
                    "similarity": similarity_score,
                    "metadata": metadata
                }
                
                if include_embeddings:
                    item["embedding"] = self.embedder.embed_single(item["text"])
                    
                final_results.append(item)
        
        self.logger.info(f"[简单相似检索]找到 {len(final_results)} 个匹配结果")
        return final_results
    
    def _time_weighted_retrieval(
        self, 
        top_k: int = 10, 
        time_decay_factor: float = 0.01,
        threshold: float = 0.0,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        时间加权检索 - 基于记忆存储时间的加权检索
        
        Args:
            top_k: 返回结果数
            time_decay_factor: 时间衰减因子
            threshold: 相似度阈值
            include_embeddings: 是否包含向量表示
            
        Returns:
            相似片段列表
        """
        # 创建一个空的查询嵌入向量以满足ChromaDB API要求
        # 这里我们使用一个模拟向量，稍后会忽略它的相似度结果，只用时间权重
        dummy_embedding = self.embedder.embed_single("dummy query for time weighted retrieval")
        
        # 获取所有记忆，使用一个模拟的查询向量满足API要求
        result = self.collection.query(
            query_embeddings=[dummy_embedding],
            n_results=top_k * 3,  # 获取更多样本进行时间排序
            include=["documents", "metadatas", "distances"]
        )
        
        final_results = []
        
        # 处理结果
        if result["ids"] and result["ids"][0]:
            candidates = []
            for i, doc_id in enumerate(result["ids"][0]):
                metadata = result["metadatas"][0][i] if "metadatas" in result and result["metadatas"][0] else {}
                
                # 获取时间戳，如果没有则使用当前时间
                timestamp = metadata.get('timestamp', datetime.datetime.now().timestamp())
                if isinstance(timestamp, str):
                    try:
                        timestamp = float(timestamp)
                    except (ValueError, TypeError):
                        timestamp = datetime.datetime.now().timestamp()
                
                # 计算时间权重: 指数衰减
                time_diff = datetime.datetime.now().timestamp() - timestamp
                time_weight = math.exp(-time_decay_factor * time_diff / (24 * 3600))  # 转换为天
                
                # 计算最终评分
                final_score = time_weight
                
                candidates.append({
                    "id": doc_id,
                    "text": result["documents"][0][i],
                    "similarity": final_score,  # 使用时间权重作为相似度
                    "metadata": metadata
                })
            
            # 按最终评分排序并取top_k
            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            candidates = candidates[:top_k]
            
            # 过滤低于阈值的结果
            filtered_candidates = [c for c in candidates if c["similarity"] >= threshold]
            
            # 添加嵌入向量(如果需要)
            if include_embeddings:
                for candidate in filtered_candidates:
                    candidate["embedding"] = self.embedder.embed_single(candidate["text"])
            
            final_results = filtered_candidates
        
        self.logger.info(f"[时间加权检索]找到 {len(final_results)} 个匹配结果")
        return final_results
    
    def _context_aware_retrieval(
        self, 
        query_text: str, 
        context_text: Optional[str] = None,
        top_k: int = 5, 
        threshold: float = 0.0,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        上下文感知检索 - 结合查询和上下文
        
        Args:
            query_text: 查询文本
            context_text: 上下文文本
            top_k: 返回结果数
            threshold: 相似度阈值
            include_embeddings: 是否包含向量表示
            
        Returns:
            相似片段列表
        """
        if not context_text:
            # 没有上下文，退化为简单检索
            return self._simple_similarity_retrieval(
                query_text, 
                top_k=top_k, 
                threshold=threshold,
                include_embeddings=include_embeddings
            )
            
        # 生成查询和上下文的向量表示
        query_embedding = self.embedder.embed_single(query_text)
        context_embedding = self.embedder.embed_single(context_text)
        
        # 组合向量 (简单平均)
        combined_embedding = [(q + c) / 2.0 for q, c in zip(query_embedding, context_embedding)]
        
        include = ["documents", "metadatas", "distances"]
            
        # 查询向量数据库
        results = self.collection.query(
            query_embeddings=[combined_embedding],
            n_results=top_k,
            include=include
        )
        
        # 处理结果
        if not results["documents"] or len(results["documents"][0]) == 0:
            self.logger.info(f"[上下文检索]未找到匹配结果")
            return []
            
        formatted_results = []
        for i in range(len(results["documents"][0])):
            similarity_score = 1.0 - min(max(0.0, float(results["distances"][0][i])), 1.0)  # ChromaDB的距离转为相似度
            
            if similarity_score < threshold:
                continue
                
            result = {
                "id": results["ids"][0][i] if "ids" in results and results["ids"][0] else f"unknown_{i}",
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if "metadatas" in results and results["metadatas"][0] else {},
                "similarity": similarity_score
            }
            
            if include_embeddings:
                result["embedding"] = self.embedder.embed_single(result["text"])
                
            formatted_results.append(result)
            
        self.logger.info(f"[上下文检索]找到 {len(formatted_results)} 个匹配结果")
        return formatted_results
    
    def _hybrid_retrieval(
        self, 
        query_text: str, 
        top_k: int = 5, 
        threshold: float = 0.0,
        include_embeddings: bool = False,
        context_texts: List[str] = None,
        keyword_weight: float = 0.2  # 降低了关键词权重
    ) -> List[Dict[str, Any]]:
        """
        混合检索 - 结合语义和关键词
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数
            threshold: 相似度阈值
            include_embeddings: 是否包含向量表示
            context_texts: 上下文文本列表，可选
            keyword_weight: 关键词权重 (0-1)
            
        Returns:
            相似片段列表
        """
        # 语义检索得分，使用更宽松的阈值来确保有足够的候选项
        semantic_threshold = max(0.0, threshold - 0.2)  # 使用比输入阈值更低的值来获取更多候选结果
        semantic_results = self._simple_similarity_retrieval(
            query_text, 
            top_k=max(top_k*3, 10),  # 获取更多结果然后重新排序
            threshold=semantic_threshold,  # 使用更宽松的阈值
            include_embeddings=include_embeddings
        )
        
        if not semantic_results:
            self.logger.info("[混合检索]语义检索未找到结果，尝试时间加权检索")
            # 如果语义检索没有结果，尝试时间加权检索
            return self._time_weighted_retrieval(
                top_k=top_k, 
                time_decay_factor=self.time_decay_factor,
                threshold=threshold,
                include_embeddings=include_embeddings
            )
        
        # 如果还有上下文，结合上下文信息
        if context_texts:
            if isinstance(context_texts, str):
                context_text = context_texts
            else:
                context_text = " ".join(context_texts)
                
            context_results = self._context_aware_retrieval(
                query_text,
                context_text,
                top_k=top_k*2,
                threshold=semantic_threshold,
                include_embeddings=False
            )
            
            # 合并两种结果，按ID去重
            id_to_result = {r["id"]: r for r in semantic_results}
            for ctx_result in context_results:
                if ctx_result["id"] not in id_to_result:
                    id_to_result[ctx_result["id"]] = ctx_result
                else:
                    # 如果两种方法都找到这条结果，取最大相似度
                    id_to_result[ctx_result["id"]]["similarity"] = max(
                        id_to_result[ctx_result["id"]]["similarity"],
                        ctx_result["similarity"]
                    )
            
            semantic_results = list(id_to_result.values())
        
        # 抽取查询中的关键词
        keywords = self._extract_keywords(query_text, max_keywords=5)  # 减少关键词数量，提高精确度
        
        # 如果没有提取到关键词或结果为空，直接返回语义结果
        if not keywords or not semantic_results:
            final_results = sorted(semantic_results, key=lambda x: x["similarity"], reverse=True)[:top_k]
            self.logger.info(f"[混合检索]找到 {len(final_results)} 个匹配结果（仅使用语义相似度）")
            return final_results
        
        # 为每个结果计算关键词匹配得分
        for result in semantic_results:
            keyword_score = self._calculate_keyword_score(result["text"], keywords)
            semantic_score = result["similarity"]
            
            # 组合得分：主要依靠语义相似度，辅以关键词匹配
            combined_score = (1 - keyword_weight) * semantic_score + keyword_weight * keyword_score
            
            # 确保组合得分不低于语义得分的0.9倍，避免关键词匹配降低太多相似度
            combined_score = max(combined_score, semantic_score * 0.9)
            
            # 更新相似度得分
            result["similarity"] = combined_score
            result["semantic_score"] = semantic_score
            result["keyword_score"] = keyword_score
        
        # 根据组合得分重新排序
        sorted_results = sorted(semantic_results, key=lambda x: x["similarity"], reverse=True)
        
        # 应用相似度阈值过滤，但如果过滤后没有结果，则忽略阈值返回前top_k个结果
        filtered_results = [r for r in sorted_results if r["similarity"] >= threshold]
        if not filtered_results and sorted_results:
            self.logger.info(f"[混合检索]所有结果低于阈值 {threshold}，返回前 {top_k} 个结果")
            filtered_results = sorted_results
        
        # 截取前 top_k 个结果
        final_results = filtered_results[:top_k]
        
        self.logger.info(f"[混合检索]找到 {len(final_results)} 个匹配结果")
        return final_results
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        从文本中提取关键词
        
        Args:
            text: 待提取关键词的文本
            max_keywords: 最大关键词数量
            
        Returns:
            关键词列表
        """
        # 简单实现：分词并过滤停用词
        words = text.lower().split()
        
        # 常见停用词
        stopwords = {"的", "了", "和", "与", "或", "在", "是", "有", "而", "也", "就", "都",
                    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                    "and", "or", "but", "if", "then", "else", "when", "at", "from", "by"}
        
        # 过滤停用词和过短的词
        filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
        
        # 计算词频
        from collections import Counter
        word_counts = Counter(filtered_words)
        
        # 返回出现频率最高的关键词
        return [word for word, _ in word_counts.most_common(max_keywords)]
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """
        计算文本与关键词的匹配得分
        
        Args:
            text: 文本内容
            keywords: 关键词列表
            
        Returns:
            匹配得分 (0-1)
        """
        # 简单实现：根据关键词在文本中的出现次数计算得分
        text_lower = text.lower()
        
        # 统计文本中出现的关键词数量
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
        
        # 计算归一化得分
        if not keywords:
            return 0.0
            
        return min(1.0, keyword_count / len(keywords))
    
    def _check_and_cleanup_memory(self):
        """检查并清理记忆库，如果超出最大容量"""
        try:
            count = self.collection.count()
            if count > self.max_memory_size:
                self.logger.info(f"记忆库条目数 ({count}) 超出最大限制 ({self.max_memory_size})，开始清理")
                self._cleanup_memory()
        except Exception as e:
            self.logger.error(f"检查记忆库大小失败: {str(e)}")
    
    def _cleanup_memory(self):
        """根据配置的清理策略清理记忆库"""
        try:
            if self.cleanup_strategy == "oldest":
                self._cleanup_by_oldest()
            elif self.cleanup_strategy == "least_used":
                self._cleanup_by_least_used()
            elif self.cleanup_strategy == "relevance":
                self._cleanup_by_relevance()
            else:
                self.logger.warning(f"未知的清理策略: {self.cleanup_strategy}，使用默认的最旧策略")
                self._cleanup_by_oldest()
        except Exception as e:
            self.logger.error(f"清理记忆库失败: {str(e)}")
    
    def _cleanup_by_oldest(self):
        """删除最早添加的记忆"""
        try:
            # 获取所有记忆
            all_memories = self.collection.get(include=["metadatas", "ids"])
            
            if not all_memories["ids"]:
                return
                
            # 获取时间戳和ID对
            timestamp_id_pairs = []
            for i, doc_id in enumerate(all_memories["ids"]):
                metadata = all_memories["metadatas"][i] if all_memories["metadatas"] else {}
                timestamp = float(metadata.get("timestamp", "0"))
                timestamp_id_pairs.append((timestamp, doc_id))
                
            # 按时间戳排序
            timestamp_id_pairs.sort()
            
            # 计算需要删除的数量
            current_count = len(timestamp_id_pairs)
            target_count = int(self.max_memory_size * 0.8)  # 清理到80%容量
            to_remove = max(0, current_count - target_count)
            
            if to_remove > 0:
                # 获取最早的N个条目的ID
                ids_to_remove = [pair[1] for pair in timestamp_id_pairs[:to_remove]]
                
                # 删除这些条目
                self.collection.delete(ids=ids_to_remove)
                self.logger.info(f"已删除 {len(ids_to_remove)} 个最早的记忆条目")
        except Exception as e:
            self.logger.error(f"按最早时间清理记忆失败: {str(e)}")
    
    def _cleanup_by_least_used(self):
        """删除最少使用的记忆"""
        try:
            # 获取所有记忆
            all_memories = self.collection.get(include=["metadatas", "ids"])
            
            if not all_memories["ids"]:
                return
                
            # 获取访问计数和ID对
            count_id_pairs = []
            for i, doc_id in enumerate(all_memories["ids"]):
                metadata = all_memories["metadatas"][i] if all_memories["metadatas"] else {}
                access_count = int(metadata.get("access_count", "0"))
                count_id_pairs.append((access_count, doc_id))
                
            # 按访问计数排序
            count_id_pairs.sort()
            
            # 计算需要删除的数量
            current_count = len(count_id_pairs)
            target_count = int(self.max_memory_size * 0.8)  # 清理到80%容量
            to_remove = max(0, current_count - target_count)
            
            if to_remove > 0:
                # 获取最少使用的N个条目的ID
                ids_to_remove = [pair[1] for pair in count_id_pairs[:to_remove]]
                
                # 删除这些条目
                self.collection.delete(ids=ids_to_remove)
                self.logger.info(f"已删除 {len(ids_to_remove)} 个最少使用的记忆条目")
        except Exception as e:
            self.logger.error(f"按使用频率清理记忆失败: {str(e)}")
    
    def _cleanup_by_relevance(self):
        """删除最不相关的记忆"""
        try:
            # 获取所有记忆
            all_memories = self.collection.get(include=["documents", "ids"])
            
            if not all_memories["ids"] or not all_memories["documents"]:
                return
                
            # 随机选择一部分文档作为查询样本
            import random
            sample_size = min(10, len(all_memories["documents"]))
            sample_indices = random.sample(range(len(all_memories["documents"])), sample_size)
            sample_docs = [all_memories["documents"][i] for i in sample_indices]
            
            # 将这些样本文档合并为一个查询
            query_text = " ".join(sample_docs)
            
            # 使用此查询检索所有文档并按相似度排序
            result = self.collection.query(
                query_texts=[query_text],
                n_results=len(all_memories["ids"]),  # 获取所有文档
                include=["ids"]
            )
            
            if not result["ids"] or not result["ids"][0]:
                return
                
            # 计算需要删除的数量
            current_count = len(result["ids"][0])
            target_count = int(self.max_memory_size * 0.8)  # 清理到80%容量
            to_remove = max(0, current_count - target_count)
            
            if to_remove > 0:
                # 获取最不相关的N个条目的ID (最后的N个结果)
                ids_to_remove = result["ids"][0][-to_remove:]
                
                # 删除这些条目
                self.collection.delete(ids=ids_to_remove)
                self.logger.info(f"已删除 {len(ids_to_remove)} 个最不相关的记忆条目")
        except Exception as e:
            self.logger.error(f"按相关性清理记忆失败: {str(e)}")
            # 如果相关性清理失败，退回到最旧策略
            self._cleanup_by_oldest()
    
    def _check_and_backup_memory(self):
        """检查并备份记忆库，如果达到备份间隔时间"""
        if not self.auto_backup:
            return
            
        current_time = time.time()
        backup_interval_seconds = self.backup_interval_days * 24 * 3600
        
        if (current_time - self.last_backup_time) >= backup_interval_seconds:
            try:
                # 创建备份目录
                backup_dir = os.path.join(os.path.dirname(self.chroma_db_path), "memory_backups")
                os.makedirs(backup_dir, exist_ok=True)
                
                # 生成备份文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(backup_dir, f"memory_backup_{timestamp}.json")
                
                # 导出记忆库
                self.export_to_json(backup_file)
                
                # 更新最后备份时间
                self.last_backup_time = current_time
                self.logger.info(f"已自动备份记忆库到: {backup_file}")
            except Exception as e:
                self.logger.error(f"自动备份记忆库失败: {str(e)}")
    
    def rebuild_memory(self, segments: List[str], metadata: Optional[List[Dict[str, str]]] = None) -> bool:
        """
        清空并重建记忆库
        
        Args:
            segments: 文本片段列表
            metadata: 元数据列表
            
        Returns:
            是否重建成功
            
        Raises:
            MemoryError: 重建过程中出现异常
        """
        try:
            self.logger.info(f"开始重建记忆库，将添加 {len(segments)} 个文本片段")
            
            # 删除现有集合
            try:
                self.client.delete_collection(self.collection_name)
                self.logger.info(f"已删除现有集合: {self.collection_name}")
            except Exception as e:
                self.logger.warning(f"删除集合时出现异常 (可能是集合不存在): {str(e)}")
            
            # 创建符合ChromaDB接口的嵌入函数
            embedding_function = KnowForgeEmbeddingFunction(self.embedder)
            
            # 重新创建集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={"description": "KnowForge记忆管理集合", "created_at": time.time()}
            )
            
            # 添加新片段
            if segments:
                self.add_segments(segments, metadata)
                
            self.logger.info("记忆库重建完成")
            return True
        
        except Exception as e:
            self.logger.error(f"重建记忆库失败: {str(e)}")
            raise MemoryError(f"重建记忆库失败: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取记忆库统计信息
        
        Returns:
            统计信息字典
            
        Raises:
            MemoryError: 获取统计信息过程中出现异常
        """
        try:
            count = self.collection.count()
            
            # 采样计算平均文本长度
            sample_size = min(count, 100)  # 最多采样100个
            if sample_size > 0:
                sample = self.collection.get(limit=sample_size)
                avg_length = sum(len(doc) for doc in sample["documents"]) / sample_size if sample["documents"] else 0
            else:
                avg_length = 0
                
            # 提取元数据统计信息
            source_counts = {}
            access_counts = []
            timestamp_range = {"min": float('inf'), "max": 0}
            
            if count > 0:
                all_metadatas = self.collection.get(include=["metadatas"])["metadatas"]
                if all_metadatas:
                    for metadata in all_metadatas:
                        # 统计来源
                        source = metadata.get("source", "unknown")
                        source_counts[source] = source_counts.get(source, 0) + 1
                        
                        # 统计访问次数
                        try:
                            access_count = int(metadata.get("access_count", "0"))
                            access_counts.append(access_count)
                        except (ValueError, TypeError):
                            pass
                            
                        # 统计时间范围
                        try:
                            timestamp = float(metadata.get("timestamp", "0"))
                            timestamp_range["min"] = min(timestamp_range["min"], timestamp)
                            timestamp_range["max"] = max(timestamp_range["max"], timestamp)
                        except (ValueError, TypeError):
                            pass
            
            # 汇总访问统计
            avg_access = sum(access_counts) / len(access_counts) if access_counts else 0
            max_access = max(access_counts) if access_counts else 0
            
            # 时间范围格式化
            import datetime
            if timestamp_range["min"] != float('inf'):
                timestamp_range["min_date"] = datetime.datetime.fromtimestamp(
                    timestamp_range["min"]
                ).strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_range["min_date"] = "N/A"
                
            if timestamp_range["max"] != 0:
                timestamp_range["max_date"] = datetime.datetime.fromtimestamp(
                    timestamp_range["max"]
                ).strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_range["max_date"] = "N/A"
                
            return {
                "count": count,
                "collection_name": self.collection_name,
                "db_path": self.chroma_db_path,
                "avg_text_length": avg_length,
                "embedding_model": self.embedder.model_name,
                "vector_size": self.embedder.vector_size,
                "sources": source_counts,
                "access_stats": {
                    "avg_access_count": avg_access,
                    "max_access_count": max_access
                },
                "time_range": timestamp_range,
                "retrieval_mode": self.retrieval_mode,
                "cleanup_strategy": self.cleanup_strategy
            }
        except Exception as e:
            self.logger.error(f"获取记忆库统计信息失败: {str(e)}")
            raise MemoryError(f"获取记忆库统计信息失败: {str(e)}")
    
    def export_to_json(self, output_path: str) -> str:
        """
        将记忆库导出为JSON文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            导出的文件路径
            
        Raises:
            MemoryError: 导出过程中出现异常
        """
        try:
            self.logger.info(f"开始导出记忆库到: {output_path}")
            
            # 获取所有内容
            all_data = self.collection.get()
            
            # 构造导出数据
            export_data = {
                "metadata": {
                    "timestamp": time.time(),
                    "collection_name": self.collection_name,
                    "count": len(all_data["ids"]) if "ids" in all_data else 0,
                    "embedding_model": self.embedder.model_name
                },
                "entries": []
            }
            
            # 添加所有条目
            if "ids" in all_data and all_data["ids"]:
                for i, doc_id in enumerate(all_data["ids"]):
                    entry = {
                        "id": doc_id,
                        "text": all_data["documents"][i] if "documents" in all_data else "",
                        "metadata": all_data["metadatas"][i] if "metadatas" in all_data and all_data["metadatas"] else {}
                    }
                    export_data["entries"].append(entry)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 写入JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"成功导出 {len(export_data['entries'])} 条记录到: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"导出记忆库失败: {str(e)}")
            raise MemoryError(f"导出记忆库失败: {str(e)}")
    
    def import_from_json(self, input_path: str, replace_existing: bool = False) -> int:
        """
        从JSON文件导入记忆库
        
        Args:
            input_path: 输入文件路径
            replace_existing: 是否替换现有内容
            
        Returns:
            导入的条目数量
            
        Raises:
            MemoryError: 导入过程中出现异常
        """
        try:
            self.logger.info(f"开始从 {input_path} 导入记忆库")
            
            # 读取JSON文件
            with open(input_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
                
            if replace_existing:
                self.logger.info("将替换现有内容")
                self.client.delete_collection(self.collection_name)
                
                # 创建符合ChromaDB接口的嵌入函数
                embedding_function = KnowForgeEmbeddingFunction(self.embedder)
                
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function,
                    metadata=import_data.get("metadata", {})
                )
            
            # 导入所有条目
            entries = import_data.get("entries", [])
            if entries:
                ids = [entry["id"] for entry in entries]
                texts = [entry["text"] for entry in entries]
                metadatas = [entry.get("metadata", {}) for entry in entries]
                
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                
            self.logger.info(f"成功导入 {len(entries)} 条记录")
            return len(entries)
            
        except Exception as e:
            self.logger.error(f"导入记忆库失败: {str(e)}")
            raise MemoryError(f"导入记忆库失败: {str(e)}")
        
    def delete_db(self) -> bool:
        """
        删除整个数据库
        
        Returns:
            是否删除成功
            
        Raises:
            MemoryError: 删除过程中出现异常
        """
        try:
            self.logger.warning(f"准备删除整个向量数据库: {self.chroma_db_path}")
            
            # 关闭客户端连接
            del self.collection
            del self.client
            
            # 强制清理内存
            import gc
            gc.collect()
            
            # 删除数据库目录
            if os.path.exists(self.chroma_db_path):
                shutil.rmtree(self.chroma_db_path)
                self.logger.info(f"已成功删除向量数据库: {self.chroma_db_path}")
                return True
            else:
                self.logger.warning(f"向量数据库路径不存在: {self.chroma_db_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"删除向量数据库失败: {str(e)}")
            raise MemoryError(f"删除向量数据库失败: {str(e)}")
            
    def __len__(self) -> int:
        """
        获取记忆库中的条目数量
        
        Returns:
            条目数量
        """
        try:
            return self.collection.count()
        except Exception:
            return 0

    def _update_metadata_on_access(self, item_ids: List[str]) -> None:
        """
        更新被访问记忆项的元数据信息（访问计数和最后访问时间）
        
        Args:
            item_ids: 被访问记忆项的ID列表
            
        Raises:
            MemoryError: 更新元数据过程中出现异常
        """
        if not item_ids:
            return
            
        try:
            self.logger.debug(f"更新 {len(item_ids)} 个记忆项的访问统计")
            
            # 获取现有元数据
            results = self.collection.get(
                ids=item_ids,
                include=["metadatas"]
            )
            
            if not results or not results['metadatas']:
                self.logger.warning(f"未找到指定ID的记忆项，无法更新访问统计: {item_ids}")
                return
                
            metadatas = results['metadatas']
            current_time = str(time.time())
            updated_metadatas = []
            
            # 更新每个记忆项的访问计数和最后访问时间
            for i, metadata in enumerate(metadatas):
                item_id = item_ids[i] if i < len(item_ids) else None
                if item_id and metadata:
                    # 更新内存中的使用计数器
                    self.usage_counter[item_id] = self.usage_counter.get(item_id, 0) + 1
                    
                    # 更新访问计数（字符串格式）
                    access_count = int(metadata.get("access_count", "0")) + 1
                    metadata["access_count"] = str(access_count)
                    
                    # 更新最后访问时间
                    metadata["last_accessed"] = current_time
                    updated_metadatas.append(metadata)
            
            # 批量更新元数据
            if updated_metadatas:
                self.collection.update(
                    ids=item_ids[:len(updated_metadatas)],
                    metadatas=updated_metadatas
                )
                self.logger.debug(f"成功更新 {len(updated_metadatas)} 个记忆项的访问统计")
                
        except Exception as e:
            self.logger.error(f"更新记忆项访问统计失败: {str(e)}")
            raise MemoryError(f"更新记忆项访问统计失败: {str(e)}")

    def get_all_segments(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """
        获取所有记忆片段
        
        Args:
            include_embeddings: 是否包含向量表示
            
        Returns:
            所有记忆片段数据
        """
        try:
            # 确定包含哪些内容
            includes = ["documents", "metadatas"]
            if include_embeddings:
                includes.append("embeddings")
                
            # 直接从集合获取所有条目
            results = self.collection.get(
                include=includes
            )
            
            return results
        except Exception as e:
            self.logger.error(f"获取所有记忆片段失败: {str(e)}")
            # 返回空结构而不是引发异常
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": [] if include_embeddings else None}
    
    def get_by_ids(self, ids: Union[str, List[str]], include_embeddings: bool = False) -> Dict[str, Any]:
        """
        根据ID获取记忆片段
        
        Args:
            ids: 记忆项ID或ID列表
            include_embeddings: 是否包含向量表示
            
        Returns:
            记忆片段数据
        """
        try:
            # 确保ids是列表
            id_list = ids if isinstance(ids, list) else [ids]
            
            # 确定包含哪些内容
            includes = ["documents", "metadatas"]
            if include_embeddings:
                includes.append("embeddings")
                
            # 获取记忆片段
            results = self.collection.get(
                ids=id_list,
                include=includes
            )
            
            return results
        except Exception as e:
            self.logger.error(f"根据ID获取记忆片段失败: {str(e)}")
            # 返回空结构而不是引发异常
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": [] if include_embeddings else None}
    
    def update_metadata(self, ids: List[str], metadatas: List[Dict[str, Any]]) -> bool:
        """
        更新记忆项的元数据
        
        Args:
            ids: 待更新记忆项的ID列表
            metadatas: 对应的元数据列表
            
        Returns:
            是否成功更新
            
        Raises:
            MemoryError: 更新过程中出现异常
        """
        try:
            self.collection.update(
                ids=ids,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            self.logger.error(f"更新记忆项元数据失败: {str(e)}")
            raise MemoryError(f"更新记忆项元数据失败: {str(e)}")
    
    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """
        更新指定ID记忆项的元数据
        
        Args:
            doc_id: 记忆项ID
            metadata: 更新后的元数据
            
        Returns:
            更新是否成功
        """
        try:
            # 确保元数据是字典类型
            if not isinstance(metadata, dict):
                self.logger.warning(f"元数据必须是字典类型，收到: {type(metadata)}")
                return False
                
            # 更新元数据
            self.collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )
            self.logger.debug(f"成功更新记忆项 {doc_id[:8]} 的元数据")
            return True
        except Exception as e:
            self.logger.error(f"更新记忆项元数据失败: {str(e)}")
            return False