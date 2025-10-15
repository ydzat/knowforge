'''
 * @Author: @ydzat, GitHub Copilot
 * @Date: 2025-05-16 11:30:00
 * @LastEditors: @ydzat, GitHub Copilot
 * @LastEditTime: 2025-05-16 11:30:00
 * @Description: 查询缓存工具 - 提高重复查询的性能
'''
import time
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import OrderedDict

from src.utils.logger import setup_logger


class QueryCacheError(Exception):
    """查询缓存处理过程中的异常"""
    pass


class QueryCache:
    """
    查询缓存类，用于存储和快速检索查询结果
    
    实现LRU（最近最少使用）缓存，提高重复查询效率
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        ttl: int = 3600,  # 过期时间，单位秒
        similarity_threshold: float = 0.95,  # 相似查询阈值
        enable_stats: bool = True
    ):
        """
        初始化查询缓存
        
        Args:
            capacity: 缓存容量（缓存条目数）
            ttl: 过期时间（秒）
            similarity_threshold: 相似查询匹配阈值
            enable_stats: 是否启用统计功能
        """
        self.logger = setup_logger()
        self.capacity = capacity
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self.enable_stats = enable_stats
        
        # LRU缓存
        self.cache = OrderedDict()  # {key: (result, timestamp, vector)}
        
        # 查询向量缓存（用于近似查询）
        self.vector_keys = {}  # {key: vector}
        
        # 统计数据
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "exact_hits": 0,
            "similar_hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }
        
        self.logger.info(f"初始化查询缓存，容量: {capacity}, TTL: {ttl}秒, 相似阈值: {similarity_threshold}")
        
    def _compute_key(self, query: str) -> str:
        """
        计算查询的哈希键
        
        Args:
            query: 查询文本
            
        Returns:
            哈希键
        """
        return hashlib.md5(query.encode()).hexdigest()
        
    def _compute_vector_key(self, vector: List[float]) -> str:
        """
        计算向量的哈希键
        
        Args:
            vector: 查询向量
            
        Returns:
            哈希键
        """
        # 转换为元组，使其可哈希
        vector_tuple = tuple(float(x) for x in vector)
        return hashlib.md5(str(vector_tuple).encode()).hexdigest()
        
    def _is_expired(self, timestamp: float) -> bool:
        """
        检查缓存条目是否过期
        
        Args:
            timestamp: 缓存时间戳
            
        Returns:
            是否过期
        """
        return (time.time() - timestamp) > self.ttl
        
    def _evict_if_needed(self) -> None:
        """
        如果缓存超过容量，移除最不常用的条目
        """
        if len(self.cache) >= self.capacity:
            # 移除最早的条目
            key, (_, _, vector_key) = self.cache.popitem(last=False)
            if vector_key in self.vector_keys:
                del self.vector_keys[vector_key]
                
            if self.enable_stats:
                self.stats["evictions"] += 1
                
            self.logger.debug(f"缓存条目 {key[:8]} 被驱逐")
                
    def _find_similar_query(self, vector: List[float]) -> Optional[str]:
        """
        查找相似的缓存查询
        
        Args:
            vector: 查询向量
            
        Returns:
            相似查询的键，如果没有则返回None
        """
        if not self.vector_keys:
            return None
            
        # 计算与所有缓存向量的相似度
        query_vector = np.array(vector)
        best_key = None
        best_similarity = 0
        
        for key, cached_vector in self.vector_keys.items():
            cached_vector = np.array(cached_vector)
            
            # 计算余弦相似度
            dot_product = np.dot(query_vector, cached_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_cached = np.linalg.norm(cached_vector)
            
            if norm_query == 0 or norm_cached == 0:
                continue
                
            similarity = dot_product / (norm_query * norm_cached)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_key = key
                
        return best_key
        
    def get(self, query: str, vector: Optional[List[float]] = None) -> Optional[Any]:
        """
        获取缓存的查询结果
        
        Args:
            query: 查询文本
            vector: 查询向量（用于相似查询匹配）
            
        Returns:
            缓存的结果，如果没有则返回None
        """
        if self.enable_stats:
            self.stats["total_queries"] += 1
            
        # 计算精确匹配的键
        key = self._compute_key(query)
        
        # 尝试精确匹配
        if key in self.cache:
            result, timestamp, vector_key = self.cache[key]
            
            # 检查是否过期
            if self._is_expired(timestamp):
                # 移除过期条目
                del self.cache[key]
                
                if vector_key in self.vector_keys:
                    del self.vector_keys[vector_key]
                    
                if self.enable_stats:
                    self.stats["expirations"] += 1
                    self.stats["misses"] += 1
                
                self.logger.debug(f"缓存条目 {key[:8]} 已过期")
                return None
                
            # 移至最近使用
            self.cache.move_to_end(key)
            
            if self.enable_stats:
                self.stats["cache_hits"] += 1
                self.stats["exact_hits"] += 1
                
            self.logger.debug(f"缓存命中: {key[:8]}")
            return result
            
        # 尝试相似匹配
        if vector is not None:
            vector_key = self._compute_vector_key(vector)
            similar_vector_key = self._find_similar_query(vector)
            
            if similar_vector_key:
                # 遍历缓存找到对应条目
                for cache_key, (result, timestamp, cached_vector_key) in list(self.cache.items()):
                    if cached_vector_key == similar_vector_key:
                        # 检查是否过期
                        if self._is_expired(timestamp):
                            # 移除过期条目
                            del self.cache[cache_key]
                            
                            if cached_vector_key in self.vector_keys:
                                del self.vector_keys[cached_vector_key]
                                
                            if self.enable_stats:
                                self.stats["expirations"] += 1
                            
                            continue
                            
                        # 移至最近使用
                        self.cache.move_to_end(cache_key)
                        
                        if self.enable_stats:
                            self.stats["cache_hits"] += 1
                            self.stats["similar_hits"] += 1
                            
                        self.logger.debug(f"相似缓存命中: {cache_key[:8]}")
                        return result
        
        # 缓存未命中
        if self.enable_stats:
            self.stats["misses"] += 1
            
        return None
        
    def put(self, query: str, result: Any, vector: Optional[List[float]] = None) -> None:
        """
        将结果放入缓存
        
        Args:
            query: 查询文本
            result: 查询结果
            vector: 查询向量（用于相似查询匹配）
        """
        # 如果缓存已满，移除最不常用的条目
        self._evict_if_needed()
        
        # 计算键
        key = self._compute_key(query)
        
        # 计算向量键
        vector_key = None
        if vector is not None:
            vector_key = self._compute_vector_key(vector)
            self.vector_keys[vector_key] = vector
            
        # 添加到缓存
        self.cache[key] = (result, time.time(), vector_key)
        
        # 移至最近使用
        self.cache.move_to_end(key)
        
        self.logger.debug(f"添加到缓存: {key[:8]}")
        
    def clear(self) -> None:
        """
        清空缓存
        """
        self.cache.clear()
        self.vector_keys.clear()
        
        if self.enable_stats:
            old_total = self.stats["total_queries"]
            self.stats = {
                "total_queries": old_total,
                "cache_hits": 0,
                "exact_hits": 0,
                "similar_hits": 0,
                "misses": 0,
                "evictions": 0,
                "expirations": 0,
            }
            
        self.logger.info("缓存已清空")
        
    def remove_expired(self) -> int:
        """
        移除所有过期的缓存条目
        
        Returns:
            移除的条目数量
        """
        removed_count = 0
        current_time = time.time()
        
        # 遍历并移除过期条目
        for key, (_, timestamp, vector_key) in list(self.cache.items()):
            if current_time - timestamp > self.ttl:
                del self.cache[key]
                
                if vector_key in self.vector_keys:
                    del self.vector_keys[vector_key]
                    
                removed_count += 1
                
        if self.enable_stats and removed_count > 0:
            self.stats["expirations"] += removed_count
            
        if removed_count > 0:
            self.logger.info(f"移除了 {removed_count} 个过期的缓存条目")
            
        return removed_count
        
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        if not self.enable_stats:
            return {"stats_enabled": False}
            
        stats = self.stats.copy()
        
        # 计算命中率
        total_requests = max(1, stats["total_queries"])
        stats["hit_ratio"] = stats["cache_hits"] / total_requests
        stats["exact_hit_ratio"] = stats["exact_hits"] / total_requests
        stats["similar_hit_ratio"] = stats["similar_hits"] / total_requests
        stats["miss_ratio"] = stats["misses"] / total_requests
        
        # 添加当前状态
        stats["current_size"] = len(self.cache)
        stats["capacity"] = self.capacity
        stats["usage_ratio"] = len(self.cache) / max(1, self.capacity)
        
        return stats
        
    def __len__(self) -> int:
        """
        获取缓存中的条目数量
        
        Returns:
            条目数量
        """
        return len(self.cache)
        
    def __contains__(self, query: str) -> bool:
        """
        检查查询是否在缓存中
        
        Args:
            query: 查询文本
            
        Returns:
            是否在缓存中
        """
        key = self._compute_key(query)
        return key in self.cache
        
    def get_with_compute(self, query: str, vector: Optional[List[float]] = None, 
                       compute_func: Callable[[], Any] = None) -> Any:
        """
        获取缓存结果，如果未命中则计算并缓存
        
        Args:
            query: 查询文本
            vector: 查询向量
            compute_func: 计算结果的函数
            
        Returns:
            缓存结果或计算结果
        """
        # 尝试从缓存获取
        result = self.get(query, vector)
        
        # 如果未命中且提供了计算函数，则计算结果并缓存
        if result is None and compute_func is not None:
            result = compute_func()
            self.put(query, result, vector)
            
        return result
