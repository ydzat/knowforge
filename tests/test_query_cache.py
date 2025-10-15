'''
 * @Author: @ydzat, GitHub Copilot
 * @Date: 2025-05-17 10:10:00
 * @LastEditors: @ydzat, GitHub Copilot
 * @LastEditTime: 2025-05-17 10:10:00
 * @Description: 查询缓存单元测试
'''
import time
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.note_generator.query_cache import QueryCache, QueryCacheError

class TestQueryCache:
    """测试查询缓存功能"""
    
    def test_init(self):
        """测试初始化"""
        cache = QueryCache(capacity=100, ttl=1800, similarity_threshold=0.92)
        assert cache.capacity == 100
        assert cache.ttl == 1800
        assert cache.similarity_threshold == 0.92
        assert len(cache.cache) == 0
        
    def test_exact_match(self):
        """测试精确匹配缓存"""
        cache = QueryCache(capacity=10)
        
        # 添加缓存项
        query = "测试查询文本"
        result = {"data": [1, 2, 3], "metadata": {"source": "test"}}
        cache.put(query, result)
        
        # 验证缓存添加成功
        assert len(cache) == 1
        
        # 测试缓存命中
        cached_result = cache.get(query)
        assert cached_result == result
        
        # 验证统计信息
        stats = cache.get_stats()
        assert stats["total_queries"] == 1
        assert stats["cache_hits"] == 1
        assert stats["exact_hits"] == 1
        assert stats["misses"] == 0
        
        # 测试缓存未命中
        missed_result = cache.get("不存在的查询")
        assert missed_result is None
        
        # 验证统计信息更新
        stats = cache.get_stats()
        assert stats["total_queries"] == 2
        assert stats["cache_hits"] == 1
        assert stats["misses"] == 1
        
    def test_similar_match(self):
        """测试相似匹配缓存"""
        cache = QueryCache(capacity=10, similarity_threshold=0.9)
        
        # 添加缓存项
        query = "机器学习基础知识"
        result = {"data": ["机器学习是人工智能的一个分支"]}
        vector = [0.1, 0.2, 0.3, 0.4]
        cache.put(query, result, vector)
        
        # 测试相似查询
        similar_query = "机器学习的基础知识介绍"
        similar_vector = [0.11, 0.19, 0.31, 0.41]  # 稍有不同但相似
        
        # 无向量时应该未命中
        missed_result = cache.get(similar_query)
        assert missed_result is None
        
        # 有向量时应该命中
        cached_result = cache.get(similar_query, similar_vector)
        assert cached_result == result
        
        # 验证统计信息
        stats = cache.get_stats()
        assert stats["similar_hits"] == 1
        assert stats["exact_hits"] == 0
        
    def test_ttl_expiration(self):
        """测试缓存过期"""
        cache = QueryCache(capacity=10, ttl=0.1)  # 设置较短的过期时间
        
        # 添加缓存项
        query = "短期缓存测试"
        result = {"data": "这是一个会很快过期的结果"}
        cache.put(query, result)
        
        # 立即获取应该命中
        assert cache.get(query) == result
        
        # 等待过期
        time.sleep(0.2)
        
        # 再次获取应该过期
        assert cache.get(query) is None
        
        # 验证统计信息
        stats = cache.get_stats()
        assert stats["expirations"] == 1
        
    def test_eviction(self):
        """测试缓存淘汰"""
        cache = QueryCache(capacity=2)  # 设置小容量以测试淘汰
        
        # 添加两个缓存项
        cache.put("query1", "result1")
        cache.put("query2", "result2")
        
        # 验证缓存命中
        assert cache.get("query1") == "result1"
        assert cache.get("query2") == "result2"
        
        # 添加第三个缓存项，应该淘汰最旧的项（query1）
        cache.put("query3", "result3")
        
        # 验证淘汰效果
        assert cache.get("query1") is None  # 已被淘汰
        assert cache.get("query2") == "result2"  # 仍在缓存中
        assert cache.get("query3") == "result3"  # 新添加的项
        
        # 验证统计信息
        stats = cache.get_stats()
        assert stats["evictions"] == 1
        
    def test_lru_behavior(self):
        """测试LRU行为"""
        cache = QueryCache(capacity=2)
        
        # 添加两个缓存项
        cache.put("query1", "result1")
        cache.put("query2", "result2")
        
        # 访问query1，使其变为最近使用
        assert cache.get("query1") == "result1"
        
        # 添加第三个缓存项，应该淘汰最不常用的（query2）
        cache.put("query3", "result3")
        
        # 验证淘汰效果
        assert cache.get("query1") == "result1"  # 保留因为最近使用
        assert cache.get("query2") is None  # 已被淘汰
        assert cache.get("query3") == "result3"  # 新添加的项
        
    def test_clear(self):
        """测试清空缓存"""
        cache = QueryCache(capacity=10)
        
        # 添加缓存项
        for i in range(5):
            cache.put(f"query{i}", f"result{i}")
        
        assert len(cache) == 5
        
        # 清空缓存
        cache.clear()
        
        # 验证清空效果
        assert len(cache) == 0
        for i in range(5):
            assert cache.get(f"query{i}") is None
            
        # 统计信息应该保留总查询数但重置其他
        stats = cache.get_stats()
        assert stats["total_queries"] == 5
        assert stats["cache_hits"] == 0
        assert stats["misses"] == 5
        
    def test_remove_expired(self):
        """测试移除过期项"""
        cache = QueryCache(capacity=10, ttl=0.1)
        
        # 添加缓存项
        for i in range(5):
            cache.put(f"query{i}", f"result{i}")
        
        assert len(cache) == 5
        
        # 等待过期
        time.sleep(0.2)
        
        # 移除过期项
        removed = cache.remove_expired()
        
        # 验证结果
        assert removed == 5
        assert len(cache) == 0
        
    def test_get_with_compute(self):
        """测试带计算功能的获取"""
        cache = QueryCache(capacity=10)
        
        # 准备计算函数
        def compute_result():
            return "计算生成的结果"
        
        # 第一次调用应该计算并缓存
        result1 = cache.get_with_compute("计算查询", compute_func=compute_result)
        assert result1 == "计算生成的结果"
        
        # 第二次调用应该直接返回缓存
        result2 = cache.get_with_compute("计算查询", compute_func=lambda: "不应该调用的结果")
        assert result2 == "计算生成的结果"
        
        # 验证只计算了一次
        assert cache.get_stats()["cache_hits"] == 1
        assert cache.get_stats()["misses"] == 1
