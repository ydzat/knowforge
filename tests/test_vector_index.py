'''
 * @Author: @ydzat, GitHub Copilot
 * @Date: 2025-05-17 10:00:00
 * @LastEditors: @ydzat, GitHub Copilot
 * @LastEditTime: 2025-05-17 10:00:00
 * @Description: 向量索引单元测试
'''
import os
import json
import pytest
import tempfile
import time
import uuid
import numpy as np
from unittest.mock import patch, MagicMock

from src.note_generator.vector_index import VectorIndex, VectorIndexError

class TestVectorIndex:
    """测试向量索引功能"""
    
    def test_init(self):
        """测试初始化"""
        index = VectorIndex(index_type="flat", vector_dim=128, max_elements=1000)
        assert index.index_type == "flat"
        assert index.vector_dim == 128
        assert index.max_elements == 1000
        assert len(index.vectors) == 0
    
    def test_add_single_vector(self):
        """测试添加单个向量"""
        index = VectorIndex(index_type="flat")
        
        # 添加向量
        vector = [0.1] * 384  # 默认维度
        index.add("test1", vector)
        
        # 验证添加成功
        assert len(index.vectors) == 1
        assert "test1" in index.id_to_index
        assert index.id_to_index["test1"] == 0
        
        # 测试更新相同ID的向量
        new_vector = [0.2] * 384
        index.add("test1", new_vector)
        
        # 验证更新成功，而非添加新项
        assert len(index.vectors) == 1
        assert index.vectors[0] == new_vector
    
    def test_batch_add(self):
        """测试批量添加向量"""
        index = VectorIndex(index_type="flat")
        
        # 准备测试数据
        ids = ["test1", "test2", "test3"]
        vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        
        # 批量添加
        count = index.batch_add(ids, vectors)
        
        # 验证添加成功
        assert count == 3
        assert len(index.vectors) == 3
        assert "test2" in index.id_to_index
        
        # 测试部分更新
        new_ids = ["test2", "test4"]
        new_vectors = [[0.5] * 384, [0.6] * 384]
        
        count = index.batch_add(new_ids, new_vectors)
        
        # 验证更新和新增成功
        assert count == 2
        assert len(index.vectors) == 4  # 一个更新，一个新增
        assert index.vectors[index.id_to_index["test2"]] == new_vectors[0]
    
    def test_remove(self):
        """测试删除向量"""
        index = VectorIndex(index_type="flat")
        
        # 添加向量
        ids = ["test1", "test2", "test3"]
        vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        index.batch_add(ids, vectors)
        
        # 删除向量
        success = index.remove("test2")
        
        # 验证删除成功
        assert success
        assert "test2" in index.id_to_index  # ID依然存在
        assert index.id_to_index["test2"] in index.deleted_indices  # 但标记为已删除
        
        # 尝试删除不存在的向量
        success = index.remove("non_existent")
        assert not success
        
        # 验证查询不会返回已删除的向量
        results = index.query([0.2] * 384, top_k=3)
        result_ids = [id for id, _ in results]
        assert "test2" not in result_ids
    
    def test_query(self):
        """测试向量查询"""
        index = VectorIndex(index_type="flat")
        
        # 添加向量
        ids = ["test1", "test2", "test3"]
        vectors = [
            [1.0, 0.0, 0.0, 0.0] + [0.0] * 380,
            [0.0, 1.0, 0.0, 0.0] + [0.0] * 380,
            [0.0, 0.0, 1.0, 0.0] + [0.0] * 380
        ]
        index.batch_add(ids, vectors)
        
        # 查询最相似的向量
        query_vector = [0.9, 0.1, 0.0, 0.0] + [0.0] * 380
        results = index.query(query_vector, top_k=2)
        
        # 验证结果
        assert len(results) == 2
        # 第一个结果应该是test1（最相似）
        assert results[0][0] == "test1"
        # 第二个结果应该是test2
        assert results[1][0] == "test2"
        
        # 验证相似度分数
        assert results[0][1] > 0.9  # test1相似度应该很高
        assert results[1][1] < 0.5  # test2相似度应该较低
        
    def test_batch_query(self):
        """测试批量查询"""
        index = VectorIndex(index_type="flat")
        
        # 添加向量
        ids = ["test1", "test2", "test3"]
        vectors = [
            [1.0, 0.0, 0.0, 0.0] + [0.0] * 380,
            [0.0, 1.0, 0.0, 0.0] + [0.0] * 380,
            [0.0, 0.0, 1.0, 0.0] + [0.0] * 380
        ]
        index.batch_add(ids, vectors)
        
        # 准备多个查询
        query_vectors = [
            [0.9, 0.1, 0.0, 0.0] + [0.0] * 380,
            [0.1, 0.9, 0.0, 0.0] + [0.0] * 380,
        ]
        
        # 执行批量查询
        results = index.batch_query(query_vectors, top_k=2)
        
        # 验证结果
        assert len(results) == 2
        assert results[0][0][0] == "test1"  # 第一个查询最相似的是test1
        assert results[1][0][0] == "test2"  # 第二个查询最相似的是test2
    
    def test_save_load(self):
        """测试索引的保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建索引并添加数据
            index_path = os.path.join(tmpdir, "test_index.pkl")
            index = VectorIndex(index_type="flat", index_path=index_path)
            
            ids = ["test1", "test2", "test3"]
            vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
            index.batch_add(ids, vectors)
            
            # 保存索引
            saved_path = index.save()
            assert saved_path == index_path
            assert os.path.exists(index_path)
            
            # 加载索引到新实例
            new_index = VectorIndex(index_type="unknown", vector_dim=100)
            loaded = new_index.load(index_path)
            
            # 验证加载成功并恢复了原始配置
            assert loaded
            assert new_index.index_type == "flat"
            assert new_index.vector_dim == 384
            assert len(new_index.vectors) == 3
            
            # 测试使用加载的索引进行查询
            results = new_index.query([0.15] * 384, top_k=1)
            assert results[0][0] == "test1" or results[0][0] == "test2"
    
    def test_hybrid_index(self):
        """测试混合索引策略"""
        # 创建混合索引，设置低阈值以便测试切换
        index = VectorIndex(index_type="hybrid", config={"hybrid_threshold": 3})
        
        # 添加少量向量，应该使用flat索引
        for i in range(3):
            index.add(f"test{i}", [float(i)/10] * 384)
        
        # 查询
        results1 = index.query([0.1] * 384, top_k=2)
        
        # 添加更多向量，超过阈值
        for i in range(3, 6):
            index.add(f"test{i}", [float(i)/10] * 384)
        
        # 再次查询，此时应该切换到HNSW索引
        results2 = index.query([0.1] * 384, top_k=2)
        
        # 验证两次查询都返回结果
        assert len(results1) > 0
        assert len(results2) > 0
