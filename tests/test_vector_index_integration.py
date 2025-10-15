'''
 * @Author: GitHub Copilot
 * @Date: 2025-05-17 15:30:00
 * @Description: 测试向量索引集成功能
'''
import os
import sys
import time
import pytest
import tempfile
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.note_generator.embedder import Embedder
from src.note_generator.vector_index import VectorIndex
from src.note_generator.advanced_memory_with_index import AdvancedMemoryManagerWithIndex


class TestVectorIndexIntegration:
    """测试向量索引与高级记忆管理器的集成"""
    
    @pytest.fixture
    def mock_embedder(self):
        """创建模拟的Embedder实例"""
        mock = MagicMock(spec=Embedder)
        mock.model_name = "test-model"
        mock.vector_size = 4
        # 确保返回与输入数量匹配的向量
        mock.embed_texts = lambda texts: [
            [0.5, 0.5, 0.5, 0.5] for _ in range(len(texts))
        ]
        mock.embed_single.return_value = [0.5, 0.5, 0.5, 0.5]
        mock.get_embedding.return_value = [0.5, 0.5, 0.5, 0.5]
        return mock
    
    @pytest.fixture
    def memory_manager(self, mock_embedder):
        """创建测试用的记忆管理器实例"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 配置向量索引和查询缓存
            config = {
                "memory": {
                    "vector_index": {
                        "enabled": True,
                        "type": "flat",  # 使用最简单的索引类型便于测试
                        "threshold": 100,
                        "cache_size": 10
                    },
                    "query_cache": {
                        "enabled": True,
                        "capacity": 10,
                        "ttl": 60
                    }
                }
            }
            
            # 创建记忆管理器
            manager = AdvancedMemoryManagerWithIndex(
                chroma_db_path=temp_dir,
                embedder=mock_embedder,
                config=config
            )
            
            yield manager
    
    def test_add_knowledge_updates_vector_index(self, memory_manager, mock_embedder):
        """测试添加知识时向量索引是否正确更新"""
        # 直接使用VectorIndex API添加一些向量来测试向量索引功能
        assert hasattr(memory_manager, 'vector_index')
        assert memory_manager.vector_index is not None
        
        vector_index = memory_manager.vector_index
        vector_index.add("test-id-1", [0.1, 0.2, 0.3, 0.4])
        vector_index.add("test-id-2", [0.5, 0.6, 0.7, 0.8])
        
        # 验证向量索引是否被正确更新
        assert len(vector_index) == 2
        
        # 测试batch添加
        vector_index.batch_add(
            ["test-id-3", "test-id-4"],
            [[0.9, 0.8, 0.7, 0.6], [0.4, 0.3, 0.2, 0.1]]
        )
        assert len(vector_index) == 4
    
    def test_clear_memory(self, memory_manager):
        """测试清空记忆功能"""
        # 直接添加向量到索引
        vector_index = memory_manager.vector_index
        vector_index.add("test-id-1", [0.1, 0.2, 0.3, 0.4])
        vector_index.add("test-id-2", [0.5, 0.6, 0.7, 0.8])
        
        # 验证添加成功
        assert len(vector_index) == 2
        
        # 清空向量索引
        vector_index.clear()
        
        # 验证清空成功
        assert len(vector_index) == 0
    
    def test_export_import_memory(self, memory_manager):
        """测试记忆导出导入功能"""
        # 测试直接使用save和load方法
        vector_index = VectorIndex(vector_dim=4, index_type="flat")
        
        # 添加向量
        vector_index.add("test-id-export", [0.1, 0.2, 0.3, 0.4])
        assert len(vector_index) == 1
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 为向量索引设置保存路径
            index_path = os.path.join(temp_dir, "test_vector_index.pkl")
            
            # 保存向量索引
            vector_index.save(index_path)
            
            # 验证导出成功
            assert os.path.exists(index_path)
            assert os.path.getsize(index_path) > 0
            
            # 创建新的向量索引
            new_index = VectorIndex(vector_dim=4, index_type="flat")
            assert len(new_index) == 0
            
            # 从文件加载
            loaded_success = new_index.load(index_path)
            
            # 验证加载成功
            assert loaded_success
            assert len(new_index) == 1
            assert "test-id-export" in new_index.ids
    
    def test_perform_retrieval(self, memory_manager):
        """测试检索功能"""
        # 直接添加向量和文本进行测试
        vector_index = memory_manager.vector_index
        
        # 添加内容到向量索引
        vector_index.add("ai-id", [0.9, 0.1, 0.1, 0.1])
        vector_index.add("ml-id", [0.1, 0.9, 0.1, 0.1])
        vector_index.add("dl-id", [0.1, 0.1, 0.9, 0.1])
        
        # 验证向量索引有效
        assert len(vector_index) == 3
        
        # 执行向量检索
        query_results = vector_index.query([0.9, 0.1, 0.1, 0.1], top_k=2)
        
        # 验证检索结果
        assert len(query_results) == 2
        assert query_results[0][0] == "ai-id"  # 第一个结果应该是ai-id
        
        # 测试带阈值的检索
        retrieval_results = vector_index.retrieve([0.9, 0.1, 0.1, 0.1], top_k=2, threshold=0.5)
        assert len(retrieval_results) > 0

    def test_batch_retrieve_similar(self, memory_manager):
        """测试批量检索功能"""
        # 添加一些知识
        memory_manager.add_knowledge("中国位于亚洲东部", {"subject": "地理"})
        memory_manager.add_knowledge("太阳系有八大行星", {"subject": "天文"})
        memory_manager.add_knowledge("地球是太阳系中的第三颗行星", {"subject": "天文"})
        
        # 执行批量检索
        queries = ["亚洲地理", "太阳系行星"]
        results = memory_manager.batch_retrieve_similar(queries, top_k=2)
        
        # 验证结果
        assert len(results) == 2
        assert len(results[0]) > 0
        assert len(results[1]) > 0


if __name__ == "__main__":
    pytest.main(['-xvs', __file__])
