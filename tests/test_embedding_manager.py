"""
测试向量嵌入管理模块
"""
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from src.note_generator.embedding_manager import EmbeddingManager, Document, EmbeddingManagerError
from src.note_generator.embedder import Embedder

class TestEmbeddingManager:
    """测试EmbeddingManager类的基本功能"""
    
    @pytest.fixture
    def mock_embedder(self):
        """创建模拟的Embedder实例"""
        mock = MagicMock(spec=Embedder)
        mock.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        mock.vector_size = 4
        mock.embed_texts.return_value = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ]
        mock.embed_single.return_value = [0.5, 0.5, 0.5, 0.5]
        return mock
    
    @pytest.fixture
    def mock_memory_manager(self):
        """创建模拟的MemoryManager实例"""
        mock = MagicMock()
        mock.query_similar.return_value = [
            {
                "id": "doc1",
                "text": "测试文档1",
                "similarity": 0.95,
                "metadata": {"source": "test"}
            },
            {
                "id": "doc2",
                "text": "测试文档2",
                "similarity": 0.85,
                "metadata": {"source": "test"}
            }
        ]
        mock.add_segments.return_value = ["id1", "id2", "id3"]
        mock.get_collection_stats.return_value = {
            "count": 10,
            "collection_name": "test_collection",
            "embedding_model": "test_model"
        }
        return mock
    
    @patch('src.note_generator.embedding_manager.Embedder')
    def test_init(self, mock_embedder_class):
        """测试初始化"""
        # 设置模拟的Embedder类返回值
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试配置
            config = {
                "embedding": {
                    "model_name": "test-model",
                    "cache_dir": os.path.join(temp_dir, "embeddings")
                },
                "memory": {
                    "enabled": True,
                    "chroma_db_path": os.path.join(temp_dir, "memory_db"),
                    "collection_name": "test_collection",
                    "top_k": 3,
                    "similarity_threshold": 0.7
                }
            }
            
            # 初始化EmbeddingManager
            manager = EmbeddingManager(temp_dir, config)
            
            # 验证Embedder类被正确初始化
            mock_embedder_class.assert_called_once_with(
                model_name="test-model", 
                cache_dir=os.path.join(temp_dir, "embeddings")
            )
            
            # 验证属性
            assert manager.workspace_dir == temp_dir
            assert manager.config == config
            assert manager.default_top_k == 3
            assert manager.similarity_threshold == 0.7
    
    @patch('src.note_generator.embedding_manager.Embedder')
    def test_init_with_invalid_model(self, mock_embedder_class):
        """测试初始化失败情况"""
        # 设置模拟Embedder类抛出EmbeddingError异常
        from src.note_generator.embedder import EmbeddingError
        mock_embedder_class.side_effect = EmbeddingError("向量化模型加载失败")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 应该抛出EmbeddingManagerError异常
            with pytest.raises(EmbeddingManagerError):
                EmbeddingManager(temp_dir)
    
    @patch('src.note_generator.embedding_manager.Embedder')
    @patch('src.note_generator.memory_manager.MemoryManager')
    def test_search_similar_content(self, mock_memory_manager_class, mock_embedder_class):
        """测试搜索相似内容"""
        # 设置模拟的Embedder和MemoryManager类返回值
        mock_embedder_instance = MagicMock()
        mock_memory_manager_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        mock_memory_manager_class.return_value = mock_memory_manager_instance
        
        # 设置mock的query_similar返回值
        mock_memory_manager_instance.query_similar.return_value = [
            {
                "id": "doc1",
                "text": "测试文档1",
                "similarity": 0.95,
                "metadata": {"source": "test"}
            },
            {
                "id": "doc2",
                "text": "测试文档2",
                "similarity": 0.85,
                "metadata": {"source": "test"}
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 初始化EmbeddingManager
            manager = EmbeddingManager(temp_dir)
            
            # 测试搜索
            results = manager.search_similar_content("测试查询", top_k=2)
            
            # 验证MemoryManager类被正确初始化
            mock_memory_manager_class.assert_called_once()
            
            # 验证query_similar方法被正确调用
            mock_memory_manager_instance.query_similar.assert_called_once_with(
                query_text="测试查询",
                top_k=2,
                threshold=manager.similarity_threshold,
                include_embeddings=False
            )
            
            # 验证结果转换
            assert len(results) == 2
            assert isinstance(results[0], Document)
            assert results[0].id == "doc1"
            assert results[0].content == "测试文档1"
            assert results[0].similarity == 0.95
            assert results[0].metadata == {"source": "test"}
    
    @patch('src.note_generator.embedding_manager.Embedder')
    @patch('src.note_generator.memory_manager.MemoryManager')
    def test_add_to_knowledge_base(self, mock_memory_manager_class, mock_embedder_class):
        """测试添加到知识库"""
        # 设置模拟的Embedder和MemoryManager类返回值
        mock_embedder_instance = MagicMock()
        mock_memory_manager_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        mock_memory_manager_class.return_value = mock_memory_manager_instance
        
        # 设置mock的add_segments返回值
        mock_memory_manager_instance.add_segments.return_value = ["id1", "id2", "id3"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 初始化EmbeddingManager
            manager = EmbeddingManager(temp_dir)
            
            # 测试添加文档
            documents = ["文档1", "文档2", "文档3"]
            ids = manager.add_to_knowledge_base(documents)
            
            # 验证add_segments方法被正确调用
            mock_memory_manager_instance.add_segments.assert_called_once()
            args, _ = mock_memory_manager_instance.add_segments.call_args
            assert args[0] == documents
            
            # 验证结果
            assert ids == ["id1", "id2", "id3"]
    
    @patch('src.note_generator.embedding_manager.Embedder')
    @patch('src.note_generator.memory_manager.MemoryManager')
    def test_empty_input(self, mock_memory_manager_class, mock_embedder_class):
        """测试空输入处理"""
        # 设置模拟的Embedder和MemoryManager类返回值
        mock_embedder_instance = MagicMock()
        mock_memory_manager_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        mock_memory_manager_class.return_value = mock_memory_manager_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 初始化EmbeddingManager
            manager = EmbeddingManager(temp_dir)
            
            # 测试空查询
            results = manager.search_similar_content("")
            assert results == []
            
            # 测试空文档列表
            ids = manager.add_to_knowledge_base([])
            assert ids == []
    
    @patch('src.note_generator.embedding_manager.Embedder')
    @patch('src.note_generator.memory_manager.MemoryManager')
    def test_get_knowledge_stats(self, mock_memory_manager_class, mock_embedder_class):
        """测试获取知识库统计信息"""
        # 设置模拟的Embedder和MemoryManager类返回值
        mock_embedder_instance = MagicMock()
        mock_memory_manager_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        mock_memory_manager_class.return_value = mock_memory_manager_instance
        
        # 设置mock的get_collection_stats返回值
        mock_memory_manager_instance.get_collection_stats.return_value = {
            "count": 10,
            "collection_name": "test_collection",
            "embedding_model": "test_model"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 初始化EmbeddingManager
            manager = EmbeddingManager(temp_dir)
            
            # 测试获取统计信息
            stats = manager.get_knowledge_stats()
            
            # 验证get_collection_stats方法被正确调用
            mock_memory_manager_instance.get_collection_stats.assert_called_once()
            
            # 验证结果
            assert stats == {
                "count": 10,
                "collection_name": "test_collection",
                "embedding_model": "test_model"
            }
