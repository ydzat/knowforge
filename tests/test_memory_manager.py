'''
 * @Author: @ydzat
 * @Date: 2025-05-14 11:20:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-05-14 13:30:00
 * @Description: 向量记忆管理模块测试用例
'''
import os
import json
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from src.note_generator.embedder import Embedder
from src.note_generator.memory_manager import MemoryManager, MemoryError

class TestMemoryManager:
    """测试MemoryManager类的基本功能"""
    
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
    def mock_chromadb_client(self):
        """创建模拟的ChromaDB客户端"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection
        return mock_client, mock_collection
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_init_default(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试默认参数初始化"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建MemoryManager实例
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 验证ChromaDB客户端被正确初始化
            mock_chroma_client.assert_called_once()
            
            # 修正这行，检查参数是否是以关键字参数方式传递
            if mock_chroma_client.call_args.kwargs:
                assert mock_chroma_client.call_args.kwargs.get('path') == temp_dir
            else:
                assert mock_chroma_client.call_args.args[0] == temp_dir
            
            # 验证尝试获取现有集合
            mock_client.get_collection.assert_called_once()
            assert manager.collection_name == MemoryManager.DEFAULT_COLLECTION
            assert manager.chroma_db_path == os.path.abspath(temp_dir)
            assert manager.embedder == mock_embedder
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_init_with_new_collection(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试初始化时创建新集合"""
        mock_client, mock_collection = mock_chromadb_client
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_chroma_client.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建MemoryManager实例
            manager = MemoryManager(temp_dir, embedder=mock_embedder, collection_name="custom_collection")
            
            # 验证尝试创建新集合
            mock_client.create_collection.assert_called_once()
            assert mock_client.create_collection.call_args[1]["name"] == "custom_collection"
            assert manager.collection_name == "custom_collection"
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_init_failure(self, mock_chroma_client):
        """测试初始化失败情况"""
        mock_chroma_client.side_effect = Exception("Failed to initialize ChromaDB")
        
        with pytest.raises(MemoryError) as excinfo:
            MemoryManager("/tmp/nonexistent_dir")
            
        assert "初始化记忆管理器失败" in str(excinfo.value)
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_embed_function(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试自定义嵌入函数"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 测试自定义嵌入函数
            texts = ["测试文本1", "测试文本2", "测试文本3"]
            vectors = manager._embed_function(texts)
            
            # 验证结果和调用
            assert len(vectors) == 3
            mock_embedder.embed_texts.assert_called_once_with(texts)
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_add_segments(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试添加片段"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 测试添加片段
            segments = ["测试片段1", "测试片段2", "测试片段3"]
            metadata = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
            
            ids = manager.add_segments(segments, metadata)
            
            # 验证ID列表长度
            assert len(ids) == 3
            
            # 验证collection.add被正确调用
            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args[1]
            assert len(call_args["ids"]) == 3
            assert call_args["documents"] == segments
            assert call_args["metadatas"] == metadata
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_add_segments_empty(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试添加空片段列表"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 测试添加空片段列表
            result = manager.add_segments([])
            assert result == []
            mock_collection.add.assert_not_called()
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_add_segments_with_default_metadata(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试使用默认元数据添加片段"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 测试添加片段（无元数据）
            segments = ["测试片段1", "测试片段2"]
            ids = manager.add_segments(segments)
            
            # 验证ID列表长度
            assert len(ids) == 2
            
            # 验证使用了默认元数据
            call_args = mock_collection.add.call_args[1]
            assert len(call_args["metadatas"]) == 2
            assert all("timestamp" in meta for meta in call_args["metadatas"])
            assert all("source" in meta for meta in call_args["metadatas"])
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_query_similar(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试检索相似片段"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        # 设置模拟查询结果
        mock_query_result = {
            "ids": [["id1", "id2", "id3"]],
            "documents": [["文本1", "文本2", "文本3"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]],
            "distances": [[0.1, 0.3, 0.5]]
        }
        mock_collection.query.return_value = mock_query_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 执行查询
            results = manager.query_similar("测试查询", top_k=3)
            
            # 验证查询参数
            mock_collection.query.assert_called_once()
            assert mock_collection.query.call_args[1]["query_texts"] == ["测试查询"]
            assert mock_collection.query.call_args[1]["n_results"] == 3
            
            # 验证结果格式和数量
            assert len(results) == 3
            assert results[0]["id"] == "id1"
            assert results[0]["text"] == "文本1"
            assert results[0]["metadata"] == {"source": "test1"}
            assert results[0]["similarity"] == pytest.approx(0.9)  # 1.0 - 0.1
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_query_similar_empty(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试空查询文本"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 测试空查询
            results = manager.query_similar("")
            assert results == []
            mock_collection.query.assert_not_called()
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_query_similar_with_threshold(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试带阈值的相似检索"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        # 设置模拟查询结果 - 3个结果，距离分别为0.1, 0.6, 0.9
        mock_query_result = {
            "ids": [["id1", "id2", "id3"]],
            "documents": [["文本1", "文本2", "文本3"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]],
            "distances": [[0.1, 0.6, 0.9]]
        }
        mock_collection.query.return_value = mock_query_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 设置阈值为0.5 (相似度高于0.5的结果)
            results = manager.query_similar("测试查询", threshold=0.5)
            
            # 应该只返回2个结果 (文本1的相似度为0.9, 文本2的相似度为0.4, 文本3的相似度为0.1)
            assert len(results) == 1
            assert results[0]["id"] == "id1"
            assert results[0]["similarity"] == pytest.approx(0.9)  # 1.0 - 0.1
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_rebuild_memory(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试重建记忆库"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 测试重建记忆库
            segments = ["测试片段1", "测试片段2"]
            success = manager.rebuild_memory(segments)
            
            # 验证操作是否成功
            assert success is True
            
            # 验证删除和创建集合操作
            mock_client.delete_collection.assert_called_once_with(MemoryManager.DEFAULT_COLLECTION)
            mock_client.create_collection.assert_called()
            
            # 验证添加新片段
            mock_collection.add.assert_called_once()
            assert mock_collection.add.call_args[1]["documents"] == segments
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_rebuild_memory_empty(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试重建空记忆库"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 测试重建空记忆库
            success = manager.rebuild_memory([])
            
            # 验证操作是否成功
            assert success is True
            
            # 验证删除和创建集合操作
            mock_client.delete_collection.assert_called_once_with(MemoryManager.DEFAULT_COLLECTION)
            mock_client.create_collection.assert_called()
            
            # 验证不添加新片段
            assert not mock_collection.add.called
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_get_collection_stats(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试获取记忆库统计信息"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        # 设置模拟的集合统计信息
        mock_collection.count.return_value = 10
        mock_collection.get.return_value = {
            "documents": ["短文本", "这是一个较长的测试文本"]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 获取统计信息
            stats = manager.get_collection_stats()
            
            # 验证结果
            assert stats["count"] == 10
            assert stats["collection_name"] == MemoryManager.DEFAULT_COLLECTION
            assert stats["db_path"] == os.path.abspath(temp_dir)
            assert stats["embedding_model"] == mock_embedder.model_name
            assert "avg_text_length" in stats
    
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_export_to_json(self, mock_chroma_client, mock_embedder, mock_chromadb_client):
        """测试导出为JSON文件"""
        mock_client, mock_collection = mock_chromadb_client
        mock_chroma_client.return_value = mock_client
        
        # 设置模拟的集合数据
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["文本1", "文本2"],
            "metadatas": [{"source": "test1"}, {"source": "test2"}]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 导出到临时文件
            output_path = os.path.join(temp_dir, "export.json")
            result_path = manager.export_to_json(output_path)
            
            # 验证文件是否创建
            assert os.path.exists(output_path)
            assert result_path == output_path
            
            # 验证文件内容
            with open(output_path, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
                
            # 检查导出数据格式
            assert "metadata" in exported_data
            assert "entries" in exported_data
            assert len(exported_data["entries"]) == 2
            assert exported_data["entries"][0]["id"] == "id1"
            assert exported_data["entries"][0]["text"] == "文本1"
            assert exported_data["entries"][0]["metadata"] == {"source": "test1"}
            
    @patch('src.note_generator.memory_manager.chromadb.PersistentClient')
    def test_delete_db(self, mock_chroma_client, mock_embedder):
        """测试删除数据库"""
        mock_chroma_client.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试目录和文件
            os.makedirs(os.path.join(temp_dir, "chroma"), exist_ok=True)
            with open(os.path.join(temp_dir, "chroma", "test.file"), "w") as f:
                f.write("test content")
                
            manager = MemoryManager(temp_dir, embedder=mock_embedder)
            
            # 执行删除操作
            result = manager.delete_db()
            
            # 验证结果
            assert result is True
            assert not os.path.exists(temp_dir)  # 确认目录已被删除