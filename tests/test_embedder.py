'''
 * @Author: @ydzat
 * @Date: 2025-05-14 11:05:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-05-14 11:05:00
 * @Description: 向量化模块测试用例
'''
import os
import pytest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

from src.note_generator.embedder import Embedder, EmbeddingError

class TestEmbedder:
    """测试Embedder类的基本功能"""
    
    def test_init_default(self):
        """测试默认参数初始化"""
        with patch('src.note_generator.embedder.SentenceTransformer') as mock_st:
            # 设置模拟的向量维度
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_instance
            
            embedder = Embedder()
            
            # 验证SentenceTransformer被正确初始化
            mock_st.assert_called_once_with(
                "sentence-transformers/all-MiniLM-L6-v2", 
                cache_folder=None
            )
            
            assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert embedder.vector_size == 384
    
    def test_init_custom(self):
        """测试自定义参数初始化"""
        with patch('src.note_generator.embedder.SentenceTransformer') as mock_st:
            # 设置模拟的向量维度
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_instance
            
            custom_model = "sentence-transformers/all-mpnet-base-v2"
            custom_cache = "/tmp/models"
            
            embedder = Embedder(model_name=custom_model, cache_dir=custom_cache)
            
            # 验证SentenceTransformer被正确初始化
            mock_st.assert_called_once_with(custom_model, cache_folder=custom_cache)
            
            assert embedder.model_name == custom_model
            assert embedder.vector_size == 768
    
    def test_init_failure(self):
        """测试初始化失败情况"""
        with patch('src.note_generator.embedder.SentenceTransformer') as mock_st:
            # 设置模拟加载失败
            mock_st.side_effect = Exception("Model not found")
            
            # 验证是否抛出正确的异常
            with pytest.raises(EmbeddingError) as excinfo:
                Embedder()
                
            assert "加载失败" in str(excinfo.value)
    
    def test_embed_texts(self):
        """测试批量文本向量化"""
        with patch('src.note_generator.embedder.SentenceTransformer') as mock_st:
            # 创建模拟的SentenceTransformer实例
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 4
            
            # 设置encode方法返回模拟的向量
            mock_vectors = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ])
            mock_instance.encode.return_value = mock_vectors
            mock_st.return_value = mock_instance
            
            # 创建Embedder实例
            embedder = Embedder()
            
            # 测试批量向量化
            texts = ["测试文本1", "测试文本2", "测试文本3"]
            vectors = embedder.embed_texts(texts)
            
            # 验证结果
            assert len(vectors) == 3
            assert len(vectors[0]) == 4
            assert isinstance(vectors, list)
            assert isinstance(vectors[0], list)
            
            # 验证模型encode方法被正确调用
            mock_instance.encode.assert_called_once_with(texts)
    
    def test_embed_single(self):
        """测试单文本向量化"""
        with patch('src.note_generator.embedder.SentenceTransformer') as mock_st:
            # 创建模拟的SentenceTransformer实例
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 4
            
            # 设置encode方法返回模拟的向量
            mock_vector = np.array([1.0, 0.0, 1.0, 0.0])
            mock_instance.encode.return_value = mock_vector
            mock_st.return_value = mock_instance
            
            # 创建Embedder实例
            embedder = Embedder()
            
            # 测试单文本向量化
            text = "测试单文本"
            vector = embedder.embed_single(text)
            
            # 验证结果
            assert len(vector) == 4
            assert isinstance(vector, list)
            
            # 验证模型encode方法被正确调用
            mock_instance.encode.assert_called_once_with(text)
    
    def test_empty_input(self):
        """测试空输入处理"""
        with patch('src.note_generator.embedder.SentenceTransformer') as mock_st:
            # 创建模拟的SentenceTransformer实例
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 4
            mock_st.return_value = mock_instance
            
            # 创建Embedder实例
            embedder = Embedder()
            
            # 测试空列表
            result = embedder.embed_texts([])
            assert result == []
            
            # 测试空字符串、None及非字符串输入
            with pytest.raises(EmbeddingError):
                embedder.embed_single("")
                
            with pytest.raises(EmbeddingError):
                embedder.embed_single(None)
                
            with pytest.raises(EmbeddingError):
                embedder.embed_single(123)
    
    def test_save_embeddings(self):
        """测试保存向量到文件"""
        with patch('src.note_generator.embedder.SentenceTransformer') as mock_st:
            # 创建模拟的SentenceTransformer实例
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 4
            
            # 设置encode方法返回模拟的向量
            mock_vectors = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])
            mock_instance.encode.return_value = mock_vectors
            mock_st.return_value = mock_instance
            
            # 创建Embedder实例
            embedder = Embedder()
            
            # 使用临时目录测试保存功能
            with tempfile.TemporaryDirectory() as temp_dir:
                texts = ["测试文本1", "测试文本2"]
                paths = embedder.save_embeddings(texts, temp_dir)
                
                # 验证返回的路径数量
                assert len(paths) == 2
                
                # 验证文件是否存在
                assert all(os.path.exists(path) for path in paths.values())
                
                # 随机验证一个文件内容
                import json
                first_path = list(paths.values())[0]
                with open(first_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    
                assert "text" in saved_data
                assert "embedding" in saved_data
                assert "model" in saved_data
                assert saved_data["model"] == "sentence-transformers/all-MiniLM-L6-v2"
                assert len(saved_data["embedding"]) == 4
    
    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        # 测试相同向量
        vec1 = [1.0, 0.0, 1.0, 0.0]
        vec2 = [1.0, 0.0, 1.0, 0.0]
        similarity = Embedder.cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0)
        
        # 测试正交向量
        vec1 = [1.0, 0.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0, 0.0]
        similarity = Embedder.cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0)
        
        # 测试完全相反的向量
        vec1 = [1.0, 0.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0, 0.0]
        similarity = Embedder.cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0)
        
        # 测试零向量
        vec1 = [0.0, 0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0, 0.0]
        similarity = Embedder.cosine_similarity(vec1, vec2)
        assert similarity == 0.0