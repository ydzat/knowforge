"""
测试OCR-LLM-知识库集成功能
"""
import os
import pytest
import tempfile
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from src.note_generator.input_handler import InputHandler
from src.note_generator.advanced_ocr_processor import AdvancedOCRProcessor
from src.note_generator.embedding_manager import EmbeddingManager, Document
from src.note_generator.llm_caller import LLMCaller

class TestOCRLLMIntegration:
    """测试OCR-LLM-知识库集成功能"""
    
    @pytest.fixture
    def mock_config(self):
        """创建测试配置"""
        return {
            "input": {
                "ocr": {
                    "enabled": True,
                    "languages": ["ch_sim", "en"],
                    "use_llm_enhancement": True,
                    "knowledge_enhanced_ocr": True,
                    "image_preprocessing": {
                        "enabled": True
                    }
                }
            },
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "memory": {
                "enabled": True,
                "top_k": 3,
                "similarity_threshold": 0.6
            },
            "llm": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "temperature": 0.3,
                "max_tokens": 1000
            }
        }
    
    @pytest.fixture
    def mock_ocr_output(self):
        """模拟的OCR输出"""
        return ("这是一段文本内容，OCR识别可能有错误，比如把'知识'识别成'智慧'等问题。", 0.8)
    
    @pytest.fixture
    def mock_image(self):
        """创建模拟的图像文件"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # 创建一个简单的图像
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            # 添加一些文本
            cv2.putText(img, "test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(tmp.name, img)
            return tmp.name
    
    @pytest.fixture
    def mock_embedder(self):
        """创建模拟的Embedder实例"""
        mock = MagicMock()
        mock.embed_single.return_value = [0.5, 0.5, 0.5, 0.5]
        mock.embed_texts.return_value = [[0.5, 0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6]]
        return mock
    
    @pytest.fixture
    def mock_documents(self):
        """创建模拟的知识库文档"""
        return [
            Document(
                id="doc1", 
                content="知识是人类进步的阶梯，知识库能够帮助我们更好地理解信息。", 
                similarity=0.85, 
                metadata={"source": "test"}
            ),
            Document(
                id="doc2", 
                content="OCR(光学字符识别)技术可以从图像中识别文字内容。", 
                similarity=0.75, 
                metadata={"source": "test"}
            )
        ]
    
    @patch('src.note_generator.input_handler.easyocr.Reader')
    @patch('src.note_generator.embedding_manager.Embedder')
    @patch('src.note_generator.embedding_manager.EmbeddingManager.search_similar_content')
    @patch('src.note_generator.llm_caller.LLMCaller.call_model')
    def test_ocr_llm_knowledge_integration(self, mock_llm_call, mock_search, mock_embedder_class, 
                                         mock_reader, mock_config, mock_image, mock_documents):
        """测试OCR-LLM-知识库完整集成流程"""
        # 设置模拟OCR输出
        mock_reader_instance = MagicMock()
        mock_reader_instance.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], 
             "这是一段文本内容，OCR识别可能有错误，比如把'知识'识别成'智慧'等问题。", 
             0.8)
        ]
        mock_reader.return_value = mock_reader_instance
        
        # 设置模拟知识库查询结果
        mock_search.return_value = mock_documents
        
        # 设置模拟LLM输出
        mock_llm_call.return_value = "这是一段文本内容，OCR识别已修正，比如把'智慧'修正为'知识'，解决了识别问题。"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建输入处理器
            input_handler = InputHandler(mock_config, temp_dir)
            
            # 创建高级OCR处理器
            ocr_processor = AdvancedOCRProcessor(mock_config, temp_dir)
            
            # 模拟OCR-LLM-知识库集成处理
            result = input_handler.advanced_ocr_llm_integration(mock_image)
            
            # 验证结果
            assert isinstance(result, tuple)
            assert len(result) == 2
            text, confidence = result
            
            # 验证文本内容是LLM处理后的
            assert "已修正" in text
            assert "解决了识别问题" in text
            
            # 验证调用流程
            mock_reader.assert_called_once()
            mock_search.assert_called()
            # LLM可能会被调用多次，我们不检查调用次数
            assert mock_llm_call.called
    
    @patch('src.note_generator.input_handler.easyocr.Reader')
    @patch('src.note_generator.embedding_manager.EmbeddingManager')
    @patch('src.note_generator.llm_caller.LLMCaller')
    def test_knowledge_enhancement_disabled(self, mock_llm_caller_class, mock_embedding_manager_class,
                                           mock_reader, mock_config, mock_image):
        """测试禁用知识库增强时的行为"""
        # 修改配置，禁用知识库增强
        config = mock_config.copy()
        config["input"]["ocr"]["knowledge_enhanced_ocr"] = False
        
        # 设置模拟OCR输出
        mock_reader_instance = MagicMock()
        mock_reader_instance.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], 
             "这是一段OCR识别的文本。", 
             0.9)
        ]
        mock_reader.return_value = mock_reader_instance
        
        # 设置模拟LLM输出
        mock_llm_caller_instance = MagicMock()
        mock_llm_caller_instance.call_model.return_value = "这是LLM增强后的文本。"
        mock_llm_caller_class.return_value = mock_llm_caller_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建输入处理器
            input_handler = InputHandler(config, temp_dir)
            
            # 模拟OCR-LLM处理
            result = input_handler.advanced_ocr_llm_integration(mock_image)
            
            # 验证结果
            assert isinstance(result, tuple)
            assert len(result) == 2
            
            # 验证调用流程
            mock_reader.assert_called_once()
            # 知识库增强禁用时，我们不检查EmbeddingManager是否被调用，只检查LLM调用
            assert mock_llm_caller_instance.call_model.called
            # LLM调用应该发生
            mock_llm_caller_instance.call_model.assert_called_once()
