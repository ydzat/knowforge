"""
测试OCR与内存系统集成的功能
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch
import tempfile
import shutil
import numpy as np
import cv2

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.advanced_ocr_processor import AdvancedOCRProcessor
from src.note_generator.advanced_memory_manager import AdvancedMemoryManager
from src.utils.config_loader import ConfigLoader

class TestOCRMemoryIntegration:
    """测试OCR与内存系统集成"""
    
    @pytest.fixture
    def setup_workspace(self):
        """设置临时工作区"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def mock_config(self):
        """创建测试配置"""
        config = {
            "locale": "zh_CN",
            "input": {
                "ocr": {
                    "enabled": True,
                    "languages": ["ch_sim", "en"],
                    "confidence_threshold": 0.6,
                    "use_llm_enhancement": True,
                    "deep_enhancement": True,
                    "knowledge_enhanced_ocr": True,
                    "image_preprocessing": {
                        "enabled": True,
                        "denoise": True,
                        "contrast_enhancement": True,
                        "adaptive_thresholding": True,
                        "deskew": True
                    }
                }
            },
            "memory": {
                "enabled": True,
                "capacity": 100,
                "working_memory_capacity": 10,
                "similarity_threshold": 0.5,
                "index_path": "memory_db/memory_index",
                "optimize_interval": 10
            }
        }
        return config
    
    @pytest.fixture
    def mock_ocr_processor(self, mock_config, setup_workspace):
        """创建OCR处理器实例"""
        return AdvancedOCRProcessor(mock_config, setup_workspace)
        
    @pytest.fixture
    def mock_memory_manager(self, mock_config, setup_workspace):
        """创建记忆管理器实例"""
        return AdvancedMemoryManager(setup_workspace, mock_config.get("memory", {}))
        
    @pytest.fixture
    def mock_image(self):
        """创建测试图像"""
        # 创建一个空白图像
        img = np.ones((300, 800), dtype=np.uint8) * 255
        
        # 添加文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'KnowForge Memory Test', (50, 150), font, 1.5, (0, 0, 0), 2)
        
        return img
    
    @patch('src.note_generator.llm_caller.LLMCaller')
    def test_ocr_memory_integration(self, mock_llm, mock_ocr_processor, mock_memory_manager, mock_image):
        """测试OCR与记忆系统的集成"""
        # 设置LLM模拟响应
        mock_llm_instance = MagicMock()
        mock_llm_instance.call_model.return_value = "KnowForge Memory Test Integration"
        mock_llm.return_value = mock_llm_instance
        
        # 向记忆管理器添加一些相关知识
        mock_memory_manager.add({
            "id": "test1",
            "content": "KnowForge是一个高级记忆管理系统",
            "metadata": {"source": "test", "type": "definition"}
        })
        
        # 保存图像到临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
            cv2.imwrite(temp_img.name, mock_image)
            image_path = temp_img.name
        
        try:
            # 模拟OCR结果
            with patch('src.note_generator.advanced_ocr_processor.AdvancedOCRProcessor.process_image') as mock_ocr:
                mock_ocr.return_value = ("KnavForge Memary Test", 0.7)  # 故意添加一些错误
                
                # 测试记忆增强功能
                with patch('src.note_generator.advanced_ocr_processor.AdvancedOCRProcessor.use_memory_for_ocr_enhancement') as mock_memory_enhance:
                    mock_memory_enhance.return_value = {
                        "original": "KnavForge Memary Test",
                        "enhanced": "KnowForge Memory Test",
                        "confidence": 0.85,
                        "references": [{"id": "test1", "content": "KnowForge是一个高级记忆管理系统", "similarity": 0.78}]
                    }
                    
                    # 处理图像并验证结果
                    result, confidence = mock_ocr_processor.process_image(image_path)
                    
                    # 验证记忆增强方法被调用
                    assert mock_memory_enhance.called
                    
                    # 验证结果修正了OCR错误
                    assert result == "KnowForge Memory Test"
                    assert confidence > 0.8
        finally:
            # 清理临时文件
            os.unlink(image_path)
    
    @patch('src.note_generator.advanced_memory_manager.LLMCaller')
    def test_enhance_ocr_with_memory(self, mock_llm, mock_memory_manager):
        """测试AdvancedMemoryManager的enhance_ocr_with_memory方法"""
        # 设置LLM模拟响应
        mock_llm_instance = MagicMock()
        mock_llm_instance.call_model.return_value = "正确识别的人工智能文本"
        mock_llm.return_value = mock_llm_instance
        
        # 在记忆管理器中添加相关知识
        mock_memory_manager.add({
            "id": "test_knowledge",
            "content": "人工智能是计算机科学的一个分支",
            "metadata": {"source": "test", "type": "definition"}
        })
        
        # 模拟检索方法
        with patch('src.note_generator.advanced_memory_manager.AdvancedMemoryManager.retrieve') as mock_retrieve:
            mock_retrieve.return_value = [{
                "id": "test_knowledge",
                "content": "人工智能是计算机科学的一个分支",
                "similarity": 0.85
            }]
            
            # 调用enhance_ocr_with_memory方法
            result = mock_memory_manager.enhance_ocr_with_memory("人エ智能文本", "OCR测试")
            
            # 验证结果
            assert result["original"] == "人エ智能文本"  # 原始文本包含OCR错误
            assert result["enhanced"] == "正确识别的人工智能文本"  # 验证返回增强后的文本
            assert result["confidence"] > 0.5
            assert len(result["references"]) == 1
            assert result["references"][0]["id"] == "test_knowledge"
            
            # 验证LLM调用
            assert mock_llm_instance.call_model.called
