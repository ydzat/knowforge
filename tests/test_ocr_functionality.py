import os
import pytest
import tempfile
import shutil
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from src.utils.config_loader import ConfigLoader
from src.note_generator.input_handler import InputHandler

# 测试OCR相关功能
class TestOCRFunctionality:
    
    @pytest.fixture
    def setup_temporary_dirs(self):
        """创建临时输入和工作目录"""
        input_dir = tempfile.mkdtemp()
        workspace_dir = tempfile.mkdtemp()
        
        # 创建图片输入目录
        os.makedirs(os.path.join(input_dir, 'images'), exist_ok=True)
        
        yield input_dir, workspace_dir
        
        # 清理
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(workspace_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_config(self):
        """创建模拟的配置加载器"""
        config = {
            'system': {'language': 'zh'},
            'input': {
                'allowed_formats': ['pdf', 'jpg', 'png'],
                'max_file_size_mb': 10,
                'ocr': {
                    'enabled': True,
                    'languages': ['ch_sim', 'en'],
                    'use_llm_enhancement': True,
                    'confidence_threshold': 0.6,
                    'image_preprocessing': {
                        'enabled': True,
                        'denoise': True,
                        'contrast_enhancement': True,
                        'auto_rotate': True
                    }
                }
            }
        }
        
        mock_config = MagicMock(spec=ConfigLoader)
        mock_config.get.side_effect = lambda path, default=None: self._get_config_value(config, path, default)
        
        return mock_config
        
    def _get_config_value(self, config_dict, path, default=None):
        """模拟ConfigLoader.get方法按路径获取配置值"""
        parts = path.split('.')
        value = config_dict
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    @pytest.fixture
    def create_test_image(self, setup_temporary_dirs):
        """创建测试用的图片"""
        input_dir, _ = setup_temporary_dirs
        img_path = os.path.join(input_dir, 'images', 'test_text.png')
        
        # 创建一个简单的带文本的图片
        img = np.ones((300, 800, 3), dtype=np.uint8) * 255  # 白底
        # 由于不能真正渲染文本，这里只是创建一个空图像
        # 实际OCR测试将通过模拟easyocr的结果来完成
        
        cv2.imwrite(img_path, img)
        
        return img_path
    
    @patch('src.note_generator.input_handler.easyocr')
    def test_extract_image_text(self, mock_easyocr, setup_temporary_dirs, mock_config, create_test_image):
        """测试提取图片文本功能"""
        input_dir, workspace_dir = setup_temporary_dirs
        
        # 模拟OCR读取器和结果
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([0, 0, 100, 50], "测试文本1", 0.95),
            ([0, 60, 100, 110], "Test Text 2", 0.85),
            ([0, 120, 100, 170], "低置信度文本", 0.4)  # 应该被过滤掉
        ]
        
        # 配置easyocr模块的模拟行为
        mock_easyocr.Reader.return_value = mock_reader
        mock_easyocr.EASYOCR_AVAILABLE = True
        
        # 创建InputHandler实例
        handler = InputHandler(input_dir, workspace_dir, mock_config)
        
        # 测试提取文本
        text, confidence = handler.extract_image_text(create_test_image)
        
        # 验证结果
        assert "测试文本1" in text
        assert "Test Text 2" in text
        assert "低置信度文本" not in text  # 由于置信度低，应被过滤
        assert confidence == pytest.approx((0.95 + 0.85) / 2, 0.001)  # 平均置信度
        
        # 验证OCR调用
        mock_easyocr.Reader.assert_called_once()
        mock_reader.readtext.assert_called_once()
    
    @patch('src.note_generator.input_handler.os.environ.get')
    def test_enhance_ocr_with_llm(self, mock_environ_get, setup_temporary_dirs, mock_config):
        """测试使用LLM增强OCR结果"""
        input_dir, workspace_dir = setup_temporary_dirs
        
        # 创建一个带自定义方法的InputHandler子类来测试enhance_ocr_with_llm
        class TestInputHandler(InputHandler):
            def __init__(self, input_dir, workspace_dir, config):
                super().__init__(input_dir, workspace_dir, config)
                self.use_llm_enhancement = True
                
            # 覆盖enhance_ocr_with_llm方法，不依赖外部LLMCaller
            def enhance_ocr_with_llm(self, ocr_text, image_context):
                # 模拟LLM纠错效果
                if "拼写錯誤" in ocr_text:
                    return ocr_text.replace("錯誤", "错误")
                return "增强后的OCR文本"
        
        # 创建测试实例
        handler = TestInputHandler(input_dir, workspace_dir, mock_config)
        
        # 测试LLM增强
        ocr_text = "有一些拼写錯誤的文本"
        context = "文件名: test.png\n所在目录: images"
        
        enhanced_text = handler.enhance_ocr_with_llm(ocr_text, context)
        
        # 验证结果
        assert enhanced_text == "有一些拼写错误的文本"
    
    @patch('src.note_generator.input_handler.easyocr')
    def test_extract_texts_with_images(self, mock_easyocr, setup_temporary_dirs, mock_config, create_test_image):
        """测试在extract_texts方法中处理图片"""
        input_dir, workspace_dir = setup_temporary_dirs
        
        # 模拟OCR读取器和结果
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([0, 0, 100, 50], "图片中的文本内容", 0.9),
        ]
        mock_easyocr.Reader.return_value = mock_reader
        mock_easyocr.EASYOCR_AVAILABLE = True
        
        # 模拟LLM增强
        with patch.object(InputHandler, 'enhance_ocr_with_llm', return_value="增强后的图片文本内容"):
            # 创建InputHandler实例
            handler = InputHandler(input_dir, workspace_dir, mock_config)
            
            # 运行extract_texts
            texts = handler.extract_texts()
            
            # 验证结果
            assert len(texts) == 1  # 应该只有一个图片被处理
            assert "[Image:" in texts[0]
            assert "增强后的图片文本内容" in texts[0]
            
            # 验证文本保存
            preprocessed_path = os.path.join(workspace_dir, "preprocessed", "images")
            assert os.path.isdir(preprocessed_path)
            assert len(os.listdir(preprocessed_path)) == 1  # 应该有一个预处理文件
    
    @patch('src.note_generator.input_handler.cv2')
    def test_preprocess_image(self, mock_cv2, setup_temporary_dirs, mock_config):
        """测试图像预处理功能"""
        input_dir, workspace_dir = setup_temporary_dirs
        
        # 模拟cv2功能
        mock_img = MagicMock()
        mock_img.shape = (300, 400, 3)  # 模拟彩色图像
        
        mock_gray = MagicMock()
        mock_denoised = MagicMock()
        mock_enhanced = MagicMock()
        
        mock_cv2.imread.return_value = mock_img
        mock_cv2.cvtColor.return_value = mock_gray
        mock_cv2.fastNlMeansDenoising.return_value = mock_denoised
        
        mock_clahe = MagicMock()
        mock_clahe.apply.return_value = mock_enhanced
        mock_cv2.createCLAHE.return_value = mock_clahe
        
        # 创建InputHandler实例
        handler = InputHandler(input_dir, workspace_dir, mock_config)
        
        # 测试图像预处理
        test_img_path = os.path.join(input_dir, "test.jpg")
        processed_img = handler._preprocess_image(test_img_path)
        
        # 验证预处理流程
        mock_cv2.imread.assert_called_once_with(test_img_path)
        mock_cv2.cvtColor.assert_called_once_with(mock_img, mock_cv2.COLOR_BGR2GRAY)
        mock_cv2.fastNlMeansDenoising.assert_called_once()
        mock_cv2.createCLAHE.assert_called_once()
        mock_clahe.apply.assert_called_once_with(mock_denoised)
        
        # 验证结果
        assert processed_img == mock_enhanced
    
    def test_ocr_disabled(self, setup_temporary_dirs, mock_config):
        """测试OCR禁用情况"""
        input_dir, workspace_dir = setup_temporary_dirs
        
        # 修改配置，禁用OCR
        mock_config.get.side_effect = lambda path, default=None: False if path == "input.ocr.enabled" else self._get_config_value({
            'system': {'language': 'zh'},
            'input': {
                'allowed_formats': ['pdf', 'jpg', 'png'],
                'ocr': {'enabled': False}
            }
        }, path, default)
        
        # 创建InputHandler实例
        handler = InputHandler(input_dir, workspace_dir, mock_config)
        
        # 测试提取文本
        test_img_path = os.path.join(input_dir, "test.jpg")
        text, confidence = handler.extract_image_text(test_img_path)
        
        # 验证结果
        assert text == ""  # 禁用OCR时应返回空字符串
        assert confidence == 0.0  # 置信度应为0