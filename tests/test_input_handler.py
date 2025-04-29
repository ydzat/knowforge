"""
InputHandler单元测试
"""
import os
import tempfile
import shutil
import pytest
from src.utils.config_loader import ConfigLoader
from src.note_generator.input_handler import InputHandler
from src.utils.exceptions import InputError


class TestInputHandler:
    """测试输入处理器"""

    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        return ConfigLoader("resources/config/config.yaml")

    @pytest.fixture
    def temp_workspace(self):
        """创建临时工作目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_input_dir(self):
        """创建临时输入目录"""
        temp_dir = tempfile.mkdtemp()
        
        # 创建目录结构
        os.makedirs(os.path.join(temp_dir, "pdf"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "codes"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "links"), exist_ok=True)
        
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_initialization(self, temp_input_dir, temp_workspace, mock_config):
        """测试初始化"""
        handler = InputHandler(temp_input_dir, temp_workspace, mock_config)
        
        # 验证预处理目录是否创建
        assert os.path.exists(os.path.join(temp_workspace, "preprocessed"))
        assert os.path.exists(os.path.join(temp_workspace, "preprocessed", "pdfs"))
        assert os.path.exists(os.path.join(temp_workspace, "preprocessed", "codes"))
        assert os.path.exists(os.path.join(temp_workspace, "preprocessed", "images"))
        assert os.path.exists(os.path.join(temp_workspace, "preprocessed", "links"))
    
    def test_scan_inputs_empty(self, temp_input_dir, temp_workspace, mock_config):
        """测试扫描空输入目录"""
        handler = InputHandler(temp_input_dir, temp_workspace, mock_config)
        result = handler.scan_inputs()
        
        # 验证返回结果
        assert len(result["pdf"]) == 0
        assert len(result["images"]) == 0
        assert len(result["codes"]) == 0
        assert len(result["links"]) == 0
    
    def test_scan_inputs_with_files(self, temp_input_dir, temp_workspace, mock_config):
        """测试扫描含有文件的输入目录"""
        # 创建测试文件
        pdf_path = os.path.join(temp_input_dir, "pdf", "test.pdf")
        with open(pdf_path, "w") as f:
            f.write("PDF content")
            
        code_path = os.path.join(temp_input_dir, "codes", "test.py")
        with open(code_path, "w") as f:
            f.write("print('Hello')")
        
        # 创建链接文件
        link_path = os.path.join(temp_input_dir, "links", "links.txt")
        with open(link_path, "w") as f:
            f.write("https://example.com\n")
        
        handler = InputHandler(temp_input_dir, temp_workspace, mock_config)
        result = handler.scan_inputs()
        
        # 验证扫描结果
        assert len(result["pdf"]) == 1
        assert pdf_path in result["pdf"]
        
        assert len(result["codes"]) == 1
        assert code_path in result["codes"]
        
        assert len(result["links"]) == 1
        assert link_path in result["links"]
    
    def test_extract_text_from_code(self, temp_input_dir, temp_workspace, mock_config):
        """测试从代码文件提取文本"""
        code_path = os.path.join(temp_input_dir, "codes", "test.py")
        code_content = "def hello():\n    print('Hello, World!')"
        
        with open(code_path, "w") as f:
            f.write(code_content)
            
        handler = InputHandler(temp_input_dir, temp_workspace, mock_config)
        result = handler.process_file(code_path)
        
        # 验证提取的文本
        assert result == code_content
    
    def test_invalid_file(self, temp_input_dir, temp_workspace, mock_config):
        """测试处理不存在的文件"""
        handler = InputHandler(temp_input_dir, temp_workspace, mock_config)
        
        with pytest.raises(InputError):
            handler.process_file("nonexistent.txt")
    
    def test_file_size_check(self, temp_input_dir, temp_workspace, mock_config):
        """测试文件大小检查"""
        # 创建一个很大的（假设限制为100MB）临时文件
        # 实际测试中，我们只创建一个小文件并检查逻辑
        handler = InputHandler(temp_input_dir, temp_workspace, mock_config)
        
        # 验证检查逻辑
        assert handler._check_file_valid(__file__)  # 当前测试文件应该有效
        
        # 验证无效扩展名
        invalid_ext_file = os.path.join(temp_input_dir, "test.invalid")
        with open(invalid_ext_file, "w") as f:
            f.write("test")
        
        assert not handler._check_file_valid(invalid_ext_file)
    
    def test_get_link_name(self, temp_input_dir, temp_workspace, mock_config):
        """测试从URL生成安全的文件名"""
        handler = InputHandler(temp_input_dir, temp_workspace, mock_config)
        
        # 测试URL转换为安全文件名
        assert handler._get_link_name("https://example.com") == "example.com"
        assert handler._get_link_name("http://test.com/path?query=1") == "test.com_path_query=1"
        
        # 测试特殊字符处理
        assert "?" not in handler._get_link_name("http://test.com/?q=test")
        assert "<" not in handler._get_link_name("http://test.com/<test>")
        assert ">" not in handler._get_link_name("http://test.com/<test>")