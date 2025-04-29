"""
Processor集成测试 - 测试核心处理器与各模块的集成
"""
import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
from src.utils.config_loader import ConfigLoader
from src.note_generator.processor import Processor
from src.note_generator.input_handler import InputHandler
from src.note_generator.splitter import Splitter
from src.note_generator.output_writer import OutputWriter


class TestProcessorIntegration:
    """测试处理器集成功能"""
    
    @pytest.fixture
    def temp_dirs(self):
        """创建临时工作目录结构"""
        base_dir = tempfile.mkdtemp()
        
        # 创建基本目录结构
        input_dir = os.path.join(base_dir, "input")
        output_dir = os.path.join(base_dir, "output")
        workspace_dir = os.path.join(base_dir, "workspace")
        config_dir = os.path.join(base_dir, "resources", "config")
        
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        os.makedirs(workspace_dir)
        os.makedirs(config_dir)
        
        # 创建输入子目录
        os.makedirs(os.path.join(input_dir, "pdf"))
        os.makedirs(os.path.join(input_dir, "codes"))
        os.makedirs(os.path.join(input_dir, "images"))
        os.makedirs(os.path.join(input_dir, "links"))
        
        # 创建配置文件
        config_path = os.path.join(config_dir, "config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("""
system:
  language: zh
  workspace_dir: workspace/
  output_dir: output/
  
splitter:
  chunk_size: 800
  overlap_size: 100
  
output:
  formats:
    - markdown
""")
        
        # 返回各个路径
        paths = {
            "base_dir": base_dir,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "workspace_dir": workspace_dir,
            "config_path": config_path
        }
        
        yield paths
        
        # 清理临时目录
        shutil.rmtree(base_dir)
    
    def test_processor_initialization(self, temp_dirs):
        """测试处理器初始化"""
        processor = Processor(
            temp_dirs["input_dir"],
            temp_dirs["output_dir"],
            temp_dirs["config_path"]
        )
        
        # 验证各个组件是否已初始化
        assert processor.input_handler is not None
        assert processor.splitter is not None
        assert processor.output_writer is not None
    
    def test_run_full_pipeline_empty(self, temp_dirs):
        """测试运行空输入的处理流程"""
        processor = Processor(
            temp_dirs["input_dir"],
            temp_dirs["output_dir"],
            temp_dirs["config_path"]
        )
        
        # 运行流程，应该返回空结果（因为没有输入文件）
        result = processor.run_full_pipeline()
        assert result == {}
    
    @patch.object(InputHandler, 'extract_texts')
    @patch.object(Splitter, 'split_text')
    @patch.object(OutputWriter, 'generate_markdown')
    def test_run_full_pipeline_mock(self, mock_generate_markdown, mock_split_text, mock_extract_texts, temp_dirs):
        """测试完整处理流程（使用模拟组件）"""
        # 设置模拟行为
        mock_extract_texts.return_value = ["测试文本1", "测试文本2"]
        mock_split_text.return_value = ["拆分后文本1", "拆分后文本2", "拆分后文本3"]
        mock_generate_markdown.return_value = os.path.join(temp_dirs["output_dir"], "markdown", "notes.md")
        
        processor = Processor(
            temp_dirs["input_dir"],
            temp_dirs["output_dir"],
            temp_dirs["config_path"]
        )
        
        # 运行流程
        result = processor.run_full_pipeline(["markdown"])
        
        # 验证各组件是否被调用
        mock_extract_texts.assert_called_once()
        mock_split_text.assert_called_once_with(["测试文本1", "测试文本2"])
        mock_generate_markdown.assert_called_once_with(["拆分后文本1", "拆分后文本2", "拆分后文本3"], "notes")
        
        # 验证结果
        assert "markdown" in result
        assert result["markdown"] == os.path.join(temp_dirs["output_dir"], "markdown", "notes.md")
    
    def test_process_file_integration(self, temp_dirs):
        """测试从文件生成笔记（实际集成）"""
        # 创建测试文件
        test_file = os.path.join(temp_dirs["input_dir"], "codes", "test.py")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("""# 测试Python文件
def hello_world():
    print("Hello, World!")
    
# 这是示例代码
hello_world()
""")
        
        processor = Processor(
            temp_dirs["input_dir"],
            temp_dirs["output_dir"],
            temp_dirs["config_path"]
        )
        
        # 运行单文件处理
        output_path = processor.generate_note(test_file)
        
        # 验证结果
        assert os.path.exists(output_path)
        assert output_path.endswith(".md")
        
        # 验证内容
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "测试Python文件" in content
            assert "hello_world" in content
    
    def test_multiple_output_formats(self, temp_dirs):
        """测试生成多种输出格式"""
        # 创建测试文件
        test_file = os.path.join(temp_dirs["input_dir"], "codes", "test.py")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("# 测试文件\nprint('多格式输出测试')")
        
        # 使用模拟配置，避免调用实际的API
        with patch.object(InputHandler, 'extract_texts') as mock_extract:
            mock_extract.return_value = ["# 测试标题\n测试内容"]
            
            processor = Processor(
                temp_dirs["input_dir"],
                temp_dirs["output_dir"],
                temp_dirs["config_path"]
            )
            
            # 运行多格式输出
            results = processor.run_full_pipeline(["markdown", "ipynb"])
            
            # 验证结果包含多种格式
            assert "markdown" in results
            assert "ipynb" in results
            
            # 验证文件是否存在
            assert os.path.exists(results["markdown"])
            assert os.path.exists(results["ipynb"])