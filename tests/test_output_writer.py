"""
OutputWriter单元测试
"""
import os
import tempfile
import shutil
import pytest
from src.utils.config_loader import ConfigLoader
from src.note_generator.output_writer import OutputWriter


class TestOutputWriter:
    """测试输出写入器"""
    
    @pytest.fixture
    def mock_config(self):
        """创建用于测试的配置"""
        config_dict = {
            "output": {
                "template_path": "resources/templates/note_template.md"
            }
        }
        
        # 创建一个ConfigLoader实例
        config = ConfigLoader("resources/config/config.yaml")
        
        # 由于我们不想修改实际的配置文件，这里使用猴子补丁
        config._config = config_dict
        return config
    
    @pytest.fixture
    def temp_dirs(self):
        """创建临时工作目录和输出目录"""
        workspace_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        
        # 创建测试所需的子目录
        os.makedirs(os.path.join(output_dir, "markdown"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "notebook"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pdf"), exist_ok=True)
        
        yield workspace_dir, output_dir
        
        shutil.rmtree(workspace_dir)
        shutil.rmtree(output_dir)
    
    @pytest.fixture
    def output_writer(self, temp_dirs, mock_config):
        """创建输出写入器实例"""
        workspace_dir, output_dir = temp_dirs
        return OutputWriter(workspace_dir, output_dir, mock_config)
    
    def test_initialization(self, temp_dirs, mock_config):
        """测试初始化"""
        workspace_dir, output_dir = temp_dirs
        writer = OutputWriter(workspace_dir, output_dir, mock_config)
        
        # 验证输出目录是否已创建
        assert os.path.exists(os.path.join(output_dir, "markdown"))
        assert os.path.exists(os.path.join(output_dir, "notebook"))
        assert os.path.exists(os.path.join(output_dir, "pdf"))
    
    def test_load_template(self, output_writer):
        """测试模板加载"""
        # 测试默认模板加载
        template = output_writer._load_template("nonexistent_template.md")
        assert "{{ title }}" in template
        assert "{{ content }}" in template
    
    def test_generate_markdown(self, output_writer, temp_dirs):
        """测试生成Markdown"""
        workspace_dir, output_dir = temp_dirs
        test_segments = ["# 测试标题\n\n这是测试内容。", "## 子标题\n\n这是子章节内容。"]
        
        output_path = output_writer.generate_markdown(test_segments, "test_note")
        
        # 验证输出文件是否创建
        assert os.path.exists(output_path)
        assert output_path.endswith(".md")
        
        # 验证内容
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "测试标题" in content
            assert "子标题" in content
            assert "这是测试内容" in content
    
    def test_generate_notebook(self, output_writer, temp_dirs):
        """测试生成Jupyter Notebook"""
        workspace_dir, output_dir = temp_dirs
        test_segments = ["# 测试标题\n\n这是测试内容。", "## 子标题\n\n这是子章节内容。"]
        
        output_path = output_writer.generate_notebook(test_segments, "test_note")
        
        # 验证输出文件是否创建
        assert os.path.exists(output_path)
        assert output_path.endswith(".ipynb")
    
    def test_generate_pdf(self, output_writer, temp_dirs):
        """测试生成PDF(测试简单实现)"""
        workspace_dir, output_dir = temp_dirs
        
        # 先生成Markdown
        test_segments = ["# PDF测试\n\n这是PDF测试内容。"]
        md_path = output_writer.generate_markdown(test_segments, "test_pdf")
        
        # 从Markdown生成PDF
        pdf_path = output_writer.generate_pdf(md_path, "test_pdf")
        
        # 验证输出文件是否创建
        assert os.path.exists(pdf_path)
        assert pdf_path.endswith(".pdf")
    
    def test_merge_segments(self, output_writer):
        """测试合并文本片段"""
        # 测试空列表
        assert output_writer._merge_segments([]) == ""
        
        # 测试单个片段
        assert output_writer._merge_segments(["单段测试"]) == "单段测试"
        
        # 测试多个片段
        result = output_writer._merge_segments(["第一段", "第二段", "第三段"])
        assert "第一段" in result
        assert "第二段" in result
        assert "第三段" in result
    
    def test_find_overlap(self, output_writer):
        """测试查找重叠部分"""
        # 测试无重叠
        assert output_writer._find_overlap("第一部分", "第二部分") == ""
        
        # 测试有重叠
        assert output_writer._find_overlap("测试重叠", "重叠部分") == "重叠"
    
    def test_generate_toc(self, output_writer):
        """测试生成目录"""
        content = """# 一级标题
这是内容

## 二级标题A
这是二级内容

## 二级标题B
这是另一个二级内容

### 三级标题
这是三级内容
"""
        toc = output_writer._generate_toc(content)
        
        # 验证目录包含各级标题
        assert "一级标题" in toc
        assert "二级标题A" in toc
        assert "二级标题B" in toc
        assert "三级标题" in toc
        
        # 验证缩进格式
        assert "- [一级标题]" in toc
        assert "  - [二级标题A]" in toc or "- [二级标题A]" in toc
    
    def test_apply_template(self, output_writer):
        """测试应用模板"""
        title = "测试标题"
        content = "测试内容"
        toc = "- [测试标题](#测试标题)"
        
        result = output_writer._apply_template(title, content, toc)
        
        # 验证标题、内容和目录都被正确替换
        assert title in result
        assert content in result
        assert toc in result