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
        """测试生成增强的PDF"""
        workspace_dir, output_dir = temp_dirs
        
        # 创建测试内容，包含表格和公式
        test_segments = [
            "# PDF增强测试\n\n这是PDF测试内容，包含表格和公式。",
            "## 表格示例\n\n| 列1 | 列2 | 列3 |\n| --- | --- | --- |\n| 数据1 | 数据2 | 数据3 |",
            "## 数学公式\n\n$E=mc^2$\n\n$$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$"
        ]
        
        # 从增强内容生成PDF
        pdf_path = output_writer.generate_pdf(test_segments, "test_enhanced_pdf")
        
        # 验证输出文件是否创建
        assert os.path.exists(pdf_path)
        # 检查文件扩展名 - 可能是.pdf或.txt（取决于是否安装了相关库）
        assert pdf_path.endswith(".pdf") or pdf_path.endswith(".txt")
    
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

    def test_generate_html(self, output_writer, temp_dirs):
        """测试生成HTML格式输出"""
        workspace_dir, output_dir = temp_dirs
        
        # 创建测试内容，包含各种格式元素
        test_segments = [
            "# HTML测试文档\n\n这是HTML测试内容，包含**粗体**和*斜体*文本。",
            "## 表格示例\n\n| 列1 | 列2 | 列3 |\n| --- | --- | --- |\n| 数据1 | 数据2 | 数据3 |\n| 行2数据1 | 行2数据2 | 行2数据3 |",
            "## 代码示例\n\n```python\ndef hello_world():\n    print('Hello, World!')\n```",
            "## 数学公式示例\n\n行内公式：$E=mc^2$\n\n块级公式：\n\n$$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$"
        ]
        
        # 生成HTML文件
        output_path = output_writer.generate_html(test_segments, "test_html")
        
        # 验证输出文件是否创建
        assert os.path.exists(output_path)
        assert output_path.endswith(".html")
        
        # 验证内容
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # 检查基本元素是否存在
            assert "<html" in content
            assert "<title>Test Html</title>" in content
            assert "HTML测试文档" in content
            
            # 检查表格
            # 可能有两种处理方式：
            # 1. 直接从Markdown渲染成HTML表格的<table>元素
            # 2. 或者在简单转换模式下保留原始Markdown表格格式
            assert ("<table" in content) or ("| 列1 | 列2 | 列3 |" in content)
            
            # 检查数学公式支持 - MathJax引用应该存在
            assert "MathJax" in content
            
            # 检查代码高亮支持
            assert "highlight.js" in content
    
    def test_generate_enhanced_notebook(self, output_writer, temp_dirs):
        """测试增强的Jupyter Notebook生成"""
        workspace_dir, output_dir = temp_dirs
        
        # 创建测试内容，包含各种格式元素
        test_segments = [
            "# Notebook测试文档\n\n这是Notebook测试内容。",
            "## 包含表格的章节\n\n| 列1 | 列2 |\n| --- | --- |\n| 数据1 | 数据2 |",
            "## 包含代码的章节\n\n```python\nprint('Hello')\n```",
            "## 包含公式的章节\n\n$$E=mc^2$$"
        ]
        
        # 生成Notebook文件
        output_path = output_writer.generate_notebook(test_segments, "test_notebook")
        
        # 验证输出文件是否创建
        assert os.path.exists(output_path)
        assert output_path.endswith(".ipynb")
        
        # 验证基本结构
        import json
        with open(output_path, "r", encoding="utf-8") as f:
            nb_content = json.load(f)
            
            # 检查是否有多个单元格(至少应该有标题单元格和内容单元格)
            assert len(nb_content.get("cells", [])) > 1
            
            # 检查第一个单元格是否包含标题
            first_cell = nb_content.get("cells", [])[0]
            assert "source" in first_cell
            source = "".join(first_cell["source"]) if isinstance(first_cell["source"], list) else first_cell["source"]
            assert "Notebook测试文档" in source or "Test Notebook" in source

    def test_enhanced_notebook(self, output_writer, temp_dirs):
        """测试增强的Jupyter Notebook生成"""
        workspace_dir, output_dir = temp_dirs
        
        # 创建测试内容，包含各种格式元素
        test_segments = [
            "# Notebook测试文档\n\n这是Notebook测试内容。",
            "## 包含表格的章节\n\n| 列1 | 列2 |\n| --- | --- |\n| 数据1 | 数据2 |",
            "## 包含代码的章节\n\n```python\nprint('Hello')\n```",
            "## 包含公式的章节\n\n$$E=mc^2$$"
        ]
        
        # 生成Notebook文件
        output_path = output_writer.generate_notebook(test_segments, "test_enhanced_notebook")
        
        # 验证输出文件是否创建
        assert os.path.exists(output_path)
        assert output_path.endswith(".ipynb")
        
        # 验证基本结构
        import json
        with open(output_path, "r", encoding="utf-8") as f:
            nb_content = json.load(f)
            
            # 检查是否有多个单元格(至少应该有标题单元格和内容单元格)
            assert len(nb_content.get("cells", [])) > 1
            
            # 检查第一个单元格是否包含标题
            first_cell = nb_content.get("cells", [])[0]
            assert "source" in first_cell
            if isinstance(first_cell["source"], list):
                source = "".join(first_cell["source"])
            else:
                source = first_cell["source"]
            assert "Notebook测试文档" in source or "Test Enhanced Notebook" in source
    
    def test_convert_toc_to_html(self, output_writer):
        """测试将目录转换为HTML格式"""
        # 创建简单的Markdown格式目录
        md_toc = "- [标题1](#标题1)\n  - [子标题1.1](#子标题1-1)\n- [标题2](#标题2)"
        
        # 转换为HTML
        html_toc = output_writer._convert_toc_to_html(md_toc)
        
        # 不管使用什么渲染方式，确保关键信息存在
        assert "标题1" in html_toc
        assert "标题2" in html_toc
        assert "子标题1.1" in html_toc
        
        # 验证链接格式 - 有可能是HTML <a> 标签或保持Markdown格式
        assert ('<a href="#标题1">标题1</a>' in html_toc) or ("[标题1](#标题1)" in html_toc)
    
    def test_markdown_to_html(self, output_writer):
        """测试将Markdown转换为HTML格式"""
        # 创建包含多种格式元素的Markdown文本
        markdown_text = """# 测试标题
        
## 子标题
        
这是**粗体**和*斜体*文本。
        
这是`行内代码`。
        
```python
def test():
    print("代码块")
```
        
| 列1 | 列2 |
|-----|-----|
| 数据1 | 数据2 |
        
数学公式: $E=mc^2$
        """
        
        # 转换为HTML
        html = output_writer._markdown_to_html(markdown_text)
        
        # 检查基本内容是否被保留 - 不管用什么方式渲染
        assert "测试标题" in html
        assert "子标题" in html
        assert "粗体" in html
        assert "斜体" in html
        assert "行内代码" in html
        
        # 检查代码块 - 可能被高亮处理，所以只检查"代码块"文本
        assert "代码块" in html
        
        # 检查函数定义 - 可能会被处理成不同格式，所以只检查原始文本是否存在
        assert ("def" in html and "test" in html) or "def test" in html
        assert "列1" in html
        assert "数据1" in html
        assert "E=mc" in html  # 数学公式
    
    def test_split_content_for_notebook(self, output_writer):
        """测试将内容分割成适合Notebook的单元格"""
        # 创建包含多种元素的内容
        content = """# 标题1
        
这是一些文本内容。

## 子标题
        
| 列1 | 列2 |
| --- | --- |
| 数据1 | 数据2 |
        
```python
print("Hello")
```
        
$$E=mc^2$$
        """
        
        # 分割内容
        cells = output_writer._split_content_for_notebook(content)
        
        # 验证结果
        assert len(cells) > 1  # 应该至少分割成几个单元格
        
        # 查找各类型的单元格
        found_header = False
        found_table = False
        found_code = False
        found_math = False
        
        for cell in cells:
            if "# 标题1" in cell:
                found_header = True
            if "| 列1 | 列2 |" in cell:
                found_table = True
            if "```python" in cell:
                found_code = True
            if "$$E=mc^2$$" in cell:
                found_math = True
        
        # 每种类型应该至少找到一个
        assert found_header
        assert found_table
        assert found_code
        assert found_math