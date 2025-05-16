#!/usr/bin/env python
# filepath: /home/ydzat/workspace/knowforge/scripts/theme_preview.py
"""
主题预览和测试工具

此脚本用于预览和测试KnowForge的输出主题。
它可以生成带有不同主题的HTML输出预览。
"""

import os
import sys
import argparse
import yaml
import shutil
from datetime import datetime

# 添加项目根目录到路径，以便导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.utils.config_loader import ConfigLoader
    from src.utils.locale_manager import LocaleManager
    from src.note_generator.output_writer import OutputWriter
except ImportError as e:
    print(f"导入模块错误: {e}")
    print("确保您正在从项目根目录运行此脚本")
    sys.exit(1)

# 示例内容
SAMPLE_CONTENT = """
# KnowForge 主题预览

这是一个示例文档，用于展示主题的渲染效果。

## 1. 标题样式

### 1.1 三级标题

#### 1.1.1 四级标题

## 2. 文本样式

这是普通文本段落。这是普通文本段落。这是普通文本段落。这是普通文本段落。这是普通文本段落。
这是普通文本段落。这是普通文本段落。这是普通文本段落。这是普通文本段落。这是普通文本段落。

**这是粗体文本** 和 *这是斜体文本*。

~~这是删除线文本~~ 和 `这是行内代码`。

> 这是引用文本块。
> 这是引用的第二行。

## 3. 列表样式

### 无序列表

- 项目一
- 项目二
  - 子项目 2.1
  - 子项目 2.2
- 项目三

### 有序列表

1. 第一步
2. 第二步
   1. 子步骤 2.1
   2. 子步骤 2.2
3. 第三步

## 4. 代码块样式

```python
def hello_world():
    print("Hello, KnowForge!")
    return True

# 这是一个Python代码示例
class ExampleClass:
    def __init__(self):
        self.value = 42
        
    def get_value(self):
        return self.value
```

## 5. 表格样式

| 名称 | 类型 | 描述 |
|------|------|------|
| 标题 | 字符串 | 文档的标题 |
| 格式 | 枚举 | 输出格式 (HTML, PDF, 等) |
| 大小 | 数字 | 文档大小 (KB) |
| 启用 | 布尔 | 是否启用此功能 |

## 6. 数学公式样式

行内公式：$E = mc^2$

块级公式：

$$
\\frac{\\partial f}{\\partial x} = 2x
$$

$$
\\begin{aligned}
a &= b + c \\\\
&= d + e
\\end{aligned}
$$

## 7. 链接和图像

[这是一个链接](https://knowforge.example.com)

"""

DEFAULT_THEMES = {
    "default": "默认主题",
    "dark": "暗黑主题",
    "light": "明亮主题",
    "minimal": "简约主题"
}

def create_temp_config(theme_name, custom_styles=None):
    """创建临时配置文件"""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "config")
    
    # 确保配置目录存在
    os.makedirs(config_dir, exist_ok=True)
    
    # 创建临时配置文件路径
    temp_config_path = os.path.join(config_dir, "temp_output_config.yaml")
    
    # 基础配置
    config = {
        "global": {
            "language": "zh",
            "generate_toc": True,
            "show_source": True,
            "show_timestamp": True,
            "show_footer": True,
            "footer_text": "由KnowForge主题预览工具生成 - {version}"
        },
        "html": {
            "theme": theme_name,
            "code_highlight_theme": "github",
            "resources": {
                "use_cdn": True
            }
        }
    }
    
    # 添加自定义样式
    if custom_styles:
        config["html"]["styles"] = custom_styles
    
    # 写入临时配置文件
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    return temp_config_path

def preview_theme(theme, custom_styles=None):
    """生成指定主题的预览"""
    try:
        # 创建临时配置文件
        temp_config_path = create_temp_config(theme, custom_styles)
        
        # 检查离线资源目录
        with open(temp_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # 如果使用离线资源，确保资源文件存在
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(script_dir)
        asset_dir = os.path.join(workspace_dir, "resources", "assets")
        
        if not config_data.get('html', {}).get('resources', {}).get('use_cdn', True):
            if not os.path.exists(asset_dir) or not os.listdir(asset_dir):
                print(f"警告: 配置为使用离线资源，但资源目录 {asset_dir} 不存在或为空")
                print("请先运行 scripts/download_resources.py 下载离线资源")
                print("或者设置 use_cdn: true 以使用在线资源")
                print("将自动切换到在线CDN资源...")
                
                # 修改配置为使用在线资源
                config_data['html']['resources']['use_cdn'] = True
                with open(temp_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)
        
        # 加载配置
        config = ConfigLoader(temp_config_path)
        
        # 设置locale和输出目录
        locale_path = os.path.join(workspace_dir, "resources", "locales")
        locale = LocaleManager("zh", locale_path)
        
        # 设置输出目录
        output_dir = os.path.join(workspace_dir, "output")
        
        # 创建OutputWriter实例
        writer = OutputWriter(workspace_dir, output_dir, config, locale)
        
        # 生成HTML预览
        file_name = f"theme_preview_{theme}"
        if custom_styles:
            file_name = "theme_preview_custom"
            
        output_path = writer.generate_html([SAMPLE_CONTENT], file_name, f"KnowForge主题预览: {theme}")
        
        print(f"主题预览已生成: {output_path}")
        
        # 清理临时文件
        os.remove(temp_config_path)
        
        return output_path
    
    except Exception as e:
        print(f"生成主题预览时出错: {e}")
        return None
    
def main():
    parser = argparse.ArgumentParser(description="KnowForge主题预览工具")
    parser.add_argument("--theme", type=str, default="default", 
                        help="要预览的主题名称 (default, dark, light, minimal)")
    parser.add_argument("--list", action="store_true", help="列出所有可用的主题")
    parser.add_argument("--preview-all", action="store_true", help="预览所有内置主题")
    parser.add_argument("--config", type=str, help="自定义配置文件路径")
    parser.add_argument("--open", action="store_true", help="生成后在浏览器中打开预览")
    
    args = parser.parse_args()
    
    if args.list:
        print("可用的主题:")
        for name, desc in DEFAULT_THEMES.items():
            print(f"  - {name}: {desc}")
        return
    
    if args.preview_all:
        print("正在生成所有主题的预览...")
        results = []
        
        for theme in DEFAULT_THEMES.keys():
            print(f"正在生成 '{theme}' 主题...")
            output_path = preview_theme(theme)
            if output_path:
                results.append((theme, output_path))
                
        print("\n所有主题预览生成完成:")
        for theme, path in results:
            print(f"  - {theme}: {path}")
            
        if args.open and results:
            try:
                import webbrowser
                for _, path in results:
                    webbrowser.open(f"file://{path}")
            except Exception as e:
                print(f"无法打开浏览器: {e}")
        return
    
    custom_styles = None
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if "html" in config_data and "styles" in config_data["html"]:
                    custom_styles = config_data["html"]["styles"]
        except Exception as e:
            print(f"无法加载自定义配置: {e}")
            return
    
    theme = args.theme
    print(f"正在生成 '{theme}' 主题的预览...")
    output_path = preview_theme(theme, custom_styles)
    
    if output_path and args.open:
        try:
            import webbrowser
            webbrowser.open(f"file://{output_path}")
        except Exception as e:
            print(f"无法打开浏览器: {e}")

if __name__ == "__main__":
    main()
