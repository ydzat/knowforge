# KnowForge 输出配置指南

**版本**: 0.1.7-final
**更新日期**: 2025年5月16日

本文档提供了关于如何配置和自定义KnowForge输出格式的详细指南，包括最新的离线资源管理和主题预览功能。

## 目录

- [配置文件位置](#配置文件位置)
- [配置结构概览](#配置结构概览)
- [全局设置](#全局设置)
- [HTML输出配置](#html输出配置)
- [PDF输出配置](#pdf输出配置)
- [Jupyter Notebook配置](#jupyter-notebook配置)
- [Markdown输出配置](#markdown输出配置)
- [离线资源管理](#离线资源管理)
- [主题预览工具](#主题预览工具)
- [主题示例](#主题示例)
- [常见问题](#常见问题)

## 配置文件位置

默认配置文件位于：`/resources/config/output_config.yaml`

您可以在KnowForge的主配置中指定自定义配置文件路径：

```yaml
output:
  config_path: "/path/to/your/custom_output_config.yaml"
```

## 配置结构概览

配置文件分为几个主要部分：

1. **全局设置** - 适用于所有输出格式的通用设置
2. **HTML输出配置** - 控制HTML输出的外观和行为
3. **PDF输出配置** - 控制PDF生成和样式
4. **Notebook配置** - 控制Jupyter Notebook的行为
5. **Markdown配置** - 控制Markdown输出的格式

## 全局设置

```yaml
global:
  language: zh  # 默认输出语言
  generate_toc: true  # 是否生成目录
  show_source: true  # 是否显示源信息
  show_timestamp: true  # 是否显示时间戳
  show_footer: true  # 是否显示页脚
  title_format: "{title}"  # 标题格式
  footer_text: "由KnowForge v{version}生成"  # 页脚文本
```

## HTML输出配置

### 主题选择

HTML输出支持多种内置主题：

```yaml
html:
  theme: default  # 可选: default, dark, light, minimal
```

### 自定义样式

```yaml
html:
  styles:
    font_family: "Segoe UI, Tahoma, Geneva, Verdana, sans-serif"
    heading_font: "Arial, sans-serif"
    code_font: "Courier New, monospace"
    text_color: "#333333"
    background_color: "#f8f9fa"
    heading_color: "#333333"
    link_color: "#007bff"
    container_width: "900px"
    container_padding: "30px"
    table_border_color: "#dddddd"
    table_header_bg: "#f2f2f2"
```

### 资源配置

可以选择使用在线CDN资源或本地资源：

```yaml
html:
  resources:
    use_cdn: true  # 是否使用CDN资源
    # CDN资源URLs
    bootstrap_css: "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
    bootstrap_js: "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"
    mathjax_js: "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
    # 本地资源目录（仅当use_cdn为false时使用）
    local_resource_dir: "resources/assets"
```

### 代码高亮主题

```yaml
html:
  code_highlight_theme: github  # 可选: github, dracula, monokai, etc.
```

## PDF输出配置

### 页面设置

```yaml
pdf:
  page_size: "A4"  # 页面大小
  page_margin: "1.5cm"  # 页边距
  preferred_engine: "weasyprint"  # 首选的PDF生成引擎，可选: weasyprint, fpdf, pdfkit
```

### 样式设置

```yaml
pdf:
  styles:
    font_family: "sans-serif"
    heading_font: "sans-serif"
    code_font: "monospace"
    base_font_size: "11pt"
    heading_font_size: "16pt"
    text_color: "#333333"
    background_color: "#ffffff"
    heading_color: "#333333"
    table_border: "1pt solid #dddddd"
    table_header_bg: "#f7f7f7"
```

### 分页设置

```yaml
pdf:
  pagination:
    show_page_numbers: true  # 是否显示页码
    page_number_format: "%d"  # 页码格式
    page_number_position: "bottom-center"  # 页码位置
```

## Jupyter Notebook配置

### Notebook格式设置

```yaml
notebook:
  nbformat_version: 4  # Notebook版本
```

### 单元格分割策略

控制如何将内容分割成多个单元格：

```yaml
notebook:
  cell_split_strategy:
    split_on_headers: true  # 在标题处分割
    split_on_code_blocks: true  # 在代码块处分割
    split_on_tables: true  # 在表格处分割
    split_on_math_blocks: true  # 在数学公式处分割
```

### 元数据设置

```yaml
notebook:
  metadata:
    kernelspec:
      display_name: "Python 3"
      language: "python"
      name: "python3"
```

## Markdown输出配置

```yaml
markdown:
  use_extended_syntax: true  # 是否使用扩展语法
  link_style: "inline"  # 链接风格，可选: inline, reference
  code_block_style: "fenced"  # 代码块风格，可选: fenced, indented
  list_indent: 2  # 列表缩进空格数
```

## 主题示例

### 暗黑主题

```yaml
html:
  theme: dark
  styles:
    background_color: "#222222"
    text_color: "#f0f0f0"
    heading_color: "#ffffff"
    link_color: "#5caefd"
    table_header_bg: "#333333"
    table_border_color: "#444444"
```

### 简约主题

```yaml
html:
  theme: minimal
  styles:
    font_family: "Arial, sans-serif"
    background_color: "#ffffff"
    container_width: "800px"
    container_padding: "20px"
    heading_font: "Georgia, serif"
```

### 学术论文主题

```yaml
html:
  theme: default
  styles:
    font_family: "Georgia, Times, serif"
    heading_font: "Georgia, Times, serif"
    container_width: "800px"
    text_color: "#222222"

pdf:
  page_size: "A4"
  page_margin: "2cm"
  styles:
    font_family: "Times"
    heading_font: "Times"
    base_font_size: "12pt"
```

## 离线资源管理

KnowForge 0.1.7版本新增了完整的离线资源支持，允许在无网络环境下使用全部功能。

### 下载离线资源

使用提供的脚本下载所需资源：

```bash
python scripts/download_resources.py
```

这将下载Bootstrap、MathJax和代码高亮所需的CSS和JS文件到`resources/assets`目录。

### 下载特定主题

如果需要下载特定的代码高亮主题：

```bash
python scripts/download_resources.py --themes "monokai,solarized-dark,solarized-light"
```

### 下载所有资源

下载所有可用的主题和资源：

```bash
python scripts/download_resources.py --all
```

### 更新离线资源

更新已下载的资源到最新版本：

```bash
python scripts/download_resources.py --update
```

### 配置离线模式

将配置文件中的`use_cdn`设置为`false`以启用离线模式：

```yaml
html:
  resources:
    use_cdn: false
    local_resource_dir: "resources/assets"  # 本地资源目录
```

## 主题预览工具

KnowForge提供了主题预览工具，帮助您测试和查看不同主题的效果。

### 列出可用主题

```bash
python scripts/theme_preview.py --list
```

### 预览特定主题

```bash
python scripts/theme_preview.py --theme dark
```

### 预览所有主题

生成所有可用主题的预览：

```bash
python scripts/theme_preview.py --preview-all
```

### 使用自定义配置预览

```bash
python scripts/theme_preview.py --theme dark --config /path/to/custom_config.yaml
```

### 预览后在浏览器中打开

```bash
python scripts/theme_preview.py --theme dark --open
```

## 常见问题

### 如何完全禁用某些元素？

您可以通过设置全局配置来控制元素的显示：

```yaml
global:
  show_timestamp: false  # 隐藏时间戳
  show_source: false  # 隐藏源信息
  generate_toc: false  # 不生成目录
  show_footer: false  # 隐藏页脚
```

### 如何调整PDF引擎优先级？

```yaml
pdf:
  preferred_engine: "fpdf"  # 优先使用fpdf
```

### 如何处理离线资源丢失的情况？

如果配置了离线模式但资源文件缺失，系统会自动回退到使用CDN并记录警告信息。您可以随时重新运行资源下载脚本。

### 配置文件不生效怎么办？

1. 确保配置文件路径正确
2. 检查YAML语法是否有误
3. 查看日志文件中是否有配置加载相关的错误
4. 确保修改后重启或重新运行KnowForge
