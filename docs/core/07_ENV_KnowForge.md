<!--
 * @Author: @ydzat
 * @Date: 2025-04-28 18:45:10
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-04-28 18:45:10
 * @Description: 环境配置指南
-->
# KnowForge 环境配置指南

本文档提供了KnowForge项目的环境配置、依赖安装和开发环境设置的详细说明，适用于开发人员和用户。

---

## 1. 系统要求

### 操作系统支持

- **Windows**: Windows 10 或更高版本
- **Linux**: Ubuntu 20.04+, Fedora 41+
- **macOS**: 11.0 (Big Sur) 或更高版本

### 基本要求

- **Python**: 3.11 或更高版本
- **磁盘空间**: 至少 1GB 可用空间（包含依赖、模型和工作目录）
- **内存**: 建议至少 4GB RAM，处理大文档时推荐 8GB+

---

## 2. 开发环境配置

### 方法一：使用Conda（推荐）

1. **安装Anaconda或Miniconda**:
   - 下载链接: [Anaconda](https://www.anaconda.com/products/individual) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - 按照官方指南安装

2. **创建虚拟环境**:
   ```bash
   # 创建新环境
   conda create -n knowforge python=3.11

   # 激活环境
   conda activate knowforge
   ```

3. **克隆项目**:
   ```bash
   git clone https://github.com/yourusername/knowforge.git
   cd knowforge
   ```

4. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

### 方法二：使用venv

1. **安装Python 3.11+**:
   - 从 [Python官网](https://www.python.org/downloads/) 下载安装

2. **创建虚拟环境**:
   ```bash
   # 在项目目录中创建venv
   python -m venv venv

   # 激活虚拟环境
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. 依赖包说明

KnowForge项目的依赖分为几个主要类别：

### 核心依赖

- **python-dotenv (≥1.0.0)**: 环境变量管理
- **pyyaml (≥6.0)**: YAML文件处理
- **typer (≥0.9.0)**: CLI接口创建

### 输入处理依赖

- **pdfplumber (≥0.10.0)**: PDF文本提取
- **easyocr (≥1.7.0)**: 图片OCR文本识别
- **beautifulsoup4 (≥4.12.0)**: 网页内容提取
- **requests (≥2.31.0)**: HTTP请求处理
- **pygments (≥2.16.0)**: 代码语法高亮

### 向量化和记忆依赖

- **sentence-transformers (≥2.3.0)**: 文本向量化
- **chromadb (≥0.4.0)**: 向量数据库

### LLM接口

- **openai (≥1.0.0)**: OpenAI兼容API调用

### 输出生成依赖

- **markdown-it-py (≥3.0.0)**: Markdown处理
- **nbformat (≥5.9.0)**: Jupyter Notebook格式处理
- **weasyprint (≥60.0)**: PDF生成

### 测试依赖

- **pytest (≥7.4.0)**: 单元测试框架
- **pytest-cov (≥4.1.0)**: 测试覆盖率统计
- **pytest-mock (≥3.11.0)**: 测试中的模拟功能

---

## 4. 环境变量配置

KnowForge使用环境变量来管理API密钥等敏感信息。**强烈建议**使用环境变量而非直接在配置文件中硬编码这些敏感信息。

### 4.1 配置API密钥（推荐方式）

**方法1：直接设置系统环境变量（更安全）**

```bash
# Linux/macOS
export DEEPSEEK_API_KEY="your-api-key-here"

# Windows (CMD)
set DEEPSEEK_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="your-api-key-here"
```

这种方式更安全，API密钥不会被保存在任何文件中。对于持久化设置，您可以将上述命令添加到您的shell配置文件中（如`.bashrc`、`.zshrc`等）。

**方法2：使用`.env`文件**

如果您需要一个更便携的方式管理环境变量：

1. **创建`.env`文件**:
   - 在项目根目录创建`.env`文件
   ```bash
   touch .env
   # 或对于Windows
   echo. > .env
   ```

2. **编辑`.env`文件**:
   ```dotenv
   # DeepSeek API Key (必需)
   DEEPSEEK_API_KEY=your-api-key-here
   
   # 如果您使用OpenAI API (可选)
   # OPENAI_API_KEY=your-openai-key-here
   
   # API基础URL (可选，使用默认值)
   # DEEPSEEK_API_BASE_URL=https://api.deepseek.com
   ```

> **安全提示**: 
> - `.env` 文件包含敏感信息，已在 `.gitignore` 中排除，确保永远不要将其提交到代码仓库。
> - 请妥善保管您的API密钥，不要在公共场合或共享环境中暴露它。
> - 在生产环境中，优先使用系统环境变量或安全的密钥管理服务。

### 4.2 密钥统一管理说明

KnowForge设计为只需设置**一次**API密钥，即可在全部组件中使用。这是通过`ConfigLoader`实现的：

- 系统优先从环境变量获取密钥
- 所有需要LLM功能的组件都使用相同的方式获取密钥
- 项目支持多种LLM提供商，会根据配置自动使用正确的环境变量

例如，如果您设置了`DEEPSEEK_API_KEY`环境变量，那么从文本拆分器到笔记生成器的所有组件都会使用这个密钥，无需重复配置。

---

## 5. 目录结构初始化

项目要求以下目录结构存在并可写入:

```bash
# 创建必要的目录结构
mkdir -p input/{pdf,images,codes,links}
mkdir -p workspace/{preprocessed,split_segments,embeddings,memory_db}
mkdir -p output/{markdown,notebook,pdf,logs}
```

这些目录会在首次运行时自动创建，但您也可以手动创建它们。

---

## 6. 运行验证

安装完成后，运行以下命令验证环境是否正确配置：

```bash
# 运行CLI帮助
python gen_notes.py --help

# 运行版本命令
python gen_notes.py version

# 运行单元测试
pytest tests/
```

如果上述命令成功执行，则表示环境配置正确。

---

## 7. 常见问题与解决方案

1. **找不到模块错误**:
   ```
   ModuleNotFoundError: No module named 'src'
   ```
   解决方案: 确保在项目根目录运行命令，或将项目根目录添加到PYTHONPATH。

2. **EasyOCR依赖问题**:
   - Windows上可能需要安装Visual C++ Build Tools
   - Linux可能需要额外的系统库: `apt-get install libgl1`

3. **ChromaDB兼容性问题**:
   - 确保使用兼容版本: `pip install chromadb==0.4.0`
   - 某些系统可能需要额外的系统依赖: `apt-get install sqlite3`

4. **WeasyPrint依赖问题**:
   - Linux: `apt-get install libpango1.0-dev libharfbuzz-dev libffi-dev`
   - Windows: 按照[官方指南](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows)安装GTK+

---

## 8. 如何升级

当有新版本发布时，按照以下步骤升级:

```bash
# 拉取最新代码
git pull origin main

# 更新依赖
pip install -r requirements.txt --upgrade
```

---

## 9. 性能优化建议

- **GPU加速**: 如果您的环境有GPU，可以安装PyTorch的CUDA版本以加速向量化处理:
  ```bash
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  ```

- **内存优化**: 处理大型文档时，可以调整`config.yaml`中的`chunk_size`参数，减小值以降低内存消耗。