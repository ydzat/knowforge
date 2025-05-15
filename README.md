# KnowForge

简体中文 | [English](README_EN.md)

![版本](https://img.shields.io/badge/版本-0.1.6-blue)
![许可证](https://img.shields.io/badge/许可证-GPL--3.0-green)
![Python](https://img.shields.io/badge/Python-3.11+-yellow)

**KnowForge** 是一个基于人工智能的自动化学习笔记生成器，能够从多种输入源（PDF、图片、代码、网页链接）整合处理，生成结构化的笔记，支持多种格式输出。

## 特性

- **多源输入支持**：处理PDF文档、图片（OCR）、网页链接、代码文件
- **智能文本拆分**：根据章节/段落自动拆分大文档
- **记忆管理系统**：通过向量化和ChromaDB实现上下文记忆和检索
- **AI生成增强**：使用大语言模型生成摘要和笔记内容，支持多种LLM接口
- **高级OCR处理**：结合LLM和知识库的OCR结果校正和增强
- **多格式输出**：支持Markdown、Jupyter Notebook、PDF格式
- **高级日志与国际化系统**：基于Logloom的全面国际化支持和日志系统
- **多语言支持**：支持中文和英文界面
- **跨平台兼容**：支持Windows和Linux系统

## 即将推出的功能

- **文档综合处理**：自动识别PDF中的文本、图像、表格和公式，并应用适当的处理方法
- **表格与公式专项处理**：提供表格数据提取和数学公式转LaTeX的功能
- **内容整合与格式保留**：确保在最终笔记中保留原始文档结构和格式

## 开发进度

参考设计文档[06_ITER_KnowForge.md](./docs/06_ITER_KnowForge.md)，当前已完成迭代2（核心功能实现）与迭代3（Logloom日志系统集成），正在进行迭代4（高级功能与优化）的开发。

最新里程碑：
- **2025年5月16日 (v0.1.6)**：完成高级PDF图像提取和OCR增强功能，实现多方法冗余提取策略和LLM增强OCR结果，提取成功率达到100%。同时完善高级记忆管理系统，实现记忆访问统计更新和工作记忆优化。
- **2025年5月14日 (v0.1.5)**：规划文档综合处理功能，设计DocumentAnalyzer、ContentExtractor等核心组件，准备开发PDF内容自动识别能力。
- **2025年5月10日 (v0.1.4)**：实现OCR-LLM-知识库集成，开发了EmbeddingManager和LLMCaller模块，大幅提升了图像文本识别质量。
- **2025年5月8日 (v0.1.3)**：将LocaleManager完全迁移至Logloom，优化了国际化资源加载机制，实现了键名智能解析功能。
- **2025年5月5日 (v0.1.2)**：向量记忆管理模块优化，修复了与ChromaDB API的兼容性问题，增强了混合检索策略。
- **2025年5月2日 (v0.1.1)**：集成Logloom日志系统，提升了系统的可靠性和国际化支持。

**下一步计划：** 开发OCR与记忆系统进一步融合功能（v0.1.7），优化OCR相关知识的记忆存取机制，实现基于历史OCR校正的自适应改进，并开发OCR结果评估系统。同时继续开发文档综合处理功能，使系统能够自动识别和处理PDF中的文本、图片、表格和公式。详细的设计方案见[13_DocumentProcessingDesign.md](./docs/others/13_DocumentProcessingDesign.md)和[13_AdvancedMemoryManager_Progress.md](./docs/modules/memory_management/13_AdvancedMemoryManager_Progress.md)。

完整开发路线图请查看[08_ROADMAP_KnowForge.md](./docs/08_ROADMAP_KnowForge.md)。

## 安装指南

### 系统要求

- **操作系统**：Windows 10+、Linux (Ubuntu 20.04+, Fedora 41+)
- **Python版本**：3.11或更高
- **内存**：建议4GB以上，处理大文档时推荐8GB+

### 方法一：使用Conda（推荐）

```bash
# 创建环境
conda create -n knowforge python=3.11
conda activate knowforge

# 克隆项目
git clone https://github.com/yourusername/knowforge.git
cd knowforge

# 安装依赖
pip install -r requirements.txt

# 安装最新版Logloom（必需）
pip install logloom

# 在conda环境中需要设置环境变量
# Linux/macOS
export DEEPSEEK_API_KEY=your-api-key-here
# Windows
set DEEPSEEK_API_KEY=your-api-key-here
```

### 方法二：使用venv

```bash
# 创建环境
python -m venv venv

# 激活环境 (Windows)
venv\Scripts\activate
# 激活环境 (Linux/macOS)
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装最新版Logloom（必需）
pip install logloom
```

## 环境配置

### 如果使用Conda

```
export DEEPSEEK_API_KEY=your-api-key-here
```

### 如果使用venv

1. 根目录(如果不存在)创建文件`.env`

2. 编辑`.env`文件，填入您的API密钥
```
DEEPSEEK_API_KEY=your-api-key-here
```

## 使用方法

### 命令行使用

基本用法:

```bash
python gen_notes.py generate --input-dir "input/" --output-dir "output/" --formats "markdown,ipynb"
```

参数说明:

- `--input-dir`: 输入文件目录
- `--output-dir`: 输出文件目录
- `--formats`: 生成的输出格式，用逗号分隔（支持 markdown, ipynb, pdf）

查看帮助:

```bash
python gen_notes.py --help
```

### 输入文件组织

将您的文件放在以下目录结构中:

```
input/
├── pdf/          # 放置PDF文档
├── images/       # 放置图片
├── codes/        # 放置代码文件
└── links/        # 放置包含URL的文本文件（每行一个链接）
```

### 输出文件位置

生成的文件将保存在:

```
output/
├── markdown/     # Markdown文件
├── notebook/     # Jupyter Notebook文件
├── pdf/          # PDF文件
└── logs/         # 日志文件
```

## 开发进度

当前项目处于迭代3（高级功能与优化阶段）开发中，已经完成了国际化系统的全面升级，下一步将重点实现OCR功能和多格式输出支持。
详细的版本历史和计划请查看[更新日志](CHANGELOG.md)和[开发路线图](docs/08_ROADMAP_KnowForge.md)。

## 开发指南

详细的开发指南请参阅 [开发者文档](docs/03_DEV_KnowForge.md)。

### 测试

运行测试:

```bash
# 运行所有测试
pytest tests/

# 检查测试覆盖率
pytest --cov=src tests/

# 运行llm集成测试(你需要在input文件夹中放入文件后，才能进行测试)
python scripts/llm_integration_check.py
```

### 国际化与日志系统 (Logloom)

项目现已完全迁移至Logloom进行国际化和日志管理。Logloom提供了以下功能：

- **统一国际化API**：使用`get_text`和`format_text`获取翻译
- **动态资源加载**：支持`register_locale_file`和`register_locale_directory`动态加载语言资源
- **多语言支持**：完整支持中英文界面和日志消息
- **智能键名解析**：处理多种键名格式，简化开发过程
- **自动日志文件轮转**：防止日志文件过大
- **可配置的日志格式和级别**：通过`resources/config/logloom_config.yaml`进行设置

新代码应直接使用Logloom API而非LocaleManager（LocaleManager现为过渡层）：

```python
# 推荐方式（直接使用Logloom）
from logloom import get_text, format_text

welcome = get_text("welcome")
error = format_text("system.error", message="发生错误")

# 过渡方式（使用LocaleManager作为Logloom封装）
from src.utils.locale_manager import LocaleManager
locale = LocaleManager("resources/locales")
welcome = locale.get("welcome")
error = locale.format("system.error", {"message": "发生错误"})
```

## 常见问题解决

### EasyOCR依赖问题
- Windows上可能需要安装Visual C++ Build Tools
- Linux可能需要额外的系统库: `apt-get install libgl1`

### WeasyPrint依赖问题
- Linux: `apt-get install libpango1.0-dev libharfbuzz-dev libffi-dev`
- Windows: 按照[官方指南](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows)安装GTK+

### Logloom问题
- 项目需要最新版本Logloom，确保使用`pip install logloom`安装了最新版本
- 如果遇到键名格式问题，请检查使用方式是否正确（详见开发指南）
- 配置文件位置应为`resources/config/logloom_config.yaml`
- 语言资源目录应为`resources/locales/`

## 项目结构

```
knowforge/
├── input/                  # 用户输入目录
├── output/                 # 最终输出目录
├── workspace/              # 中间缓存区
├── src/                    # 源代码
│   ├── note_generator/     # 核心逻辑模块
│   │   ├── advanced_memory_manager.py # 高级记忆管理系统
│   │   ├── advanced_ocr_processor.py # 高级OCR处理器 
│   │   ├── llm_caller.py   # LLM调用模块
│   │   ├── document_analyzer.py # 文档分析器(规划中)
│   │   ├── content_extractor.py # 内容提取器(规划中)
│   │   └── content_processor.py # 内容处理器(规划中)
│   ├── cli/                # CLI界面
│   └── utils/              # 工具类
│       ├── locale_manager.py # Logloom封装（过渡层）
│       └── logger.py       # 日志系统
├── resources/              # 静态资源
│   ├── config/             # 配置文件
│   │   ├── config.yaml     # 主配置文件
│   │   └── logloom_config.yaml # Logloom配置
│   └── locales/            # 语言资源
│       ├── zh.yaml         # 中文语言资源
│       ├── en.yaml         # 英文语言资源
│       ├── logloom_zh.yaml # Logloom中文资源
│       └── logloom_en.yaml # Logloom英文资源
├── tests/                  # 单元测试
└── scripts/                # 工具脚本
```

## 贡献指南

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

本项目采用GNU通用公共许可证v3.0 (GPL-3.0) - 详情参见 [LICENSE](LICENSE) 文件。该许可证确保软件及其衍生作品始终保持开源，并要求任何修改或衍生作品同样以GPL-3.0协议发布。

## 作者

- **@ydzat** - [GitHub](https://github.com/ydzat)
