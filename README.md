# KnowForge

简体中文 | [English](README_EN.md)

![版本](https://img.shields.io/badge/版本-0.1.2--beta-blue)
![许可证](https://img.shields.io/badge/许可证-GPL--3.0-green)
![Python](https://img.shields.io/badge/Python-3.11+-yellow)

**KnowForge** 是一个基于人工智能的自动化学习笔记生成器，能够从多种输入源（PDF、图片、代码、网页链接）整合处理，生成结构化的笔记，支持多种格式输出。

## 特性

- **多源输入支持**：处理PDF文档、图片（OCR）、网页链接、代码文件
- **智能文本拆分**：根据章节/段落自动拆分大文档
- **记忆管理系统**：通过向量化和ChromaDB实现上下文记忆和检索
- **AI生成增强**：使用大语言模型生成摘要和笔记内容，支持多种LLM接口
- **多格式输出**：支持Markdown、Jupyter Notebook、PDF格式
- **高级日志系统**：集成Logloom，支持多语言日志消息和文件自动轮转
- **多语言支持**：支持中文和英文界面
- **跨平台兼容**：支持Windows和Linux系统

## 开发进度

参考设计文档[06_ITER_KnowForge.md](./docs/06_ITER_KnowForge.md)，当前已完成迭代2：核心功能实现，实现主要业务功能模块，可处理基本输入并生成输出的系统。

最新里程碑：
- **2025年5月14日**：向量记忆管理模块优化，修复了与ChromaDB API的兼容性问题，增强了混合检索策略，优化了阈值处理逻辑。
- **2025年5月13日**：集成Logloom日志系统，提升了系统的可靠性和国际化支持。

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

# 安装Logloom
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

# 安装Logloom
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

当前项目处于迭代3（高级功能与优化阶段）开发中，正在实现记忆管理、多格式输出支持和高级输入处理功能。
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

### 日志系统 (Logloom)

Logloom日志系统通过配置文件 `resources/config/logloom_config.yaml` 进行设置，支持：

- 多语言日志消息 (中/英)
- 自动文件轮转，防止日志文件过大
- 可配置的日志格式和级别
- 控制台和文件双通道输出

## 常见问题解决

### EasyOCR依赖问题
- Windows上可能需要安装Visual C++ Build Tools
- Linux可能需要额外的系统库: `apt-get install libgl1`

### WeasyPrint依赖问题
- Linux: `apt-get install libpango1.0-dev libharfbuzz-dev libffi-dev`
- Windows: 按照[官方指南](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows)安装GTK+

### Logloom问题
- 如果遇到"无法解析导入logloom"错误，请确认已安装logloom：`pip install logloom`
- 配置文件位置错误：检查 `resources/config/logloom_config.yaml` 是否存在

## 项目结构

```
knowforge/
├── input/                  # 用户输入目录
├── output/                 # 最终输出目录
├── workspace/              # 中间缓存区
├── src/                    # 源代码
│   ├── note_generator/     # 核心逻辑模块
│   ├── cli/                # CLI界面
│   └── utils/              # 工具类
├── resources/              # 静态资源
│   ├── config/             # 配置文件
│   │   ├── config.yaml     # 主配置文件
│   │   └── logloom_config.yaml # Logloom配置
│   └── locales/            # 语言资源
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
