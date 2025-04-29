<!--
 * @Author: @ydzat
 * @Date: 2025-04-28 15:31:28
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-04-28 16:27:33
 * @Description: 高阶设计文档
-->
# KnowForge: AI助力学习笔记生成器 设计文档

---

## 项目总览

**KnowForge** 是一个基于人工智能的自动化学习笔记生成器，目标是把多种输入源（PDF、图片、代码、网页链接）整合处理，生成结构化的笔记，支持 PDF、Jupyter Notebook(.ipynb)、Markdown(.md) 等格式输出。

项目目标：
- 支持跨平台发布（Windows、Linux）
- 支持多语言（中文，英文）
- 支持多模态运行（CLI 与 Python库调用）
- 支持大文档处理与记忆管理
- 支持新增模型（DeepSeek Chat / Reasoner）
- 未来可接入 MoeAI-C 系统

---

## 全局设计概览

### 输入源
- PDF文档
- 图片（支持OCR）
- 网页链接（自动抓取网页内容）
- 代码文件（Python等）

### 内容处理流程
1. 输入分析：检测文件/图片/网页/代码
2. 自动拆分：根据章节/小节拆分大文档
3. 向量化：用 sentence-transformers 转换成Embedding
4. 记忆管理：保存到 ChromaDB
5. 调用DeepSeek模型：生成摘要和添加设计笔记
6. 输出格式生成：Markdown -> Notebook/PDF

### 输出格式
- Markdown (.md)
- Jupyter Notebook (.ipynb)
- PDF (.pdf)

---

## 技术栏目

| 模块 | 技术工具 |
|:-----|:---------|
| 输入处理 | pdfplumber, easyocr, requests+beautifulsoup4, pygments |
| 向量化处理 | sentence-transformers (all-MiniLM-L6-v2) |
| 调用LLM模型 | openai-python SDK + DeepSeek API |
| 向量存储 | ChromaDB |
| 输出生成 | markdown-it-py, weasyprint, nbformat |
| CLI控制器 | Typer |
| 打包发布 | PyInstaller |
| 安全管理 | python-dotenv |
| 日志系统 | logging |
| 异常处理 | 自定义异常类组 |

---

## 工程化优化

- 静态信息与动态逻辑分离：增加resources/ 目录，管理配置、文本模板、多语言资源
- 固定错误处理组：定义统一异常类，日志系统，保证系统稳定性
- 支持CLI + Python Library 双模式：支持装配成为库，符合 MoeAI-C 未来调用需求

---

## 项目目录结构

```bash
knowforge/
├── input/                  # 用户输入目录
│   ├── pdf/                # 原始课件或文档PDF
│   ├── images/             # 讲义截图、扫描图片
│   ├── codes/              # 参考代码文件（支持Python等）
│   └── links/              # 包含网址链接的文本文件（每行一个链接）
├── workspace/              # 中间缓存区
│   ├── preprocessed/       # 已解析原始文本
│   ├── split_segments/     # 拆分后的文本片段
│   ├── embeddings/         # 生成的文本向量缓存
│   └── memory_db/          # ChromaDB本地向量数据库文件
├── output/                 # 最终输出目录
│   ├── markdown/           # 生成的Markdown笔记
│   ├── notebook/           # 生成的Jupyter Notebook (.ipynb)
│   ├── pdf/                # 生成的最终版PDF文档
│   └── logs/               # 运行日志文件
├── docs/                   # 工程设计文档 
├── src/
│   ├── note_generator/     # 核心逻辑模块
│   │   ├── __init__.py
│   │   ├── processor.py          # 主流程控制器
│   │   ├── input_handler.py      # 输入文件预处理
│   │   ├── splitter.py           # 文档拆分
│   │   ├── embedder.py           # 文本向量化模块
│   │   ├── memory_manager.py     # 向量记忆检索
│   │   ├── llm_caller.py         # DeepSeek API封装器
│   │   └── output_writer.py      # Markdown/Notebook/PDF生成器
│   ├── cli/
│   │   ├── __init__.py
│   │   └── cli_main.py           # CLI命令行界面
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── exceptions.py         # 自定义异常类
│   │   ├── logger.py             # 日志管理
│   │   ├── config_loader.py      # 加载配置和环境变量
│   │   └── file_utils.py         # 文件操作辅助工具
├── resources/             # 静态资源目录
│   ├── config/
│   │   ├── config.yaml            # 系统基础配置文件
│   │   ├── llm_profiles.yaml      # LLM模型参数配置
│   ├── locales/
│   │   ├── en.yaml                # 英文界面语言包
│   │   └── zh.yaml                # 中文界面语言包
│   ├── templates/
│   │   ├── note_template.md       # 笔记正文模板
│   │   └── system_prompts.yaml    # 交互提示模板
├── tests/                  # 单元测试目录
│   ├── __init__.py
│   ├── test_processor.py
│   ├── test_embedder.py
│   ├── test_output_writer.py
│   └── test_memory_manager.py
├── scripts/                # 工具脚本目录
│   ├── clean_workspace.py
│   ├── rebuild_memory.py
│   └── export_config_docs.py
├── gen_notes.py            # Typer CLI入口文件
├── requirements.txt
├── README.md
├── LICENSE
└── .env.example            # 环境变量示例文件
```

---

## 配置管理系统设计（Config Management）

### 目标
- 将系统运行时所需的所有参数、路径、开关项集中管理。
- 方便调整、复现、扩展，不需要修改程序内部代码。
- 保持用户配置（如API密钥、选择的LLM模型）安全独立。

### 组成部分

1. `resources/config/config.yaml`
    - 主配置文件，集中管理默认设置。
2. `.env` 文件
    - 存放敏感信息（如API密钥），避免硬编码。
3. `src/utils/config_loader.py`
    - 负责统一加载、校验、合并配置。


### config.yaml 示例结构

```yaml
# 系统通用配置
system:
  language: "zh"
  workspace_dir: "workspace/"
  output_dir: "output/"

# 输入设置
input:
  allowed_formats: ["pdf", "jpg", "png", "txt", "md"]

# 文本拆分器设置
splitter:
  chunk_size: 800
  overlap_size: 100

# 向量化器设置
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

# 记忆数据库设置
memory:
  chroma_db_path: "workspace/memory_db/"

# LLM调用设置
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  temperature: 0.5

# 输出设置
output:
  formats: ["markdown", "ipynb", "pdf"]
```


### .env 示例内容

```dotenv
# DeepSeek API Key
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# 可预留其他密钥
```


### config_loader.py 设计

```python
import yaml
import os
from dotenv import load_dotenv

class ConfigLoader:
    def __init__(self, config_path: str):
        load_dotenv()
        self.config = self._load_yaml(config_path)

    def _load_yaml(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default=None):
        """通过 system.language 这样的路径访问配置项"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key, {})
        return value if value != {} else default

    def get_env(self, env_var: str, default=None):
        """获取环境变量（用于API Key等）"""
        return os.getenv(env_var, default)
```

---

# 环境管理系统设计（Environment Management）

### 目标
- 保护敏感信息，避免API密钥硬编码。
- 保持开发环境与生产环境灵活切换。
- 支持本地、云端部署统一管理。

### 组成部分
- `.env` 文件（放在项目根目录，不纳入版本控制）
- `python-dotenv` 自动加载环境变量
- `config_loader.get_env()` 方法统一访问环境变量

### 示例：敏感信息管理流程

1. 开发者创建 `.env` 文件：

```dotenv
DEEPSEEK_API_KEY=your-secret-key-here
```

2. 在程序初始化时调用：

```python
from utils.config_loader import ConfigLoader

config = ConfigLoader("resources/config/config.yaml")
api_key = config.get_env("DEEPSEEK_API_KEY")
```

3. 生产部署时，服务器上直接设置环境变量，无需`.env`文件。

### 安全注意事项
- `.env`文件加入`.gitignore`，禁止提交到Git仓库。
- 通过 `os.getenv` 动态读取，支持Docker/Kubernetes等环境变量注入。
- 日志中禁止打印敏感字段内容。

---

# 日志与异常处理体系设计（Logging & Exception Handling）

### 目标
- 统一捕获并记录程序运行过程中的重要信息和异常错误。
- 提供开发者调试依据，提升用户使用体验（优雅提示而不是程序崩溃）。
- 支持日志文件与终端输出双通道，便于定位问题。

### 日志系统设计

| 内容 | 说明 |
|:-----|:-----|
| 日志框架 | Python标准库 `logging` |
| 日志输出 | 终端输出 + 文件持久化（output/logs/note_gen.log）|
| 日志等级 | DEBUG / INFO / WARNING / ERROR / CRITICAL |
| 日志格式 | 时间戳 + 日志等级 + 模块名 + 消息内容 |

#### logger.py 示例

```python
import logging
import os

def setup_logger(log_dir="output/logs", log_level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "note_gen.log")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("knowforge")
```

使用示例：

```python
from utils.logger import setup_logger

logger = setup_logger()
logger.info("程序启动")
```

---

### 异常处理系统设计

| 内容 | 说明 |
|:-----|:-----|
| 异常分层 | 定义基础异常类和细分子类 |
| 捕获机制 | 封装统一 try-except 模板 |
| 日志记录 | 出错时自动记录详细traceback到日志 |
| 用户提示 | 仅输出简洁友好的错误提示到终端 |

#### exceptions.py 示例

```python
class NoteGenError(Exception):
    """基础异常：所有可预期错误继承它"""
    pass

class InputError(NoteGenError):
    """输入处理相关错误"""
    pass

class APIError(NoteGenError):
    """API调用相关错误"""
    pass

class MemoryError(NoteGenError):
    """向量记忆相关错误"""
    pass

class OutputError(NoteGenError):
    """输出生成相关错误"""
    pass
```

#### 应用示例（主程序中）

```python
from utils.logger import setup_logger
from utils.exceptions import NoteGenError

logger = setup_logger()

try:
    # 主程序逻辑
    ...
except NoteGenError as e:
    logger.error(f"Known Error: {str(e)}")
    print("发生错误，请查看logs文件了解详细信息。")
except Exception as e:
    logger.exception(f"Unexpected Error: {str(e)}")
    print("程序遇到未处理的异常，已记录日志。")
```

---

## 多语言支持与静态数据分离设计（Localization & Static-Dynamic Separation）

### 目标

- 支持界面信息（提示、日志、异常提示）动态切换语言。
- 所有文字内容（如提示信息、错误消息）统一维护，不硬编码在程序中。
- 动态调用静态资源，便于后期维护和扩展其他语言。

### 组成部分

- `resources/locales/zh.yaml` （中文语言文件）
- `resources/locales/en.yaml` （英文语言文件）
- `src/utils/locale_manager.py` （语言管理模块）

### 语言文件（示例）

**resources/locales/zh.yaml**

```yaml
system:
  start_message: "程序启动"
  error_occurred: "发生错误，请查看logs文件了解详细信息。"
  unexpected_error: "程序遇到未处理的异常，已记录日志。"
```

**resources/locales/en.yaml**

```yaml
system:
  start_message: "Program started"
  error_occurred: "An error occurred. Please check the logs for details."
  unexpected_error: "Unexpected error occurred. Logged for review."
```

### 语言管理模块（locale\_manager.py）示例

```python
import yaml

class LocaleManager:
    def __init__(self, locale_path: str, language: str = "zh"):
        self.language = language
        self.messages = self._load_locale(locale_path)

    def _load_locale(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key_path: str) -> str:
        keys = key_path.split('.')
        value = self.messages.get(self.language, {})
        for key in keys:
            value = value.get(key, {})
        if isinstance(value, str):
            return value
        return ""
```

### 使用示例

```python
from utils.locale_manager import LocaleManager

locale = LocaleManager("resources/locales/locale.yaml", language="zh")
print(locale.get("system.start_message"))
```

---

## 日志与异常处理体系设计（Logging & Exception Handling）

### 目标

- 统一捕获并记录程序运行过程中的重要信息和异常错误。
- 提供开发者调试依据，提升用户使用体验（优雅提示而不是程序崩溃）。
- 支持日志文件与终端输出双通道，便于定位问题。
- 所有提示文字通过 `LocaleManager` 动态加载。

### 日志系统设计

| 内容   | 说明                                        |
| ---- | ----------------------------------------- |
| 日志框架 | Python标准库 `logging`                       |
| 日志输出 | 终端输出 + 文件持久化（output/logs/note\_gen.log）   |
| 日志等级 | DEBUG / INFO / WARNING / ERROR / CRITICAL |
| 日志格式 | 时间戳 + 日志等级 + 模块名 + 消息内容                   |

#### logger.py 示例

```python
import logging
import os
from utils.locale_manager import LocaleManager

locale = LocaleManager("resources/locales/locale.yaml", language="zh")

def setup_logger(log_dir="output/logs", log_level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "note_gen.log")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("knowforge")
```

使用示例：

```python
from utils.logger import setup_logger

logger = setup_logger()
logger.info(locale.get("system.start_message"))
```

---

### 异常处理系统设计

| 内容   | 说明                         |
| ---- | -------------------------- |
| 异常分层 | 定义基础异常类和细分子类               |
| 捕获机制 | 封装统一 try-except 模板         |
| 日志记录 | 出错时自动记录详细traceback到日志      |
| 用户提示 | 仅输出LocaleManager加载的友好信息到终端 |

#### exceptions.py 示例

```python
class NoteGenError(Exception):
    """基础异常：所有可预期错误继承它"""
    pass

class InputError(NoteGenError):
    """输入处理相关错误"""
    pass

class APIError(NoteGenError):
    """API调用相关错误"""
    pass

class MemoryError(NoteGenError):
    """向量记忆相关错误"""
    pass

class OutputError(NoteGenError):
    """输出生成相关错误"""
    pass
```

#### 应用示例（主程序中）

```python
from utils.logger import setup_logger
from utils.exceptions import NoteGenError
from utils.locale_manager import LocaleManager

locale = LocaleManager("resources/locales/locale.yaml", language="zh")
logger = setup_logger()

try:
    # 主程序逻辑
    ...
except NoteGenError as e:
    logger.error(f"Known Error: {str(e)}")
    print(locale.get("system.error_occurred"))
except Exception as e:
    logger.exception(f"Unexpected Error: {str(e)}")
    print(locale.get("system.unexpected_error"))
```

---

## 记忆管理系统设计（Memory Management）

### 目标

- 处理超大文档，合理拆分，逐步向LLM输入，避免超出上下文窗口限制。
- 在拆分后维持前后文关联性，保证笔记连贯性。
- 支持动态记忆检索，在生成过程中按需引用已有上下文。

### 核心思路

| 环节    | 设计策略                                                |
| ----- | --------------------------------------------------- |
| 文档拆分  | 采用智能分段，优先以自然段落/章节标题为分界点；无法明确分段时采用固定长度滑动窗口策略。        |
| 向量化存储 | 使用 `sentence-transformers` 生成Embedding，存入ChromaDB。  |
| 动态记忆  | 在需要追加生成时，通过Embedding相似度检索前文相关片段作为上下文辅助。             |
| 窗口管理  | 根据不同模型（如DeepSeek Chat/Reasoner）上下文长度，动态控制输入片段数量与大小。 |

### 文档拆分模块（splitter.py）设计

#### 拆分逻辑

- 优先：使用LLM智能分析文档结构，确定最佳拆分点
- 次之：按段落空行拆分（当LLM拆分失败时作为后备方案）
- 最后：按字符长度硬性切片（如每800字符，作为最终兜底方案）

#### 拆分配置参数（config.yaml）

```yaml
splitter:
  chunk_size: 800
  overlap_size: 100
  use_llm: true
  llm_provider: "deepseek"
```

chunk\_size 控制单片最大长度，overlap\_size 控制片段间重叠，帮助保持上下文连续性。use_llm 控制是否启用LLM辅助拆分（默认开启），llm_provider 指定使用的LLM提供商。

### 向量存储与检索模块（memory\_manager.py）设计

#### 核心接口

```python
class MemoryManager:
    def __init__(self, chroma_db_path: str, embedding_model: str):
        pass

    def add_segments(self, segments: list) -> None:
        """将文本片段向量化并存储到ChromaDB"""
        pass

    def query_similar(self, query_text: str, top_k: int = 5) -> list:
        """检索与query_text最相似的若干片段"""
        pass

    def rebuild_memory(self, segments: list) -> None:
        """清空重建记忆库（可用于重新整理）"""
        pass
```

#### 使用流程示例

```python
memory = MemoryManager("workspace/memory_db/", embedding_model="sentence-transformers/all-MiniLM-L6-v2")

# 添加记忆
memory.add_segments(["本章介绍强化学习的基本概念。", "马尔可夫决策过程（MDP）是..."])

# 检索记忆
related_context = memory.query_similar("什么是MDP？", top_k=3)
```

### 智能窗口管理设计（Window Manager）

为不同模型配置最大上下文长度，拆分与组织输入，防止超长报错。

配置示例（llm\_profiles.yaml）：

```yaml
deepseek-chat:
  max_context_length: 8000

deepseek-reasoner:
  max_context_length: 32000
```

窗口管理逻辑示例：

- 估算每段文本编码后Token数量
- 总Token数不得超过 `max_context_length`
- 超出时优先丢弃相似度最低的片段

---

## 输入源处理系统设计（Input Handler）

### 目标

- 支持多类型输入源（PDF文档、图片、网页链接、代码文件）。
- 统一格式标准化处理，将所有输入转为纯文本片段供后续拆分和记忆处理。
- 自动识别输入目录结构，灵活适配不同课程/资料组织方式。

### 输入类型与处理策略

| 输入类型 | 处理工具或方法                   | 输出结果     |
| ---- | ------------------------- | -------- |
| PDF  | pdfplumber                | 提取文本     |
| 图片   | easyocr                   | OCR识别为文本 |
| 网页链接 | requests + BeautifulSoup4 | 提取网页正文   |
| 代码文件 | 直接读取文本，使用Pygments高亮（可选）   | 保留格式化代码  |

### 目录扫描与组织规则

1. 输入根目录：`input/`
2. 子目录分类型：
   - `input/pdf/`
   - `input/images/`
   - `input/links/`
   - `input/codes/`
3. 自动根据子目录推断输入类型，不需用户标注。

支持如下两种常见组织结构：

- **结构一**：课件为独立文件，如 ch1.pdf, ch2.pdf, ch3.pdf
- **结构二**：每章一个子目录，如 ch1/01\_intro.pdf, ch1/02\_concepts.pdf

解析逻辑根据目录/文件名自然排序（数字优先，字母次之）。

### 核心模块（input\_handler.py）设计

```python
class InputHandler:
    def __init__(self, input_dir: str, workspace_dir: str):
        pass

    def scan_inputs(self) -> dict:
        """扫描并分类整理所有输入文件，返回 {'pdf': [...], 'images': [...], 'links': [...], 'codes': [...]}"""
        pass

    def extract_texts(self) -> list:
        """按类别批量提取文本内容，返回统一格式的文本片段列表"""
        pass

    def save_preprocessed(self) -> None:
        """将提取出的文本保存到 workspace/preprocessed/ 目录，便于后续处理"""
        pass
```

### 文本提取示例

```python
handler = InputHandler("input/", "workspace/")
handler.scan_inputs()
segments = handler.extract_texts()
handler.save_preprocessed()
```

- PDF文本提取到 `workspace/preprocessed/pdfs/`
- 图片OCR提取到 `workspace/preprocessed/images/`
- 网页正文提取到 `workspace/preprocessed/links/`
- 代码提取到 `workspace/preprocessed/codes/`

### 配置支持（可在config.yaml中设定）

```yaml
input:
  allowed_formats: ["pdf", "jpg", "png", "txt", "md"]
  max_file_size_mb: 100
  enable_ocr_languages: ["ch_sim", "en"]
```

- 限制输入文件最大体积（防止异常超大文件）
- OCR支持多语言（如中文简体+英文）

---
## 输出生成系统设计（Output Writer）

### 目标

- 支持多种格式输出，覆盖Markdown、Jupyter Notebook和PDF。
- 保证输出文件结构清晰、排版美观、易于后续编辑。
- 与LocaleManager集成，支持中英文笔记输出。
- 可灵活选择输出格式，支持单一/多种格式同时生成。

### 支持的输出格式

| 输出格式                      | 说明                          |
| ------------------------- | --------------------------- |
| Markdown (.md)            | 标准Markdown文档，支持纯文本编辑        |
| Jupyter Notebook (.ipynb) | 包含Markdown单元，便于交互式阅读与运行     |
| PDF (.pdf)                | 通过HTML或Markdown渲染生成高质量PDF文件 |

### 核心模块（output\_writer.py）设计

```python
class OutputWriter:
    def __init__(self, workspace_dir: str, output_dir: str, locale_manager):
        pass

    def generate_markdown(self, notes: list, filename: str) -> str:
        """根据笔记片段生成Markdown文件，返回生成路径"""
        pass

    def generate_notebook(self, notes: list, filename: str) -> str:
        """根据笔记片段生成.ipynb文件，返回生成路径"""
        pass

    def generate_pdf(self, markdown_path: str, filename: str) -> str:
        """将Markdown渲染为PDF文件，返回生成路径"""
        pass
```

### 输出流程示例

```python
writer = OutputWriter("workspace/", "output/", locale)

md_path = writer.generate_markdown(notes, "my_notes")
nb_path = writer.generate_notebook(notes, "my_notes")
writer.generate_pdf(md_path, "my_notes")
```

### Markdown模板支持（resources/templates/note\_template.md）

- 使用Jinja2或手动模板，统一渲染风格。
- 插入章节标题、时间戳、来源信息等元数据。

示例占位符：

```markdown
# {{ title }}

**生成时间**: {{ timestamp }}

{{ content }}
```

### 输出目录组织结构

- Markdown文件 -> `output/markdown/`
- Jupyter Notebook文件 -> `output/notebook/`
- PDF文件 -> `output/pdf/`

### 配置支持（可在config.yaml中设定）

```yaml
output:
  formats: ["markdown", "ipynb", "pdf"]
  pdf:
    page_size: "A4"
    margin: "1cm"
```

### 依赖库

- Markdown处理：`markdown-it-py`
- Notebook处理：`nbformat`
- PDF渲染：`weasyprint`

---
## CLI与Python Library接口设计（CLI & Library Interfaces）

### 目标

- 支持通过命令行一键式操作（适合普通用户使用）。
- 支持作为Python库调用，供高级用户或其他系统（如 MoeAI-C）集成。
- 保持两种接口风格统一，避免功能重复开发。

### CLI接口设计（基于 Typer）

#### 入口文件 `gen_notes.py`

```python
import typer
from src.cli.cli_main import cli

def main():
    typer.run(cli)

if __name__ == "__main__":
    main()
```

#### 主命令入口 `cli_main.py`

```python
import typer
from src.note_generator.processor import Processor

cli = typer.Typer()

@cli.command()
def generate(
    input_dir: str = typer.Option("input/", help="输入文件目录"),
    output_dir: str = typer.Option("output/", help="输出文件目录"),
    config_path: str = typer.Option("resources/config/config.yaml", help="配置文件路径"),
    formats: str = typer.Option("markdown,ipynb,pdf", help="生成的输出格式，逗号分隔")
):
    processor = Processor(input_dir, output_dir, config_path)
    processor.run_full_pipeline(formats.split(","))
```

#### CLI调用示例

```bash
python gen_notes.py generate --input-dir "input/" --output-dir "output/" --formats "markdown,ipynb"
```

### Python Library接口设计

#### 直接调用Processor模块

```python
from src.note_generator.processor import Processor

processor = Processor(
    input_dir="input/",
    output_dir="output/",
    config_path="resources/config/config.yaml"
)
processor.run_full_pipeline(["markdown", "pdf"])
```

### 统一性设计原则

- CLI与Library共用Processor主流程类，避免代码分叉。
- 错误与提示信息均通过LocaleManager统一管理。
- 运行日志统一接入Logger体系。

### 配置支持（config.yaml片段）

```yaml
cli:
  default_input_dir: "input/"
  default_output_dir: "output/"
  default_formats: ["markdown", "ipynb", "pdf"]
```

---
## 测试与工具脚本体系设计（Testing & Utilities）

### 目标

- 保证核心模块（如Processor、InputHandler、OutputWriter等）具备完善的单元测试覆盖。
- 提供一组常用的运维辅助脚本（如清理缓存、重建记忆库、导出配置信息等）。
- 测试与脚本保持模块化、可扩展、易维护。

### 测试体系设计（tests/）

| 模块    | 测试文件示例                   | 测试内容                    |
| ----- | ------------------------ | ----------------------- |
| 输入处理  | test\_input\_handler.py  | 输入扫描、文本提取准确性            |
| 文本拆分  | test\_splitter.py        | 拆分粒度与重叠逻辑验证             |
| 向量化处理 | test\_embedder.py        | Embedding生成正确性          |
| 记忆管理  | test\_memory\_manager.py | 向量存取与相似性检索准确性           |
| LLM调用 | test\_llm\_caller.py     | Prompt构建与调用异常处理         |
| 输出生成  | test\_output\_writer.py  | Markdown/ipynb/pdf生成正确性 |

#### 测试框架

- 使用 `pytest`
- 使用 `pytest-mock` 或内置 `unittest.mock` 进行外部API/IO的模拟测试

#### 示例：test\_input\_handler.py

```python
import pytest
from src.note_generator.input_handler import InputHandler

def test_scan_inputs(tmp_path):
    (tmp_path / "pdf").mkdir()
    (tmp_path / "pdf/sample.pdf").write_text("dummy pdf")
    handler = InputHandler(str(tmp_path), str(tmp_path))
    inputs = handler.scan_inputs()
    assert "pdf" in inputs
    assert len(inputs["pdf"]) == 1
```

### 辅助脚本体系设计（scripts/）

| 脚本                      | 功能                     |
| ----------------------- | ---------------------- |
| clean\_workspace.py     | 清理workspace/下所有中间文件与缓存 |
| rebuild\_memory.py      | 重新提取文本并重建ChromaDB向量数据库 |
| export\_config\_docs.py | 导出当前生效配置，生成markdown文档  |

#### 示例：clean\_workspace.py

```python
import shutil
import os

def clean_workspace(workspace_dir="workspace/"):
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
        os.makedirs(workspace_dir)
    print("✅ Workspace已清空并重新初始化")

if __name__ == "__main__":
    clean_workspace()
```

### 规范要求

- 所有脚本放置在 `scripts/` 目录下
- 所有测试用例放置在 `tests/` 目录下，保持清晰分层
- 测试覆盖率要求核心模块至少达到80%

---

## 系统工作流与模块交互设计（System Workflow & Interaction Design）

### 目标

- 全面描述KnowForge项目的整体执行流程，涵盖各功能模块与支撑模块。
- 体现配置管理、多语言支持、异常日志体系、记忆管理等工程细节。
- 便于新开发者快速理解系统运行机制与扩展方法。


### 主流程总结（包含辅助模块）

1. **初始化阶段**
   - 加载 `.env` 环境变量（API Key、安全配置）
   - 加载 `config.yaml` 配置文件（系统参数、模型选择）
   - 初始化 `Logger`（日志系统）
   - 初始化 `LocaleManager`（多语言系统）

2. **输入源处理阶段**
   - 使用 `InputHandler` 扫描 `input/` 目录，分类识别 PDF、图片、代码、链接。
   - 提取纯文本，存入 `workspace/preprocessed/`

3. **文本拆分阶段**
   - 使用 `Splitter` 按章节/段落/长度智能拆分，生成标准文本片段。

4. **记忆与向量化阶段**
   - 使用 `Embedder` 将片段转为向量表示。
   - 使用 `MemoryManager` 将向量存储到本地 `ChromaDB`。
   - 支持检索召回相关记忆片段，辅助生成。

5. **笔记生成阶段**
   - 使用 `LLMCaller` 调用 DeepSeek Chat/Reasoner，根据文本片段逐步生成总结笔记。
   - 处理超长文档时，按智能窗口策略分批生成并拼接。

6. **输出生成阶段**
   - 使用 `OutputWriter` 将生成内容输出为 Markdown。
   - 可选择性转换为 Jupyter Notebook (.ipynb) 和 PDF文件。

7. **清理与后处理阶段**（可选）
   - 使用 `scripts/clean_workspace.py` 清理中间缓存。
   - 使用 `scripts/rebuild_memory.py` 重建记忆数据库。

8. **异常与日志全程管理**
   - 统一捕获异常（NoteGenError体系）并优雅提示。
   - 详细记录各阶段日志到 `output/logs/` 目录。


### 模块交互完整流程图（文字版）

```plaintext
[配置管理 ConfigLoader] + [环境管理 dotenv] + [多语言支持 LocaleManager]
           ↓
        [日志系统 Logger]
           ↓
[输入源 (PDF/图片/链接/代码)]
           ↓
        [InputHandler] → 提取纯文本
           ↓
          [Splitter] → 智能拆分片段
           ↓
[Embedder] → 文本向量化 → [MemoryManager] 向量存储 & 检索
           ↓
        [LLMCaller] → 调用DeepSeek生成笔记
           ↓
        [OutputWriter]
           ↓
 [输出Markdown/ipynb/PDF → output目录]

【辅助流程】
- [scripts/clean_workspace.py] → 清理缓存
- [scripts/rebuild_memory.py] → 记忆重建
- [tests/*] → 单元测试保障各模块稳定
```


### 关键设计原则总结

- **模块解耦，接口清晰**：每个模块职责单一，便于扩展和替换。
- **配置驱动，灵活可调**：通过配置文件与环境变量集中管理参数。
- **多语言支持，国际友好**：所有提示信息国际化处理，支持中英文切换。
- **异常与日志统一**：确保程序在各类异常下稳定运行，日志可溯源。
- **记忆管理智能化**：处理大文档、支持上下文记忆检索，提升生成连贯性。
- **双接口支持（CLI+Library）**：兼顾命令行用户与开发者集成需求。
- **测试与工具完善**：提供标准化测试用例与运维辅助脚本。


