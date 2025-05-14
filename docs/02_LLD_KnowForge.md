<!--
 * @Author: @ydzat
 * @Date: 2025-04-28 16:25:30
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-04-28 16:51:02
 * @Description: 详细设计文档
-->
# KnowForge 详细设计文档（Low-Level Design, LLD）

---

## 文档说明

本文件为KnowForge项目的详细设计文档，针对每个核心模块给出：
- 类定义（Class Definition）
- 主要属性（Attributes）
- 核心方法（Methods）
- 输入输出规范（Input/Output Specification）
- 简要的内部流程说明（Internal Workflow）

遵循模块化、高内聚、低耦合的工程设计标准。


## 模块清单（按调用顺序排列）

| 模块 | 简述 |
|:----|:----|
| 1. ConfigLoader | 配置加载器，统一管理配置文件与环境变量 |
| 2. LocaleManager | 多语言管理器，统一管理提示文字与异常信息 |
| 3. Logger（封装） | 日志系统，负责标准化日志输出 |
| 4. InputHandler | 输入源处理器，扫描并提取输入内容（PDF、图片、代码、链接） |
| 5. Splitter | 文本拆分器，按章节/段落/长度智能拆分文本 |
| 6. Embedder | 向量化工具，负责文本向量生成 |
| 7. MemoryManager | 向量存储管理器，负责向量存取与检索（ChromaDB） |
| 8. LLMCaller | 大模型调用器，封装与DeepSeek Chat/Reasoner交互 |
| 9. OutputWriter | 输出生成器，负责生成Markdown、ipynb、PDF文件 |
| 10. Processor | 主流程控制器，调度各子模块完成整个处理流程 |
| 11. Exception Classes | 自定义异常体系，负责标准化错误处理 |
| 12. Scripts工具集 | 辅助脚本，如清理缓存、重建记忆库等 |


---


# 1. ConfigLoader 模块

## 类定义
```python
class ConfigLoader:
    def __init__(self, config_path: str)
```

## 主要属性
| 属性名 | 类型 | 说明 |
|:-----|:-----|:-----|
| config_path | str | 配置文件路径 |
| config | dict | 加载后的配置字典 |

## 核心方法
| 方法名 | 参数 | 返回 | 说明 |
|:------|:----|:----|:----|
| get | key_path: str, default: Any = None | Any | 获取配置项，支持点号路径访问（如 system.language） |
| get_env | env_var: str, default: Any = None | Any | 获取环境变量（如API Key） |

## 输入输出规范
- 输入：配置文件路径 (`resources/config/config.yaml`)
- 输出：配置字典对象，供其他模块调用

## 内部流程说明
1. 加载 `.env` 文件
2. 加载 `config.yaml` 文件
3. 提供统一的访问接口


---

# 2. LocaleManager 模块

## 类定义

```python
class LocaleManager:
    def __init__(self, locale_path: str, language: str = "zh")
```

## 主要属性

| 属性名          | 类型   | 说明                                        |
| ------------ | ---- | ----------------------------------------- |
| locale\_path | str  | 语言文件路径（如 `resources/locales/locale.yaml`） |
| language     | str  | 当前语言（"zh" 或 "en"）                         |
| messages     | dict | 加载后的语言数据字典                                |

## 核心方法

| 方法名 | 参数             | 返回  | 说明                                  |
| --- | -------------- | --- | ----------------------------------- |
| get | key\_path: str | str | 通过路径访问提示信息（如 system.start\_message） |

## 输入输出规范

- 输入：语言文件（YAML格式，包含中英文内容）
- 输出：指定键路径对应的本地化字符串

## 内部流程说明

1. 解析指定语言的YAML文件
2. 支持点号分隔的路径访问（多级嵌套）
3. 当查询失败时返回空字符串

---

# 3. Logger 模块

## 类定义

```python
class LoggerManager:
    def __init__(self, log_dir: str = "output/logs", log_level: int = logging.INFO)
```

## 主要属性

| 属性名        | 类型             | 说明                           |
| ---------- | -------------- | ---------------------------- |
| log\_dir   | str            | 日志文件输出目录                     |
| log\_level | int            | 日志级别（默认INFO，可设为DEBUG、ERROR等） |
| logger     | logging.Logger | Python标准Logger对象             |

## 核心方法

| 方法名         | 参数 | 返回             | 说明              |
| ----------- | -- | -------------- | --------------- |
| get\_logger | 无  | logging.Logger | 返回已配置好的Logger实例 |

## 输入输出规范

- 输入：日志目录路径与日志等级
- 输出：标准Logger对象

## 内部流程说明

1. 创建日志目录（如果不存在）
2. 配置日志格式（时间戳 + 级别 + 模块名 + 消息内容）
3. 同时输出到文件与终端
4. 返回统一管理的Logger对象

---

# 4. InputHandler 模块

## 类定义

```python
class InputHandler:
    def __init__(self, input_dir: str, workspace_dir: str)
```

## 主要属性

| 属性名            | 类型  | 说明                    |
| -------------- | --- | --------------------- |
| input\_dir     | str | 输入源目录（如 `input/`）     |
| workspace\_dir | str | 工作区目录（如 `workspace/`） |

## 核心方法

| 方法名                | 参数 | 返回   | 说明                        |
| ------------------ | -- | ---- | ------------------------- |
| scan\_inputs       | 无  | dict | 扫描并分类所有输入文件（返回dict，按类型分组） |
| extract\_texts     | 无  | list | 提取纯文本片段（统一格式返回）           |
| save\_preprocessed | 无  | None | 保存提取出的文本到预处理目录            |

## 输入输出规范

- 输入：`input/` 目录下的PDF、图片、链接、代码等文件
- 输出：
  - `scan_inputs()` 返回分类好的文件路径字典
  - `extract_texts()` 返回标准化文本列表
  - `save_preprocessed()` 保存提取文本到 `workspace/preprocessed/`

## 内部流程说明

1. 遍历输入目录，根据文件夹/文件扩展名推断类别。
2. 分类型调用处理器：
   - PDF → pdfplumber提取文本
   - 图片 → easyocr识别
   - 链接 → requests+BeautifulSoup提取正文
   - 代码 → 直接读取文本（可选Pygments高亮）
3. 提取出的文本标准化保存，供Splitter模块后续处理。

---
# 5. Splitter 模块

## 类定义

```python
class Splitter:
    def __init__(self, config: ConfigLoader)
```

## 主要属性

| 属性名           | 类型  | 说明                       |
| ------------- | --- | ------------------------ |
| chunk_size   | int | 单个文本片段最大长度（字符数，仅用于备选方案）          |
| overlap_size | int | 相邻片段之间的重叠长度（字符数，防止上下文断裂） |
| use_llm | bool | 是否启用LLM辅助拆分（默认为True） |
| llm_provider | str | 使用的LLM服务提供商（默认为"deepseek"） |
| llm_api_key | str | LLM服务的API密钥 |

## 核心方法

| 方法名          | 参数               | 返回   | 说明                |
| ------------ | ---------------- | ---- | ----------------- |
| split_text | text_segments: list[str] | list | 输入文本列表，返回拆分后的片段列表 |
| _split_by_structure | text: str | list | 按文本结构拆分单段文本 |
| _split_with_llm_assistance | text: str, detected_patterns: list, headers: list | list | 使用LLM辅助进行拆分 |
| _split_with_deepseek | text: str, detected_patterns: list, headers: list | list | 使用DeepSeek模型进行拆分 |
| _split_by_length | text: str, chunk_size: int, overlap_size: int | list | 按固定长度拆分（备选方案） |

## 输入输出规范

- 输入：提取的标准化纯文本列表（通常来自InputHandler）
- 输出：分割后的文本片段列表，每个片段满足语义完整性，适合后续向量化处理

## 内部流程说明

1. 遍历文本列表，逐个进行拆分：
   - 优先使用LLM分析文档结构，智能识别章节和逻辑段落
   - 向LLM提供文本样本和检测到的可能标题，请求其分析文档结构
   - LLM可通过两种方式拆分：
     - 方法一：提供正则表达式匹配主要章节标题
     - 方法二：直接提供行号作为拆分点
   - LLM拆分失败时，回退到段落级别的拆分（基于空行）
   - 最后兜底方案为按chunk_size固定长度切割，保留overlap_size重叠
2. 所有片段确保具有语义完整性，避免在不合理的位置断开，提升后续生成质量。
3. 与配置系统和多语言系统集成，支持动态调整拆分策略。

---

# 6. Embedder 模块

## 类定义

```python
class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2")
```

## 主要属性

| 属性名         | 类型                  | 说明                          |
| ----------- | ------------------- | --------------------------- |
| model\_name | str                 | 预训练Embedding模型名称            |
| model       | SentenceTransformer | 加载后的SentenceTransformer模型实例 |

## 核心方法

| 方法名           | 参数               | 返回                | 说明               |
| ------------- | ---------------- | ----------------- | ---------------- |
| embed\_texts  | texts: list[str] | list[list[float]] | 输入文本列表，返回对应的向量列表 |
| embed\_single | text: str        | list[float]       | 输入单段文本，返回其对应的向量  |

## 输入输出规范

- 输入：纯文本（单条或批量）
- 输出：文本对应的高维向量表示（list或list of list）

## 内部流程说明

1. 加载指定的Sentence-Transformer预训练模型（默认MiniLM-L6-v2）。
2. 将文本输入模型，生成Embedding。
3. 返回统一格式的向量数据，供MemoryManager使用。

---

# 7. MemoryManager 模块

## 类定义

```python
class MemoryManager:
    def __init__(self, chroma_db_path: str, embedder: Embedder, collection_name: str = DEFAULT_COLLECTION, config: Dict[str, Any] = None)
```

## 主要属性

| 属性名              | 类型                | 说明                          |
| ---------------- | ----------------- | --------------------------- |
| chroma\_db\_path | str               | ChromaDB存储路径                |
| embedder         | Embedder          | 向量化工具实例（用于新增片段时生成Embedding） |
| collection       | Chroma Collection | ChromaDB中的集合实例              |
| retrieval\_mode  | str               | 当前检索模式（simple, time_weighted, context_aware, hybrid） |
| cleanup\_strategy | str              | 记忆清理策略（oldest, least_used, relevance） |

## 核心方法

| 方法名             | 参数                                | 返回   | 说明                 |
| --------------- | --------------------------------- | ---- | ------------------ |
| add\_segments   | segments: list[str], metadata: Optional[List[Dict[str, str]]] = None | List[str] | 将文本片段向量化并添加到向量数据库中，返回ID列表 |
| query\_similar  | query\_text: str, top\_k: int = None, threshold: float = None, include_embeddings: bool = False, context_texts: List[str] = None, retrieval_mode: str = None | List[Dict] | 根据查询文本检索相似的片段   |
| rebuild\_memory | segments: list[str], metadata: Optional[List[Dict[str, str]]] = None | bool | 清空原数据库，重新建立新的记忆库   |
| export\_to\_json | output_path: str | str | 将记忆库导出为JSON文件 |
| import\_from\_json | input_path: str, replace_existing: bool = False | int | 从JSON文件导入记忆库 |
| get\_collection\_stats | | Dict[str, Any] | 获取记忆库统计信息 |

### 检索策略方法

| 方法名                      | 参数                    | 返回          | 说明                     |
| ------------------------- | --------------------- | ----------- | ---------------------- |
| \_simple\_similarity\_retrieval | query\_text: str, top\_k: int = 5, threshold: float = 0.0, include_embeddings: bool = False | List[Dict] | 基于语义相似度的简单检索 |
| \_time\_weighted\_retrieval | top\_k: int = 10, time\_decay\_factor: float = 0.01, threshold: float = 0.0, include\_embeddings: bool = False | List[Dict] | 基于时间权重的检索，更倾向于最近添加的内容 |
| \_context\_aware\_retrieval | query\_text: str, context\_text: Optional[str] = None, top\_k: int = 5, threshold: float = 0.0, include\_embeddings: bool = False | List[Dict] | 结合上下文的语义检索，提高相关性 |
| \_hybrid\_retrieval | query\_text: str, top\_k: int = 5, threshold: float = 0.0, include\_embeddings: bool = False, context\_texts: List[str] = None, keyword\_weight: float = 0.2 | List[Dict] | 混合检索模式，结合语义和关键词匹配 |

### 记忆管理方法

| 方法名                   | 参数 | 返回   | 说明                |
| ---------------------- | -- | ---- | ----------------- |
| \_cleanup\_memory      | 无  | None | 根据配置的策略清理记忆库      |
| \_cleanup\_by\_oldest  | 无  | None | 删除最早添加的记忆        |
| \_cleanup\_by\_least\_used | 无 | None | 删除最少使用的记忆        |
| \_cleanup\_by\_relevance | 无 | None | 删除与核心内容最不相关的记忆 |

## 输入输出规范

- 输入：
  - 添加记忆: 文本片段列表、元数据列表(可选)
  - 查询记忆: 查询文本、检索参数(top_k, threshold等)
- 输出：
  - `add_segments()` 返回添加的条目ID列表
  - `query_similar()` 返回包含相似段落、相似度分数和元数据的字典列表
  - `rebuild_memory()` 返回布尔值，表示操作是否成功
  - `get_collection_stats()` 返回包含记忆库统计信息的字典

## 内部流程说明

1. 初始化或连接本地Chroma数据库，建立默认集合（如 `knowforge_memory`）。
2. 添加片段时：
   - 调用Embedder生成向量
   - 将向量+原始文本+元数据(时间戳、来源等)存入ChromaDB
   - 如超出最大容量，触发清理策略
3. 查询时根据指定的retrieval_mode选择检索策略：
   - simple: 纯语义相似度检索，使用向量相似度计算
   - time_weighted: 基于时间的加权检索，优先返回最近添加的内容
   - context_aware: 结合上下文的检索，合并查询文本和上下文文本的向量
   - hybrid: 混合检索策略，结合语义相似度和关键词匹配，增强结果相关性
4. 查询结果处理:
   - 应用相似度阈值过滤
   - 格式化标准返回结构(id, text, similarity, metadata)
   - 在hybrid模式下，若无匹配结果则自动回退到time_weighted策略
5. 记忆库管理:
   - 支持记忆库统计、导入导出、重建
   - 根据清理策略(oldest, least_used, relevance)维护记忆库大小

### 检索策略详细流程

#### 混合检索策略 (_hybrid_retrieval)

1. 首先进行语义检索获取候选结果，使用较宽松阈值
2. 提取查询文本中的关键词
3. 为每个候选结果计算关键词匹配得分
4. 组合语义相似度(80%)与关键词匹配度(20%)
5. 重新排序并应用最终阈值过滤
6. 若结果为空，自动回退到时间加权检索

#### 上下文感知检索 (_context_aware_retrieval)

1. 将查询文本与上下文文本向量化
2. 计算两者向量的平均值作为组合查询向量
3. 使用组合向量进行检索，增强上下文关联性
4. 无上下文时退化为简单相似度检索

---
# 8. LLMCaller 模块

## 类定义

```python
class LLMCaller:
    def __init__(self, model_name: str, api_key: str, api_base_url: str)
```

## 主要属性

| 属性名         | 类型  | 说明                           |
| ------------- | ---- | ---------------------------- |
| model_name    | str  | 使用的DeepSeek模型名称（chat或reasoner） |
| api_key       | str  | DeepSeek API密钥                |
| api_base_url  | str  | API调用的基础URL                 |

## 核心方法

| 方法名         | 参数                                         | 返回   | 说明                  |
| ------------ | ------------------------------------------ | ---- | ------------------- |
| call_model   | prompt: str, memory: list[str] = None      | str  | 调用模型，生成笔记文本       |
| build_prompt | input_text: str, memory: list[str] = None | str  | 构建最终prompt（带记忆片段） |

## 输入输出规范

- 输入：
  - prompt文本（当前片段）
  - （可选）辅助记忆内容列表
- 输出：
  - DeepSeek返回的总结/笔记文本（纯字符串）

## 内部流程说明

1. 构建Prompt：
   - 将当前文本片段作为主体内容
   - （可选）在Prompt中追加历史相关记忆片段
2. 调用DeepSeek API：
   - 使用requests.post发送请求
   - 包含模型名称、请求体、API Key等信息
3. 处理API返回：
   - 提取生成文本，返回给调用方
   - 出错时优雅异常处理并记录日志

---

# 9. OutputWriter 模块

## 类定义

```python
class OutputWriter:
    def __init__(self, workspace_dir: str, output_dir: str, locale_manager)
```

## 主要属性

| 属性名             | 类型            | 说明                      |
| --------------- | ------------- | ----------------------- |
| workspace\_dir  | str           | 工作区目录路径（如 `workspace/`） |
| output\_dir     | str           | 输出目录路径（如 `output/`）     |
| locale\_manager | LocaleManager | 语言管理器实例（用于本地化提示信息）      |

## 核心方法

| 方法名                | 参数                                 | 返回  | 说明                          |
| ------------------ | ---------------------------------- | --- | --------------------------- |
| generate\_markdown | notes: list[str], filename: str    | str | 生成Markdown文件并返回文件路径         |
| generate\_notebook | notes: list[str], filename: str    | str | 生成Jupyter Notebook文件并返回文件路径 |
| generate\_pdf      | markdown\_path: str, filename: str | str | 将Markdown渲染为PDF文件并返回文件路径    |

## 输入输出规范

- 输入：
  - notes：文本段落列表（通常是生成的笔记内容）
  - filename：输出文件名（不带扩展名）
- 输出：
  - 对应格式的文件（.md、.ipynb、.pdf）保存到 `output/` 下，并返回文件路径

## 内部流程说明

1. Markdown生成：
   - 逐段插入文本，按照预设模板组织内容。
2. Notebook生成：
   - 将每段文本封装成一个Markdown单元，生成标准.ipynb结构。
3. PDF生成：
   - 将Markdown文件转为HTML，再渲染为高质量PDF（使用weasyprint）。
4. 输出过程中记录日志与异常捕获，保证中断恢复性。

---

# 10. Processor 模块

## 类定义

```python
class Processor:
    def __init__(self, input_dir: str, output_dir: str, config_path: str)
```

## 主要属性

| 属性名             | 类型             | 说明                      |
| --------------- | -------------- | ----------------------- |
| input\_dir      | str            | 输入源目录路径（如 `input/`）     |
| output\_dir     | str            | 输出目录路径（如 `output/`）     |
| config\_loader  | ConfigLoader   | 配置加载器实例                 |
| locale\_manager | LocaleManager  | 多语言管理器实例                |
| logger          | logging.Logger | 日志对象                    |
| workspace\_dir  | str            | 工作区目录路径（如 `workspace/`） |

## 核心方法

| 方法名                 | 参数                 | 返回   | 说明                       |
| ------------------- | ------------------ | ---- | ------------------------ |
| run\_full\_pipeline | formats: list[str] | None | 执行完整处理流程（从输入到输出，支持多格式生成） |

## 输入输出规范

- 输入：
  - 输入目录中的各种文件（PDF、图片、代码、链接）
  - 用户指定的输出格式（如markdown、ipynb、pdf）
- 输出：
  - 指定格式的笔记文件保存到输出目录

## 内部流程说明

1. 初始化核心模块：InputHandler、Splitter、Embedder、MemoryManager、LLMCaller、OutputWriter。
2. 调用InputHandler提取纯文本。
3. 使用Splitter拆分文本片段。
4. 使用MemoryManager存储记忆。
5. 遍历片段，调用LLMCaller生成笔记内容（支持参考记忆检索）。
6. 调用OutputWriter生成指定格式输出文件。
7. 记录完整日志，异常统一捕获处理。

---

# 11. Exception Classes 模块

## 基础异常类

```python
class NoteGenError(Exception):
    """KnowForge项目的基础异常类型，所有自定义异常继承它"""
    pass
```

## 具体异常子类

| 类名          | 继承自          | 说明        |
| ----------- | ------------ | --------- |
| InputError  | NoteGenError | 输入处理相关错误  |
| APIError    | NoteGenError | API调用相关错误 |
| MemoryError | NoteGenError | 记忆管理相关错误  |
| OutputError | NoteGenError | 输出生成相关错误  |

## 主要属性与特性

- 继承自Python标准Exception类。
- 允许自定义异常消息内容。
- 支持统一异常捕获与分类型日志记录。

## 使用示例

```python
from src.utils.locale_manager import LocaleManager

try:
    locale = LocaleManager("resources/locales/locale.yaml", "zh")
    # 可能抛出异常的操作
    raise InputError(locale.get("errors.input_error"))
except NoteGenError as e:
    logger.error(f"Known Error: {str(e)}")
    print(locale.get("system.error_occurred"))
```

## Locale文件新增字段要求（resources/locales/locale.yaml）

```yaml
errors:
  input_error: "输入文件格式不受支持"
system:
  error_occurred: "系统运行过程中出现错误"
```

## 内部流程说明

1. 所有可预期错误通过NoteGenError及其子类抛出。
2. Processor主流程或CLI入口统一捕获NoteGenError：
   - 记录错误日志（本地化）
   - 向用户输出友好提示（使用LocaleManager）
3. 未知错误（Exception）则进入兜底异常处理流程。

---

# 12. Scripts工具集模块

## 设计目标

- 提供运维辅助工具，简化开发与测试过程。
- 包括清理工作区、重建记忆库、导出配置信息等常用功能。
- 保持脚本模块化、轻量、易调用。
- 严格遵循静态数据与动态数据分离原则，所有提示文本通过LocaleManager加载。

## 主要脚本列表

| 脚本文件                    | 主要功能                 |
| ----------------------- | -------------------- |
| clean\_workspace.py     | 清空并重建workspace/工作目录  |
| rebuild\_memory.py      | 重新提取文本并重建ChromaDB记忆库 |
| export\_config\_docs.py | 导出当前配置为Markdown文档    |

## 典型脚本设计示例（国际化修正版）

### clean\_workspace.py

```python
import shutil
import os
from src.utils.locale_manager import LocaleManager

def clean_workspace(workspace_dir="workspace/"):
    locale = LocaleManager("resources/locales/locale.yaml", "zh")
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    print(locale.get("scripts.clean_workspace_success"))

if __name__ == "__main__":
    clean_workspace()
```

### rebuild\_memory.py

```python
from src.note_generator.input_handler import InputHandler
from src.note_generator.splitter import Splitter
from src.note_generator.memory_manager import MemoryManager
from src.note_generator.embedder import Embedder
from src.utils.locale_manager import LocaleManager

def rebuild_memory(input_dir="input/", workspace_dir="workspace/"):
    locale = LocaleManager("resources/locales/locale.yaml", "zh")
    handler = InputHandler(input_dir, workspace_dir)
    texts = handler.extract_texts()
    splitter = Splitter()
    segments = splitter.split_texts(texts)
    embedder = Embedder()
    memory = MemoryManager("workspace/memory_db/", embedder)
    memory.rebuild_memory(segments)
    print(locale.get("scripts.rebuild_memory_success"))

if __name__ == "__main__":
    rebuild_memory()
```

### export\_config\_docs.py

```python
from src.utils.config_loader import ConfigLoader
from src.utils.locale_manager import LocaleManager

def export_config(config_path="resources/config/config.yaml", output_path="docs/config_reference.md"):
    locale = LocaleManager("resources/locales/locale.yaml", "zh")
    config = ConfigLoader(config_path).config
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"### {key}\n\n{value}\n\n")
    print(locale.get("scripts.export_config_success"))

if __name__ == "__main__":
    export_config()
```

## Locale文件新增字段要求（resources/locales/locale.yaml）

```yaml
scripts:
  clean_workspace_success: "✅ Workspace已清空并重新初始化"
  rebuild_memory_success: "✅ Memory数据库重建完成"
  export_config_success: "✅ 配置文档导出完成"
```
