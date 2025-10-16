# KnowForge 重构指南（与 LingCraft 解耦版）

> 本文档目标是将 KnowForge 明确重构为一个"专注于多源内容处理与结构化笔记生成的知识内核模块"，符合单一责任原则，并能被 LingCraft 或其他上层系统通过接口调用。

---

## LingCraft 简介（非 KnowForge 组成部分）

**LingCraft 是一个独立的智能学习系统**，聚焦于语言与知识的个性化学习路径规划、错题管理、内容生成、练习反馈等任务。它通过调用 KnowForge 作为知识处理内核，完成内容结构化工作，但自身并不属于 KnowForge 的组成部分。两者为模块化解耦、可独立部署的程序。

| 项目        | 职责                   | 是否包含于 KnowForge |
| --------- | -------------------- | --------------- |
| LingCraft | 用户交互、学习流程、输出控制、反馈生成  | ❌ 不属于           |
| KnowForge | 多源输入解析、笔记结构化、向量知识库生成 | ✅ 本文档核心内容       |

---

## 一. 重构目标与原因

### 目标

* 将 KnowForge 转化为 **纯粹的知识处理库/服务层**：解耦输入文档，转为结构化知识
* 别于 LingCraft：KnowForge **不再负责最终输出** (Markdown/文章/笔记格式)，这些由 LingCraft 负责格式化和展示
* 支持作为 **Python SDK + REST API** 被外部调用

### 原因

* 简化架构，减轻重处功能，使其更适合做为预处理系统/后端服务
* 遵循 **SRP (单一责任原则)** ：LingCraft 进行专注的学习调度，KnowForge 负责知识分析
* 方便后续的 **跨项目重用 / 插件化打包**

---

## 二. 新性质定位

**KnowForge = “AI 知识分析 + 分组 + 向量化结构化库”**

* 输入类型：PDF、图片(包含OCR)、网页、代码
* 输出类型：**结构化知识对象 (NoteBlock / Section / TopicGraph)**
* 定位类型：Python SDK 、Headless Service 、LLM Agent Plugin

---

## 三. 重构后模块分工

### 保留模块

| 模块                  | 功能                              |
| ------------------- | ------------------------------- |
| `input_handler.py`  | 解析输入文档/图片/网页                    |
| `splitter.py`       | 智能拆分文本，根据段落/Token缩界             |
| `embedder.py`       | 文本向量化                           |
| `memory_manager.py` | ChromaDB 向量记忆库操作                |
| `llm_caller.py`     | 集成 GPT / DeepSeek API 等         |
| `note_generator.py` | 根据分段 + 向量 + LLM 生成 NoteBlock\[] |

### 移除/被别系统接管的模块

| 模块                 | 处理                    |
| ------------------ | --------------------- |
| `output_writer.py` | 移除，后续输出由 LingCraft 接管 |
| `nbformat/文档/模板`   | 别离并进入 LingCraft 展示模块  |

### 新增接口

* `get_structured_notes(input_dir) -> List[NoteBlock]`
* `get_embedding(text: str) -> np.array`
* `query_similar(query: str) -> List[RelatedBlock]`

---

## 四. 对外调用 API/接口设计

### 类型定义

```python
class NoteBlock:
    id: str
    section_title: str
    content: str
    keywords: List[str]
    embedding: np.array
    metadata: dict  # source, timestamp, language, etc
```

### Python SDK

```python
from knowforge.interface import KnowForgeProcessor

kf = KnowForgeProcessor(config_path="resources/config.yaml")
notes = kf.get_structured_notes("input/ch2.pdf")
for nb in notes:
    print(nb.section_title, nb.keywords)
```

### REST API (Flask FastAPI 样式)

```http
POST /process
Content-Type: multipart/form-data

[file: pdf/image/link.txt]

-->
Response:
{
  "notes": [
    {"title": "...", "content": "...", "keywords": [...], "embedding": [...]},
    ...
  ]
}
```

---

## 五. 处理流程重构图

```plaintext
[Input: PDF / IMG / Link]
      |
      v
[InputHandler] → [Splitter] → [Embedder]
      |                  ↓
      |            [MemoryManager] <===> [query_similar()]
      v
[LLMCaller] → generate structured notes (NoteBlock[])
      v
[Return: Structured Notes → LingCraft]
```

---

## 六. 功能模块合并和整理性性推荐

| 模块                               | 处理方式                                |
| -------------------------------- | ----------------------------------- |
| `processor.py`                   | 组装成一个 `KnowForgeProcessor` 类，提供外部接口 |
| `tests/`                         | 保留，简化成单模块单次功能测试                     |
| `cli/`                           | 可选，作为 debug/本地使用控制器                 |
| `logger`, `exceptions`, `locale` | 保留，进行输出处理缩约                         |

---

## 七. 应用场景和环境场景

| 场景         | 接入方式                                  |
| ---------- | ------------------------------------- |
| LingCraft  | Python SDK (local) or REST API (远程部署) |
| VSCode插件   | 调用 REST API 进行“学术PDF笔记提取”             |
| Obsidian插件 | 基于本地 SDK 生成结构卡片嵌入 markdown            |
| 教学 SaaS    | 后端部署 KnowForge，接收课程资料生成课件笔记           |

---

## 八. 重构后成果

* 更轻量级的知识处理工具
* 可提供 SDK 和 REST API 被外部调用
* 使用环境可以是小型定制应用，也可以是公有运维服务
* 为 LingCraft 和未来的后端知识系统打造了根基

---

# KnowForge 功能需求文档（支持 LingCraft 多笔记本/知识空间场景）

## 🧭 文档目的

本功能需求文档旨在明确：为配合 LingCraft 的多笔记本架构，KnowForge 必须支持“**逻辑上独立、语义上可关联**”的知识空间（Knowledge Space）管理功能，并通过 API 形式暴露相关能力，同时满足 LingCraft 的学习与复习模式对高效知识检索的需求。

---

## 🔰 补充目标：支持 LingCraft 的学习/复习模式

LingCraft 将提供学习模式（总结、结构化知识点）与复习模式（卡片回顾、自适应练习）。这要求 KnowForge：

* 提供**高效的向量检索与标签筛选能力**
* 支持根据知识类型（如错题、定义、例题、难点）快速调用相关内容
* 返回结构化格式，便于 LingCraft 生成卡片或交互式练习单元

---

## 🧱 一、核心概念定义

| 概念                    | 说明                                        |
| --------------------- | ----------------------------------------- |
| 笔记本（Notebook）         | 用户在 LingCraft 中创建的学习主题单位，如“线性代数”、“计算机网络”等 |
| 知识空间（Knowledge Space） | KnowForge 内部针对每个笔记本生成的独立向量库或命名空间          |
| 跨知识空间语义关联             | 不同知识空间中知识片段的相似性或引用关系，允许建立非显式连接            |

---

## 🧩 二、KnowForge 必须支持的功能点

### 1. 多知识空间管理能力

* 创建/删除知识空间
* 切换当前激活知识空间（用于当前导入/查询）
* 获取当前系统中全部知识空间列表（含元信息）

### 2. 每个知识空间支持：

* 多模态文档输入（PDF、图像、网页、代码）
* 文本拆分、向量化、笔记生成
* 向量检索（仅在当前空间内或跨空间）
* 获取该空间内所有笔记索引、标签结构、关键词聚类

### 3. 知识空间间的语义关联（新增）

* 提供跨知识空间相似查询（如：“本知识空间中某段内容是否与其他空间内容重复？”）
* 提供基于向量相似度的“知识桥”结构：用于建立弱连接图谱（后期供 LingCraft 可视化）

### 4. 支持 LingCraft 的知识检索优化（新增）

* 提供基于**关键词、标签、结构层级**的快速筛选
* 提供“难度标签”、“错题标记”、“定义类”、“例题类”等类型维度检索
* 支持“知识点回顾模式”：

  * 输入：`topic=Gradient Descent`
  * 输出：结构化知识片段 + 示例句 + 典型错解（若有）
* 查询方式：基于向量 + 分类标签的**组合式多通道检索**

---

## 📤 三、API 设计需求

### 1. 知识空间管理接口

```http
POST /api/spaces/create
{
  "space_id": "ml2025",
  "display_name": "机器学习",
  "description": "2025春季机器学习课程笔记"
}

GET /api/spaces/list
→ 返回所有知识空间及元数据

POST /api/spaces/delete
{
  "space_id": "ml2025"
}
```

### 2. 激活知识空间（上下文切换）

```http
POST /api/spaces/activate
{
  "space_id": "ml2025"
}
```

或通过 SDK：

```python
kf.set_active_space("ml2025")
kf.ingest("pdfs/ch1.pdf")  # 自动进入当前空间
```

### 3. 跨空间向量检索

```http
POST /api/query/cross_space
{
  "query": "线性回归的基本假设",
  "top_k": 3,
  "include_spaces": ["ml2025", "linear_algebra"],
  "mode": "semantic"
}
```

### 4. 标签/结构化主题查询（面向复习模式）

```http
POST /api/query/by_tag
{
  "tag": "definition",
  "topic": "Bayes Theorem",
  "space_id": "probability101",
  "top_k": 5
}
```

### 5. 知识点“复习包”生成接口（实验性）

```http
GET /api/review/pack?topic=Decision+Trees&space_id=ml2025
```

返回：

* 概念定义块
* 关键推导
* 示例题
* 易错点

---

## 🧠 四、内部存储策略建议

| 层级          | 描述                                           |
| ----------- | -------------------------------------------- |
| 工作空间路径      | 每个知识空间对应独立向量索引与文本缓存路径，如：`/workspace/ml2025/` |
| ChromaDB 分区 | 支持按空间名分区存储，避免混淆（当前已使用 ChromaDB）              |
| Metadata 路径 | 每个空间独立管理笔记索引、标签、拆分结果、历史摘要等                   |

### 对 ChromaDB 的使用建议

* 每个知识空间对应一个 Chroma collection
* 向量插入时应包含 metadata：`{"tag": "definition", "difficulty": 2, "source": "ocr", ...}`
* 查询时支持多条件过滤（vector + tag）

---

## 🔐 五、权限与安全建议（未来）

* 支持为每个空间设置访问 token / 权限标识符（对接团队、多人协作）
* 日志记录每次空间切换与数据注入行为
* 所有查询需声明作用空间（或显式跨空间）

---

## 🎯 六、未来扩展建议（面向 LingCraft 深度协同）

* 支持知识空间之间的“依赖图谱”：如课程 A 显式声明依赖课程 B
* 为每个知识空间生成自动“学习路径图”（由 LingCraft 使用）
* 支持在知识空间中嵌入“Notebook 笔记引用”回链，作为结构化链接
* 增加“主题标签树”结构，实现更细粒度的复习模式匹配（如“卷积神经网络 > 反向传播 > 权重梯度消失”）

---

本功能需求文档作为 KnowForge 的中期重构目标，旨在为 LingCraft 构建“多笔记本、弱关联、强推理”的知识支撑底座。如果你需要，我可以进一步起草 API 路由定义文档（OpenAPI）或修改当前 KnowForge 接口结构。
