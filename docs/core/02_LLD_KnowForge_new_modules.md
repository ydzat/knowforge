# 文档综合处理模块详细设计

---

# 5. DocumentAnalyzer 模块

## 类定义

```python
class DocumentAnalyzer:
    def __init__(self, config: Dict[str, Any] = None)
```

## 主要属性

| 属性名 | 类型 | 说明 |
|:----|:----|:----|
| config | Dict[str, Any] | 配置选项 |
| min_text_block_size | int | 最小文本块尺寸 |
| min_image_area | int | 最小图像区域 |
| table_detection_confidence | float | 表格检测置信度阈值 |
| formula_detection_confidence | float | 公式检测置信度阈值 |
| layout_analysis_mode | str | 布局分析模式 ('simple', 'advanced', 'ml') |

## 核心方法

| 方法名 | 参数 | 返回 | 说明 |
|:----|:----|:----|:----|
| analyze_document | file_path: str | List[Dict[str, Any]] | 分析文档结构，识别内容类型和区域 |
| _identify_content_type | content_area: Region | str | 识别内容区域类型 |
| _detect_tables | page | List[Dict] | 检测页面中的表格区域 |
| _detect_formulas | page | List[Dict] | 检测页面中的数学公式 |
| _detect_images | page | List[Dict] | 检测页面中的图像区域 |
| _detect_text_blocks | page | List[Dict] | 检测页面中的文本块 |
| get_document_structure | file_path: str | Dict[str, Any] | 获取文档大纲结构 |

## 输入输出规范

- **输入**：PDF文档文件路径
- **输出**：内容块列表，每个块包含以下信息：
  ```python
  {
      "type": "text"|"image"|"table"|"formula", # 内容类型
      "page_num": int,                         # 页码
      "bbox": (x0, y0, x1, y1),                # 边界框
      "content_id": str,                       # 内容唯一ID
      "metadata": {                            # 元数据
          "is_header": bool,                   # 是否为标题
          "level": int,                        # 标题级别(仅对标题有效)
          "confidence": float,                 # 识别置信度
          "context": str                       # 上下文信息
      }
  }
  ```

## 内部流程说明

1. 加载PDF文档，遍历每个页面
2. 对每个页面应用布局分析算法
3. 识别文本区域，判断是否为标题、段落等
4. 识别图像区域，提取位置信息
5. 应用表格检测算法，识别表格区域
6. 应用公式检测算法，识别数学公式
7. 对所有识别的内容区域按顺序排列
8. 建立内容区域之间的层级和关联关系
9. 返回结构化的内容块列表

---

# 6. ContentExtractor 模块

## 类定义

```python
class ContentExtractor:
    def __init__(self, config: Dict[str, Any] = None)
```

## 主要属性

| 属性名 | 类型 | 说明 |
|:----|:----|:----|
| config | Dict[str, Any] | 配置选项 |
| image_format | str | 提取图像的格式 ('PNG', 'JPEG', 'TIFF') |
| image_resolution | int | 提取图像分辨率DPI |
| table_extraction_method | str | 表格提取方法 ('lattice', 'stream') |
| formula_extraction_method | str | 公式提取方法 ('image', 'mathml', 'latex') |

## 核心方法

| 方法名 | 参数 | 返回 | 说明 |
|:----|:----|:----|:----|
| extract_content | doc, content_blocks: List[Dict] | Dict[str, Any] | 根据内容块列表提取所有内容 |
| extract_text | doc, content_block: Dict | str | 提取纯文本内容 |
| extract_image | doc, content_block: Dict | np.ndarray | 提取图像内容为NumPy数组 |
| extract_table | doc, content_block: Dict | Dict | 提取表格内容和结构 |
| extract_formula | doc, content_block: Dict | Dict | 提取公式内容 |
| get_surrounding_context | doc, content_block: Dict, context_range: int | str | 获取内容周围的上下文文本 |

## 输入输出规范

- **输入**：
  - PyMuPDF文档对象
  - DocumentAnalyzer生成的内容块列表

- **输出**：
  - 提取的内容字典，格式如下：
  ```python
  {
      "content_id": {
          "type": "text"|"image"|"table"|"formula",
          "content": str|np.ndarray|Dict,    # 根据类型不同而不同
          "page_num": int,
          "metadata": Dict                   # 原始元数据加提取相关信息
      }
  }
  ```

## 内部流程说明

1. 遍历DocumentAnalyzer提供的内容块列表
2. 根据内容块的类型，调用对应的提取方法
3. 文本提取：使用PyMuPDF的文本提取功能，保留格式信息
4. 图像提取：转换为标准格式的NumPy数组，应用分辨率调整
5. 表格提取：
   - 'lattice'模式：基于表格线条提取
   - 'stream'模式：基于空间关系提取
6. 公式提取：
   - 'image'模式：作为图像提取
   - 'mathml'/'latex'模式：尝试提取为结构化格式
7. 为每种内容类型添加周围的上下文信息
8. 返回统一格式的内容字典

---

# 7. ContentProcessor 模块

## 类定义

```python
class ContentProcessor:
    def __init__(self, ocr_llm_processor, config: Dict[str, Any] = None)
```

## 主要属性

| 属性名 | 类型 | 说明 |
|:----|:----|:----|
| ocr_llm_processor | OCRLLMProcessor | OCR-LLM处理器实例 |
| config | Dict[str, Any] | 配置选项 |
| table_formatter | TableFormatter | 表格格式化工具 |
| formula_processor | FormulaProcessor | 公式处理工具 |
| embedding_manager | EmbeddingManager | 向量嵌入管理器(用于相关内容检索) |
| splitter | Splitter | 文本拆分器实例 |

## 核心方法

| 方法名 | 参数 | 返回 | 说明 |
|:----|:----|:----|:----|
| process_all | extracted_contents: Dict[str, Any] | Dict[str, Any] | 处理所有提取的内容 |
| process_text | text: str, metadata: Dict = None | str | 处理纯文本内容 |
| process_image | image: np.ndarray, context: str = None, metadata: Dict = None | str | 处理图像内容 |
| process_table | table_data: Dict, context: str = None, metadata: Dict = None | str | 处理表格内容 |
| process_formula | formula_data: Dict, context: str = None, metadata: Dict = None | str | 处理公式内容 |
| _enhance_with_knowledge | content: str, content_type: str, context: str = None | str | 用知识库增强处理结果 |
| _apply_llm_processing | content: str, prompt_template: str, context: str = None | str | 应用LLM处理增强内容 |

## 输入输出规范

- **输入**：
  - ContentExtractor提取的内容字典
  
- **输出**：
  - 处理后的内容字典，格式如下：
  ```python
  {
      "content_id": {
          "type": "text"|"image"|"table"|"formula",
          "original_content": <原始内容>,
          "processed_content": str,          # 处理后的文本内容
          "metadata": Dict,                  # 包含处理信息的元数据
          "page_num": int
      }
  }
  ```

## 内部流程说明

1. 遍历ContentExtractor提供的各类型内容
2. 文本处理：
   - 使用Splitter按语义进行拆分
   - 保留标题和段落结构
3. 图像处理：
   - 应用现有OCR-LLM流程
   - 利用上下文信息增强识别准确性
4. 表格处理：
   - 识别表头和数据关系
   - 转换为Markdown/HTML表格格式
   - 应用LLM解释表格内容和结构
5. 公式处理：
   - 转换为LaTeX格式
   - 应用LLM解释公式含义
6. 知识库增强：
   - 检索相关领域知识
   - 补充专业术语解释
   - 纠正潜在识别错误
7. 保存原始内容和处理后内容，便于后续调整

---

# 8. ContentIntegrator 模块

## 类定义

```python
class ContentIntegrator:
    def __init__(self, config: Dict[str, Any] = None)
```

## 主要属性

| 属性名 | 类型 | 说明 |
|:----|:----|:----|
| config | Dict[str, Any] | 配置选项 |
| structure_preservation_level | str | 结构保留级别 ('strict', 'balanced', 'content_first') |
| connection_templates | Dict[str, str] | 不同内容类型间的连接模板 |
| heading_format | Dict[str, str] | 标题格式化模板 |
| output_format | str | 输出格式 ('markdown', 'html', 'text') |

## 核心方法

| 方法名 | 参数 | 返回 | 说明 |
|:----|:----|:----|:----|
| integrate | processed_contents: Dict[str, Any], doc_structure: Dict[str, Any] = None | List[str] | 整合处理后的内容 |
| _maintain_structure | processed_contents: Dict[str, Any], doc_structure: Dict[str, Any] | List[Dict] | 保持原始文档结构 |
| _order_content_blocks | processed_contents: Dict[str, Any] | List[str] | 按页面和位置排序内容块 |
| _insert_connective_text | ordered_blocks: List[Dict] | List[Dict] | 在内容块之间插入连接文本 |
| _format_for_output | integrated_content: List[Dict] | List[str] | 根据输出格式生成最终内容 |
| _handle_cross_references | content: List[Dict] | List[Dict] | 处理内容之间的交叉引用 |
| generate_toc | content: List[Dict] | str | 生成内容目录 |

## 输入输出规范

- **输入**：
  - ContentProcessor处理后的内容字典
  - 可选的文档结构信息

- **输出**：
  - 整合后的内容段落列表，每段对应文档中的一个语义完整部分
  - 保持了原始文档的结构和顺序
  - 同时包含所有类型的内容（文本、图像描述、表格、公式）

## 内部流程说明

1. 加载文档结构信息（如果有）
2. 按页码和位置对内容块进行排序
3. 识别标题层级，建立内容层次结构
4. 根据内容类型选择合适的展示格式：
   - 文本：直接使用处理后内容
   - 图像：添加标题和描述
   - 表格：使用Markdown/HTML表格格式
   - 公式：使用适合输出格式的公式表示
5. 在内容块之间添加自然过渡文本
6. 处理内容之间的交叉引用
7. 生成文档目录（可选）
8. 根据指定格式生成最终输出
9. 确保输出内容保持原始文档的结构和顺序
