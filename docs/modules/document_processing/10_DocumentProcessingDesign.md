# 文档综合处理设计文档

## 1. 背景与目的

当前的KnowForge系统需要用户手动区分不同类型的输入（如PDF、图片等），且在处理PDF文档时无法自动识别和处理其中包含的非文本内容（如图片、表格、公式等）。这限制了系统的易用性和功能完整性。本文档旨在设计一个文档综合处理系统，使KnowForge能够自动识别和处理文档中的多种内容类型，无需用户手动划分。

## 2. 设计目标

1. 自动识别和提取PDF文档中的文本、图片、表格和公式内容
2. 为每种内容类型应用适当的处理方法
3. 整合处理结果，保持原始文档的结构和语义连贯性
4. 与现有的OCR-LLM-知识库集成无缝对接
5. 在v1.0正式版前完成实现

## 3. 系统架构

### 3.1 核心模块

1. **DocumentAnalyzer**：文档结构分析器
   - 负责识别文档结构，划分内容区域类型
   - 将文档分解为不同类型的内容块

2. **ContentExtractor**：内容提取器
   - 提取纯文本内容
   - 提取图像内容
   - 提取表格内容
   - 提取公式内容

3. **ContentProcessor**：内容处理器
   - 文本处理：直接传递给Splitter
   - 图像处理：应用OCR-LLM流程
   - 表格处理：表格识别与结构化
   - 公式处理：数学公式识别与LaTeX转换

4. **ContentIntegrator**：内容整合器
   - 按原始顺序整合各类型内容
   - 保持原始文档结构
   - 生成统一格式的处理结果

### 3.2 工作流程

```
文档输入 -> 文档分析 -> 内容提取 -> 类型分发 -> 专项处理 -> 内容整合 -> 统一输出
```

## 4. 详细设计

### 4.1 DocumentAnalyzer类

```python
class DocumentAnalyzer:
    def __init__(self, config: Dict[str, Any] = None):
        """初始化文档分析器"""
        
    def analyze_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        分析文档结构，识别内容类型
        
        Args:
            file_path: 文档路径
            
        Returns:
            List[Dict]: 内容块列表，每个块包含类型、位置、内容等信息
        """
        
    def _identify_content_type(self, content_area) -> str:
        """
        识别内容区域类型
        
        Returns:
            str: 'text', 'image', 'table', 'formula'中的一种
        """
```

### 4.2 ContentExtractor类

```python
class ContentExtractor:
    def __init__(self, config: Dict[str, Any] = None):
        """初始化内容提取器"""
        
    def extract_text(self, doc, content_block: Dict) -> str:
        """提取纯文本内容"""
        
    def extract_image(self, doc, content_block: Dict) -> np.ndarray:
        """提取图像内容"""
        
    def extract_table(self, doc, content_block: Dict) -> Dict:
        """提取表格内容"""
        
    def extract_formula(self, doc, content_block: Dict) -> Dict:
        """提取公式内容"""
```

### 4.3 ContentProcessor类

```python
class ContentProcessor:
    def __init__(self, ocr_llm_processor, config: Dict[str, Any] = None):
        """
        初始化内容处理器
        
        Args:
            ocr_llm_processor: OCR-LLM处理器实例
            config: 配置选项
        """
        
    def process_text(self, text: str) -> str:
        """处理纯文本内容"""
        
    def process_image(self, image: np.ndarray, context: str = None) -> str:
        """
        处理图像内容
        
        Args:
            image: 图像数据
            context: 图像上下文（周围的文本）
        """
        
    def process_table(self, table_data: Dict) -> str:
        """处理表格内容，返回结构化文本"""
        
    def process_formula(self, formula_data: Dict) -> str:
        """处理公式内容，返回LaTeX或其他格式"""
```

### 4.4 ContentIntegrator类

```python
class ContentIntegrator:
    def __init__(self, config: Dict[str, Any] = None):
        """初始化内容整合器"""
        
    def integrate(self, processed_blocks: List[Dict[str, Any]]) -> List[str]:
        """
        整合处理后的内容块
        
        Args:
            processed_blocks: 处理后的内容块列表
            
        Returns:
            List[str]: 整合后的内容段落
        """
        
    def _maintain_structure(self, processed_blocks: List[Dict[str, Any]]) -> List[str]:
        """保持原始文档结构"""
```

## 5. 技术选型

### 5.1 PDF处理技术

- **PyMuPDF (fitz)**: 用于提取PDF中的文本、图像和结构信息
- **PDF元素定位算法**：基于页面坐标系统，识别文本块、图像块、表格区域

### 5.2 表格识别技术

- **Camelot**: Python库，专门用于PDF表格提取
- **Tabula-py**: 另一个表格提取备选方案
- **自定义表格识别算法**：结合边界框检测和格线分析

### 5.3 公式识别技术

- **Math OCR**: 开源数学公式OCR系统
- **MathPix API**: 商业API，可选用于高质量公式识别
- **LaTeX转换工具**: 将识别后的公式转为LaTeX格式

## 6. 实现计划

### 6.1 v0.1.5 (PDF内容综合提取功能)

- 实现DocumentAnalyzer基本功能：PDF文本和图像区域识别
- 实现ContentExtractor的文本和图像提取功能
- 集成现有OCR-LLM流程处理提取的图像

### 6.2 v0.1.6 (表格与公式专项处理)

- 完善DocumentAnalyzer：添加表格和公式区域识别
- 实现ContentExtractor的表格和公式提取功能
- 开发ContentProcessor的表格处理功能
- 初步实现公式识别与处理功能

### 6.3 v0.1.7-v0.2.0 (内容整合与格式保留)

- 完善所有处理器功能
- 实现ContentIntegrator整合功能
- 确保在最终笔记中保留原始文档结构
- 优化表格和公式在不同输出格式中的展示

## 7. 评估指标

- **识别准确率**: 内容类型识别准确率 > 90%
- **处理完整性**: 至少处理95%的文档内容(无遗漏)
- **结构保持**: 输出保持原始文档的主要结构和信息流
- **处理时间**: 平均每页处理时间 < 30秒

## 8. 风险与挑战

1. **复杂布局处理**: 杂志、学术论文等复杂布局文档的正确识别
2. **非标准表格**: 无边界或合并单元格表格的准确识别
3. **手写公式**: 手写数学公式的识别难度较高
4. **多语言处理**: 不同语言混合文档的处理

## 9. 与现有系统的集成

本设计将通过以下方式与现有系统集成：

1. 扩展InputHandler以支持新的综合处理流程
2. 重用现有的OCR-LLM-知识库集成流程处理图像内容
3. 增强Splitter以保持不同类型内容之间的语义连贯性
4. 修改OutputWriter以支持表格和公式的特殊展示需求
