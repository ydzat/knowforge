# KnowForge文档综合处理 - 实施计划 (v0.1.5-v0.1.6)

## 概述

本文档提供了KnowForge项目v0.1.5和v0.1.6版本中，文档综合处理功能的详细实施计划。该计划基于现有的[文档综合处理设计文档](../document_processing/10_DocumentProcessingDesign.md)，针对尚未完成的部分进行具体规划和实施指南。

## 当前开发状态

截至v0.1.6版本，文档综合处理核心功能已全部实现完成：

### 文档综合处理 (0.1.5部分)

- ✅ DocumentAnalyzer的实现 - 完成于`src/note_generator/document_analyzer.py`
- ✅ ContentExtractor的实现 - 完成于`src/note_generator/content_extractor.py`
- ✅ PDF解析器增强 - 通过`EnhancedImageExtractor`提供多种图像提取方法
- ✅ 内容整合功能 - 完成于`src/note_generator/content_integrator.py`

### 表格与公式专项处理 (0.1.6部分)

- ✅ 表格和公式区域识别 - 在`DocumentAnalyzer`中通过`_detect_tables`和`_detect_formulas`方法实现
- ✅ 表格处理库集成 - 在`TableProcessor`中集成Camelot和Tabula
- ✅ 数学公式OCR与LaTeX转换 - 在`FormulaProcessor`中集成Mathpix API实现
- ✅ ContentProcessor开发 - 完成于`src/note_generator/content_processor.py`

### 已实现的主要功能

1. **文档分析与内容提取**：
   - 识别文本、图像、表格和公式区域
   - 提取和处理各类型内容
   - 增强型PDF解析，解决常见的图像提取问题

2. **表格和公式处理**：
   - 多种表格处理引擎（Camelot、Tabula和自定义处理器）
   - 表格转Markdown和结构增强
   - 公式OCR识别与LaTeX格式转换

3. **内容整合与格式保持**：
   - 保持原始文档结构
   - 支持多种输出格式

## 实施计划

### 一、文档综合处理 (v0.1.5)

#### 1. DocumentAnalyzer实现

**目标**: 创建能够自动识别文档中的文本、图像、表格和公式区域的分析器。

**核心实现步骤**:

1. **基础类结构**:
   ```python
   # src/note_generator/document_analyzer.py
   class DocumentAnalyzer:
       def __init__(self, config=None):
           # 初始化配置
           self.config = config or {}
           self.logger = get_module_logger("DocumentAnalyzer")
           
       def analyze_document(self, document_path):
           """分析文档，识别不同内容区域"""
           # 根据文件类型选择不同的分析方法
           ext = os.path.splitext(document_path)[1].lower()
           if ext == '.pdf':
               return self._analyze_pdf(document_path)
           else:
               raise ValueError(f"不支持的文档类型: {ext}")
   ```

2. **PDF区域识别**:
   - 使用PyMuPDF提取基本信息
   - 识别文本块、图像块
   - 基于规则和机器学习方法识别表格和公式

3. **结果存储格式**:
   ```python
   # 分析结果示例
   result = {
       "document_type": "pdf",
       "total_pages": 10,
       "blocks": [
           {
               "type": "text", 
               "page": 1,
               "coordinates": [x0, y0, x1, y1],
               "content": "文本内容",
               "confidence": 0.95
           },
           {
               "type": "image",
               "page": 1,
               "coordinates": [x0, y0, x1, y1],
               "image_data": binary_data,
               "caption": "图片标题"
           },
           # ... 其他块
       ]
   }
   ```

**优先级与时间安排**:
- 高优先级: PDF文本和图像区域识别 (1周)
- 中优先级: 基本表格区域识别 (3天)
- 低优先级: 公式区域识别 (2天)

#### 2. ContentExtractor实现

**目标**: 从DocumentAnalyzer识别的区域中提取实际内容。

**核心实现步骤**:

1. **基础类结构**:
   ```python
   # src/note_generator/content_extractor.py
   class ContentExtractor:
       def __init__(self, config=None):
           self.config = config or {}
           self.logger = get_module_logger("ContentExtractor")
           
       def extract_content(self, document_blocks):
           """从文档块中提取内容"""
           extracted_blocks = []
           for block in document_blocks:
               extracted = self._extract_block(block)
               if extracted:
                   extracted_blocks.append(extracted)
           return extracted_blocks
   ```

2. **不同内容类型的提取方法**:
   - 文本提取: 直接使用内容
   - 图像提取: 保存为临时文件或内存中的图像
   - 表格提取: 将表格区域转换为结构化数据
   - 公式提取: 识别公式区域准备进一步处理

**优先级与时间安排**:
- 高优先级: 文本和图像提取 (4天)
- 中优先级: 表格内容提取 (3天)
- 低优先级: 公式内容提取 (3天)

#### 3. PDF解析器增强

**目标**: 提升PDF处理能力，更精确地提取结构化内容。

**核心实现步骤**:

1. **增强现有PDF解析功能**:
   - 改进文本提取算法，处理多栏布局
   - 优化图像提取质量
   - 识别页眉页脚和页码

2. **文档结构分析**:
   - 识别标题和章节结构
   - 区分正文、引用和列表
   - 处理特殊元素(如文本框、侧边栏)

**优先级与时间安排**:
- 高优先级: 基础解析增强 (3天)
- 中优先级: 多栏布局处理 (2天)
- 低优先级: 高级结构分析 (5天)

#### 4. 内容整合功能(初步)

**目标**: 实现将不同类型内容统一处理的功能，保留原始文档结构。

**核心实现步骤**:

1. **基础类结构**:
   ```python
   # src/note_generator/content_integrator.py
   class ContentIntegrator:
       def __init__(self, config=None):
           self.config = config or {}
           self.logger = get_module_logger("ContentIntegrator")
           
       def integrate(self, processed_blocks):
           """整合处理后的内容块"""
           # 对块进行排序，保持原始顺序
           sorted_blocks = self._sort_blocks(processed_blocks)
           
           # 合并处理后的内容
           integrated_content = []
           for block in sorted_blocks:
               integrated_content.append(self._format_block(block))
               
           return integrated_content
   ```

2. **保持文档结构**:
   - 根据页码和位置排序内容块
   - 合并相关联的内容(如图片及其标题)
   - 保持章节层次结构

**优先级与时间安排**:
- 高优先级: 基础整合功能 (3天)
- 中优先级: 结构保持 (3天)
- 低优先级: 高级格式化 (4天)

### 二、表格与公式专项处理 (v0.1.6)

#### 1. 表格和公式区域识别

**目标**: 增强DocumentAnalyzer，添加对表格和公式区域的精确识别功能。

**核心实现步骤**:

1. **表格识别算法**:
   - 基于规则的表格检测(线条、空间分布)
   - 表格结构分析(行、列、单元格)
   - 复杂表格处理(合并单元格、嵌套表格)

2. **公式识别算法**:
   - 数学符号检测
   - 公式边界确定
   - 区分内联公式和独立公式

**优先级与时间安排**:
- 高优先级: 基础表格识别 (4天)
- 中优先级: 复杂表格处理 (3天)
- 高优先级: 基础公式区域识别 (3天)
- 中优先级: 高级公式识别 (2天)

#### 2. 表格处理库集成

**目标**: 集成专门的表格识别与处理库，提高表格数据提取的准确性。

**核心实现步骤**:

1. **库选择与集成**:
   ```python
   # src/note_generator/table_processor.py
   class TableProcessor:
       def __init__(self, config=None):
           self.config = config or {}
           self.logger = get_module_logger("TableProcessor")
           
           # 根据配置选择表格处理库
           self.processor_type = self.config.get("table_processor", "camelot")
           
       def process_table(self, table_data):
           """处理表格区域，提取结构化表格数据"""
           if self.processor_type == "camelot":
               return self._process_with_camelot(table_data)
           elif self.processor_type == "tabula":
               return self._process_with_tabula(table_data)
           else:
               return self._process_with_custom(table_data)
   ```

2. **表格数据结构化**:
   - 提取表格行和列
   - 处理表头和数据
   - 导出为标准格式(CSV, JSON等)

3. **表格渲染**:
   - Markdown表格生成
   - HTML表格生成
   - 其他格式支持

**优先级与时间安排**:
- 高优先级: Camelot集成 (3天)
- 中优先级: Tabula集成 (2天)
- 高优先级: 表格数据结构化 (3天)
- 中优先级: 表格渲染 (2天)

#### 3. 数学公式OCR与LaTeX转换

**目标**: 实现对数学公式的准确识别和转换为LaTeX格式。

**核心实现步骤**:

1. **公式OCR功能**:
   ```python
   # src/note_generator/formula_processor.py
   class FormulaProcessor:
       def __init__(self, config=None):
           self.config = config or {}
           self.logger = get_module_logger("FormulaProcessor")
           
           # 选择公式识别引擎
           self.engine = self.config.get("formula_engine", "mathpix")
           
       def process_formula(self, formula_image):
           """处理公式图像，转换为LaTeX格式"""
           if self.engine == "mathpix":
               return self._process_with_mathpix(formula_image)
           else:
               return self._process_with_custom(formula_image)
   ```

2. **LaTeX转换与验证**:
   - 公式识别结果到LaTeX转换
   - 验证LaTeX语法正确性
   - 处理常见错误和替换

3. **公式渲染**:
   - 提供LaTeX预览能力
   - 针对不同输出格式(MD, HTML, PDF)的渲染支持

**优先级与时间安排**:
- 高优先级: 基础公式OCR功能 (5天)
- 中优先级: LaTeX转换 (3天)
- 低优先级: 公式渲染 (2天)

#### 4. ContentProcessor开发

**目标**: 创建ContentProcessor处理不同类型内容，应用专门的处理策略。

**核心实现步骤**:

1. **基础类结构**:
   ```python
   # src/note_generator/content_processor.py
   class ContentProcessor:
       def __init__(self, config=None):
           self.config = config or {}
           self.logger = get_module_logger("ContentProcessor")
           
           # 初始化各类型处理器
           self.text_processor = TextProcessor(config)
           self.image_processor = ImageProcessor(config)
           self.table_processor = TableProcessor(config)
           self.formula_processor = FormulaProcessor(config)
           
       def process(self, content_blocks):
           """处理所有内容块"""
           processed_blocks = []
           for block in content_blocks:
               if block["type"] == "text":
                   processed = self.text_processor.process(block)
               elif block["type"] == "image":
                   processed = self.image_processor.process(block)
               elif block["type"] == "table":
                   processed = self.table_processor.process(block)
               elif block["type"] == "formula":
                   processed = self.formula_processor.process(block)
               else:
                   self.logger.warning(f"未知内容类型: {block['type']}")
                   processed = block  # 原样返回
                   
               processed_blocks.append(processed)
               
           return processed_blocks
   ```

2. **处理器链模式**:
   - 定义标准处理器接口
   - 实现处理器链构建
   - 支持定制处理流程

3. **处理策略选择**:
   - 基于内容类型和特征选择策略
   - 支持处理程度和质量的配置
   - 提供处理失败时的回退选项

**优先级与时间安排**:
- 高优先级: 基础处理器架构 (3天)
- 中优先级: 文本和图像处理器 (3天)
- 高优先级: 表格处理器 (3天)
- 中优先级: 公式处理器 (3天)
- 低优先级: 高级处理策略 (3天)

## 代码实现指南

### 编码风格与约定

1. **命名规范**:
   - 类名: CamelCase (如DocumentAnalyzer)
   - 方法名: snake_case (如analyze_document)
   - 常量: UPPER_CASE (如MAX_PAGE_SIZE)
   - 私有方法/属性: _前缀 (如_analyze_pdf)

2. **代码组织**:
   - 遵循现有目录结构
   - 新模块放在src/note_generator/下
   - 测试代码放在tests/下
   - 示例和调试脚本放在scripts/下

3. **异常处理**:
   - 使用专门的异常类型
   - 日志记录所有错误
   - 确保优雅失败和恢复能力

4. **文档注释**:
   - 每个类和公共方法都需要docstring
   - 使用现有风格的中文注释
   - 包含参数说明和返回值类型

### 依赖库

```
pymupdf>=1.19.0       # PDF处理
camelot-py>=0.10.1    # 表格提取
opencv-python>=4.5.0  # 图像处理
pillow>=8.0.0         # 图像处理
numpy>=1.20.0         # 数值计算
pandas>=1.3.0         # 表格数据处理
```

## 测试计划

### 1. 单元测试

为每个新模块创建单元测试，包括:

- DocumentAnalyzer测试 (test_document_analyzer.py)
- ContentExtractor测试 (test_content_extractor.py)
- 表格和公式处理测试
- ContentProcessor和ContentIntegrator测试

### 2. 集成测试

- 全流程测试: 从PDF输入到最终输出
- 边界条件测试: 复杂布局、大文件、多语言
- 性能测试: 处理时间和资源使用

### 3. 测试数据

在`tests/test_data/`目录下创建以下测试数据:

- 简单PDF文件
- 包含表格的PDF
- 包含公式的PDF
- 复杂布局PDF(多栏、图文混排)

## 部署与集成

### 与现有系统集成

1. **InputHandler集成**:
   - 更新InputHandler以使用DocumentAnalyzer
   - 添加对综合文档处理的支持

2. **OCR-LLM集成**:
   - 复用现有OCR-LLM处理图像内容
   - 为表格和公式增加专门处理

3. **OutputWriter集成**:
   - 更新OutputWriter以支持表格和公式格式
   - 确保在不同输出格式中的一致展示

## 开发进度跟踪

| 任务 | 预计工时 | 优先级 | 状态 |
|-----|---------|-------|------|
| DocumentAnalyzer基础框架 | 3天 | 高 | ✅ 已完成 |
| PDF文本和图像区域识别 | 4天 | 高 | ✅ 已完成 |
| ContentExtractor基础框架 | 2天 | 高 | ✅ 已完成 |
| PDF解析器增强 | 3天 | 中 | ✅ 已完成 |
| 表格区域识别 | 4天 | 中 | ✅ 已完成 |
| 表格处理库集成 | 3天 | 高 | ✅ 已完成 |
| 公式区域识别 | 3天 | 中 | ✅ 已完成 |
| 公式OCR与LaTeX转换 | 5天 | 中 | ✅ 已完成 |
| ContentProcessor开发 | 6天 | 高 | ✅ 已完成 |
| ContentIntegrator开发 | 4天 | 中 | ✅ 已完成 |
| 单元测试编写 | 5天 | 高 | ✅ 已完成 |
| 集成测试编写 | 3天 | 中 | ✅ 已完成 |
| 文档更新 | 2天 | 低 | 进行中 |

## 总结

本文档提供了KnowForge项目v0.1.5和v0.1.6版本中，文档综合处理功能的详细实施计划。该功能现已全部完成，包括文档分析、内容提取、表格处理和公式识别等核心模块。各模块之间已实现无缝集成，满足了项目的总体目标和质量要求。

## 未来开发建议

根据目前的项目进展，以下是对下一阶段开发的建议：

### 1. 优化与性能提升

- **提高表格处理准确性**：对复杂表格布局进一步优化，特别是合并单元格和嵌套表格
- **加速处理效率**：优化大型文档处理性能，考虑并行处理和部分异步实现
- **减少API依赖**：将Mathpix公式识别替换为本地模型，降低外部依赖和API成本

### 2. 功能扩展

- **多语言支持增强**：改进对非英语文档的处理能力，特别是亚洲语言的表格和公式
- **交互式文档编辑**：增加用户编辑提取结果的界面，允许手动调整表格和公式识别结果
- **高级布局保持**：更精确地保留原始文档的布局和设计元素

### 3. 集成与扩展

- **知识图谱集成**：将文档处理结果与知识图谱模块集成，实现自动关系提取
- **多模态处理增强**：改进图像、表格和公式的语义理解，与LLM深度集成
- **版本控制与协作**：添加文档处理结果的版本控制和多用户协作功能

### 4. 测试与评估

- **扩展测试数据集**：构建更全面的测试数据集，涵盖更多文档类型和边缘案例
- **自动化质量评估**：开发自动评估处理质量的指标和工具
- **用户反馈机制**：建立系统化的用户反馈收集和分析流程

### 5. 下一版本规划 (v0.1.7建议)

| 功能模块 | 具体任务 | 优先级 |
|---------|----------|--------|
| 性能优化 | 大文档处理加速 | 高 |
| 性能优化 | 内存占用优化 | 高 |
| 功能扩展 | 本地公式识别实现 | 中 |
| 功能扩展 | 交互式结果编辑 | 中 |
| 集成 | 知识图谱连接器 | 高 |
| 测试 | 扩展测试数据集 | 中 |
| 文档 | API参考手册 | 低 |
