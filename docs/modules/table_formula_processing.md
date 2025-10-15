# 表格和公式处理功能文档

## 表格处理功能

KnowForge 的表格处理功能通过 `TableProcessor` 类实现，支持多种表格处理引擎，包括：

1. **Camelot 引擎**：专为 PDF 文件中的表格提取和处理优化
2. **Tabula 引擎**：支持复杂表格结构的处理
3. **自定义引擎**：通过正则表达式和文本处理处理简单表格

### 主要功能：

- **表格结构标准化**：确保所有行具有相同数量的列
- **空行和列处理**：移除空行并适当填充空单元格
- **表格结构增强**：处理多级表头和合并单元格
- **表格到多种格式转换**：支持 Markdown 和 HTML 输出

### 使用示例：

```python
from src.note_generator.content_processor import TableProcessor

# 初始化表格处理器
config = {
    "table.processor": "custom",  # 可选: "camelot", "tabula", "custom"
    "table.clean_empty_rows": True,
    "table.normalize_columns": True,
    "table.enhance_structure": True,
}
processor = TableProcessor(config)

# 处理表格
table_data = """
| Product | Price | Qty | Total |
| --- | --- | --- | --- |
| Laptop | 6000 | 2 | 12000 |
| Monitor | 1500 | 3 | 4500 |
| Mouse | 80 | 5 | 400 |
| Total | | | 16900 |
"""
result = processor.process({"type": "table", "table_data": table_data})

# 获取处理后的 Markdown 表格
markdown_table = result["markdown"]
```

## 公式处理功能

KnowForge 的公式处理功能通过 `FormulaProcessor` 类实现，支持多种公式处理引擎，包括：

1. **Mathpix 引擎**：支持通过 API 识别图片中的数学公式
2. **自定义引擎**：使用正则表达式和文本处理来转换简单的数学表达式到 LaTeX

### 主要功能：

- **公式类型检测**：自动区分内联公式和块级公式
- **简单表达式转换**：将简单的数学表达式（如指数、分数）转换为 LaTeX 格式
- **LaTeX 格式化**：根据公式类型（内联/块级）添加适当的 LaTeX 分隔符

### 使用示例：

```python
from src.note_generator.content_processor import FormulaProcessor

# 初始化公式处理器
config = {
    "formula.engine": "custom",  # 可选: "mathpix", "custom"
    "formula.detect_formula_type": True,
    "formula.convert_simple_expressions": True,
}
processor = FormulaProcessor(config)

# 处理公式
formula_text = "E=mc^2"
result = processor.process({"type": "formula", "formula_text": formula_text})

# 获取处理后的 LaTeX 公式
latex_formula = result["latex"]
```

## 表格和公式演示脚本

在 `scripts/table_formula_demo.py` 中提供了一个完整的演示脚本，展示了表格和公式处理功能的使用方法。此脚本包含多个表格和公式示例，对它们进行处理，并输出结果。

### 运行方式：

```bash
python scripts/table_formula_demo.py
```

## 文档处理测试脚本

在 `scripts/test_document_processing.py` 中提供了一个测试脚本，用于测试完整的文档处理流程，包括表格和公式的处理。该脚本支持处理 PDF 和纯文本文件。

### 运行方式：

```bash
python scripts/test_document_processing.py 文档路径 [选项]
```

### 选项：

- `--output-dir`：指定输出目录
- `--table-processor`：选择表格处理器 (camelot, tabula, custom)
- `--formula-engine`：选择公式处理引擎 (mathpix, custom)
- `--api-key` 和 `--app-id`：如果使用 Mathpix 引擎，提供 API 凭据
