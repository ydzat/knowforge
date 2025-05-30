# KnowForge 文档结构重组记录

*注意：此文件仅记录KnowForge项目文档结构的重组变更，不是项目功能的变更记录。项目功能变更请查看项目根目录下的[CHANGELOG.md](../CHANGELOG.md)。*

## 2025-05-15：文档结构重组

### 变更概要

- 实现了更加模块化的文档组织结构
- 创建了专门的目录用于不同类型的文档
- 添加了各目录的README文件，提供导航和概览
- 统一了文档命名规则和引用路径
- 明确了不同项目文档的边界（KnowForge vs Logloom）

### 具体变更

1. **新增目录结构**:
   - `core/` - 核心设计文档
   - `modules/` - 模块特定的设计文档
     - `ocr_llm/` - OCR与LLM集成相关文档
     - `document_processing/` - 文档处理相关文档
     - `memory_management/` - 内存管理相关文档（规划中）
   - `assets/` - 图表和资源文件
   - `releases/` - 发布文档

2. **文档移动**:
   - 将核心设计文档(01-08系列)移动到`core/`目录
   - 将模块设计文档(09-12系列)分类移动到`modules/`下对应子目录
   - 明确`others/`目录仅包含Logloom项目文档

3. **新增文档**:
   - 添加英文版OCR-LLM集成文档
   - 添加内存管理模块占位文档
   - 为所有目录创建README文件

4. **交叉引用更新**:
   - 更新高级设计文档中对模块文档的引用路径
   - 添加模块之间的相互引用
   - 创建主索引文档以便导航

### 统计信息

- 目录数量: 9
- 文档数量: 
  - 中文文档: 35
  - 英文文档: 12
  - README文件: 9

### 后续计划

1. 创建更多模块文档，特别是内存管理模块的详细设计
2. 添加更多图表和可视化资源
3. 完善英文文档的覆盖率
4. 实现文档版本控制机制
