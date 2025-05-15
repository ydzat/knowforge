# KnowForge 模块文档

本目录包含KnowForge项目的各个模块的详细设计文档和技术规范。每个子目录对应一个功能模块，包含该模块的所有相关文档。

## 模块列表

### [OCR-LLM集成模块](./ocr_llm/)

该模块负责提高图像文本识别质量，并利用大语言模型和领域知识库对OCR结果进行增强和校正。

主要文档：
- [OCR-LLM-知识库集成技术文档](./ocr_llm/09_OCR_LLM_Integration.md)

### [文档处理模块](./document_processing/)

该模块负责自动识别和处理文档中的多种内容类型，包括文本、图片、表格和公式，提高系统的易用性和功能完整性。

主要文档：
- [文档综合处理设计文档](./document_processing/10_DocumentProcessingDesign.md)
- [文档处理变更摘要](./document_processing/11_DocumentProcessing_ChangesSummary.md)

### [内存管理模块](./memory_management/)

该模块负责实现高级知识记忆管理系统，提高知识的存储效率和检索准确性。

主要文档：
- [高级知识记忆管理系统设计](./memory_management/12_MemoryManagement.md) (占位文档)

## 文档命名规则

模块文档使用以下命名规则：

- 序号前缀：用于表示文档的创建顺序和关联性
- 描述性名称：清晰地表明文档的主题和内容
- 语言后缀：中文文档无后缀，英文文档添加`_EN`后缀

例如：`09_OCR_LLM_Integration.md` 和 `09_OCR_LLM_Integration_EN.md`
