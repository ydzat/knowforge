# KnowForge 文档索引

此目录包含 KnowForge 项目的所有设计文档和技术规范。

## 文档结构更新记录

当前文档结构于2025-05-15完成重组，详细的变更记录请参见：[文档结构更新记录](./DOC_STRUCTURE_UPDATE.md)。

## 目录结构

每个子目录中都包含一个README文件，提供该目录内容的概览：
- [core/README.md](./core/README.md) - 核心设计文档目录概览
- [modules/README.md](./modules/README.md) - 模块文档目录概览
- [modules/ocr_llm/README.md](./modules/ocr_llm/README.md) - OCR-LLM集成模块文档概览
- [modules/document_processing/README.md](./modules/document_processing/README.md) - 文档处理模块文档概览
- [modules/memory_management/README.md](./modules/memory_management/README.md) - 内存管理模块文档概览
- [releases/README.md](./releases/README.md) - 发布文档目录概览

- **核心设计文档** [core/](./core/)
  - [高级设计](./core/01_HLD_KnowForge.md) | [英文版](./core/01_HLD_KnowForge_EN.md)
  - [低级设计](./core/02_LLD_KnowForge.md) | [英文版](./core/02_LLD_KnowForge_EN.md)
  - [新模块低级设计](./core/02_LLD_KnowForge_new_modules.md)
  - [开发指南](./core/03_DEV_KnowForge.md) | [英文版](./core/03_DEV_KnowForge_EN.md)
  - [测试计划](./core/04_TEST_KnowForge.md) | [英文版](./core/04_TEST_KnowForge_EN.md)
  - [发布指南](./core/05_REL_KnowForge.md) | [英文版](./core/05_REL_KnowForge_EN.md)
  - [迭代计划](./core/06_ITER_KnowForge.md) | [英文版](./core/06_ITER_KnowForge_EN.md)
  - [环境配置](./core/07_ENV_KnowForge.md) | [英文版](./core/07_ENV_KnowForge_EN.md)
  - [项目路线图](./core/08_ROADMAP_KnowForge.md) | [英文版](./core/08_ROADMAP_KnowForge_EN.md)

- **模块设计文档** [modules/](./modules/)
  - **OCR和LLM集成** [modules/ocr_llm/](./modules/ocr_llm/)
    - [OCR与LLM集成](./modules/ocr_llm/09_OCR_LLM_Integration.md) | [英文版](./modules/ocr_llm/09_OCR_LLM_Integration_EN.md)
  - **文档处理** [modules/document_processing/](./modules/document_processing/)
    - [文档处理设计](./modules/document_processing/10_DocumentProcessingDesign.md) | [英文版](./modules/document_processing/10_DocumentProcessingDesign_EN.md)
    - [文档处理变更摘要](./modules/document_processing/11_DocumentProcessing_ChangesSummary.md) | [英文版](./modules/document_processing/11_DocumentProcessing_ChangesSummary_EN.md)
  - **内存管理** [modules/memory_management/](./modules/memory_management/)
    - [高级知识记忆管理系统设计](./modules/memory_management/12_MemoryManagement.md) | [英文版](./modules/memory_management/12_MemoryManagement_EN.md) (占位文档)

- **发布文档** [releases/](./releases/)

- **图表和资源** [assets/](./assets/)

## 其他资源

- [其他相关文档](./others/) - 这些文件主要与Logloom项目相关
