# OCR-LLM集成模块文档

本目录包含KnowForge项目中OCR（光学字符识别）和LLM（大语言模型）集成模块的设计文档和技术规范。此模块负责提高图像文本识别质量，并利用大语言模型和领域知识库对OCR结果进行增强和校正。

## 文档列表

- [OCR-LLM-知识库集成技术文档](./09_OCR_LLM_Integration.md) | [英文版](./09_OCR_LLM_Integration_EN.md)
  - 详细描述了OCR-LLM-知识库集成的架构、核心组件、处理流程和性能评估

## 关键技术

- 图像预处理和质量增强
- OCR引擎集成（Tesseract, Paddle OCR）
- 向量嵌入和知识库检索
- 大语言模型（LLM）增强
- 领域知识整合

## 与其他模块的关系

- 为[文档处理模块](../document_processing/)提供图像中文本的识别和增强服务
- 与[内存管理模块](../memory_management/)协作，优化知识检索和存储

## 进一步阅读

- [KnowForge核心设计文档](../../core/)中的相关章节
