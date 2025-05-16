<!--
 * @Author: @ydzat
 * @Date: 2025-04-29 01:30:33
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-05-16 23:59:59
 * @Description: 项目更新日志
-->

# Changelog

This document records all major version updates and changes of the KnowForge project.

## [0.1.7] - 2025-05-16 (已完成)

### Output System Enhancement with Configuration and Offline Support

#### Features Added
- Implemented user configuration system for output customization
  - Created unified configuration schema in `/resources/config/output_config.yaml`
  - Added support for HTML, PDF, and Jupyter Notebook output customization
  - Implemented theme system with multiple built-in themes (default, dark, light, minimal)
  - Added comprehensive style configuration options for all output formats
- Developed offline resource management system
  - Created resource download script (`scripts/download_resources.py`)
  - Added support for Bootstrap, MathJax, and Highlight.js offline use
  - Implemented automatic CDN fallback when offline resources unavailable
  - Added command-line options for custom theme downloading
- Enhanced output generation with multi-engine support
  - Improved PDF generation with multiple rendering engines (weasyprint, fpdf)
  - Fixed HTML to plain text conversion with proper regex handling
  - Added automatic engine fallback mechanism with graceful degradation
  - Enhanced error handling and reporting in output generation
- Developed theme preview and testing tools
  - Enhanced `scripts/theme_preview.py` with theme comparison options
  - Added proper locale support and resource detection
  - Implemented offline/online mode auto-switching
  - Added options for batch theme testing

#### Technical Details
- Fixed syntax errors and structural issues in output generation methods
- Enhanced HTML template system with conditional resource loading
- Improved error handling with multi-level try/except structures
- Fixed regex patterns for HTML parsing and content extraction

#### Documentation
- Enhanced output enhancement progress documentation
- Updated output configuration guide with offline resource information
- Added usage instructions for theme preview and resource management tools
- Created detailed configuration examples for different use cases

## [0.1.6] - 2025-05-12 (已完成)

### Document Processing System with Table and Formula Support

#### Features Added
- Completed document processing framework with full content type support
  - Implemented integrated document processing pipeline
  - Added specialized table and formula processing capabilities
  - Enhanced content integration with structure preservation
- Table processing functionality (已完成)
  - Added table area detection and extraction
  - Implemented multi-processor architecture (Camelot, Tabula, Custom)
  - Added table structure normalization and enhancement
  - Implemented table-to-markdown/HTML conversion
  - Added automatic column standardization and empty cell handling
- Formula processing functionality (已完成)
  - Added formula detection in document with pattern matching
  - Implemented LaTeX conversion with multiple engines
  - Added formula type detection (inline vs. block)
  - Integrated optional Mathpix API support for image-based formulas
  - Added support for complex LaTeX expressions and mathematical notation
- Content integration enhancements
  - Added structure-preserving content integration
  - Implemented Markdown and HTML output formats
  - Enhanced position-based content ordering

#### Technical Details
- Designed modular processor architecture with configurable components
- Implemented fallback mechanisms for robust processing
- Added comprehensive demo scripts for testing features
- Enhanced error handling and reporting system

#### Documentation
- Added detailed document processing architecture documentation
- Created integration guides for table and formula processing
- Updated test cases for all new components
- Added comprehensive module documentation in docs/modules/table_formula_processing.md

### PDF Image Extraction and OCR Enhancement

#### Features Added
- Enhanced PDF image extraction with multi-method redundant strategy
  - Implemented 3 extraction methods with automatic fallback mechanism
  - Achieved 100% extraction success rate on test documents
  - Added image quality verification and enhancement capabilities
- Advanced OCR processing with LLM and memory system integration
  - Developed `AdvancedOCRProcessor` with multi-stage processing pipeline
  - Implemented LLM-based OCR result correction
  - Integrated with `AdvancedMemoryManager` for knowledge-enhanced OCR
  - Added confidence estimation algorithm combining multiple factors
- Improved memory management system
  - Implemented `_update_metadata_on_access` method for access statistics
  - Enhanced working memory capacity management algorithm
  - Added priority queue optimization for better performance

#### Technical Details
- Added comprehensive test suite for OCR-memory integration
- Implemented transparent error handling for image extraction failures
- Optimized image preprocessing for better OCR results
- Added configurable OCR enhancement parameters
- Updated configuration schema for OCR settings

#### Documentation
- Added detailed PDF extraction and OCR processing workflow documentation
- Updated development roadmap with version 0.1.7 plans
- Created integration guide for OCR-memory system

## [0.1.5] - 2025-05-14

### Advanced Memory Management System

#### Features Added
- Implemented `AdvancedMemoryManager` class with multi-tier memory architecture
  - `ShortTermMemory`: Fast, temporary storage for recent interactions
  - `WorkingMemory`: Active knowledge management with priority-based access
  - `LongTermMemory`: Persistent storage with advanced retrieval capabilities
- Developed dynamic memory management mechanisms
  - Importance scoring algorithm based on content, usage and external factors
  - Forgetting mechanism based on Ebbinghaus forgetting curve model
  - Memory reinforcement for important knowledge items
  - Context-aware retrieval with adaptive weights
- Added integration interfaces for document processing and OCR-LLM modules
- Implemented knowledge graph and association network capabilities

#### Technical Details
- Created comprehensive testing suite with 90+ test cases
- Optimized retrieval performance with hybrid retrieval strategies
- Implemented advanced configuration options for memory management behavior
- Added detailed memory system statistics and diagnostics
- Extended config.yaml with advanced memory management settings

#### Documentation
- Comprehensive design documentation in Chinese and English
- Updated implementation roadmap with phased development approach
- Added API reference with usage examples

## [0.1.4] - 2025-05-15

### Documentation Structure Reorganization

#### Changes Made
- Reorganized documentation into a modular structure with specialized directories
- Created dedicated folders for core design documents and module-specific documentation
- Added comprehensive README files for improved navigation
- Separated KnowForge and Logloom documentation
- Full details available in [docs/DOC_STRUCTURE_UPDATE.md](./docs/DOC_STRUCTURE_UPDATE.md)

### OCR-LLM-Knowledge Base Integration

#### Features Added
- Implemented `EmbeddingManager` class for knowledge retrieval capabilities
- Developed `LLMCaller` module with support for DeepSeek and OpenAI APIs
- Created advanced OCR processing pipeline with image preprocessing
- Added knowledge base enhancement for OCR results
- Implemented comprehensive test scripts for OCR-LLM integration
- Developed knowledge base content management tools

#### Improvements
- Significantly improved OCR text quality through LLM enhancement
- Added error handling and retry logic for API calls
- Implemented confidence level estimation for enhanced OCR results
- Created debug visualization for OCR preprocessing steps
- Optimized vector similarity search for contextual knowledge retrieval

#### Technical Details
- Average 1200% increase in effective content extraction from poor quality OCR
- Average 25% increase in confidence scores for OCR results
- Processing time of ~20 seconds per image (7-10s for LLM, 1-2s for knowledge base, 3s for OCR)
- Successful retrieval and application of domain knowledge for specialized content

#### Usage Requirements
- Requires a valid DeepSeek API key for LLM enhancement
- Requires activation of the knowforge conda environment

## [0.1.3] - 2025-05-14

### LocaleManager Migration to Logloom

#### Improvements
- Completely migrated LocaleManager to use Logloom's internationalization features
- Leveraged new Logloom APIs including `register_locale_file` and `register_locale_directory`
- Implemented intelligent key name resolution to handle various key formats
- Added single instance pattern for safe function access
- Enhanced error handling and fallback mechanisms
- Updated all tests to verify proper Logloom integration

#### Technical Debt Reduction
- Removed all legacy implementation code and backup logic
- Simplified the codebase by relying completely on Logloom
- Improved maintainability by removing duplicate functionality

#### Known Issues
- Some key formatting issues may still exist with certain key patterns
- Integration tests show possible recursive error in warning handling
- Documentation needs to be updated to reflect the new API

## [0.1.2] - 2025-05-14

### Vector Memory Management Enhancement

#### Improvements
- Fixed compatibility issues with ChromaDB's latest API in various retrieval methods
- Enhanced the `_hybrid_retrieval` method with more robust fallback strategies
- Added missing `_extract_keywords` and `_calculate_keyword_score` functions
- Optimized threshold handling in retrieval methods to ensure results even with low similarity
- Improved the balance between semantic similarity and keyword matching in hybrid retrieval
- Added missing datetime and math module imports for time-based retrievals

#### Bug Fixes
- Fixed ChromaDB query API parameter issues in `_time_weighted_retrieval`
- Fixed `_context_aware_retrieval` to properly handle combined embeddings
- Resolved issues where hybrid retrieval would return no results
- Ensured proper error handling for all retrieval methods

## [0.1.1] - 2025-05-13

### Logloom Integration

#### New Features
- Integrated Logloom logging system, replacing basic logging
- Added support for multilingual log messages
- Implemented automatic log file rotation
- Enhanced log format standardization
- Added configurable log levels and outputs

#### Improvements
- Updated logger.py to use Logloom native API
- Added language resource files for log messages
- Enhanced test coverage for the logging system
- Fixed issues with logger behavior in multi-threading contexts
- Added detailed documentation for Logloom integration

#### Configuration
- Added logloom_config.yaml for centralized configuration
- Added language resources: logloom_zh.yaml and logloom_en.yaml

#### Development Tools
- Enhanced llm_integration_check.py to test logging system

## [0.1.0] - 2025-04-29

### Initial Release

#### New Features
- Multi-source input support (PDF, web links, code files (txt))
- Intelligent text splitting (based on chapters/paragraphs)
- DeepSeek LLM integration
- Multi-format output (Markdown)
- Multilingual UI support (Chinese, English)
- Command-line interface (based on Typer)

#### Core Modules
- ConfigLoader - Configuration loading and management
- LocaleManager - Multilingual support
- Logger - Logging system
- InputHandler - Input processing
- Splitter - Text splitting
- LLMCaller - DeepSeek API integration
- OutputWriter - Output generation
- Processor - Main flow controller

#### Development Tools
- Complete test suite (unit tests + LLM integration tests)
- Utility scripts

---

# 更新日志

本文档记录KnowForge项目的所有重要版本更新和变更。

## [0.1.3] - 2025-05-14

### LocaleManager迁移至Logloom

#### 改进
- 完全迁移LocaleManager至Logloom国际化功能
- 利用新的Logloom API，包括`register_locale_file`和`register_locale_directory`
- 实现智能键名解析功能，处理各种键名格式
- 添加单例模式以支持安全函数访问
- 增强错误处理和回退机制
- 更新所有测试，验证Logloom集成正常工作

#### 技术债务减少
- 移除所有遗留实现代码和备用逻辑
- 通过完全依赖Logloom简化代码库
- 通过移除重复功能提高可维护性

#### 已知问题
- 某些键格式可能仍存在格式化问题
- 集成测试显示警告处理中可能存在递归错误
- 文档需要更新以反映新的API

## [0.1.2] - 2025-05-14

### 向量记忆管理优化

#### 改进
- 修复与ChromaDB最新API在各种检索方法中的兼容性问题
- 增强`_hybrid_retrieval`方法，提供更强大的后备策略
- 添加缺失的`_extract_keywords`和`_calculate_keyword_score`函数
- 优化检索方法中的阈值处理，以确保即使在相似度较低的情况下也能得到结果
- 改善混合检索中语义相似性与关键词匹配之间的平衡
- 为基于时间的检索添加缺失的datetime和math模块导入

#### Bug 修复
- 修复`_time_weighted_retrieval`中ChromaDB查询API参数问题
- 修复`_context_aware_retrieval`以正确处理组合嵌入
- 解决混合检索返回空结果的问题
- 确保所有检索方法的错误处理正常

## [0.1.1] - 2025-05-13

### Logloom集成

#### 新增功能
- 集成Logloom日志系统，替代基础日志系统
- 添加多语言日志消息支持
- 实现日志文件自动轮转
- 增强日志格式标准化
- 添加可配置的日志级别和输出通道

#### 改进
- 更新logger.py以使用Logloom原生API
- 添加日志消息的语言资源文件
- 增强日志系统的测试覆盖率
- 修复多线程环境下的日志行为问题
- 添加Logloom集成的详细文档

#### 配置
- 添加logloom_config.yaml进行集中配置
- 添加语言资源：logloom_zh.yaml和logloom_en.yaml

#### 开发工具
- 增强llm_integration_check.py以测试日志系统

## [0.1.0] - 2025-04-29

### 初始版本发布

#### 新增功能
- 多源输入支持（PDF、网页链接、代码文件(txt)）
- 智能文本拆分（基于章节/段落）
- 集成DeepSeek大语言模型
- 多格式输出（Markdown）
- 多语系界面支持（中文、英文）
- 命令行界面（基于Typer）

#### 核心模块
- ConfigLoader - 配置加载与管理
- LocaleManager - 多语系支持
- Logger - 日志系统
- InputHandler - 输入处理
- Splitter - 文本拆分
- LLMCaller - DeepSeek API调用
- OutputWriter - 输出生成
- Processor - 主流程控制器

#### 开发工具
- 完整测试套件（单元测试+LLM集成测试）
- 工具脚本