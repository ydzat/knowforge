<!--
 * @Author: @ydzat
 * @Date: 2025-04-29 01:30:33
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-05-14 10:00:00
 * @Description: 
-->

# Changelog

This document records all major version updates and changes of the KnowForge project.

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