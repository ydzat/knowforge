<!--
 * @Author: @ydzat
 * @Date: 2025-04-29 01:30:33
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-04-29 02:00:15
 * @Description: 
-->

# Changelog

This document records all major version updates and changes of the KnowForge project.

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