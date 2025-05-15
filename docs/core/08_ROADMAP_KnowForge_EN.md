<!--
 * @Author: @ydzat
 * @Date: 2025-05-14 14:50:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-05-14 14:50:00
 * @Description: KnowForge Project Development Roadmap
-->

# KnowForge Project Development Roadmap

## Current Development Stage Overview

The KnowForge project is currently in the early development stage (0.1.4), with the basic architecture established and core functionalities gradually being improved. The major work completed so far includes:

1. **Basic Architecture Setup** (0.1.0):
   - Core module structure design and implementation
   - Multilingual support system
   - Basic configuration management
   - Multi-source input processing (PDF, web links, code files)
   - Command-line interface

2. **Logging System Upgrade** (0.1.1):
   - Integration of Logloom logging system
   - Implementation of log internationalization support
   - Addition of log file rotation functionality

3. **Vector Memory Management Optimization** (0.1.2):
   - ChromaDB API compatibility fixes
   - Enhanced hybrid retrieval strategies
   - Threshold processing logic optimization
   - Keyword matching functionality improvements

4. **Internationalization System Update** (0.1.3):
   - Complete migration of LocaleManager to Logloom
   - Optimization of internationalization resource loading mechanism
   - Implementation of smart key name parsing functionality
   - Establishment of universal internationalization interfaces

5. **OCR-LLM-Knowledge Base Integration** (0.1.4):
   - Implementation of EmbeddingManager class for knowledge retrieval
   - Development of LLMCaller module supporting DeepSeek and OpenAI APIs
   - Implementation of advanced OCR processing pipeline with image preprocessing
   - Integration of knowledge base to enhance OCR results
   - Establishment of complete OCR-LLM-knowledge base processing pipeline

2. **Logging System Upgrade** (0.1.1):
   - Integration of Logloom logging system
   - Implementation of internationalized log messages
   - Addition of log file rotation functionality

3. **Vector Memory Management Optimization** (0.1.2):
   - ChromaDB API compatibility fixes
   - Enhanced hybrid retrieval strategies
   - Optimized threshold handling logic
   - Improved keyword matching functionality

## Development Priorities

### 1. Short-term Goals (0.1.4 - 0.1.6)

#### 1.1 Continue Improving OCR-LLM Integration
- **Optimize Current Implementation** (Current focus-0.1.4):
  - Optimize OCR image preprocessing parameters to improve recognition accuracy
  - Improve LLM prompt templates for different document types
  - Expand knowledge base content to enhance professional domain text recognition
  - Provide user-friendly knowledge base management tools
  - Improve confidence calculation mechanisms to enhance result reliability

#### 1.2 Document Comprehensive Processing (Prioritized)
- **PDF Content Comprehensive Extraction** (Next development-0.1.5):
  - Implement DocumentAnalyzer to automatically identify text and image areas in documents
  - Implement ContentExtractor to extract text and image content
  - Enhance PDF parser to recognize document structure (chapters, paragraphs, figure positions)
  - Process extracted images through the existing OCR-LLM pipeline
  - Develop initial content integration functionality to maintain original document structure

- **Table and Formula Specialized Processing** (Planned for 0.1.6):
  - Improve DocumentAnalyzer to add table and formula region recognition
  - Add specialized table recognition and processing libraries (Camelot/tabula-py)
  - Implement mathematical formula OCR and LaTeX conversion
  - Develop ContentProcessor to handle different content types

- **Enhanced Image OCR Capabilities** (Continuous improvement):
  - Implement multilingual OCR support (Chinese, English, formula recognition)
  - Add image preprocessing (denoising, contrast enhancement) to improve OCR accuracy
  - Develop image analysis capability to recognize chart structures
  - Implement LLM-assisted OCR result correction and enhancement

#### 1.3 LLM Calling Module Reinforcement
- **DeepSeek Model Call Optimization**:
  - Add more model options (DeepSeek Chat/Reasoner)
  - Optimize prompt templates to improve generation quality
  - Implement context manager to avoid excessively long inputs
  - Add dynamic adjustment of model parameters (temperature, etc.)
  - Add dynamic adjustment functionality for model parameters (e.g., temperature)

#### 1.4 Content Integration and Output Format Extension
- **Content Integration and Format Preservation** (Planned for 0.1.7-0.2.0):
  - Implement ContentIntegrator functionality to integrate different content types
  - Ensure preservation of original document structure in the final notes
  - Optimize table and formula display in different output formats
  - Strengthen contextual associations between different content types

- **Enriched Output Capabilities**:
  - Improve PDF generation (supporting table and formula rendering)
  - Optimize Jupyter Notebook output format (supporting interactive tables and LaTeX formulas)
  - Add HTML output option (supporting responsive tables and MathJax formulas)
  - Support code block syntax highlighting and formatting

#### 1.5 Web Interface Basic Development
- **Provide Simple Web Operation Interface**:
  - Basic file upload and processing
  - Note preview and download
  - Progress display and status feedback

#### 1.6 Test Coverage Enhancement
- **Expand Test System**:
  - Add mock tests for OCR and LLM calls
  - Enhance integration test coverage
  - Develop end-to-end test cases

### 2. Mid-term Goals (0.2.0 - 0.3.0)

#### 2.1 Advanced Memory Management Features
- **Memory Library Management Enhancement**:
  - Implement memory expiration management
  - Add memory priority sorting
  - Develop memory categorization tag system
  - Support memory library query and browsing interface

#### 2.2 User Interface Development
- **Add Graphical Interface**:
  - Develop Web-based user interface (e.g., using Flask or Streamlit)
  - Design intuitive operation workflows
  - Add real-time processing progress display
  - Provide result preview functionality

#### 2.3 Plugin System Design
- **Implement Extensible Plugin Architecture**:
  - Design unified plugin interfaces
  - Support custom input source plugins
  - Support custom output format plugins
  - Implement model provider plugins (supporting more LLMs)

#### 2.4 MoeAI-C System Integration
- **Interface with MoeAI-C System**:
  - Implement standard API interface
  - Support remote calling
  - Implement data synchronization mechanism
  - Design secure authentication process

### 3. Long-term Goals (0.3.0+)

#### 3.1 Advanced Multimodal Capabilities
- **Process More Complex Multimodal Inputs**:
  - Deep image semantic understanding (chart analysis, visualization interpretation)
  - Audio transcription and summarization (long-term plan only, not core functionality)
  - Video content analysis (long-term plan only, not core functionality)
  - Complex scientific document processing (including advanced mathematical formulas, professional diagrams)

#### 3.2 Collaborative Workflow
- **Support Multi-user Collaboration**:
  - Develop user permission management
  - Implement shared memory library
  - Support collaborative note editing
  - Add annotation and feedback mechanism

#### 3.3 Offline Mode and Performance Optimization
- **Enhance Offline Capabilities**:
  - Support local small models
  - Optimize vector database performance
  - Implement incremental processing mechanism
  - Reduce resource consumption

## Development Recommendations and Best Practices

### Code Organization and Style
1. **Follow the established modular structure**, maintaining high cohesion and low coupling
2. **Improve documentation**, adding detailed comments and design documents for each new feature
3. **Maintain internationalization support**, manage all user-visible text with LocaleManager
4. **Standardize error handling**, using custom exception classes (NoteGenError and subclasses)

### Testing Strategy
1. **Write unit tests**, ensuring test coverage for each new feature
2. **Use mock objects**, simulating external dependencies (such as LLM API)
3. **Run integration tests regularly**, verifying overall system functionality
4. **Create regression tests for discovered bugs**, preventing recurrence

### Version Control and Release
1. **Follow semantic versioning** (SemVer):
   - Minor versions (0.1.x) for bug fixes and small feature enhancements
   - Medium versions (0.x.0) for new functional modules
   - Major versions (x.0.0) for significant architectural changes
2. **Maintain CHANGELOG**, recording detailed changes for each version
3. **Regular releases**, maintaining small update steps for easier testing and feedback

## Next Step Action Plan

### Immediate Tasks (Priority from high to low)
1. **Implement Image OCR Functionality**
   - Integrate easyocr library, supporting Chinese and English recognition
   - Add image preprocessing functionality
   - Write test cases

2. **Optimize LLM Call Logic**
   - Enhance prompt engineering to improve generation quality
   - Implement dynamic context window management
   - Add more model option configurations

3. **Improve PDF Output Functionality**
   - Implement conversion from Markdown to PDF
   - Add table of contents generation
   - Support custom styling

4. **Develop Basic Web Interface**
   - Design simple user interaction flow
   - Implement file upload and result display
   - Add basic progress display functionality

### Technical Debt Management
1. **Refactor the memory_manager module** to further optimize code organization
2. **Optimize vector retrieval performance**, especially for large memory libraries
3. **Improve test coverage**, particularly for recently modified parts

## Current Development Status

### OCR-LLM-Knowledge Base Integration (0.1.4)

#### Completed Work

1. **EmbeddingManager Class Implementation**:
   - Created knowledge retrieval capabilities
   - Implemented efficient vector embedding management
   - Added error handling and logging
   - Developed seamless integration with MemoryManager

2. **LLMCaller Module Development**:
   - Support for DeepSeek and OpenAI APIs
   - Implemented retry logic and error handling
   - Added automatic API key rotation functionality
   - Optimized request parameters and timeout handling

3. **OCR-LLM-Knowledge Base Testing and Verification**:
   - Verified complete OCR-LLM-Knowledge base processing pipeline
   - Developed comprehensive test scripts and tools

### Next Phase Focus: Document Comprehensive Processing

Upon evaluation, to achieve the goal of "users not needing to manually categorize file types," we will focus on developing document comprehensive processing functionality in v0.1.5-v0.2.0, including:

1. **Document Structure Analysis**:
   - Automatically distinguish text, images, tables, and formulas in PDF documents
   - Develop DocumentAnalyzer and ContentExtractor modules
   - Preserve original document structure information for subsequent integration

2. **Specialized Content Processing**:
   - Table recognition and structured processing
   - Mathematical formula recognition and LaTeX conversion
   - Preservation and rendering of special formats

3. **Content Integration Mechanism**:
   - Integrate various types of content according to original order and structure
   - Ensure semantic continuity between different content types
   - Guarantee format uniformity and structural integrity of note output

## Conclusion

The KnowForge project has established a solid foundational architecture and implemented early versions of core functionalities. Through continuous iteration and module enhancement, the project is moving toward becoming a full-featured AI-assisted learning note generator. Current priorities include enhancing input processing capabilities, optimizing the LLM calling module, expanding output format options, and improving test coverage.

Future development should maintain modular design, following established code organization and internationalization support principles, while continuously improving documentation and testing. Progress should be made according to feature priorities, maintaining small update steps to facilitate timely feedback and adjustments.