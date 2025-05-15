<!--
 * @Author: @ydzat
 * @Date: 2025-04-28 16:25:30
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-04-29 02:16:59
 * @Description: Low-Level Design Document
-->
# KnowForge Low-Level Design Document (LLD)

---

## Document Description

This document provides detailed design specifications for the KnowForge project, including for each core module:
- Class Definitions
- Main Attributes
- Core Methods
- Input/Output Specifications
- Brief Internal Workflow Descriptions

Follows engineering design principles of modularity, high cohesion, and low coupling.

## Module List (Ordered by Execution Flow)

| Module | Brief Description |
|:------|:-----------------|
| 1. ConfigLoader | Configuration loader, manages config files and environment variables |
| 2. LocaleManager | Multilingual manager, handles prompt texts and exception messages |
| 3. Logger (Wrapper) | Logging system, standardizes log output |
| 4. InputHandler | Input processor, scans and extracts content (PDFs, images, code, links) |
| 5. Splitter | Text splitter, intelligently splits text by sections/paragraphs/length |
| 6. Embedder | Vectorization tool, generates text embeddings |
| 7. MemoryManager | Vector storage manager, handles vector storage/retrieval (ChromaDB) |
| 8. LLMCaller | LLM caller, interfaces with DeepSeek Chat/Reasoner |
| 9. OutputWriter | Output generator, creates Markdown, ipynb, and PDF files |
| 10. Processor | Main workflow controller, coordinates all submodules |
| 11. Exception Classes | Custom exception system, standardizes error handling |
| 12. Scripts Utilities | Helper scripts (clean cache, rebuild memory, etc.) |

---

# 1. ConfigLoader Module

## Class Definition
```python
class ConfigLoader:
    def __init__(self, config_path: str)
```

## Main Attributes
| Attribute | Type | Description |
|:---------|:-----|:-----------|
| config_path | str | Path to config file |
| config | dict | Loaded configuration dictionary |

## Core Methods
| Method | Parameters | Returns | Description |
|:-------|:----------|:-------|:-----------|
| get | key_path: str, default: Any = None | Any | Get config item using dot notation (e.g. system.language) |
| get_env | env_var: str, default: Any = None | Any | Get environment variable (e.g. API keys) |

## Input/Output Specification
- Input: Config file path (`resources/config/config.yaml`)
- Output: Configuration dictionary for other modules to use

## Internal Workflow
1. Load `.env` file
2. Load `config.yaml` file
3. Provide unified access interface

---

# 2. LocaleManager Module

## Class Definition

```python
class LocaleManager:
    def __init__(self, locale_path: str, language: str = "zh")
```

## Main Attributes

| Attribute | Type | Description |
|:---------|:-----|:-----------|
| locale_path | str | Path to locale files (e.g. `resources/locales/locale.yaml`) |
| language | str | Current language ("zh" or "en") |
| messages | dict | Loaded locale message dictionary |

## Core Methods

| Method | Parameters | Returns | Description |
|:-------|:----------|:-------|:-----------|
| get | key_path: str | str | Get localized message by path (e.g. system.start_message) |

## Input/Output Specification

- Input: Locale files (YAML format with zh/en content)
- Output: Localized string for specified key path

## Internal Workflow

1. Parse specified language YAML file
2. Support dot notation path access (nested levels)
3. Return empty string if query fails

---

# 3. Logger Module

## Class Definition

```python
class LoggerManager:
    def __init__(self, log_dir: str = "output/logs", log_level: int = logging.INFO)
```

## Main Attributes

| Attribute | Type | Description |
|:---------|:-----|:-----------|
| log_dir | str | Log file output directory |
| log_level | int | Log level (default INFO, can be DEBUG/ERROR etc.) |
| logger | logging.Logger | Python standard Logger instance |

## Core Methods

| Method | Parameters | Returns | Description |
|:-------|:----------|:-------|:-----------|
| get_logger | None | logging.Logger | Returns configured Logger instance |

## Input/Output Specification

- Input: Log directory path and log level
- Output: Standard Logger object

## Internal Workflow

1. Create log directory if not exists
2. Configure log format (timestamp + level + module + message)
3. Output to both file and terminal
4. Return managed Logger instance

---

# 4. InputHandler Module

## Class Definition

```python
class InputHandler:
    def __init__(self, input_dir: str, workspace_dir: str)
```

## Main Attributes

| Attribute | Type | Description |
|:---------|:-----|:-----------|
| input_dir | str | Input source directory (e.g. `input/`) |
| workspace_dir | str | Workspace directory (e.g. `workspace/`) |

## Core Methods

| Method | Parameters | Returns | Description |
|:-------|:----------|:-------|:-----------|
| scan_inputs | None | dict | Scan and categorize input files (returns dict grouped by type) |
| extract_texts | None | list | Extract plain text segments (standardized format) |
| save_preprocessed | None | None | Save extracted texts to preprocessing directory |

## Input/Output Specification

- Input: PDFs, images, links, code files in `input/` directory
- Output:
  - `scan_inputs()` returns categorized file paths
  - `extract_texts()` returns standardized text list
  - `save_preprocessed()` saves texts to `workspace/preprocessed/`

## Internal Workflow

1. Traverse input directory, infer types by folder/extensions
2. Process by type:
   - PDF → pdfplumber text extraction
   - Images → easyocr recognition
   - Links → requests+BeautifulSoup content extraction
   - Code → direct text read (optional Pygments highlighting)
3. Save standardized texts for Splitter module

---
# 5. Splitter Module

## Class Definition

```python
class Splitter:
    def __init__(self, config: ConfigLoader)
```

## Main Attributes

| Attribute | Type | Description |
|:---------|:-----|:-----------|
| chunk_size | int | Max length per text segment (chars, fallback only) |
| overlap_size | int | Overlap between adjacent segments (chars) |
| use_llm | bool | Whether to use LLM-assisted splitting (default True) |
| llm_provider | str | LLM service provider (default "deepseek") |
| llm_api_key | str | LLM service API key |

## Core Methods

| Method | Parameters | Returns | Description |
|:-------|:----------|:-------|:-----------|
| split_text | text_segments: list[str] | list | Split input texts into segments |
| _split_by_structure | text: str | list | Split single text by structure |
| _split_with_llm_assistance | text: str, detected_patterns: list, headers: list | list | Use LLM to assist splitting |
| _split_with_deepseek | text: str, detected_patterns: list, headers: list | list | Use DeepSeek model for splitting |
| _split_by_length | text: str, chunk_size: int, overlap_size: int | list | Fallback fixed-length splitting |

## Input/Output Specification

- Input: Standardized plain texts (from InputHandler)
- Output: List of semantically coherent text segments for vectorization

## Internal Workflow

1. Process each text:
   - First try LLM-assisted structural analysis
   - Provide text samples and detected headers to LLM
   - LLM can split via:
     - Method 1: Regex patterns for section headers
     - Method 2: Direct line numbers as split points
   - Fallback to paragraph splitting (by empty lines)
   - Final fallback: Fixed-length chunks with overlap
2. Ensure semantic coherence in all segments
3. Integrates with config system for dynamic strategies

---

# 6. Embedder Module

## Class Definition

```python
class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2")
```

## Main Attributes

| Attribute | Type | Description |
|:---------|:-----|:-----------|
| model_name | str | Pretrained embedding model name |
| model | SentenceTransformer | Loaded SentenceTransformer instance |

## Core Methods

| Method | Parameters | Returns | Description |
|:-------|:----------|:-------|:-----------|
| embed_texts | texts: list[str] | list[list[float]] | Generate embeddings for text list |
| embed_single | text: str | list[float] | Generate embedding for single text |

## Input/Output Specification

- Input: Plain text (single or batch)
- Output: High-dimensional vector representations

## Internal Workflow

1. Load specified Sentence-Transformer model (default MiniLM-L6-v2)
2. Generate embeddings for input texts
3. Return standardized vector data for MemoryManager

---

# 7. MemoryManager Module

## Class Definition

```python
class MemoryManager:
    def __init__(self, chroma_db_path: str, embedder: Embedder)
```

## Main Attributes

| Attribute | Type | Description |
|:---------|:-----|:-----------|
| chroma_db_path | str | ChromaDB storage path |
| embedder | Embedder | Embedder instance (for new segments) |
| collection | Chroma Collection | ChromaDB collection instance |

## Core Methods

| Method | Parameters | Returns | Description |
|:-------|:----------|:-------|:-----------|
| add_segments | segments: list[str] | None | Vectorize and add segments to DB |
| query_similar | query_text: str, top_k: int = 5 | list | Retrieve top-k similar segments |
| rebuild_memory | segments: list[str] | None | Clear and rebuild memory DB |

## Input/Output Specification

- Input: Text segments (batch add or single query)
- Output:
  - `add_segments()` updates DB only
  - `query_similar()` returns similar segments
  - `rebuild_memory()` reinitializes memory DB

## Internal Workflow

1. Initialize/connect to local ChromaDB (default collection `knowforge_memory`)
2. For adding segments:
   - Call Embedder to generate vectors
   - Store vectors + original texts in ChromaDB
3. For queries:
   - Vectorize query text
   - Perform nearest neighbor search (TopK), return texts

---

# 8. LLMCaller Module

## Class Definition

```python
class LLMCaller:
    def __init__(self, model_name: str, api_key: str, api_base_url: str)
```

## Main Attributes

| Attribute | Type | Description |
|:---------|:-----|:-----------|
| model_name | str | DeepSeek model name (chat/reasoner) |
| api_key | str | DeepSeek API key |
| api_base_url | str | API base URL |

## Core Methods

| Method | Parameters | Returns | Description |
|:-------|:----------|:-------|:-----------|
| call_model | prompt: str, memory: list[str] = None | str | Call model to generate notes |
| build_prompt | input_text: str, memory: list[str] = None | str | Build final prompt (with memory) |

## Input/Output Specification

- Input:
  - Prompt text (current segment)
  - Optional memory context list
- Output:
  - Generated note text from DeepSeek (plain string)

## Internal Workflow

1. Build Prompt:
   - Current text segment as main content
   - Optionally append relevant memory segments
2. Call DeepSeek API:
   - Send request via requests.post
   - Include model name, request body, API key
3. Process API response:
   - Extract generated text
   - Graceful error handling and logging

---

# 9. OutputWriter Module

## Class Definition

```python
class OutputWriter:
    def __init__(self, workspace_dir: str, output_dir: str, locale_manager)
```

## Main Attributes

| Attribute | Type | Description |
|:---------|:-----|:-----------|
| workspace_dir | str | Workspace directory path (e.g. `workspace/`) |
| output_dir | str | Output directory path (e.g. `output/`) |
| locale_manager | LocaleManager | Locale manager instance (for localized prompts) |

## Core Methods

| Method | Parameters | Returns | Description |
|:-------|:----------|:-------|:-----------|
| generate_markdown | notes: list[str], filename: str | str | Generate Markdown file |
| generate_notebook | notes: list[str], filename: str | str | Generate Jupyter Notebook |
| generate_pdf | markdown_path: str, filename: str | str | Render PDF from Markdown |

## Input/Output Specification

- Input:
  - notes: Text segments (generated note content)
  - filename: Output filename (no extension)
- Output:
  - Generated files (.md, .ipynb, .pdf) saved to `output/`

## Internal Workflow

1. Markdown generation:
   - Insert segments following template
2. Notebook generation:
   - Wrap each segment in Markdown cell
3. PDF generation:
   - Convert Markdown to HTML then PDF (weasyprint)
4. Logging and error handling throughout

---

# 10. Processor Module

## Class Definition

```python
class Processor:
    def __init__(self, input_dir: str, output_dir: str, config_path: str)
```

## Main Attributes

| Attribute | Type | Description |
|:---------|:-----|:-----------|
| input_dir | str | Input directory path (e.g. `input/`) |
| output_dir | str | Output directory path (e.g. `output/`) |
| config_loader | ConfigLoader | Config loader instance |
| locale_manager | LocaleManager | Locale manager instance |
| logger | logging.Logger | Logger instance |
| workspace_dir | str | Workspace directory path (e.g. `workspace/`) |

## Core Methods

| Method | Parameters | Returns | Description |
|:-------|:----------|:-------|:-----------|
| run_full_pipeline | formats: list[str] | None | Execute full workflow (input to output) |

## Input/Output Specification

- Input:
  - Various files in input directory
  - Requested output formats (markdown/ipynb/pdf)
- Output:
  - Generated note files in output directory

## Internal Workflow

1. Initialize core modules
2. Call InputHandler to extract texts
3. Use Splitter to segment texts
4. Store segments in MemoryManager
5. Generate notes via LLMCaller (with memory retrieval)
6. Generate outputs via OutputWriter
7. Comprehensive logging and error handling

---

# 11. Exception Classes Module

## Base Exception Class

```python
class NoteGenError(Exception):
    """Base exception type for KnowForge, all custom exceptions inherit it"""
    pass
```

## Specific Exception Subclasses

| Class | Inherits From | Description |
|:-----|:------------|:-----------|
| InputError | NoteGenError | Input processing errors |
| APIError | NoteGenError | API call errors |
| MemoryError | NoteGenError | Memory management errors |
| OutputError | NoteGenError | Output generation errors |

## Main Attributes & Features

- Inherits from Python standard Exception
- Supports custom error messages
- Unified exception catching and typed logging

## Usage Example

```python
from src.utils.locale_manager import LocaleManager

try:
    locale = LocaleManager("resources/locales/locale.yaml", "zh")
    # Operation that may raise exception
    raise InputError(locale.get("errors.input_error"))
except NoteGenError as e:
    logger.error(f"Known Error: {str(e)}")
    print(locale.get("system.error_occurred"))
```

## Locale File Additions (resources/locales/locale.yaml)

```yaml
errors:
  input_error: "Input file format not supported"
system:
  error_occurred: "An error occurred during system operation"
```

## Internal Workflow

1. All expected errors raised via NoteGenError subclasses
2. Processor/CLI catches NoteGenError:
   - Log error (localized)
   - Show user-friendly message (via LocaleManager)
3. Unknown errors go to fallback handling

---

# 12. Scripts Utilities Module

## Design Goals

- Provide maintenance tools for development/testing
- Includes workspace cleanup, memory rebuild, config export
- Keep scripts modular and lightweight
- Follow static/dynamic data separation, use LocaleManager

## Main Scripts

| Script | Main Function |
|:------|:-------------|
| clean_workspace.py | Clean and recreate workspace/ directory |
| rebuild_memory.py | Rebuild ChromaDB memory from texts |
| export_config_docs.py | Export config as Markdown docs |

## Example Script Designs (Internationalized)

### clean_workspace.py

```python
import shutil
import os
from src.utils.locale_manager import LocaleManager

def clean_workspace(workspace_dir="workspace/"):
    locale = LocaleManager("resources/locales/locale.yaml", "zh")
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    print(locale.get("scripts.clean_workspace_success"))

if __name__ == "__main__":
    clean_workspace()
```

### rebuild_memory.py

```python
from src.note_generator.input_handler import InputHandler
from src.note_generator.splitter import Splitter
from
