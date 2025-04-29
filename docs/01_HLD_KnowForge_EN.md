<!--
 * @Author: @ydzat 
 * @Date: 2025-04-28 15:31:28
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-04-29 02:14:07
 * @Description: High-Level Design Document
-->
# KnowForge: AI-Powered Learning Note Generator Design Document

---

## Project Overview

**KnowForge** is an AI-powered automated learning note generator that integrates multiple input sources (PDFs, images, code, web links) and processes them to generate structured notes, supporting output formats like PDF, Jupyter Notebook (.ipynb), and Markdown (.md).

Project Goals:
- Cross-platform support (Windows, Linux)
- Multi-language support (Chinese, English)
- Multi-modal operation (CLI and Python library calls)
- Large document processing and memory management
- Support for adding new models (DeepSeek Chat / Reasoner)
- Future integration with MoeAI-C system

---

## Global Design Overview

### Input Sources
- PDF documents
- Images (with OCR support)
- Web links (automatic web content scraping)
- Code files (Python, etc.)

### Content Processing Flow
1. Input analysis: Detect files/images/webpages/code
2. Automatic splitting: Split large documents by chapters/sections
3. Vectorization: Convert to embeddings using sentence-transformers
4. Memory management: Save to ChromaDB
5. Call DeepSeek model: Generate summaries and add design notes
6. Output format generation: Markdown → Notebook/PDF

### Output Formats
- Markdown (.md)
- Jupyter Notebook (.ipynb)
- PDF (.pdf)

---

## Technology Stack

| Module | Technologies |
|:-------|:------------|
| Input Processing | pdfplumber, easyocr, requests+beautifulsoup4, pygments |
| Vectorization | sentence-transformers (all-MiniLM-L6-v2) |
| LLM Model Calls | openai-python SDK + DeepSeek API |
| Vector Storage | ChromaDB |
| Output Generation | markdown-it-py, weasyprint, nbformat |
| CLI Controller | Typer |
| Packaging | PyInstaller |
| Security Management | python-dotenv |
| Logging System | logging |
| Exception Handling | Custom exception classes |

---

## Engineering Optimizations

- Separation of static info and dynamic logic: Added resources/ directory to manage configs, templates, and multilingual resources
- Standardized error handling: Defined unified exception classes and logging system for stability
- Dual-mode support (CLI + Python Library): Supports integration as a library for future MoeAI-C system needs

---

## Project Directory Structure

```bash
knowforge/
├── input/                  # User input directory
│   ├── pdf/                # Original PDF documents
│   ├── images/             # Lecture screenshots/scanned images
│   ├── codes/              # Reference code files (Python, etc.)
│   └── links/              # Text files containing web URLs (one per line)
├── workspace/              # Intermediate cache
│   ├── preprocessed/       # Parsed raw text
│   ├── split_segments/     # Split text segments
│   ├── embeddings/         # Generated text vector cache
│   └── memory_db/         # ChromaDB local vector database files
├── output/                # Final output directory
│   ├── markdown/          # Generated Markdown notes
│   ├── notebook/          # Generated Jupyter Notebook (.ipynb)
│   ├── pdf/               # Final PDF documents
│   └── logs/              # Runtime logs
├── docs/                  # Engineering design documents
├── src/
│   ├── note_generator/    # Core logic modules
│   │   ├── __init__.py
│   │   ├── processor.py          # Main workflow controller
│   │   ├── input_handler.py      # Input file preprocessing
│   │   ├── splitter.py           # Document splitting
│   │   ├── embedder.py           # Text vectorization
│   │   ├── memory_manager.py     # Vector memory retrieval
│   │   ├── llm_caller.py         # DeepSeek API wrapper
│   │   └── output_writer.py      # Markdown/Notebook/PDF generator
│   ├── cli/
│   │   ├── __init__.py
│   │   └── cli_main.py           # CLI interface
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── exceptions.py         # Custom exceptions
│   │   ├── logger.py             # Log management
│   │   ├── config_loader.py      # Config and env var loading
│   │   └── file_utils.py         # File operation utilities
├── resources/             # Static resources
│   ├── config/
│   │   ├── config.yaml            # System config
│   │   ├── llm_profiles.yaml      # LLM model configs
│   ├── locales/
│   │   ├── en.yaml                # English language pack
│   │   └── zh.yaml                # Chinese language pack
│   ├── templates/
│   │   ├── note_template.md       # Note template
│   │   └── system_prompts.yaml    # Interaction prompt templates
├── tests/                 # Unit tests
│   ├── __init__.py
│   ├── test_processor.py
│   ├── test_embedder.py
│   ├── test_output_writer.py
│   └── test_memory_manager.py
├── scripts/               # Utility scripts
│   ├── clean_workspace.py
│   ├── rebuild_memory.py
│   └── export_config_docs.py
├── gen_notes.py           # Typer CLI entry point
├── requirements.txt
├── README.md
├── LICENSE
└── .env.example           # Environment variable template
```

---

## Configuration Management System Design

### Goals
- Centralize all runtime parameters, paths, and switches
- Enable adjustments without modifying core code
- Keep sensitive configs (API keys, model choices) secure

### Components

1. `resources/config/config.yaml`
   - Main config file for default settings
2. `.env` file
   - Stores sensitive info (API keys), avoiding hardcoding
3. `src/utils/config_loader.py`
   - Handles unified config loading, validation, and merging

### config.yaml Example Structure

```yaml
# System config
system:
  language: "zh"
  workspace_dir: "workspace/"
  output_dir: "output/"

# Input settings
input:
  allowed_formats: ["pdf", "jpg", "png", "txt", "md"]

# Text splitter settings
splitter:
  chunk_size: 800
  overlap_size: 100

# Embedding settings
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

# Memory database settings
memory:
  chroma_db_path: "workspace/memory_db/"

# LLM call settings
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  temperature: 0.5

# Output settings
output:
  formats: ["markdown", "ipynb", "pdf"]
```

### .env Example

```dotenv
# DeepSeek API Key
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# Other keys can be added
```

### config_loader.py Design

```python
import yaml
import os
from dotenv import load_dotenv

class ConfigLoader:
    def __init__(self, config_path: str):
        load_dotenv()
        self.config = self._load_yaml(config_path)

    def _load_yaml(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default=None):
        """Access config items via dot notation (e.g. system.language)"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key, {})
        return value if value != {} else default

    def get_env(self, env_var: str, default=None):
        """Get environment variables (for API keys, etc.)"""
        return os.getenv(env_var, default)
```

---

# Environment Management System Design

### Goals
- Protect sensitive info, avoiding API key hardcoding
- Support flexible dev/prod environment switching
- Unified management for local and cloud deployments

### Components
- `.env` file (in project root, excluded from version control)
- `python-dotenv` for auto-loading env vars
- `config_loader.get_env()` for unified access

### Example: Sensitive Info Management

1. Developer creates `.env`:

```dotenv
DEEPSEEK_API_KEY=your-secret-key-here
```

2. Program initialization:

```python
from utils.config_loader import ConfigLoader

config = ConfigLoader("resources/config/config.yaml")
api_key = config.get_env("DEEPSEEK_API_KEY")
```

3. Production deployment uses system env vars directly

### Security Notes
- Add `.env` to `.gitignore`
- Use `os.getenv` for Docker/Kubernetes compatibility
- Never log sensitive field contents

---

# Logging & Exception Handling System Design

### Goals
- Unified capture and recording of runtime info and errors
- Better debugging and user experience (graceful errors vs crashes)
- Dual-channel logging (file + terminal) for troubleshooting

### Logging System Design

| Aspect | Details |
|:-------|:--------|
| Framework | Python standard `logging` |
| Output | Terminal + file (output/logs/note_gen.log) |
| Levels | DEBUG / INFO / WARNING / ERROR / CRITICAL |
| Format | Timestamp + Level + Module + Message |

#### logger.py Example

```python
import logging
import os

def setup_logger(log_dir="output/logs", log_level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "note_gen.log")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("knowforge")
```

Usage:

```python
from utils.logger import setup_logger

logger = setup_logger()
logger.info("Program started")
```

---

### Exception Handling System Design

| Aspect | Details |
|:-------|:--------|
| Hierarchy | Base exception class with specialized subclasses |
| Capture | Unified try-except templates |
| Logging | Automatic traceback recording |
| User Feedback | Clean error messages only |

#### exceptions.py Example

```python
class NoteGenError(Exception):
    """Base exception for all expected errors"""
    pass

class InputError(NoteGenError):
    """Input processing errors"""
    pass

class APIError(NoteGenError):
    """API call errors"""
    pass

class MemoryError(NoteGenError):
    """Vector memory errors"""
    pass

class OutputError(NoteGenError):
    """Output generation errors"""
    pass
```

#### Usage Example

```python
from utils.logger import setup_logger
from utils.exceptions import NoteGenError

logger = setup_logger()

try:
    # Main logic
    ...
except NoteGenError as e:
    logger.error(f"Known Error: {str(e)}")
    print("Error occurred, see logs for details.")
except Exception as e:
    logger.exception(f"Unexpected Error: {str(e)}")
    print("Unexpected error, logged for review.")
```

---

## Memory Management System Design

### Goals
- Handle large documents with smart splitting
- Maintain context coherence across splits
- Support dynamic memory retrieval during generation

### Core Approach

| Phase | Strategy |
|:------|:---------|
| Splitting | Smart segmentation by natural breaks, fallback to sliding window |
| Vectorization | `sentence-transformers` embeddings stored in ChromaDB |
| Dynamic Memory | Embedding similarity search for relevant context |
| Window Management | Dynamic input sizing based on model context limits |

### Splitter Module (splitter.py) Design

#### Splitting Logic
- Primary: LLM-assisted structural analysis
- Secondary: Paragraph breaks
- Fallback: Fixed-length chunks (800 chars with 100 overlap)

#### Config (config.yaml)

```yaml
splitter:
  chunk_size: 800
  overlap_size: 100
  use_llm: true
  llm_provider: "deepseek"
```

### Memory Manager (memory_manager.py) Design

#### Core Interface

```python
class MemoryManager:
    def __init__(self, chroma_db_path: str, embedding_model: str):
        pass

    def add_segments(self, segments: list) -> None:
        """Vectorize and store text segments"""
        pass

    def query_similar(self, query_text: str, top_k: int = 5) -> list:
        """Retrieve similar segments"""
        pass

    def rebuild_memory(self, segments: list) -> None:
        """Reset and rebuild memory"""
        pass
```

#### Usage Example

```python
memory = MemoryManager("workspace/memory_db/", "sentence-transformers/all-MiniLM-L6-v2")

# Add memory
memory.add_segments(["This chapter introduces RL basics.", "MDP is..."])

# Query memory
related = memory.query_similar("What's MDP?", top_k=3)
```

### Window Management
Config (llm_profiles.yaml):

```yaml
deepseek-chat:
  max_context_length: 8000

deepseek-reasoner:
  max_context_length: 32000
```

Window logic:
- Estimate token counts per segment
- Stay under `max_context_length`
- Drop least relevant segments when over limit

---

## Input Processing System Design

### Goals
- Support multiple input types (PDFs, images, web links, code)
- Standardize to plain text segments for downstream processing
- Auto-detect directory structures

### Input Types & Processing

| Type | Tools/Methods | Output |
|:-----|:-------------|:-------|
| PDF | pdfplumber | Extracted text |
| Image | easyocr | OCR text |
| Web Link | requests+BeautifulSoup | Web content |
| Code | Direct read + Pygments (optional) | Formatted code |

### Directory Scanning Rules

1. Root: `input/`
2. Subdirs by type:
   - `input/pdf/`
   - `input/images/`
   - `input/links/`
   - `input/codes/`

Supports two common structures:
- Flat files (ch1.pdf, ch2.pdf)
- Chapter subdirs (ch1/01_intro.pdf)

### InputHandler (input_handler.py) Design

```python
class InputHandler:
    def __init__(self, input_dir: str, workspace_dir: str):
        pass

    def scan_inputs(self) -> dict:
        """Scan and categorize inputs"""
        pass

    def extract_texts(self) -> list:
        """Extract text content"""
        pass

    def save_preprocessed(self) -> None:
        """Save extracted texts"""
        pass
```

### Usage Example

```python
handler = InputHandler("input/", "workspace/")
handler.scan_inputs()
segments = handler.extract_texts()
handler.save_preprocessed()
```

### Config Support

```yaml
input:
  allowed_formats: ["pdf", "jpg", "png", "txt", "md"]
  max_file_size_mb: 100
  enable_ocr_languages: ["ch_sim", "en"]
```

---

## Output Generation System Design

### Goals
- Support multiple output formats (Markdown, Notebook, PDF)
- Ensure clear structure and good formatting
- Integrate with LocaleManager for multilingual output
- Flexible format selection

### Supported Formats

| Format | Description |
|:-------|:-----------|
| Markdown | Standard .md files |
| Notebook | .ipynb with Markdown cells |
| PDF | High-quality rendered PDFs |

### OutputWriter (output_writer.py) Design

```python
class OutputWriter:
    def __init__(self, workspace_dir: str, output_dir: str, locale_manager):
        pass

    def generate_markdown(self, notes: list, filename: str) -> str:
        """Generate Markdown file"""
        pass

    def generate_notebook(self, notes: list, filename: str) -> str:
        """Generate .ipynb file"""
        pass

    def generate_pdf(self, markdown_path: str, filename: str) -> str:
        """Render PDF from Markdown"""
        pass
```

### Usage Example

```python
writer = OutputWriter("workspace/", "output/", locale)

md_path = writer.generate_markdown(notes, "my_notes")
nb_path = writer.generate_notebook(notes, "my_notes")
writer.generate_pdf(md_path, "my_notes")
```

### Template Support (note_template.md)

```markdown
# {{ title }}

**Generated**: {{ timestamp }}

{{ content }}
```

### Output Directory Structure

- Markdown -> `output/markdown/`
- Notebook -> `output/notebook/`
- PDF -> `output/pdf/`

### Config Support

```yaml
output:
  formats: ["markdown", "ipynb", "pdf"]
  pdf:
    page_size: "A4"
    margin: "1cm"
```

### Dependencies

- Markdown: `markdown-it-py`
- Notebook: `nbformat`
- PDF: `weasyprint`

---

## CLI & Library Interface Design

### Goals
- Support CLI one-click operation
- Support Python library integration
- Maintain consistent interfaces

### CLI Design (Typer-based)

#### Entry Point (gen_notes.py)

```python
import typer
from src.cli.cli_main import cli

def main():
    ty
