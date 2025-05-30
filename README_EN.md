# KnowForge

[简体中文](README.md) | English

![Version](https://img.shields.io/badge/version-0.1.7-blue)
![License](https://img.shields.io/badge/license-GPL--3.0-green)
![Python](https://img.shields.io/badge/Python-3.11+-yellow)

**KnowForge** is an AI-powered automated learning notes generator that integrates and processes multiple input sources (PDF, images, code, web links) to create structured notes in various output formats.

## Features

- **Multi-source Input Support**: Process PDF documents, images (OCR), web links, and code files
- **Intelligent Text Splitting**: Automatically split large documents based on chapters/paragraphs
- **Memory Management System**: Context memory and retrieval through vectorization and ChromaDB
- **AI-Generated Enhancement**: Generate summaries and note content using large language models, supporting multiple LLM interfaces
- **Advanced PDF Image Extraction**: Multi-method redundant extraction strategy with 100% success rate
- **Advanced OCR Processing**: OCR with LLM enhancement and memory system integration for improved accuracy
- **Multiple Output Formats**: Support for Markdown, Jupyter Notebook, and PDF formats
- **Advanced Logging System**: Integrated Logloom with multilingual log messages and automatic file rotation
- **Multilingual Support**: Chinese and English interfaces
- **Cross-platform Compatibility**: Support for Windows and Linux systems

## Version Planning

- **Current Version**: v0.1.7 (Output System Enhancement Completed)
- **v1.0.0**: Final vision version with complete functionality (including Web interface)
- **v2.0.0**: MoeAI-C system integration version

## Development Roadmap

- **v0.2.0** (Expected: June 2025): Advanced Memory Management & LLM Enhancement
- **v0.3.0** (Expected: August 2025): Knowledge Graph & Cross-Document Relations
- **v0.4.0** (Expected: October 2025): Learning Assistance & Intelligent Assessment
- **v0.5.0-v1.0.0** (Expected: December 2025-March 2026): Web Interface & User Experience

## Development Progress

Refer to the [KnowForge Roadmap Document](./docs/core/08_ROADMAP_KnowForge_EN.md). Currently completed version v0.1.7 (Output System Enhancement) and working on version v0.2.0 (Advanced Memory Management & LLM Enhancement).

Latest milestones:
- **May 20, 2025 (v0.1.7)**: Completed output system enhancement with user configuration system, offline resource management, and multi-engine output support.
- **May 16, 2025 (v0.1.6)**: Implemented advanced PDF image extraction and OCR enhancement features with multi-method redundant extraction strategy and improved memory management system.
- **May 14, 2025 (v0.1.5)**: Developed document comprehensive processing functionality with DocumentAnalyzer and ContentExtractor components.
- **May 8, 2025 (v0.1.3)**: Completely migrated LocaleManager to Logloom, optimized internationalization resource loading mechanism, implemented smart key name parsing functionality.
- **May 5, 2025 (v0.1.2)**: Vector Memory Management optimization, fixed compatibility issues with ChromaDB API, enhanced hybrid retrieval strategies.
- **May 2, 2025 (v0.1.1)**: Integration of the Logloom logging system, enhancing system reliability and internationalization support.

**Next Steps:** We've now completed the v0.1.7 output enhancement features and are moving into the v0.2.0 development phase, focusing on advanced memory management and LLM module enhancement. Our updated roadmap includes clear milestones from the current version to v1.0.0, with Web interface development scheduled for completion before v1.0.0. Detailed design documentation is available at [13_DocumentProcessingDesign.md](./docs/others/13_DocumentProcessingDesign.md) and [13_AdvancedMemoryManager_Progress.md](./docs/modules/memory_management/13_AdvancedMemoryManager_Progress.md).

For the complete updated development roadmap with detailed milestones, please check [08_ROADMAP_KnowForge.md](./docs/core/08_ROADMAP_KnowForge.md).

## Installation Guide

### System Requirements

- **Operating System**: Windows 10+, Linux (Ubuntu 20.04+, Fedora 41+)
- **Python Version**: 3.11 or higher
- **Memory**: 4GB+ recommended, 8GB+ for processing large documents

### Method 1: Using Conda (Recommended)

```bash
# Create environment
conda create -n knowforge python=3.11
conda activate knowforge

# Clone project
git clone https://github.com/yourusername/knowforge.git
cd knowforge

# Install dependencies
pip install -r requirements.txt

# Install Logloom
pip install logloom

# Set environment variables in conda environment
# Linux/macOS
export DEEPSEEK_API_KEY=your-api-key-here
# Windows
set DEEPSEEK_API_KEY=your-api-key-here
```

### Method 2: Using venv

```bash
# Create environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate
# Activate environment (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Logloom
pip install logloom
```

## Environment Configuration

### If Using Conda

```
export DEEPSEEK_API_KEY=your-api-key-here
```

### If Using venv

1. Create a `.env` file in the root directory (if it doesn't exist)

2. Edit the `.env` file, enter your API key
```
DEEPSEEK_API_KEY=your-api-key-here
```

## Usage

### Command Line Usage

Basic usage:

```bash
python gen_notes.py generate --input-dir "input/" --output-dir "output/" --formats "markdown,ipynb"
```

Parameter description:

- `--input-dir`: Input file directory
- `--output-dir`: Output file directory
- `--formats`: Generated output formats, comma-separated (supports markdown, ipynb, pdf)

View help:

```bash
python gen_notes.py --help
```

### Input File Organization

Place your files in the following directory structure:

```
input/
├── pdf/          # Place PDF documents
├── images/       # Place images
├── codes/        # Place code files
└── links/        # Place text files containing URLs (one link per line)
```

### Output File Location

Generated files will be saved in:

```
output/
├── markdown/     # Markdown files
├── notebook/     # Jupyter Notebook files
├── pdf/          # PDF files
└── logs/         # Log files
```

## Development Progress

The project is currently in Iteration 3 (Advanced Features and Optimization) development, implementing memory management, multi-format output support, and advanced input processing capabilities.
For detailed version history and plans, please check the [Changelog](CHANGELOG.md) and [Development Roadmap](docs/08_ROADMAP_KnowForge_EN.md).

## Development Guide

For detailed development guidelines, please refer to the [Developer Documentation](docs/03_DEV_KnowForge_EN.md).

### Testing

Run tests:

```bash
# Run all tests
pytest tests/

# Check test coverage
pytest --cov=src tests/

# Run LLM integration test (you need to place files in the input folder before testing)
python scripts/llm_integration_check.py
```

### Logging System (Logloom)

The Logloom logging system is configured through the `resources/config/logloom_config.yaml` file and supports:

- Multilingual log messages (Chinese/English)
- Automatic file rotation to prevent log files from growing too large
- Configurable log formats and levels
- Console and file dual-channel output

## Troubleshooting

### EasyOCR Dependency Issues
- Windows may require Visual C++ Build Tools
- Linux may require additional system libraries: `apt-get install libgl1`

### WeasyPrint Dependency Issues
- Linux: `apt-get install libpango1.0-dev libharfbuzz-dev libffi-dev`
- Windows: Follow the [official guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows) to install GTK+

### Logloom Issues
- If you encounter "cannot resolve import logloom" errors, make sure logloom is installed: `pip install logloom`
- Configuration file location error: Check that `resources/config/logloom_config.yaml` exists

## Project Structure

```
knowforge/
├── input/                  # User input directory
├── output/                 # Final output directory
├── workspace/              # Intermediate cache area
├── src/                    # Source code
│   ├── note_generator/     # Core logic modules
│   ├── cli/                # CLI interface
│   └── utils/              # Utility classes
├── resources/              # Static resources
│   ├── config/             # Configuration files
│   │   ├── config.yaml     # Main configuration
│   │   └── logloom_config.yaml # Logloom configuration
│   └── locales/            # Language resources
│       ├── logloom_zh.yaml # Logloom Chinese resource
│       └── logloom_en.yaml # Logloom English resource
├── tests/                  # Unit tests
└── scripts/                # Tool scripts
```

## Contribution Guidelines

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## License

This project uses the GNU General Public License v3.0 (GPL-3.0) - see the [LICENSE](LICENSE) file for details. This license ensures that the software and its derivative works remain open source and requires any modifications or derivative works to also be published under the GPL-3.0 license.

## Author

- **@ydzat** - [GitHub](https://github.com/ydzat)