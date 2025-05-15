<!--
 * @Author: @ydzat
 * @Date: 2025-04-28 18:45:10
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-04-29 02:23:13
 * @Description: Environment Configuration Guide
-->
# KnowForge Environment Setup Guide

This document provides detailed instructions for environment configuration, dependency installation and development setup for the KnowForge project, suitable for both developers and users.

---

## 1. System Requirements

### Supported Operating Systems

- **Windows**: Windows 10 or later
- **Linux**: Ubuntu 20.04+, Fedora 41+
- **macOS**: 11.0 (Big Sur) or later

### Basic Requirements

- **Python**: 3.11 or higher
- **Disk Space**: Minimum 1GB free space (including dependencies, models and working directories)
- **Memory**: Recommended minimum 4GB RAM, 8GB+ recommended for large documents

---

## 2. Development Environment Setup

### Method 1: Using Conda (Recommended)

1. **Install Anaconda or Miniconda**:
   - Download links: [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - Follow official installation guide

2. **Create Virtual Environment**:
   ```bash
   # Create new environment
   conda create -n knowforge python=3.11

   # Activate environment
   conda activate knowforge
   ```

3. **Clone Project**:
   ```bash
   git clone https://github.com/yourusername/knowforge.git
   cd knowforge
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Using venv

1. **Install Python 3.11+**:
   - Download from [Python.org](https://www.python.org/downloads/)

2. **Create Virtual Environment**:
   ```bash
   # Create venv in project directory
   python -m venv venv

   # Activate virtual environment
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Dependency Packages

KnowForge dependencies are categorized as follows:

### Core Dependencies

- **python-dotenv (≥1.0.0)**: Environment variable management
- **pyyaml (≥6.0)**: YAML file processing
- **typer (≥0.9.0)**: CLI interface creation

### Input Processing

- **pdfplumber (≥0.10.0)**: PDF text extraction
- **easyocr (≥1.7.0)**: Image OCR text recognition
- **beautifulsoup4 (≥4.12.0)**: Web content extraction
- **requests (≥2.31.0)**: HTTP request handling
- **pygments (≥2.16.0)**: Code syntax highlighting

### Vectorization & Memory

- **sentence-transformers (≥2.3.0)**: Text vectorization
- **chromadb (≥0.4.0)**: Vector database

### LLM Interface

- **openai (≥1.0.0)**: OpenAI-compatible API calls

### Output Generation

- **markdown-it-py (≥3.0.0)**: Markdown processing
- **nbformat (≥5.9.0)**: Jupyter Notebook format
- **weasyprint (≥60.0)**: PDF generation

### Testing

- **pytest (≥7.4.0)**: Unit testing framework
- **pytest-cov (≥4.1.0)**: Test coverage
- **pytest-mock (≥3.11.0)**: Mocking for tests

---

## 4. Environment Variables

KnowForge uses environment variables to manage API keys and other sensitive information. It is **strongly recommended** to use environment variables rather than hardcoding sensitive information in configuration files.

### 4.1 Configuring API Keys (Recommended Approach)

**Method 1: Set System Environment Variables Directly (More Secure)**

```bash
# Linux/macOS
export DEEPSEEK_API_KEY="your-api-key-here"

# Windows (CMD)
set DEEPSEEK_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="your-api-key-here"
```

This approach is more secure as API keys are not saved in any files. For persistent settings, you can add these commands to your shell configuration file (like `.bashrc`, `.zshrc`, etc.).

**Method 2: Using a `.env` File**

If you need a more portable way to manage environment variables:

1. **Create a `.env` file**:
   - Create a `.env` file in the project root directory
   ```bash
   touch .env
   # or for Windows
   echo. > .env
   ```

2. **Edit the `.env` file**:
   ```dotenv
   # DeepSeek API Key (required)
   DEEPSEEK_API_KEY=your-api-key-here
   
   # If you're using OpenAI API (optional)
   # OPENAI_API_KEY=your-openai-key-here
   
   # API Base URL (optional, uses default)
   # DEEPSEEK_API_BASE_URL=https://api.deepseek.com
   ```

> **Security Tips**: 
> - The `.env` file contains sensitive information and is excluded in `.gitignore`. Ensure you never commit it to code repositories.
> - Keep your API keys secure and don't expose them in public places or shared environments.
> - In production environments, prefer using system environment variables or secure key management services.

### 4.2 Unified Key Management

KnowForge is designed to require setting up API keys **only once** for use across all components. This is achieved through the `ConfigLoader`:

- The system prioritizes getting keys from environment variables
- All components that need LLM functionality use the same method to retrieve keys
- The project supports multiple LLM providers and will automatically use the correct environment variables based on configuration

For example, if you set the `DEEPSEEK_API_KEY` environment variable, all components from the text splitter to the note generator will use this key without requiring additional configuration.

---

## 5. Directory Structure Initialization

Project requires these writable directories:

```bash
# Create required directory structure
mkdir -p input/{pdf,images,codes,links}
mkdir -p workspace/{preprocessed,split_segments,embeddings,memory_db}
mkdir -p output/{markdown,notebook,pdf,logs}
```

These will be auto-created on first run but can be created manually.

---

## 6. Verification

After installation, verify setup with:

```bash
# Show CLI help
python gen_notes.py --help

# Check version
python gen_notes.py version

# Run unit tests
pytest tests/
```

Successful execution indicates proper setup.

---

## 7. Troubleshooting

1. **Module Not Found**:
   ```
   ModuleNotFoundError: No module named 'src'
   ```
   Solution: Run commands from project root or add root to PYTHONPATH.

2. **EasyOCR Dependencies**:
   - Windows: May need Visual C++ Build Tools
   - Linux: May need `apt-get install libgl1`

3. **ChromaDB Issues**:
   - Use compatible version: `pip install chromadb==0.4.0`
   - Some systems need: `apt-get install sqlite3`

4. **WeasyPrint Dependencies**:
   - Linux: `apt-get install libpango1.0-dev libharfbuzz-dev libffi-dev`
   - Windows: Follow [official guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows) for GTK+

---

## 8. Upgrading

When new version is released:

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

## 9. Performance Optimization

- **GPU Acceleration**: Install PyTorch with CUDA for faster vectorization:
  ```bash
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  ```

- **Memory Optimization**: For large documents, adjust `chunk_size` in `config.yaml` to reduce memory usage.
