# KnowForge Test Plan Document

## 1. Purpose

This document details the testing strategy, test case design methodology, coverage requirements, and testing tools specifications for the KnowForge project to ensure quality and stability.

Intended audience: Developers, QA engineers, maintainers.

---

## 2. Testing Objectives

- Ensure all core functionalities work correctly and defects are identified/fixed.
- Achieve specified code coverage standards for critical modules.
- Validate system stability, reliability and performance.
- Ensure quality and stability of LLM integration.

---

## 3. Test Scope

### 3.1 Functional Testing

- **Input Processing (InputHandler)**: Support for PDFs, images, web links, code files.
- **Text Splitting (Splitter)**: Validate intelligent text splitting logic.
- **Vectorization & Memory Management (Embedder & MemoryManager)**: Verify vector storage and similarity search.
- **Model Calling (LLMCaller)**: Validate DeepSeek API calls and exception handling.
- **Output Generation (OutputWriter)**: Verify Markdown, Jupyter Notebook and PDF generation.
- **CLI Interaction (cli_main.py)**: Validate command line argument parsing and execution flow.

### 3.2 Non-Functional Testing

- **Performance Testing**: Large document processing (memory usage, execution time).
- **Compatibility Testing**: Cross-platform consistency (Windows, Linux).
- **Error Handling**: System error handling and logging.

---

## 4. Testing Strategy

KnowForge adopts a dual-track testing approach:

1. **Unit & Integration Tests** (pytest): Functional tests for individual modules
2. **LLM Integration E2E Tests** (dedicated scripts): Real-environment tests for LLM-dependent features

Benefits:
- Standardized unit/integration tests ensure basic functionality
- Dedicated scripts test LLM features in real environments
- Easy CI/CD integration for automated quality control
- Isolates external dependencies for efficient testing

### 4.1 Unit Tests (Pytest)

- Use `pytest` framework to test individual functions/modules.
- 5-10 test cases per core module minimum.
- Mock LLM API calls and filesystem operations.

### 4.2 Integration Tests (Pytest)

- Test module interactions and data flow.
- Verify component integration.
- Simulate LLM responses for various scenarios.

### 4.3 LLM Integration Tests (Dedicated Scripts)

- Use `scripts/llm_integration_check.py` for real-environment LLM testing.
- End-to-end validation from input to final output.
- Test LLM-assisted splitting and content analysis.

---

## 5. Testing Tools

| Tool | Purpose |
|------|---------|
| pytest | Execute unit/integration tests |
| pytest-cov | Code coverage analysis |
| mock | Mock external dependencies (APIs, filesystem) |
| llm_integration_check.py | LLM integration testing |
| GitLab CI/CD | Future CI/CD integration |

---

## 6. Test Environment Setup

1. Create environment:
   ```bash
   conda create -n knowforge python=3.11
   conda activate knowforge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure LLM credentials:
   ```bash
   export DEEPSEEK_API_KEY="your_api_key"
   # Or modify resources/config/config.yaml
   ```

---

## 7. Test Execution

### 7.1 Run Unit/Integration Tests

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_splitter.py
```

### 7.2 Check Coverage

Minimum 80% coverage for core modules.

```bash
pytest --cov=src tests/
```

### 7.3 Generate Coverage Report

```bash
pytest --cov-report=html --cov=src tests/
```

### 7.4 Run LLM Integration Tests

```bash
python scripts/llm_integration_check.py
```

---

## 8. Test Case Examples

### 8.1 Python Test Cases (pytest)

#### Example: InputHandler Unit Test

```python
import pytest
from src.note_generator.input_handler import InputHandler

def test_scan_inputs(tmp_path):
    (tmp_path / "pdf").mkdir()
    (tmp_path / "pdf/sample.pdf").write_text("dummy pdf")
    handler = InputHandler(str(tmp_path), str(tmp_path))
    inputs = handler.scan_inputs()
    assert "pdf" in inputs
    assert len(inputs["pdf"]) == 1
```

#### Example: Splitter Integration Test (Mocked LLM)

```python
import pytest
from unittest.mock import patch, MagicMock
from src.note_generator.splitter import Splitter

@patch('src.note_generator.splitter.LLMCaller')
def test_llm_assisted_splitting(mock_llm_caller):
    mock_instance = MagicMock()
    mock_instance.call.return_value = {"splits": [100, 250, 400]}
    mock_llm_caller.return_value = mock_instance
    
    splitter = Splitter(chunk_size=500, overlap_size=50)
    text = "This is a long text..." * 20
    chunks = splitter.split_with_llm_assistance(text)
    
    assert len(chunks) == 4
    mock_instance.call.assert_called_once()
```

### 8.2 LLM Integration Test Cases

llm_integration_check.py performs:
1. Full note generation workflow
2. Output file validation
3. Log verification for "LLM split successful"
4. Content completeness checks
5. Performance monitoring

Sample output:
```
[OK] Output file exists: output/markdown/notes.md
[OK] Log contains LLM split confirmation
[OK] Output content is non-empty
Execution time: 89.89 seconds
```

---

## 9. Dual-Track Testing Benefits

### 9.1 Python Tests (pytest) Advantages

- **Fast execution**: No external API dependencies
- **High coverage**: Tests edge cases and errors
- **Stable**: Unaffected by network/service status
- **Precise**: Pinpoints issues to specific modules
- **CI/CD friendly**: Easy automation integration

### 9.2 LLM Integration Tests Advantages

- **Real environment**: Tests actual LLM API interactions
- **End-to-end validation**: Full workflow verification
- **Integration issues**: Catches Python-test-invisible issues
- **Performance monitoring**: Real-world metrics
- **Quality assurance**: Validates LLM feature effectiveness

### 9.3 Test Responsibility Matrix

| Test Type | Primary Responsibility | When to Run |
|-----------|------------------------|-------------|
| Python Tests | Module functionality, logic, edge cases | Code commits, CI |
| LLM Tests | LLM quality, E2E flow, performance | Feature changes, releases, env changes |

---

## 10. Test Results & Defect Tracking

- Use GitHub/GitLab Issues for defect tracking.
- Prioritize issues (Critical, High, Medium, Low).
- Distinguish between functional defects and LLM-related issues.

---

## 11. CI/CD & Automation (Future Plan)

- Integrate GitLab CI/CD for automated Python tests and coverage.
- Schedule regular LLM integration tests.
- Run Python tests pre-merge for all pull requests.
