# KnowForge Developer Guide

## 1. Purpose and Scope

This document provides detailed guidelines for KnowForge project developers regarding environment setup, development workflow, testing standards, logging and exception handling.

Scope:
- Primarily for developers, contributors, and testers.
- Detailed specifications for all project modules, features, and dependencies to ensure unified development and operation standards.

---

## 2. Development Environment Setup Guide

### Required Tools and Environment

1. **Operating System**
   - Supported platforms: Windows 10+, Linux (Fedora 41 recommended)
   
2. **Anaconda**
   - Recommended for Python environment management.
   - [Installation Guide](https://www.anaconda.com/products/individual)

3. **Python Version**
   - Python 3.11+
   ```bash
   conda create -n knowforge python=3.11
   conda activate knowforge
   ```

---

## 3. Project Structure and Module Overview

Refer to [HLD](./01_HLD_KnowForge_EN.md) for project directory structure.

---

## 4. Dependency Installation and Execution Commands

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python gen_notes.py generate --input-dir "input/" --output-dir "output/" --formats "markdown,ipynb"
```

### Unit Tests

```bash
pytest tests/
```

---

## 5. Development Workflow Standards

### Code Standards

- **Naming Conventions**: CamelCase for classes, snake_case for functions/variables.
- **Comments & Documentation**: Clear docstrings for all classes and functions.
- **Formatting**: Use `black` for auto-formatting, follow PEP8.
- **Branch Management**: Git flow strategy.

### Commit Standards

- Clear and descriptive commit messages.
- All PRs must pass code review and unit tests.

---

## 6. Testing Standards

- Use `pytest` for unit and integration tests.
- Use `mock` for external API calls and data interactions.

### Coverage Check

```bash
pytest --cov=src tests/
```

- Core modules require at least 80% test coverage.

---

## 7. Logging and Exception Handling Guidelines

- Must use `LoggerManager` for logging.
- Unified exception handling with `NoteGenError` and subclasses.
- All log messages and exceptions must be internationalized via `LocaleManager`, no hardcoded strings.

---

## 8. Localization and Static Resources Standards

- All messages and exceptions loaded via `LocaleManager.get()`.
- Static resources (templates, images) stored in `resources/templates/` and `resources/assets/`.

---

## 9. Packaging and Release Process (PyInstaller)

Generate single-file executable with PyInstaller:

```bash
pyinstaller --onefile gen_notes.py
```

Built executable located in `dist/`, supports Windows and Linux.

---

## 10. Future Extension Interfaces (e.g. MoeAI-C Integration)

- Reserved RESTful API/GraphQL interfaces for future integration with external AI modules/tools.
