# KnowForge Release Guide
## 1. Purpose

This document provides clear release and deployment guidelines for the KnowForge project, assisting developers and maintainers with cross-platform distribution.

Intended audience: Release engineers, system administrators, operations personnel.

---

## 2. Release Preparation

### 2.1 Environment Verification

- Confirm Python version: Python 3.11+
- Ensure complete dependency installation
  ```bash
  pip install -r requirements.txt
  ```

### 2.2 Functionality Verification

- Pass all unit and integration tests with coverage ≥80%.
- Run tests:
  ```bash
  pytest --cov=src tests/
  ```

---

## 3. Packaging Process

### 3.1 PyInstaller Packaging

- Package project as single executable using PyInstaller

#### Packaging Command Example

```bash
pyinstaller --onefile gen_notes.py
```

### 3.2 Output Verification

- Packaged files located in `dist/` directory.
- Validate release package functionality:

```bash
./dist/gen_notes generate --input-dir "input/" --output-dir "output/" --formats "markdown,ipynb"
```

---

## 4. Release File Structure

Example release structure:

```plaintext
KnowForge_v1.0.0/
├── gen_notes                 # Executable (Windows: gen_notes.exe)
├── input/                    # Default input directory (empty in release)
├── output/                   # Default output directory (empty in release)
├── resources/                # Required resource directory
│   ├── config/
│   ├── locales/
│   └── templates/
├── README.md                 # User documentation
└── LICENSE                   # Open source license
```

---

## 5. Cross-Platform Considerations

### 5.1 Windows Platform

- Recommended to build on Windows environment.
- Windows build command:

```bash
pyinstaller --onefile --windowed gen_notes.py
```

### 5.2 Linux Platform

- Recommended to build on Linux (e.g. Fedora 41) for compatibility.
- Linux build command:

```bash
pyinstaller --onefile gen_notes.py
```

---

## 6. Docker Deployment (Optional Advanced Solution)

### Dockerfile Example

```Dockerfile
FROM python:3.11

WORKDIR /app

COPY . /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "gen_notes.py", "generate", "--input-dir", "input/", "--output-dir", "output/", "--formats", "markdown,ipynb"]
```

### Docker Build & Run

```bash
docker build -t knowforge:v1.0 .
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output knowforge:v1.0
```

---

## 7. Documentation & Release Notes

- Provide detailed user manual (README.md).
- Record version updates and feature changes in CHANGELOG.md.

---

## 8. Post-Release Verification

- Run release version on target platforms.
- Verify all core functionality:
  - Input file processing
  - Output format generation (Markdown, Notebook, PDF)
  - Logging and exception handling

---

## 9. Issue Tracking & Feedback

- Use GitHub/GitLab Issues for user feedback and issue tracking.
- Regularly address user feedback for continuous improvement.
