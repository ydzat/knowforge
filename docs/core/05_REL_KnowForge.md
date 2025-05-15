# KnowForge 发布指南
## 1. 文档目的

本文件为KnowForge项目提供明确的发布和部署指南，帮助开发人员与维护人员完成跨平台发布。

适用范围：项目发布工程师、系统管理员、运维人员。

---

## 2. 发布准备

### 2.1 环境确认

- 确认Python版本：Python 3.11+
- 确认依赖项安装完整
  ```bash
  pip install -r requirements.txt
  ```

### 2.2 功能完整性确认

- 确保通过所有单元测试与集成测试，覆盖率满足标准（≥80%）。
- 执行测试命令：
  ```bash
  pytest --cov=src tests/
  ```

---

## 3. 打包发布流程

### 3.1 使用PyInstaller打包

- 使用PyInstaller将项目打包为单一可执行文件

#### 打包命令示例

```bash
pyinstaller --onefile gen_notes.py
```

### 3.2 输出与验证

- 打包后的文件存放于 `dist/` 目录下。
- 验证发布包功能完整性：

```bash
./dist/gen_notes generate --input-dir "input/" --output-dir "output/" --formats "markdown,ipynb"
```

---

## 4. 发布文件结构

发布目录结构示例：

```plaintext
KnowForge_v1.0.0/
├── gen_notes                 # 可执行程序（Windows: gen_notes.exe）
├── input/                    # 默认输入目录（发布时可为空）
├── output/                   # 默认输出目录（发布时可为空）
├── resources/                # 必须随附的资源目录
│   ├── config/
│   ├── locales/
│   └── templates/
├── README.md                 # 用户使用说明
└── LICENSE                   # 开源协议文件
```

---

## 5. 跨平台发布注意事项

### 5.1 Windows平台

- 推荐使用PyInstaller在Windows环境下生成。
- Windows上生成命令：

```bash
pyinstaller --onefile --windowed gen_notes.py
```

### 5.2 Linux平台

- 推荐在Linux（如Fedora 41）环境下打包，确保可执行文件兼容性。
- Linux生成命令：

```bash
pyinstaller --onefile gen_notes.py
```

---

## 6. Docker化部署（可选高级方案）

### Dockerfile示例

```Dockerfile
FROM python:3.11

WORKDIR /app

COPY . /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "gen_notes.py", "generate", "--input-dir", "input/", "--output-dir", "output/", "--formats", "markdown,ipynb"]
```

### Docker构建与运行

```bash
docker build -t knowforge:v1.0 .
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output knowforge:v1.0
```

---

## 7. 用户文档与发布说明

- 提供详细的用户使用手册（README.md）。
- 版本更新与功能变动记录在发布说明（CHANGELOG.md）。

---

## 8. 发布后验证

- 在目标平台运行发布版。
- 确认所有核心功能正常，尤其是：
  - 输入文件识别与处理
  - 输出格式生成（Markdown、Notebook、PDF）
  - 日志记录与异常处理

---

## 9. 问题追踪与反馈机制

- 用户反馈与问题追踪统一使用GitHub/GitLab Issues进行。
- 定期处理用户反馈，持续迭代优化产品。
