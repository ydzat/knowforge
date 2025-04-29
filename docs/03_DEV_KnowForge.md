# KnowForge 开发者指南

## 1. 文档目的与适用范围

本文件旨在为KnowForge项目的开发人员提供详细的环境搭建、开发流程、测试规范、日志异常处理等操作指南。

适用范围：
- 主要面向开发人员、贡献者、测试人员。
- 对项目的所有模块、功能、依赖等进行详细说明，确保统一开发与运行标准。

---

## 2. 开发环境搭建指南

### 必要工具与环境

1. **操作系统**
   - 支持平台：Windows 10+、Linux（推荐Fedora 41）
   
2. **Anaconda**
   - 推荐使用Anaconda管理Python环境。
   - [安装指南](https://www.anaconda.com/products/individual)

3. **Python版本**
   - Python 3.11+
   ```bash
   conda create -n knowforge python=3.11
   conda activate knowforge
   ```

---

## 3. 项目结构与主要模块说明

项目目录结构请参考[HLD](./01_HLD_KnowForge.md)。

---

## 4. 安装依赖与运行指令

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行项目

```bash
python gen_notes.py generate --input-dir "input/" --output-dir "output/" --formats "markdown,ipynb"
```

### 单元测试

```bash
pytest tests/
```

---

## 5. 开发流程规范

### 代码规范

- **命名规范**：类名使用驼峰式，函数和变量名使用蛇形命名。
- **注释与文档**：每个类与函数需清晰的docstring。
- **格式工具**：使用`black`自动格式化代码，遵循PEP8。
- **分支管理**：采用Git管理策略。

### 提交规范

- 提交信息需清晰明确。
- 每次PR合并需通过代码审核与单元测试。

---

## 6. 测试流程规范

- 使用`pytest`进行单元和集成测试。
- 使用`mock`模拟外部API调用与数据交互。

### 覆盖率检查

```bash
pytest --cov=src tests/
```

- 主要模块需达到至少80%的覆盖率。

---

## 7. 日志与异常处理注意事项

- 必须使用`LoggerManager`进行日志管理。
- 异常处理统一使用`NoteGenError`及其子类。
- 所有日志与异常提示文字需通过`LocaleManager`进行国际化处理，避免硬编码。

---

## 8. 本地化与静态资源使用规范

- 所有提示信息与异常消息通过`LocaleManager.get()`加载。
- 静态资源（模板、图片）统一放置在`resources/templates/`与`resources/assets/`。

---

## 9. 打包与发布流程（PyInstaller）

使用PyInstaller生成单文件可执行程序：

```bash
pyinstaller --onefile gen_notes.py
```

打包后的程序位于`dist/`目录，支持Windows和Linux。

---

## 10. 未来扩展接口预留（如MoeAI-C对接）

- 预留RESTful API或GraphQL接口，未来可与外部AI模块或工具集成。
