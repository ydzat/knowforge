# KnowForge 测试计划文档

## 1. 文档目的

本文件详细描述KnowForge项目的测试策略、测试用例设计方法、覆盖率要求以及测试工具使用规范，确保项目质量与稳定性。

适用人员：开发人员、测试工程师、维护人员。

---

## 2. 测试目标

- 保证所有核心功能正常运行，发现并修复缺陷。
- 确保关键模块的代码覆盖率达到指定标准。
- 验证系统稳定性、可靠性及性能。
- 确保LLM集成的质量和稳定性。

---

## 3. 测试范围

### 3.1 功能测试

- **输入处理（InputHandler）**：支持PDF、图片、网页链接、代码文件输入。
- **文本拆分（Splitter）**：验证智能拆分文本逻辑正确性。
- **向量化与记忆管理（Embedder & MemoryManager）**：验证向量存储和相似性检索正确性。
- **模型调用（LLMCaller）**：验证DeepSeek API调用正常，异常处理得当。
- **输出生成（OutputWriter）**：Markdown、Jupyter Notebook与PDF文件的生成及格式正确。
- **CLI交互（cli_main.py）**：命令行参数解析与执行流程正确性。

### 3.2 非功能测试

- **性能测试**：检查大文档处理性能（内存占用、执行时间）。
- **兼容性测试**：确保跨平台（Windows、Linux）功能一致。
- **异常与错误处理**：系统错误处理与日志记录。

---

## 4. 测试策略

KnowForge项目采用双轨测试策略，将测试分为两个主要部分：

1. **单元和集成测试**（使用pytest）：针对各个模块和组件的功能测试
2. **LLM集成端到端测试**（使用专用脚本）：针对LLM依赖功能的真实环境测试

这种双轨测试策略的优势在于：
- 使用pytest进行标准化的单元和集成测试，保障基础功能和组件级质量
- 使用专用脚本进行LLM相关功能的真实环境测试，降低开发与部署环境差异带来的风险
- 易于CI/CD集成，支持自动化质量控制
- 隔离对外部依赖的测试，提高测试效率和准确性

### 4.1 单元测试（Unit Tests）- Pytest

- 使用`pytest`框架，确保单一函数或模块功能正确。
- 每个核心模块至少提供5-10个测试用例。
- 使用Mock模拟LLM API调用和文件系统操作，关注内部逻辑。

### 4.2 集成测试（Integration Tests）- Pytest

- 测试模块间的交互逻辑，验证整体流程。
- 关注组件之间的衔接和数据流转。
- 模拟LLM返回结果，测试各种情况下的处理逻辑。

### 4.3 LLM集成测试（LLM Integration Tests）- 专用脚本

- 使用`scripts/llm_integration_check.py`进行真实环境下的LLM调用测试。
- 端到端流程验证，从输入文件到最终输出的完整流程。
- 测试LLM辅助拆分、内容分析等依赖真实LLM的功能。

---

## 5. 测试工具

| 工具名称        | 作用                             |
|--------------|--------------------------------|
| pytest       | 执行单元与集成测试用例                  |
| pytest-cov   | 统计并检查代码覆盖率                    |
| mock         | 模拟外部依赖（如API、文件系统）             |
| llm_integration_check.py | LLM集成端到端测试专用脚本 |
| GitLab CI/CD | 自动化持续集成、执行自动化测试（未来可选）    |

---

## 6. 测试环境搭建

1. 创建并激活环境：
   ```bash
   conda create -n knowforge python=3.11
   conda activate knowforge
   ```

2. 安装测试依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 配置LLM访问凭证：
   ```bash
   # 根据实际情况配置环境变量或配置文件
   export DEEPSEEK_API_KEY="your_api_key"
   # 或修改resources/config/config.yaml中的相关配置
   ```

---

## 7. 测试执行

### 7.1 执行单元和集成测试（Python Tests）

```bash
# 执行所有Python测试
pytest tests/

# 执行特定模块测试
pytest tests/test_splitter.py
```

### 7.2 检查覆盖率

要求核心模块覆盖率至少80%。

```bash
pytest --cov=src tests/
```

### 7.3 查看覆盖率报告

```bash
pytest --cov-report=html --cov=src tests/
```

### 7.4 执行LLM集成测试

```bash
# 确保input/目录下有测试样本
python scripts/llm_integration_check.py
```

---

## 8. 测试用例分类与示例

### 8.1 Python测试用例（使用pytest）

这些测试主要关注内部逻辑和模块行为，通常使用模拟(mock)代替真实的LLM调用。

#### 示例：输入处理模块单元测试

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

#### 示例：文本拆分模块集成测试（模拟LLM）

```python
import pytest
from unittest.mock import patch, MagicMock
from src.note_generator.splitter import Splitter

@patch('src.note_generator.splitter.LLMCaller')
def test_llm_assisted_splitting(mock_llm_caller):
    # 模拟LLM返回的拆分建议
    mock_instance = MagicMock()
    mock_instance.call.return_value = {"splits": [100, 250, 400]}
    mock_llm_caller.return_value = mock_instance
    
    splitter = Splitter(chunk_size=500, overlap_size=50)
    text = "这是一段长文本..." * 20  # 创建测试文本
    chunks = splitter.split_with_llm_assistance(text)
    
    assert len(chunks) == 4  # 预期生成4个片段
    mock_instance.call.assert_called_once()  # 确认LLM被调用
```

### 8.2 LLM集成测试用例（使用llm_integration_check.py）

这些测试关注在真实环境中的端到端流程和LLM集成质量，使用真实的LLM API调用。

#### 示例：LLM辅助拆分功能检查

llm_integration_check.py脚本会：
1. 执行完整的笔记生成流程
2. 验证输出文件存在
3. 检查日志中是否存在"LLM成功拆分文本"的标记
4. 验证生成内容的完整性和非空性
5. 检测LLM响应时间和系统性能

例如：
```
[OK] 输出文件存在: output/markdown/notes.md
[OK] 日志中包含 LLM 拆分成功 相关信息
[OK] 输出笔记内容非空
执行完成，耗时: 89.89秒
```

---

## 9. 为什么采用双轨测试策略

### 9.1 Python测试（pytest）的优势

- **快速执行**：不依赖外部API，可以快速运行
- **高覆盖率**：可以测试各种边界条件和错误情况
- **稳定性**：不受网络和外部服务状态影响
- **精确定位**：可以精确定位问题模块和函数
- **适合CI/CD**：可以集成到自动化流程中

### 9.2 LLM集成测试的优势

- **真实环境**：测试真实LLM API的交互和响应
- **端到端验证**：验证完整流程的功能正确性
- **发现集成问题**：发现Python测试中无法发现的集成问题
- **性能监控**：监控真实环境中的性能和响应时间
- **质量保障**：确保LLM辅助功能的实际效果满足需求

### 9.3 测试职责划分

| 测试类型 | 主要职责 | 何时执行 |
|---------|---------|---------|
| Python测试 | 模块功能、内部逻辑、异常处理、边界条件 | 每次代码提交、持续集成 |
| LLM集成测试 | LLM拆分质量、端到端流程、性能监控、真实环境验证 | 功能变更后、版本发布前、环境变更后 |

---

## 10. 测试结果记录与缺陷跟踪

- 使用GitHub/GitLab的Issue功能进行缺陷跟踪。
- 明确问题优先级（Critical, High, Medium, Low），便于快速定位和修复。
- 区分普通功能缺陷和LLM相关问题，便于针对性解决。

---

## 11. 持续集成与自动化（未来规划）

- 集成GitLab CI/CD Pipeline，自动执行Python测试并生成覆盖率报告。
- 定期执行LLM集成测试，监控LLM功能质量。
- 每次合并请求（Merge Request）前自动执行Python测试。