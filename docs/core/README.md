# KnowForge 核心设计文档

本目录包含KnowForge项目的核心设计文档和技术规范。这些文档描述了整个系统的架构、设计原则、开发流程和迭代计划等。

## 文档列表

### 设计文档

- [高级设计](./01_HLD_KnowForge.md) | [英文版](./01_HLD_KnowForge_EN.md)
  - 描述系统的整体架构、主要组件和设计原则
- [低级设计](./02_LLD_KnowForge.md) | [英文版](./02_LLD_KnowForge_EN.md)
  - 详细描述各个组件的实现细节、数据结构和算法
- [新模块低级设计](./02_LLD_KnowForge_new_modules.md)
  - 描述新增模块的详细设计

### 流程文档

- [开发指南](./03_DEV_KnowForge.md) | [英文版](./03_DEV_KnowForge_EN.md)
  - 开发流程、代码规范和贡献指南
- [测试计划](./04_TEST_KnowForge.md) | [英文版](./04_TEST_KnowForge_EN.md)
  - 测试策略、测试用例和测试流程
- [发布指南](./05_REL_KnowForge.md) | [英文版](./05_REL_KnowForge_EN.md)
  - 版本发布流程和要求

### 规划文档

- [迭代计划](./06_ITER_KnowForge.md) | [英文版](./06_ITER_KnowForge_EN.md)
  - 迭代开发计划和里程碑
- [环境配置](./07_ENV_KnowForge.md) | [英文版](./07_ENV_KnowForge_EN.md)
  - 开发和生产环境的配置要求
- [项目路线图](./08_ROADMAP_KnowForge.md) | [英文版](./08_ROADMAP_KnowForge_EN.md)
  - 长期发展规划和功能路线图

## 文档命名规则

核心文档使用以下命名规则：

- 序号前缀：用于区分不同类型的文档，并确保它们在文件列表中的顺序
  - 01-02：设计文档
  - 03-05：流程文档
  - 06-08：规划文档
- 文档类型缩写：如HLD（高级设计）、LLD（低级设计）、DEV（开发指南）等
- 项目名：KnowForge
- 语言后缀：中文文档无后缀，英文文档添加`_EN`后缀

例如：`01_HLD_KnowForge.md` 和 `01_HLD_KnowForge_EN.md`
