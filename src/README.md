# src 目录说明

本目录包含TrialAgents项目的核心源代码。

## 目录结构

```
src/
├── agents/                 # 智能体实现
│   ├── __init__.py
│   ├── base_agent.py       # 基础Agent类 (应从utils/framework.py迁移)
│   ├── clinician_agent.py     # Clinician Agent实现 (应从utils/framework.py迁移)
│   ├── supervisor_agent.py     # Supervisor + Trialist Agent实现 (应从utils/framework.py迁移)
│   └── simulator_agent.py  # Simulator Agent实现 (新增)
│
├── pipeline/               # 工作流管道实现
│   ├── __init__.py
│   ├── phase1_review.py    # 阶段一：临床综述生成 (应从collaboration_pipeline.py迁移)
│   ├── phase2_collaborate.py  # 阶段二：多轮协同诊断 (应从utils/framework.py迁移)
│   └── phase3_execute.py   # 阶段三：多样化任务完成 (新增)
│
└── collaboration/          # 协作机制
    ├── __init__.py
    └── collaboration_engine.py  # 协作引擎 (应从utils/framework.py迁移)
```

## 迁移指南

### 从utils/framework.py迁移

`utils/framework.py` 包含以下类，应迁移到src目录：

1. **Agent基类** → `src/agents/base_agent.py`
   - 包含通用的LLM客户端初始化
   - API配置管理

2. **ClinicianAgent类** → `src/agents/clinician_agent.py`
   - Clinician Agent的完整实现
   - 包含analysis()和collaborate()方法

3. **SupervisorAgent类** → `src/agents/supervisor_agent.py`
   - 包含Trialist和Supervisor的功能
   - summary()和revise_logits()方法

4. **Collaboration类** → `src/collaboration/collaboration_engine.py`
   - 多轮协作讨论的核心逻辑

### 从collaboration_pipeline.py迁移

`collaboration_pipeline.py` 包含完整的三阶段流程，应拆分到：

1. **阶段一逻辑** → `src/pipeline/phase1_review.py`
   - 多Clinician并行分析
   - 综述生成

2. **阶段二逻辑** → `src/pipeline/phase2_collaborate.py`
   - Trialist综述合成
   - 多轮协作讨论

3. **阶段三逻辑** → `src/pipeline/phase3_execute.py`
   - 任务路由
   - LNL模型调用
   - Simulator Agent调用

4. **主流程** → `src/pipeline/main_pipeline.py`
   - 整合三个阶段的完整流程

## 模块依赖关系

```
main_pipeline.py
    ├── phase1_review.py
    │       └── clinician_agent.py
    ├── phase2_collaborate.py
    │       ├── supervisor_agent.py
    │       └── collaboration_engine.py
    └── phase3_execute.py
            ├── simulator_agent.py
            └── lnl_model (from utils/training/)

All agents:
    └── base_agent.py
```

## 使用示例

```python
# 导入新的src模块结构
from src.agents import ClinicianAgent, SupervisorAgent, SimulatorAgent
from src.pipeline import Phase1Review, Phase2Collaboration, Phase3Execution
from src.collaboration import CollaborationEngine

# 运行完整流程
from src.pipeline.main_pipeline import TrialAgentsPipeline

pipeline = TrialAgentsPipeline(config)
result = pipeline.run(patient_data)
```

## 注意事项

1. 迁移时需保持向后兼容，原文件暂时保留
2. 逐步更新import路径
3. 测试确保功能一致性
4. 完成迁移后，原utils/framework.py可以标记为deprecated
