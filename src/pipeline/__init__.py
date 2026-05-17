"""
TrialAgents工作流管道模块

包含三阶段工作流的实现：
- Phase 1: Clinical Review Generation (临床综述生成)
- Phase 2: Multi-round Collaborative Diagnosis (多轮协同诊断)
- Phase 3: Diverse Task Completion (多样化任务完成)

迁移说明：
本模块代码从 m-agents/collaboration_pipeline.py 迁移至此
"""

from .phase1_review import run_phase1
from .phase2_collaborate import run_phase2
from .phase3_execute import run_phase3

__all__ = [
    'run_phase1',
    'run_phase2',
    'run_phase3',
]
