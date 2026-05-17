"""
TrialAgents协作引擎模块

包含多智能体协作的核心机制：
- CollaborationEngine: 管理多轮讨论流程

迁移说明：
本模块代码从 m-agents/utils/framework.py 中的 Collaboration 类迁移至此
"""

from .collaboration_engine import CollaborationEngine, Collaboration

__all__ = [
    'CollaborationEngine',
    'Collaboration',
]
