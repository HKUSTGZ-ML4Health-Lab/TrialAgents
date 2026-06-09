"""
TrialAgents Workflow Pipeline Module

Three-phase workflow implementation:
- Phase 1: Clinical Review Generation
- Phase 2: Multi-round Collaborative Diagnosis
- Phase 3: Diverse Task Completion

Migrated from m-agents/collaboration_pipeline.py
"""
from .phase1_review import run_phase1
from .phase2_collaborate import run_phase2
from .phase3_execute import run_phase3

__all__ = [
    'run_phase1',
    'run_phase2',
    'run_phase3',
]
