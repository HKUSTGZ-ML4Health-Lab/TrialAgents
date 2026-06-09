"""
TrialAgents Collaboration Engine Module

Core multi-agent collaboration mechanism:
- CollaborationEngine: Manages multi-round discussion workflow

Migrated from the Collaboration class in m-agents/utils/framework.py
"""
from .collaboration_engine import CollaborationEngine, Collaboration

__all__ = [
    'CollaborationEngine',
    'Collaboration',
]
