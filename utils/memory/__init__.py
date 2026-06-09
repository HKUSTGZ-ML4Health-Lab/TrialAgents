"""
TrialAgents Memory Bank

Long-term memory module enabling agents to learn from past errors.

Core components:
- MemoryStore: Storage management (ErrorCase, SuccessCase, PatternMeta)
- MemoryEngine: FAISS vector search engine
- MemoryInjector: Prompt injector
- MemoryUpdater: Auto updater

Usage example:
    from utils.memory import MemoryStore, MemoryEngine, MemoryInjector, MemoryUpdater

    store = MemoryStore()
    engine = MemoryEngine(store=store)
    injector = MemoryInjector(engine)
    updater = MemoryUpdater(store, engine)

    # Search similar errors
    similar_errors = engine.search_similar_errors(patient_text, k=5)

    # Inject into prompt
    enhanced_prompt = injector.inject_for_clinician(base_prompt, patient_text)

    # Register prediction result
    is_error, case_id = updater.register_prediction_result(...)
"""
from .schema import (
    ErrorType,
    PatientProfile,
    ErrorCase,
    SuccessCase,
    PatternMeta,
)

from .store import MemoryStore

from .engine import MemoryEngine, SearchResult

from .injector import MemoryInjector, PromptEnhancer

from .updater import MemoryUpdater, create_memory_updater


__all__ = [
    # Schema
    'ErrorType',
    'PatientProfile',
    'ErrorCase',
    'SuccessCase',
    'PatternMeta',
    # Store
    'MemoryStore',
    # Engine
    'MemoryEngine',
    'SearchResult',
    # Injector
    'MemoryInjector',
    'PromptEnhancer',
    # Updater
    'MemoryUpdater',
    'create_memory_updater',
]
