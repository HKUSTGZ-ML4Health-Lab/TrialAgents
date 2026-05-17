"""
TrialAgents Memory Bank

长期记忆模块，让Agent能够从过去的错误中学习

主要组件：
- MemoryStore: 存储管理（ErrorCase, SuccessCase, PatternMeta）
- MemoryEngine: FAISS向量检索引擎
- MemoryInjector: Prompt注入器
- MemoryUpdater: 自动更新器

用法示例：
    from utils.memory import MemoryStore, MemoryEngine, MemoryInjector, MemoryUpdater

    # 初始化
    store = MemoryStore()
    engine = MemoryEngine(store=store)
    injector = MemoryInjector(engine)
    updater = MemoryUpdater(store, engine)

    # 检索相似错误
    similar_errors = engine.search_similar_errors(patient_text, k=5)

    # 注入到prompt
    enhanced_prompt = injector.inject_for_clinician(base_prompt, patient_text)

    # 登记预测结果
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
