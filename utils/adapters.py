"""
TrialAgents 工具函数适配层

提供统一的接口，已完成阶段1、2、3的迁移：
- utils.metrics: 指标计算函数
- utils.retrieval: 检索系统
- utils.context: 医疗上下文构建 (datasets_info + process_data + healthcare_context_utils)
- utils.data: 数据处理工具

使用方式:
    from utils.adapters import RetrievalSystem, wrap_data, ContextBuilder
"""

from typing import Any

# ============================================================
# 已迁移的模块 - 直接从本地导入
# ============================================================

try:
    # 阶段1: metrics 模块
    from .metrics import get_final_result as _get_final_result

    # 阶段2: retrieval 模块
    from .retrieval import RetrievalSystem as _RetrievalSystem

    # 阶段3: context 模块
    from .context import ContextBuilder as _ContextBuilder
    from .context import (
        process_patient_df,
        load_clinical_trial_datasets,
    )

    # data 模块
    from .data_utils.runner_utils import process_time_series as _process_time_series
    from .data_utils.paths import build_experiment_path, ensure_dir

    LOCAL_IMPORT_SUCCESS = True

except ImportError as e:
    print(f"Warning: Failed to import from local modules: {e}")
    LOCAL_IMPORT_SUCCESS = False
    _get_final_result = None
    _RetrievalSystem = None
    _ContextBuilder = None
    _process_time_series = None

# ============================================================
# 检索系统适配器
# ============================================================

class RetrievalSystem:
    """
    检索系统适配器

    继承自 utils.retrieval.RetrievalSystem
    """

    def __init__(self, retriever_name: str = "MedCPT", corpus_name: str = "MSD", **kwargs):
        """
        初始化检索系统

        Args:
            retriever_name: 检索器名称
            corpus_name: 语料库名称
            **kwargs: 其他参数
        """
        if _RetrievalSystem is None:
            raise NotImplementedError("RetrievalSystem not available. Please check installation.")
        self._system = _RetrievalSystem(retriever_name=retriever_name, corpus_name=corpus_name, **kwargs)

    def retrieve(self, query: str, k: int = 1):
        """检索相关文档"""
        return self._system.retrieve(query, k=k)

    def retrieve_with_metadata(self, query: str, k: int = 5) -> dict:
        """
        带元数据的检索

        Args:
            query: 查询字符串
            k: 返回结果数量

        Returns:
            包含检索结果和元数据的字典
        """
        texts, scores, indices = self.retrieve(query, k=k)
        return {
            "texts": texts,
            "scores": scores,
            "indices": indices,
            "query": query,
            "k": k,
        }


# ============================================================
# 上下文构建器
# ============================================================

class ContextBuilder:
    """
    医疗上下文构建器

    封装 utils.context.ContextBuilder
    """

    def __init__(
        self,
        dataset_name: str,
        task: str,
        model_name: str,
        mode: str
    ):
        """
        初始化上下文构建器

        Args:
            dataset_name: 数据集名称
            task: 任务名称
            model_name: 模型名称
            mode: 模式 (test/val)
        """
        if _ContextBuilder is None:
            raise NotImplementedError("ContextBuilder not available. Please check installation.")
        self._builder = _ContextBuilder(
            dataset_name=dataset_name,
            task=task,
            model_name=model_name,
            mode=mode
        )

    def build(self, patient_index: int, patient_id: int):
        """构建基础上下文，返回 (basic_context, subcontext, hcontext) 三元组"""
        return self._builder.generate_context(patient_index, patient_id)

    def build_enhanced(self, patient_index: int, patient_id: int) -> dict:
        """增强版上下文构建"""
        basic_context, subcontext, hcontext = self.build(patient_index, patient_id)
        return {
            "basic_context": basic_context,
            "subcontext": subcontext,
            "hcontext": hcontext,
            "patient_id": patient_id,
            "enhanced": True,
        }


# ============================================================
# 函数透传
# ============================================================

def get_final_result(logits: dict) -> tuple:
    """
    获取最终结果

    Args:
        logits: 预测结果字典

    Returns:
        (final_logit, final_conf)
    """
    if _get_final_result is None:
        raise NotImplementedError("get_final_result not available.")
    return _get_final_result(logits)


def wrap_data(config: dict) -> list:
    """
    包装数据函数

    委托到 utils.data.wrap_data 的完整实现。
    支持所有 6 个 NCT 数据集（NCT00003299, NCT03041311, NCT00041119,
    NCT00079274, NCT00694382, NCT00174655）。

    Args:
        config: 配置字典，包含 dataset_name, task, trial_path 等

    Returns:
        患者数据列表
    """
    from .data_utils.wrap_data import wrap_data as _wrap_data
    return _wrap_data(config)


def process_time_series(dataset: str, patient_list: list) -> None:
    """处理时间序列数据"""
    if _process_time_series is None:
        raise NotImplementedError("process_time_series not available.")
    return _process_time_series(dataset, patient_list)


def load_results(*args, **kwargs):
    """加载结果函数"""
    raise NotImplementedError("load_results not available.")


# ============================================================
# 导出和状态检查
# ============================================================

__all__ = [
    'RetrievalSystem',
    'wrap_data',
    'ContextBuilder',
    'get_final_result',
    'process_time_series',
    'load_results',
    'build_experiment_path',
    'ensure_dir',
    'LOCAL_IMPORT_SUCCESS',
]


def check_adapters() -> dict:
    """
    检查适配器状态

    Returns:
        适配器状态字典
    """
    return {
        "local_import_success": LOCAL_IMPORT_SUCCESS,
        "available_adapters": [a for a in __all__ if not a.startswith('_')],
        "migrated_modules": ["metrics", "retrieval", "context", "data"],
    }


if __name__ == "__main__":
    print("Checking adapters...")
    status = check_adapters()
    print(f"Local import success: {status['local_import_success']}")
    print(f"Migrated modules: {status['migrated_modules']}")
    print(f"Available adapters: {status['available_adapters']}")
