"""
usage:
    from utils.adapters import RetrievalSystem, wrap_data, ContextBuilder
"""

from typing import Any

# ============================================================
# Migrated modules - imported directly from local
# ============================================================

# Phase 1: metrics module
from .metrics import get_final_result as _get_final_result

# Phase 2: retrieval module
from .others.retrieval import RetrievalSystem as _RetrievalSystem

# Phase 3: context module
from .context import ContextBuilder as _ContextBuilder
from .context import (
    process_patient_df,
    load_clinical_trial_datasets,
)

# data module
from .data_utils.runner_utils import process_time_series as _process_time_series
from .data_utils.paths import build_experiment_path, ensure_dir

try:

    LOCAL_IMPORT_SUCCESS = True

except ImportError as e:
    print(f"Warning: Failed to import from local modules: {e}")
    LOCAL_IMPORT_SUCCESS = False
    _get_final_result = None
    _RetrievalSystem = None
    _ContextBuilder = None
    _process_time_series = None

# ============================================================
# Retrieval System Adapter
# ============================================================

class RetrievalSystem:
    """
    Retrieval System Adapter

    Inherits from utils.retrieval.RetrievalSystem
    """

    def __init__(self, retriever_name: str = "MedCPT", corpus_name: str = "MSD", **kwargs):
        """
        Initialize the retrieval system

        Args:
            retriever_name: Retriever name
            corpus_name: Corpus name
            **kwargs: Other parameters
        """
        if _RetrievalSystem is None:
            raise NotImplementedError("RetrievalSystem not available. Please check installation.")
        self._system = _RetrievalSystem(retriever_name=retriever_name, corpus_name=corpus_name, **kwargs)

    def retrieve(self, query: str, k: int = 1):
        """Retrieve relevant documents"""
        return self._system.retrieve(query, k=k)

    def retrieve_with_metadata(self, query: str, k: int = 5) -> dict:
        """
        Retrieve with metadata

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            Dictionary containing retrieval results and metadata
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
# Context Builder
# ============================================================

class ContextBuilder:
    """
    Medical context builder

    Wraps utils.context.ContextBuilder
    """

    def __init__(
        self,
        dataset_name: str,
        task: str,
        model_name: str,
        mode: str
    ):
        """
        Initialize the context builder

        Args:
            dataset_name: Dataset name
            task: Task name
            model_name: Model name
            mode: Mode (test/val)
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
        """Build basic context, returns (basic_context, subcontext, hcontext) triplet"""
        return self._builder.generate_context(patient_index, patient_id)

    def build_enhanced(self, patient_index: int, patient_id: int) -> dict:
        """Enhanced context builder"""
        basic_context, subcontext, hcontext = self.build(patient_index, patient_id)
        return {
            "basic_context": basic_context,
            "subcontext": subcontext,
            "hcontext": hcontext,
            "patient_id": patient_id,
            "enhanced": True,
        }


# ============================================================
# Function Passthrough
# ============================================================

def get_final_result(logits: dict) -> tuple:
    """
    Get the final result

    Args:
        logits: Prediction result dictionary

    Returns:
        (final_logit, final_conf)
    """
    if _get_final_result is None:
        raise NotImplementedError("get_final_result not available.")
    return _get_final_result(logits)


def wrap_data(config: dict) -> list:
    """
    Data wrapping function

    Delegates to the full implementation in utils.data.wrap_data.
    Supports all 6 NCT datasets (NCT00003299, NCT03041311, NCT00041119,
    NCT00079274, NCT00694382, NCT00174655).

    Args:
        config: Configuration dictionary containing dataset_name, task, trial_path, etc.

    Returns:
        List of patient data
    """
    from .data_utils.wrap_data import wrap_data as _wrap_data
    return _wrap_data(config)


def process_time_series(dataset: str, patient_list: list) -> None:
    """Process time series data"""
    if _process_time_series is None:
        raise NotImplementedError("process_time_series not available.")
    return _process_time_series(dataset, patient_list)


def load_results(*args, **kwargs):
    """Load results function"""
    raise NotImplementedError("load_results not available.")


# ============================================================
# Exports and Status Check
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
    Check adapter status

    Returns:
        Adapter status dictionary
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
