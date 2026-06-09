"""
TrialAgents Data Path Management Utility

Unified management of all data paths, supporting symlinks and environment variable overrides
"""

import os
from pathlib import Path
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Default data path configuration (relative paths)
DEFAULT_PATHS = {
    "raw": "data/raw",
    "processed": "data/processed/trials",
    "outputs": "data/outputs",
    "responses": "data/outputs/responses/response",
    "digital_twins": "data/outputs/digital_twins/digital_twins",
    "checkpoints": "data/checkpoints",
    "logs": "logs",
    "training_runs": "logs/training",
    "baselines_runs": "logs/baselines",
    "baselines_saved_models": "baselines/saved_models",
    "baselines_response": "data/outputs/baselines_response",
}


def get_project_root() -> Path:
    """Get the project root directory"""
    return PROJECT_ROOT


def get_data_path(data_type: str, *subpaths: str) -> str:
    """
    Get data path

    Args:
        data_type: Data type (raw, processed, outputs, responses, digital_twins, checkpoints, logs)
        *subpaths: Subpath components

    Returns:
        Full path string
    """
    # First check environment variables
    env_var = f"TRIALAGENTS_{data_type.upper()}_PATH"
    if env_var in os.environ:
        base_path = Path(os.environ[env_var])
    else:
        # Use default configuration
        relative_path = DEFAULT_PATHS.get(data_type, f"data/{data_type}")
        base_path = PROJECT_ROOT / relative_path

    # Append subpath components
    if subpaths:
        base_path = base_path.joinpath(*subpaths)

    # Ensure the directory exists
    base_path.parent.mkdir(parents=True, exist_ok=True)

    return str(base_path)


def get_raw_data_path(*subpaths: str) -> str:
    """Get raw data path"""
    return get_data_path("raw", *subpaths)


def get_processed_data_path(*subpaths: str) -> str:
    """Get processed data path"""
    return get_data_path("processed", *subpaths)


def get_outputs_path(*subpaths: str) -> str:
    """Get outputs path"""
    return get_data_path("outputs", *subpaths)


def get_responses_path(*subpaths: str) -> str:
    """Get responses output path"""
    return get_data_path("responses", *subpaths)


def get_digital_twins_path(*subpaths: str) -> str:
    """Get digital twins path"""
    return get_data_path("digital_twins", *subpaths)


def get_checkpoints_path(*subpaths: str) -> str:
    """Get checkpoints path"""
    return get_data_path("checkpoints", *subpaths)


def get_logs_path(*subpaths: str) -> str:
    """Get logs path"""
    return get_data_path("logs", *subpaths)


def ensure_dir(path: str) -> str:
    """
    Ensure the directory exists

    Args:
        path: Path string

    Returns:
        Path string
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def build_experiment_path(
    dataset_name: str,
    task: str,
    llm: str,
    corpus_name: str,
    group: str,
    suffix: str = "morpos"
) -> str:
    """
    Build experiment output path

    Format: data/outputs/responses/{dataset_name}_{task}_{llm}_{corpus_name}_{group}_{suffix}

    Args:
        dataset_name: Dataset name
        task: Task name
        llm: LLM model name
        corpus_name: Corpus name
        group: Group (control/experimental)
        suffix: Suffix

    Returns:
        Experiment output path
    """
    experiment_name = f"{dataset_name}_{task}_{llm}_{corpus_name}_{group}_{suffix}"
    return get_responses_path(experiment_name)


# Compatibility function for the original project's data paths
def get_m_agents_data_path(relative_path: str) -> str:
    """
    Get data path from the original m-agents project

    Used by adapters to read data from the original project

    Args:
        relative_path: Path relative to m-agents

    Returns:
        Full path
    """
    m_agents_root = Path(os.getenv("MAS_M_AGENTS_ROOT", str(PROJECT_ROOT.parent / "m-agents")))
    return str(m_agents_root / relative_path)


# Convenience functions: get common data file paths
def get_trial_background_path(dataset_name: str) -> str:
    """Get trial background file path"""
    return get_raw_data_path("ctgov", dataset_name, "bg.json")


def get_patient_data_path(dataset_name: str, patient_id: str) -> str:
    """Get patient data file path"""
    return get_processed_data_path(dataset_name, f"{patient_id}.json")


def get_patient_response_path(experiment_name: str, patient_id: str) -> str:
    """Get patient response directory path"""
    return get_responses_path(experiment_name, patient_id)


def get_training_runs_path(*subpaths: str) -> str:
    """Get LNL training run path"""
    return get_data_path("training_runs", *subpaths)


def get_baselines_runs_path(*subpaths: str) -> str:
    """Get baseline training run path"""
    return get_data_path("baselines_runs", *subpaths)


def get_baselines_saved_models_path(*subpaths: str) -> str:
    """Get baseline model save path"""
    return get_data_path("baselines_saved_models", *subpaths)


def get_baselines_response_path(*subpaths: str) -> str:
    """Get baseline LLM response output path"""
    return get_data_path("baselines_response", *subpaths)
