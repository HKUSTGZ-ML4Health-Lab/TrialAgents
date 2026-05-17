"""
TrialAgents 数据路径管理工具

统一管理所有数据路径，支持符号链接和环境变量覆盖
"""

import os
from pathlib import Path
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# 默认数据路径配置（相对路径）
DEFAULT_PATHS = {
    "raw": "data/raw",
    "processed": "data/processed/trials",
    "outputs": "data/outputs",
    "responses": "data/outputs/responses",
    "digital_twins": "data/outputs/digital_twins",
    "checkpoints": "data/checkpoints",
    "logs": "logs",
    "training_runs": "logs/training",
    "baselines_runs": "logs/baselines",
    "baselines_saved_models": "baselines/saved_models",
    "baselines_response": "data/outputs/baselines_response",
}


def get_project_root() -> Path:
    """获取项目根目录"""
    return PROJECT_ROOT


def get_data_path(data_type: str, *subpaths: str) -> str:
    """
    获取数据路径
    
    Args:
        data_type: 数据类型（raw, processed, outputs, responses, digital_twins, checkpoints, logs）
        *subpaths: 子路径组件
        
    Returns:
        完整路径字符串
    """
    # 首先检查环境变量
    env_var = f"TRIALAGENTS_{data_type.upper()}_PATH"
    if env_var in os.environ:
        base_path = Path(os.environ[env_var])
    else:
        # 使用默认配置
        relative_path = DEFAULT_PATHS.get(data_type, f"data/{data_type}")
        base_path = PROJECT_ROOT / relative_path
    
    # 添加子路径
    if subpaths:
        base_path = base_path.joinpath(*subpaths)
    
    # 确保目录存在
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    return str(base_path)


def get_raw_data_path(*subpaths: str) -> str:
    """获取原始数据路径"""
    return get_data_path("raw", *subpaths)


def get_processed_data_path(*subpaths: str) -> str:
    """获取处理后数据路径"""
    return get_data_path("processed", *subpaths)


def get_outputs_path(*subpaths: str) -> str:
    """获取输出路径"""
    return get_data_path("outputs", *subpaths)


def get_responses_path(*subpaths: str) -> str:
    """获取响应输出路径"""
    return get_data_path("responses", *subpaths)


def get_digital_twins_path(*subpaths: str) -> str:
    """获取数字孪生路径"""
    return get_data_path("digital_twins", *subpaths)


def get_checkpoints_path(*subpaths: str) -> str:
    """获取检查点路径"""
    return get_data_path("checkpoints", *subpaths)


def get_logs_path(*subpaths: str) -> str:
    """获取日志路径"""
    return get_data_path("logs", *subpaths)


def ensure_dir(path: str) -> str:
    """
    确保目录存在
    
    Args:
        path: 路径字符串
        
    Returns:
        路径字符串
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
    构建实验输出路径
    
    格式: data/outputs/responses/{dataset_name}_{task}_{llm}_{corpus_name}_{group}_{suffix}
    
    Args:
        dataset_name: 数据集名称
        task: 任务名称
        llm: LLM模型名称
        corpus_name: 语料库名称
        group: 组别（control/experimental）
        suffix: 后缀
        
    Returns:
        实验输出路径
    """
    experiment_name = f"{dataset_name}_{task}_{llm}_{corpus_name}_{group}_{suffix}"
    return get_responses_path(experiment_name)


# 与原项目的数据路径兼容函数
def get_m_agents_data_path(relative_path: str) -> str:
    """
    获取原项目m-agents的数据路径
    
    用于适配器从原项目读取数据
    
    Args:
        relative_path: 相对于m-agents的路径
        
    Returns:
        完整路径
    """
    m_agents_root = Path("/home/chunyu.lei/DEAP/SELF_DATA/mas/m-agents")
    return str(m_agents_root / relative_path)


# 便捷函数：获取常用数据文件路径
def get_trial_background_path(dataset_name: str) -> str:
    """获取试验背景文件路径"""
    return get_raw_data_path("ctgov", dataset_name, "bg.json")


def get_patient_data_path(dataset_name: str, patient_id: str) -> str:
    """获取患者数据文件路径"""
    return get_processed_data_path(dataset_name, f"{patient_id}.json")


def get_patient_response_path(experiment_name: str, patient_id: str) -> str:
    """获取患者响应目录路径"""
    return get_responses_path(experiment_name, patient_id)


def get_training_runs_path(*subpaths: str) -> str:
    """获取 LNL 训练日志路径"""
    return get_data_path("training_runs", *subpaths)


def get_baselines_runs_path(*subpaths: str) -> str:
    """获取 baseline 训练日志路径"""
    return get_data_path("baselines_runs", *subpaths)


def get_baselines_saved_models_path(*subpaths: str) -> str:
    """获取 baseline 模型保存路径"""
    return get_data_path("baselines_saved_models", *subpaths)


def get_baselines_response_path(*subpaths: str) -> str:
    """获取 baseline LLM 响应输出路径"""
    return get_data_path("baselines_response", *subpaths)
