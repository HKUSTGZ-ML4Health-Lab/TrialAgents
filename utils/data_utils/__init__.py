"""
TrialAgents 数据处理模块

包含数据路径管理、运行工具和数据包装
"""

from .paths import (
    build_experiment_path,
    ensure_dir,
    get_data_path,
    get_raw_data_path,
)
from .runner_utils import process_time_series
from .wrap_data import wrap_data, process_time_series as serialize_time_series

__all__ = [
    'build_experiment_path',
    'ensure_dir',
    'get_data_path',
    'get_raw_data_path',
    'process_time_series',
    'wrap_data',
    'serialize_time_series',
]
