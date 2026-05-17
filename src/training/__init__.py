"""
TrialAgents 训练模块

包含 LNL (Learning with Noisy Labels) 训练相关组件：
- lnl_train: DivideMix 训练核心
- lnl_dataloader: 数据加载器
- fusion: 多模态融合模型
- soft_label_utils: 软标签工具函数
"""

from .lnl_train import run_dividemix, create_parser
from .lnl_dataloader import nct_dataloader, nct_dataset, process_data
from .fusion import Fusion, Merger, Predictor, TransformerEncoder, TransformerEncoderLayer
from .soft_label_utils import (
    build_soft_labels,
    hard_to_onehot,
    confidence_to_soft_labels,
    label_smoothing,
    temperature_scaling,
    continuous_to_soft_labels,
    mix_labels,
    validate_soft_labels,
)

__all__ = [
    'run_dividemix',
    'create_parser',
    'nct_dataloader',
    'nct_dataset',
    'process_data',
    'Fusion',
    'Merger',
    'Predictor',
    'TransformerEncoder',
    'TransformerEncoderLayer',
    'build_soft_labels',
    'hard_to_onehot',
    'confidence_to_soft_labels',
    'label_smoothing',
    'temperature_scaling',
    'continuous_to_soft_labels',
    'mix_labels',
    'validate_soft_labels',
]
