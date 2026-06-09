"""
TrialAgents Training Module

LNL (Learning with Noisy Labels) training components:
- lnl_train: DivideMix training core
- lnl_dataloader: Data loader
- fusion: Multi-modal fusion model
- soft_label_utils: Soft label utilities
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
