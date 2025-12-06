"""
CORY复现项目 - src模块初始化
"""

from .model import CORYAgent, CORYDualAgent
from .reward import RewardModel, ValueFunction, SequentialRewardAggregator
from .trainer import PPOTrainer
from .data import (
    MathReasoningDataset,
    load_gsm8k_data,
    generate_mock_math_data,
    create_dataloader,
    TextPromptFormatter
)
from .evaluation import Evaluator

__all__ = [
    "CORYAgent",
    "CORYDualAgent",
    "RewardModel",
    "ValueFunction",
    "SequentialRewardAggregator",
    "PPOTrainer",
    "MathReasoningDataset",
    "load_gsm8k_data",
    "generate_mock_math_data",
    "create_dataloader",
    "TextPromptFormatter",
    "Evaluator"
]
