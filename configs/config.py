"""
CORY论文复现 - 配置文件
定义训练和模型的各种超参数
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """模型配置类"""
    # 基础模型名称 - 使用7B模型以获得数学推理能力
    # 推荐选项：
    # - "meta-llama/Llama-2-7b-hf" (需HuggingFace token)
    # - "mistralai/Mistral-7B-v0.1" (开源，推荐)
    # - "gpt2" (仅用于快速测试)
    model_name: str = "mistralai/Mistral-7B-v0.1"  
    
    # 模型参数
    max_length: int = 2048  # 7B模型支持更长序列
    use_lora: bool = True  # 使用LoRA节省显存（7B模型必须启用）
    
    # LoRA参数（针对7B模型优化）
    lora_r: int = 16  # 增大秩以提升表达能力
    lora_alpha: int = 32  # LoRA缩放因子
    lora_dropout: float = 0.1  # LoRA dropout率
    
    # 量化配置（7B模型建议启用以节省显存）
    load_in_8bit: bool = True  # 使用8位量化（推荐）
    load_in_4bit: bool = False  # 4位量化（显存极度受限时使用）


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 基本训练参数（针对7B模型调整）
    num_epochs: int = 5  # 7B模型收敛更快
    batch_size: int = 1  # 7B模型显存占用大，使用小批量
    gradient_accumulation_steps: int = 16  # 增大累积步数以模拟更大批量
    learning_rate: float = 2e-5  # 7B模型使用稍大学习率
    warmup_ratio: float = 0.1  # 预热比例
    weight_decay: float = 0.01  # 权重衰减
    
    # PPO相关参数（CORY核心）
    ppo_epochs: int = 4  # PPO更新轮数
    clip_epsilon: float = 0.2  # PPO裁剪参数
    value_loss_coef: float = 0.5  # 价值损失系数
    entropy_coef: float = 0.01  # 熵正则化系数
    max_grad_norm: float = 1.0  # 梯度裁剪阈值
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    
    # CORY特有参数
    num_agents: int = 2  # 智能体数量（双智能体协作）
    cooperation_weight: float = 0.3  # 协作奖励权重
    kl_penalty: float = 0.1  # KL散度惩罚系数
    
    # ===== 角色交换配置（Role Swap）- CORY核心机制 =====
    # 两个智能体周期性交换参数，互换Pioneer和Observer角色
    # 这是CORY论文的关键创新点之一
    role_swap_enabled: bool = True  # 是否启用角色交换
    role_swap_frequency: int = 100  # 每N步交换一次角色
    role_swap_type: str = "soft"  # 交换类型: "hard"(完全交换), "soft"(插值混合), "lora"(仅LoRA)
    role_swap_factor: float = 0.3  # 软交换的插值因子 (0-1, 0.5为完全平均)
    
    # 输出设置
    output_dir: str = "./outputs"
    save_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    
    # 随机种子
    seed: int = 42


@dataclass
class DataConfig:
    """数据配置类"""
    # 数据集设置（使用简单数据集便于演示）
    dataset_name: str = "gsm8k"  # 数学推理数据集
    train_split: str = "train"
    eval_split: str = "test"
    max_train_samples: int = 1000  # 限制训练样本数（便于快速运行）
    max_eval_samples: int = 200  # 限制评估样本数
    
    # 数据处理
    num_workers: int = 4
    max_prompt_length: int = 256
    max_response_length: int = 256


@dataclass
class CORYConfig:
    """CORY算法总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # 设备配置
    device: str = "cuda"  # 或 "cpu"
    
    # 实验名称
    experiment_name: str = "cory_reproduction"
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建输出目录
        os.makedirs(self.training.output_dir, exist_ok=True)
        
        # 如果没有GPU，强制使用CPU
        import torch
        if not torch.cuda.is_available():
            self.device = "cpu"
            print("警告: 未检测到GPU，将使用CPU进行训练")


def get_default_config() -> CORYConfig:
    """获取默认配置"""
    return CORYConfig()


def get_lightweight_config() -> CORYConfig:
    """
    获取轻量级配置（用于快速测试和低显存环境）
    适合普通电脑运行
    """
    config = CORYConfig()
    
    # 使用更小的模型
    config.model.model_name = "gpt2"
    config.model.max_length = 256
    
    # 减少训练量
    config.training.num_epochs = 2
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 2
    
    # 减少数据量
    config.data.max_train_samples = 500
    config.data.max_eval_samples = 100
    
    return config


# 预定义配置字典
CONFIGS = {
    "default": get_default_config,
    "lightweight": get_lightweight_config,
}
