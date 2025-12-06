"""
CORY论文复现 - 工具函数模块
包含各种辅助功能函数
"""

import os
import random
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


def set_seed(seed: int) -> None:
    """
    设置随机种子，确保实验可复现
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 尝试导入torch并设置种子
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # 确保CUDA卷积运算的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    创建实验输出目录
    
    参数:
        base_dir: 基础目录路径
        experiment_name: 实验名称
    
    返回:
        创建的实验目录路径
    """
    # 生成带时间戳的目录名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    # 创建目录
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    
    return exp_dir


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    保存数据为JSON文件
    
    参数:
        data: 要保存的数据字典
        filepath: 保存路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载数据
    
    参数:
        filepath: 文件路径
    
    返回:
        加载的数据字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """
    将秒数格式化为可读的时间字符串
    
    参数:
        seconds: 秒数
    
    返回:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}小时{minutes}分钟{secs}秒"
    elif minutes > 0:
        return f"{minutes}分钟{secs}秒"
    else:
        return f"{secs}秒"


def compute_moving_average(values: List[float], window_size: int = 10) -> List[float]:
    """
    计算移动平均值
    
    参数:
        values: 数值列表
        window_size: 窗口大小
    
    返回:
        移动平均值列表
    """
    if len(values) < window_size:
        return values
    
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window = values[start_idx:i + 1]
        result.append(sum(window) / len(window))
    
    return result


def print_training_info(config) -> None:
    """
    打印训练配置信息
    
    参数:
        config: 配置对象
    """
    print("=" * 60)
    print("CORY论文复现 - 训练配置信息")
    print("=" * 60)
    print(f"模型名称: {config.model.model_name}")
    print(f"设备: {config.device}")
    print(f"训练轮数: {config.training.num_epochs}")
    print(f"批大小: {config.training.batch_size}")
    print(f"学习率: {config.training.learning_rate}")
    print(f"智能体数量: {config.training.num_agents}")
    print(f"协作权重: {config.training.cooperation_weight}")
    print(f"数据集: {config.data.dataset_name}")
    print(f"训练样本数: {config.data.max_train_samples}")
    print("=" * 60)


class AverageMeter:
    """
    计算并存储平均值和当前值的工具类
    用于跟踪训练过程中的各种指标
    """
    
    def __init__(self, name: str = ""):
        """
        初始化
        
        参数:
            name: 指标名称
        """
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """重置所有统计值"""
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数
        self.history = []  # 历史记录
    
    def update(self, val: float, n: int = 1) -> None:
        """
        更新统计值
        
        参数:
            val: 新的值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)
    
    def __str__(self) -> str:
        return f"{self.name}: 当前={self.val:.4f}, 平均={self.avg:.4f}"


class Logger:
    """
    简单的日志记录器
    用于记录训练过程中的各种信息
    """
    
    def __init__(self, log_dir: str, filename: str = "training.log"):
        """
        初始化日志记录器
        
        参数:
            log_dir: 日志目录
            filename: 日志文件名
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, filename)
        
        # 初始化日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"CORY训练日志 - 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
    
    def log(self, message: str, print_msg: bool = True) -> None:
        """
        记录日志信息
        
        参数:
            message: 日志消息
            print_msg: 是否同时打印到控制台
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")
        
        # 打印到控制台
        if print_msg:
            print(log_line)
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        记录指标
        
        参数:
            metrics: 指标字典
            step: 当前步数
        """
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log(f"Step {step} | {metrics_str}")
