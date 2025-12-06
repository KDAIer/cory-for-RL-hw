"""
CORY论文复现 - 奖励模型模块
实现多维度奖励函数，包括任务奖励、协作奖励和KL惩罚
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import re


class RewardModel(nn.Module):
    """
    CORY奖励模型
    整合多种奖励信号以指导双智能体的协作学习
    """
    
    def __init__(
        self,
        cooperation_weight: float = 0.3,
        kl_penalty: float = 0.1,
        task_weight: float = 0.6,
        device: str = "cuda"
    ):
        """
        初始化奖励模型
        
        参数:
            cooperation_weight: 协作奖励权重
            kl_penalty: KL散度惩罚系数
            task_weight: 任务奖励权重
            device: 计算设备
        """
        super().__init__()
        
        self.cooperation_weight = cooperation_weight
        self.kl_penalty = kl_penalty
        self.task_weight = task_weight
        self.device = device
    
    def compute_task_reward(
        self,
        responses: List[str],
        ground_truths: List[str],
        task_type: str = "math"
    ) -> torch.Tensor:
        """
        计算任务奖励
        根据响应与标准答案的匹配程度给予奖励
        
        参数:
            responses: 模型生成的响应列表
            ground_truths: 标准答案列表
            task_type: 任务类型（"math"数学推理，"qa"问答等）
        
        返回:
            任务奖励张量
        """
        rewards = []
        
        for response, truth in zip(responses, ground_truths):
            if task_type == "math":
                reward = self._math_reward(response, truth)
            else:
                reward = self._text_similarity_reward(response, truth)
            rewards.append(reward)
        
        return torch.tensor(rewards, device=self.device)
    
    def _math_reward(self, response: str, truth: str) -> float:
        """
        数学任务奖励计算
        检查响应中是否包含正确答案
        
        参数:
            response: 模型响应
            truth: 正确答案
        
        返回:
            奖励值（0.0-1.0）
        """
        # 从响应中提取数字答案
        response_numbers = re.findall(r'-?\d+\.?\d*', response)
        truth_numbers = re.findall(r'-?\d+\.?\d*', truth)
        
        if not response_numbers or not truth_numbers:
            return 0.0
        
        # 检查最后一个数字是否匹配（通常是最终答案）
        try:
            response_answer = float(response_numbers[-1])
            truth_answer = float(truth_numbers[-1])
            
            # 完全匹配给满分
            if abs(response_answer - truth_answer) < 1e-6:
                return 1.0
            
            # 部分匹配给部分分数
            relative_error = abs(response_answer - truth_answer) / (abs(truth_answer) + 1e-8)
            if relative_error < 0.1:
                return 0.5
            
            return 0.0
            
        except (ValueError, IndexError):
            return 0.0
    
    def _text_similarity_reward(self, response: str, truth: str) -> float:
        """
        文本相似度奖励计算
        使用简单的词重叠度量
        
        参数:
            response: 模型响应
            truth: 标准答案
        
        返回:
            奖励值（0.0-1.0）
        """
        # 简单的词重叠计算
        response_words = set(response.lower().split())
        truth_words = set(truth.lower().split())
        
        if not truth_words:
            return 0.0
        
        overlap = len(response_words & truth_words)
        precision = overlap / len(response_words) if response_words else 0
        recall = overlap / len(truth_words) if truth_words else 0
        
        # F1分数
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            return f1
        
        return 0.0
    
    def compute_cooperation_reward(
        self,
        pioneer_response: str,
        observer_response: str,
        ground_truth: str
    ) -> float:
        """
        计算协作奖励
        衡量Observer是否成功改进了Pioneer的响应
        
        参数:
            pioneer_response: Pioneer的响应
            observer_response: Observer的改进响应
            ground_truth: 标准答案
        
        返回:
            协作奖励值
        """
        # 计算Pioneer的任务分数
        pioneer_score = self._math_reward(pioneer_response, ground_truth)
        
        # 计算Observer的任务分数
        observer_score = self._math_reward(observer_response, ground_truth)
        
        # 如果Observer改进了结果，给予正向协作奖励
        improvement = observer_score - pioneer_score
        
        if improvement > 0:
            # 改进幅度越大，奖励越高
            cooperation_reward = improvement * 2.0
        elif improvement == 0 and observer_score > 0:
            # 保持正确答案也有小奖励
            cooperation_reward = 0.1
        else:
            # 退步则给予惩罚
            cooperation_reward = improvement * 0.5
        
        return cooperation_reward
    
    def compute_batch_cooperation_rewards(
        self,
        pioneer_responses: List[str],
        observer_responses: List[str],
        ground_truths: List[str]
    ) -> torch.Tensor:
        """
        批量计算协作奖励
        
        参数:
            pioneer_responses: Pioneer响应列表
            observer_responses: Observer响应列表
            ground_truths: 标准答案列表
        
        返回:
            协作奖励张量
        """
        rewards = []
        
        for pioneer_resp, observer_resp, truth in zip(
            pioneer_responses, observer_responses, ground_truths
        ):
            reward = self.compute_cooperation_reward(
                pioneer_resp, observer_resp, truth
            )
            rewards.append(reward)
        
        return torch.tensor(rewards, device=self.device)
    
    def compute_kl_penalty_reward(
        self,
        kl_divergence: torch.Tensor
    ) -> torch.Tensor:
        """
        计算KL散度惩罚
        防止策略偏离参考模型太远
        
        参数:
            kl_divergence: KL散度值
        
        返回:
            KL惩罚（负奖励）
        """
        return -self.kl_penalty * kl_divergence
    
    def compute_total_reward(
        self,
        task_reward: torch.Tensor,
        cooperation_reward: torch.Tensor,
        kl_divergence: torch.Tensor
    ) -> torch.Tensor:
        """
        计算总奖励
        综合任务奖励、协作奖励和KL惩罚
        
        参数:
            task_reward: 任务奖励
            cooperation_reward: 协作奖励
            kl_divergence: KL散度
        
        返回:
            总奖励
        """
        total = (
            self.task_weight * task_reward +
            self.cooperation_weight * cooperation_reward +
            self.compute_kl_penalty_reward(kl_divergence)
        )
        
        return total


class ValueFunction(nn.Module):
    """
    价值函数网络
    用于估计状态价值，辅助PPO训练
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化价值函数网络
        
        参数:
            hidden_size: 隐藏层大小
            num_layers: 层数
            dropout: Dropout率
        """
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(hidden_size, hidden_size // 2))
            else:
                layers.append(nn.Linear(hidden_size // 2, hidden_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_size // 2, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            hidden_states: 隐藏状态张量
        
        返回:
            价值估计
        """
        # 取序列的最后一个隐藏状态
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        
        return self.network(hidden_states).squeeze(-1)


class SequentialRewardAggregator:
    """
    序贯奖励聚合器
    用于处理多轮协作中的奖励分配
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        pioneer_share: float = 0.5
    ):
        """
        初始化奖励聚合器
        
        参数:
            gamma: 折扣因子
            pioneer_share: Pioneer奖励分配比例
        """
        self.gamma = gamma
        self.pioneer_share = pioneer_share
    
    def aggregate_rewards(
        self,
        pioneer_rewards: List[float],
        observer_rewards: List[float],
        cooperation_bonus: float
    ) -> Tuple[List[float], List[float]]:
        """
        聚合序贯协作的奖励
        
        参数:
            pioneer_rewards: Pioneer各步骤的奖励
            observer_rewards: Observer各步骤的奖励
            cooperation_bonus: 协作加成
        
        返回:
            分配给Pioneer和Observer的最终奖励
        """
        # 计算总奖励
        total_pioneer = sum([r * (self.gamma ** i) for i, r in enumerate(pioneer_rewards)])
        total_observer = sum([r * (self.gamma ** i) for i, r in enumerate(observer_rewards)])
        
        # 加入协作加成
        total_reward = total_pioneer + total_observer + cooperation_bonus
        
        # 按比例分配
        final_pioneer_reward = total_reward * self.pioneer_share
        final_observer_reward = total_reward * (1 - self.pioneer_share)
        
        return final_pioneer_reward, final_observer_reward
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[List[float], List[float]]:
        """
        计算广义优势估计（GAE）
        
        参数:
            rewards: 奖励序列
            values: 价值估计序列
            next_value: 下一状态价值
            gamma: 折扣因子
            gae_lambda: GAE参数
        
        返回:
            优势值列表和回报列表
        """
        advantages = []
        returns = []
        gae = 0
        
        # 反向计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
