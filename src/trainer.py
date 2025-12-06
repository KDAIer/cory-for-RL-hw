"""
CORY论文复现 - PPO训练器模块
实现基于PPO的双智能体协作训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np

# 导入本地模块
import sys
sys.path.append("..")
from src.model import CORYDualAgent
from src.reward import RewardModel, SequentialRewardAggregator


class PPOTrainer:
    """
    PPO训练器
    实现CORY的序贯协作PPO训练
    
    核心特性:
    - Pioneer-Observer双智能体协作
    - PPO策略优化
    - 角色交换（Role Swap）机制
    """
    
    def __init__(
        self,
        dual_agent: CORYDualAgent,
        reward_model: RewardModel,
        config: dict,
        device: str = "cuda"
    ):
        """
        初始化PPO训练器
        
        参数:
            dual_agent: CORY双智能体系统
            reward_model: 奖励模型
            config: 训练配置
            device: 计算设备
        """
        self.dual_agent = dual_agent
        self.reward_model = reward_model
        self.config = config
        self.device = device
        
        # PPO超参数
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        
        # 角色交换（Role Swap）配置 - CORY核心机制
        self.role_swap_enabled = config.get("role_swap_enabled", True)
        self.role_swap_frequency = config.get("role_swap_frequency", 100)  # 每N步交换一次
        self.role_swap_type = config.get("role_swap_type", "soft")  # "hard", "soft", "lora"
        self.role_swap_factor = config.get("role_swap_factor", 0.3)  # 软交换的插值因子
        
        # 训练步数计数器
        self.global_step = 0
        self.swap_count = 0
        
        # 初始化优化器
        trainable_params = self.dual_agent.get_trainable_parameters()
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config.get("learning_rate", 1e-5),
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        # 奖励聚合器
        self.reward_aggregator = SequentialRewardAggregator(
            gamma=self.gamma,
            pioneer_share=0.5
        )
        
        # 训练统计
        self.train_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_reward": [],
            "pioneer_reward": [],
            "observer_reward": [],
            "cooperation_bonus": []
        }
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计（GAE）
        
        参数:
            rewards: 奖励张量
            values: 价值估计张量
            dones: 结束标志张量
        
        返回:
            优势值和回报值
        """
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # 对每个样本计算GAE
        for b in range(batch_size):
            last_gae = 0
            for t in reversed(range(rewards.shape[1])):
                if t == rewards.shape[1] - 1:
                    next_value = 0
                else:
                    next_value = values[b, t + 1]
                
                delta = rewards[b, t] + self.gamma * next_value * (1 - dones[b, t]) - values[b, t]
                last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[b, t]) * last_gae
                
                advantages[b, t] = last_gae
                returns[b, t] = last_gae + values[b, t]
        
        # 标准化优势值
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        计算PPO策略损失
        
        参数:
            log_probs: 当前策略的对数概率
            old_log_probs: 旧策略的对数概率
            advantages: 优势值
        
        返回:
            策略损失
        """
        # 计算概率比
        ratio = torch.exp(log_probs - old_log_probs)
        
        # PPO裁剪
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.clip_epsilon,
            1 + self.clip_epsilon
        )
        
        # 取两者最小值作为损失
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        return policy_loss
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        计算价值函数损失
        
        参数:
            values: 价值预测
            returns: 实际回报
        
        返回:
            价值损失
        """
        return nn.functional.mse_loss(values, returns)
    
    def compute_entropy(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算策略熵（用于探索）
        
        参数:
            logits: 策略logits
        
        返回:
            熵值
        """
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy
    
    def train_step(
        self,
        prompts: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        参数:
            prompts: 输入提示列表
            ground_truths: 标准答案列表
        
        返回:
            训练统计信息字典
        """
        # 更新全局步数
        self.global_step += 1
        
        # 检查是否需要进行角色交换
        role_swapped = self._maybe_swap_roles()
        
        # 第一阶段：Pioneer生成
        with torch.no_grad():
            pioneer_responses, pioneer_ids = self.dual_agent.pioneer_generate(
                prompts,
                max_new_tokens=128
            )
        
        # 第二阶段：Observer改进
        with torch.no_grad():
            observer_responses, observer_ids = self.dual_agent.observer_refine(
                prompts,
                pioneer_responses,
                max_new_tokens=128
            )
        
        # 计算各类奖励
        # 1. Pioneer任务奖励
        pioneer_task_rewards = self.reward_model.compute_task_reward(
            pioneer_responses, ground_truths, task_type="math"
        )
        
        # 2. Observer任务奖励
        observer_task_rewards = self.reward_model.compute_task_reward(
            observer_responses, ground_truths, task_type="math"
        )
        
        # 3. 协作奖励
        cooperation_rewards = self.reward_model.compute_batch_cooperation_rewards(
            pioneer_responses, observer_responses, ground_truths
        )
        
        # 计算总奖励
        total_rewards = (
            pioneer_task_rewards * 0.3 +
            observer_task_rewards * 0.5 +
            cooperation_rewards * 0.2
        )
        
        # PPO更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.ppo_epochs):
            # 编码Pioneer输入
            pioneer_inputs = self.dual_agent.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # 获取Pioneer的输出
            pioneer_outputs = self.dual_agent.pioneer(
                input_ids=pioneer_inputs.input_ids,
                attention_mask=pioneer_inputs.attention_mask
            )
            
            # 计算Pioneer的对数概率
            pioneer_logits = pioneer_outputs["logits"]
            pioneer_log_probs = torch.nn.functional.log_softmax(pioneer_logits, dim=-1)
            
            # 简化：使用平均奖励作为优势估计
            advantages = total_rewards.unsqueeze(1).expand(-1, pioneer_logits.shape[1])
            
            # 计算损失
            # 使用简化的策略梯度
            selected_log_probs = pioneer_log_probs.mean(dim=-1).mean(dim=-1)
            policy_loss = -(selected_log_probs * total_rewards).mean()
            
            # 熵正则化
            entropy = self.compute_entropy(pioneer_logits)
            
            # 总损失
            loss = policy_loss - self.entropy_coef * entropy
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.dual_agent.get_trainable_parameters(),
                self.max_grad_norm
            )
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()
        
        # 记录统计信息
        avg_policy_loss = total_policy_loss / self.ppo_epochs
        avg_entropy = total_entropy / self.ppo_epochs
        
        stats = {
            "policy_loss": avg_policy_loss,
            "entropy": avg_entropy,
            "total_reward": total_rewards.mean().item(),
            "pioneer_reward": pioneer_task_rewards.mean().item(),
            "observer_reward": observer_task_rewards.mean().item(),
            "cooperation_bonus": cooperation_rewards.mean().item(),
            "role_swapped": role_swapped,
            "swap_count": self.swap_count,
            "global_step": self.global_step
        }
        
        # 更新历史统计
        for key, value in stats.items():
            if key in self.train_stats:
                self.train_stats[key].append(value)
        
        return stats
    
    def _maybe_swap_roles(self) -> bool:
        """
        检查并执行角色交换
        
        CORY论文核心机制：周期性地交换Pioneer和Observer的参数，
        使两个智能体互换角色。这样可以：
        1. 避免角色固化，促进双向学习
        2. Pioneer学习Observer的改进能力
        3. Observer学习Pioneer的探索能力
        4. 增强整体协作效果
        
        返回:
            是否执行了角色交换
        """
        if not self.role_swap_enabled:
            return False
        
        # 检查是否到达交换频率
        if self.global_step % self.role_swap_frequency != 0:
            return False
        
        # 执行角色交换
        print(f"\n[Step {self.global_step}] 触发角色交换 (第 {self.swap_count + 1} 次)")
        
        # 计算交换前的相似度
        pre_similarity = self.dual_agent.get_role_similarity()
        print(f"  交换前智能体相似度: {pre_similarity:.4f}")
        
        # 根据配置选择交换类型
        if self.role_swap_type == "hard":
            # 硬交换：完全交换所有参数
            self.dual_agent.swap_roles()
        elif self.role_swap_type == "lora":
            # LoRA交换：只交换LoRA参数（更轻量）
            self.dual_agent.swap_lora_parameters()
        elif self.role_swap_type == "soft":
            # 软交换：参数插值混合
            self.dual_agent.soft_role_swap(self.role_swap_factor)
        else:
            print(f"  未知的交换类型: {self.role_swap_type}")
            return False
        
        # 计算交换后的相似度
        post_similarity = self.dual_agent.get_role_similarity()
        print(f"  交换后智能体相似度: {post_similarity:.4f}")
        
        self.swap_count += 1
        return True
    
    def set_role_swap_config(
        self,
        enabled: bool = None,
        frequency: int = None,
        swap_type: str = None,
        factor: float = None
    ) -> None:
        """
        动态设置角色交换配置
        
        参数:
            enabled: 是否启用角色交换
            frequency: 交换频率（每N步）
            swap_type: 交换类型 ("hard", "soft", "lora")
            factor: 软交换的插值因子
        """
        if enabled is not None:
            self.role_swap_enabled = enabled
            print(f"角色交换已{'启用' if enabled else '禁用'}")
        
        if frequency is not None:
            self.role_swap_frequency = frequency
            print(f"角色交换频率设置为: 每 {frequency} 步")
        
        if swap_type is not None:
            self.role_swap_type = swap_type
            print(f"角色交换类型设置为: {swap_type}")
        
        if factor is not None:
            self.role_swap_factor = factor
            print(f"软交换因子设置为: {factor}")
    
    def train_epoch(
        self,
        dataloader,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        参数:
            dataloader: 数据加载器
            epoch: 当前epoch编号
        
        返回:
            epoch统计信息
        """
        self.dual_agent.train()
        
        epoch_stats = {
            "policy_loss": 0,
            "entropy": 0,
            "total_reward": 0,
            "pioneer_reward": 0,
            "observer_reward": 0,
            "cooperation_bonus": 0
        }
        
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            prompts = batch["question"]
            ground_truths = batch["answer"]
            
            # 执行训练步骤
            stats = self.train_step(prompts, ground_truths)
            
            # 累积统计
            for key in epoch_stats:
                if key in stats:
                    epoch_stats[key] += stats[key]
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{stats['policy_loss']:.4f}",
                "reward": f"{stats['total_reward']:.4f}"
            })
        
        # 计算平均值
        for key in epoch_stats:
            epoch_stats[key] /= max(num_batches, 1)
        
        return epoch_stats
    
    def save_checkpoint(self, path: str) -> None:
        """
        保存检查点
        
        参数:
            path: 保存路径
        """
        checkpoint = {
            "pioneer_state_dict": self.dual_agent.pioneer.state_dict(),
            "observer_state_dict": self.dual_agent.observer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_stats": self.train_stats
        }
        torch.save(checkpoint, path)
        print(f"检查点已保存至: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        加载检查点
        
        参数:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.dual_agent.pioneer.load_state_dict(checkpoint["pioneer_state_dict"])
        self.dual_agent.observer.load_state_dict(checkpoint["observer_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_stats = checkpoint["train_stats"]
        print(f"检查点已从 {path} 加载")
