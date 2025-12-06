"""
CORY论文复现 - 创新拓展模块
===============================
在原论文基础上提出以下创新改进:

1. 三智能体架构 (TriAgent): 引入Critic智能体进行质量评估
2. 动态协作权重: 根据训练进度自适应调整协作强度
3. 课程学习策略: 从简单到复杂的问题安排
4. 多轮迭代改进: 支持多轮Pioneer-Observer交替
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import random
import math


class CriticAgent(nn.Module):
    """
    创新1: Critic智能体
    在Pioneer和Observer基础上，增加一个负责质量评估的Critic智能体。
    
    Critic的作用:
    - 评估Pioneer和Observer响应的质量
    - 提供更精确的奖励信号
    - 帮助决定是否接受Observer的改进
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_quality_levels: int = 5
    ):
        """
        初始化Critic智能体
        
        参数:
            hidden_size: 隐藏层大小
            num_quality_levels: 质量等级数量
        """
        super().__init__()
        
        # 质量评估网络
        self.quality_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_quality_levels)
        )
        
        # 比较网络：判断Observer是否比Pioneer更好
        self.comparison_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def evaluate_quality(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        评估响应质量
        
        参数:
            hidden_states: 响应的隐藏状态表示
        
        返回:
            质量分数（0-4的等级）
        """
        # 取序列的平均表示
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.mean(dim=1)
        
        quality_logits = self.quality_net(hidden_states)
        quality_probs = torch.softmax(quality_logits, dim=-1)
        
        # 计算期望质量分数
        levels = torch.arange(quality_probs.shape[-1], device=quality_probs.device).float()
        expected_quality = (quality_probs * levels).sum(dim=-1)
        
        return expected_quality
    
    def compare_responses(
        self,
        pioneer_hidden: torch.Tensor,
        observer_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        比较Pioneer和Observer的响应质量
        
        参数:
            pioneer_hidden: Pioneer响应的隐藏状态
            observer_hidden: Observer响应的隐藏状态
        
        返回:
            Observer更好的概率
        """
        # 取平均表示
        if pioneer_hidden.dim() == 3:
            pioneer_hidden = pioneer_hidden.mean(dim=1)
        if observer_hidden.dim() == 3:
            observer_hidden = observer_hidden.mean(dim=1)
        
        # 拼接并比较
        combined = torch.cat([pioneer_hidden, observer_hidden], dim=-1)
        comparison_prob = self.comparison_net(combined)
        
        return comparison_prob


class DynamicCooperationScheduler:
    """
    创新2: 动态协作权重调度器
    根据训练进度自适应调整协作强度，实现从探索到利用的平滑过渡。
    
    设计理念:
    - 训练初期：高探索，低协作约束
    - 训练中期：平衡探索和利用
    - 训练后期：高利用，强协作约束
    """
    
    def __init__(
        self,
        initial_coop_weight: float = 0.1,
        final_coop_weight: float = 0.5,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        schedule_type: str = "cosine"
    ):
        """
        初始化调度器
        
        参数:
            initial_coop_weight: 初始协作权重
            final_coop_weight: 最终协作权重
            warmup_steps: 预热步数
            total_steps: 总训练步数
            schedule_type: 调度类型（"linear", "cosine", "exponential"）
        """
        self.initial_weight = initial_coop_weight
        self.final_weight = final_coop_weight
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.current_step = 0
    
    def get_cooperation_weight(self) -> float:
        """
        获取当前协作权重
        
        返回:
            当前步骤的协作权重
        """
        if self.current_step < self.warmup_steps:
            # 预热阶段：线性增加
            progress = self.current_step / self.warmup_steps
            return self.initial_weight + progress * (self.final_weight - self.initial_weight) * 0.3
        
        # 主训练阶段
        main_progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        main_progress = min(1.0, main_progress)
        
        if self.schedule_type == "linear":
            weight = self.initial_weight + main_progress * (self.final_weight - self.initial_weight)
        
        elif self.schedule_type == "cosine":
            # 余弦退火式增长
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * \
                     (1 - math.cos(math.pi * main_progress)) / 2
        
        elif self.schedule_type == "exponential":
            # 指数增长
            weight = self.initial_weight * (self.final_weight / self.initial_weight) ** main_progress
        
        else:
            weight = self.final_weight
        
        return weight
    
    def step(self) -> float:
        """
        前进一步并返回当前权重
        
        返回:
            当前协作权重
        """
        self.current_step += 1
        return self.get_cooperation_weight()
    
    def get_kl_penalty(self) -> float:
        """
        获取动态KL惩罚系数
        训练后期增加KL惩罚以稳定训练
        
        返回:
            当前KL惩罚系数
        """
        progress = self.current_step / self.total_steps
        # 从0.05增加到0.2
        return 0.05 + 0.15 * progress


class CurriculumLearningManager:
    """
    创新3: 课程学习管理器
    实现从简单到复杂的问题安排，提高训练效率。
    
    难度评估标准:
    - 问题长度
    - 计算步骤数
    - 数字大小
    - 问题类型复杂度
    """
    
    def __init__(
        self,
        difficulty_levels: int = 5,
        initial_level: int = 1,
        promotion_threshold: float = 0.7
    ):
        """
        初始化课程学习管理器
        
        参数:
            difficulty_levels: 难度等级数
            initial_level: 初始难度等级
            promotion_threshold: 升级阈值（准确率达到此值后升级）
        """
        self.difficulty_levels = difficulty_levels
        self.current_level = initial_level
        self.promotion_threshold = promotion_threshold
        
        # 各难度等级的准确率历史
        self.level_accuracy_history = {i: [] for i in range(1, difficulty_levels + 1)}
    
    def assess_difficulty(self, question: str, answer: str) -> int:
        """
        评估问题难度
        
        参数:
            question: 问题文本
            answer: 答案文本
        
        返回:
            难度等级（1-5）
        """
        difficulty = 1
        
        # 根据问题长度增加难度
        if len(question) > 100:
            difficulty += 1
        if len(question) > 200:
            difficulty += 1
        
        # 根据答案中的数字大小增加难度
        import re
        numbers = re.findall(r'\d+', answer)
        if numbers:
            max_num = max(int(n) for n in numbers)
            if max_num > 100:
                difficulty += 1
            if max_num > 1000:
                difficulty += 1
        
        return min(difficulty, self.difficulty_levels)
    
    def filter_by_difficulty(
        self,
        data: List[Dict],
        target_level: Optional[int] = None
    ) -> List[Dict]:
        """
        根据难度筛选数据
        
        参数:
            data: 原始数据列表
            target_level: 目标难度等级（如果为None则使用当前等级）
        
        返回:
            筛选后的数据
        """
        if target_level is None:
            target_level = self.current_level
        
        filtered = []
        for item in data:
            item_difficulty = self.assess_difficulty(item["question"], item["answer"])
            if item_difficulty <= target_level:
                filtered.append(item)
        
        return filtered
    
    def update_performance(self, accuracy: float) -> bool:
        """
        更新当前难度等级的表现，并决定是否升级
        
        参数:
            accuracy: 当前准确率
        
        返回:
            是否升级到下一等级
        """
        self.level_accuracy_history[self.current_level].append(accuracy)
        
        # 检查是否应该升级
        if len(self.level_accuracy_history[self.current_level]) >= 3:
            recent_accs = self.level_accuracy_history[self.current_level][-3:]
            avg_acc = sum(recent_accs) / len(recent_accs)
            
            if avg_acc >= self.promotion_threshold and self.current_level < self.difficulty_levels:
                self.current_level += 1
                print(f"课程学习: 升级到难度等级 {self.current_level}!")
                return True
        
        return False


class MultiRoundRefiner:
    """
    创新4: 多轮迭代改进器
    支持多轮Pioneer-Observer交替，进一步提升响应质量。
    
    工作流程:
    1. Pioneer生成初始响应
    2. Observer进行第一轮改进
    3. Pioneer观察Observer的改进并进行第二轮优化
    4. ...循环直到收敛或达到最大轮数
    """
    
    def __init__(
        self,
        max_rounds: int = 3,
        convergence_threshold: float = 0.95,
        diminishing_returns_factor: float = 0.8
    ):
        """
        初始化多轮改进器
        
        参数:
            max_rounds: 最大改进轮数
            convergence_threshold: 收敛阈值（质量分数比值）
            diminishing_returns_factor: 递减奖励因子
        """
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.diminishing_returns_factor = diminishing_returns_factor
    
    def should_continue(
        self,
        current_quality: float,
        previous_quality: float,
        current_round: int
    ) -> bool:
        """
        判断是否应该继续改进
        
        参数:
            current_quality: 当前轮次质量分数
            previous_quality: 上一轮质量分数
            current_round: 当前轮次
        
        返回:
            是否继续
        """
        if current_round >= self.max_rounds:
            return False
        
        if previous_quality > 0:
            improvement_ratio = current_quality / previous_quality
            # 如果改进不明显，停止
            if improvement_ratio < self.convergence_threshold:
                return False
        
        return True
    
    def compute_round_weight(self, round_num: int) -> float:
        """
        计算各轮次的奖励权重（递减）
        
        参数:
            round_num: 轮次编号（从0开始）
        
        返回:
            该轮次的权重
        """
        return self.diminishing_returns_factor ** round_num


class EnhancedRewardModel:
    """
    增强版奖励模型
    整合以上所有创新改进的奖励计算
    """
    
    def __init__(
        self,
        base_task_weight: float = 0.5,
        base_coop_weight: float = 0.3,
        base_kl_penalty: float = 0.1,
        diversity_bonus: float = 0.1
    ):
        """
        初始化增强奖励模型
        
        参数:
            base_task_weight: 基础任务奖励权重
            base_coop_weight: 基础协作奖励权重
            base_kl_penalty: 基础KL惩罚
            diversity_bonus: 多样性奖励
        """
        self.base_task_weight = base_task_weight
        self.base_coop_weight = base_coop_weight
        self.base_kl_penalty = base_kl_penalty
        self.diversity_bonus = diversity_bonus
        
        # 动态调度器
        self.scheduler = DynamicCooperationScheduler()
    
    def compute_diversity_reward(
        self,
        responses: List[str]
    ) -> float:
        """
        计算多样性奖励
        鼓励生成多样化的响应
        
        参数:
            responses: 响应列表
        
        返回:
            多样性分数
        """
        if len(responses) < 2:
            return 0.0
        
        # 简单的词汇多样性计算
        all_words = []
        for resp in responses:
            words = set(resp.lower().split())
            all_words.append(words)
        
        # 计算不同响应之间的差异
        diversity = 0.0
        count = 0
        for i in range(len(all_words)):
            for j in range(i + 1, len(all_words)):
                if all_words[i] and all_words[j]:
                    # Jaccard距离
                    intersection = len(all_words[i] & all_words[j])
                    union = len(all_words[i] | all_words[j])
                    jaccard_dist = 1 - (intersection / union if union > 0 else 0)
                    diversity += jaccard_dist
                    count += 1
        
        return diversity / count if count > 0 else 0.0
    
    def compute_enhanced_reward(
        self,
        task_reward: float,
        cooperation_reward: float,
        kl_divergence: float,
        responses: List[str],
        round_num: int = 0
    ) -> float:
        """
        计算增强版总奖励
        
        参数:
            task_reward: 任务奖励
            cooperation_reward: 协作奖励
            kl_divergence: KL散度
            responses: 所有响应（用于计算多样性）
            round_num: 当前轮次
        
        返回:
            总奖励
        """
        # 获取动态权重
        coop_weight = self.scheduler.get_cooperation_weight()
        kl_weight = self.scheduler.get_kl_penalty()
        
        # 计算多样性奖励
        diversity_reward = self.compute_diversity_reward(responses)
        
        # 轮次衰减
        round_weight = MultiRoundRefiner().compute_round_weight(round_num)
        
        # 总奖励
        total = (
            self.base_task_weight * task_reward * round_weight +
            coop_weight * cooperation_reward +
            self.diversity_bonus * diversity_reward -
            kl_weight * kl_divergence
        )
        
        # 更新调度器
        self.scheduler.step()
        
        return total


class TriAgentCORY:
    """
    三智能体CORY系统
    整合所有创新改进的完整系统
    """
    
    def __init__(
        self,
        pioneer,
        observer,
        critic: CriticAgent,
        enhanced_reward: EnhancedRewardModel,
        curriculum: CurriculumLearningManager,
        multi_round: MultiRoundRefiner
    ):
        """
        初始化三智能体系统
        
        参数:
            pioneer: Pioneer智能体
            observer: Observer智能体
            critic: Critic智能体
            enhanced_reward: 增强奖励模型
            curriculum: 课程学习管理器
            multi_round: 多轮改进器
        """
        self.pioneer = pioneer
        self.observer = observer
        self.critic = critic
        self.enhanced_reward = enhanced_reward
        self.curriculum = curriculum
        self.multi_round = multi_round
    
    def generate_with_multi_round(
        self,
        prompt: str,
        max_rounds: int = 3
    ) -> Dict:
        """
        多轮协作生成
        
        参数:
            prompt: 输入提示
            max_rounds: 最大轮数
        
        返回:
            包含各轮生成结果的字典
        """
        results = {
            "prompt": prompt,
            "rounds": [],
            "final_response": None
        }
        
        current_response = None
        previous_quality = 0.0
        
        for round_num in range(max_rounds):
            round_result = {
                "round": round_num + 1,
                "pioneer_response": None,
                "observer_response": None,
                "quality_score": None
            }
            
            # Pioneer生成（或基于Observer改进后再生成）
            # 这里使用模拟，实际实现需要调用真实模型
            if current_response is None:
                pioneer_resp = f"[Pioneer Round {round_num + 1}] 初始响应"
            else:
                pioneer_resp = f"[Pioneer Round {round_num + 1}] 基于上轮改进的响应"
            
            round_result["pioneer_response"] = pioneer_resp
            
            # Observer改进
            observer_resp = f"[Observer Round {round_num + 1}] 改进后的响应"
            round_result["observer_response"] = observer_resp
            
            # Critic评估（模拟）
            quality_score = 0.5 + 0.1 * round_num + random.uniform(-0.05, 0.05)
            round_result["quality_score"] = quality_score
            
            results["rounds"].append(round_result)
            current_response = observer_resp
            
            # 检查是否应该继续
            if not self.multi_round.should_continue(quality_score, previous_quality, round_num + 1):
                break
            
            previous_quality = quality_score
        
        results["final_response"] = current_response
        
        return results


# 模拟演示函数
def demo_innovations():
    """
    演示所有创新改进
    """
    print("\n" + "=" * 60)
    print("CORY创新拓展 - 演示")
    print("=" * 60)
    
    # 1. 动态协作权重演示
    print("\n【创新1】动态协作权重调度")
    print("-" * 40)
    scheduler = DynamicCooperationScheduler(
        initial_coop_weight=0.1,
        final_coop_weight=0.5,
        warmup_steps=10,
        total_steps=100
    )
    
    for step in [0, 10, 25, 50, 75, 100]:
        scheduler.current_step = step
        weight = scheduler.get_cooperation_weight()
        kl = scheduler.get_kl_penalty()
        print(f"  步骤 {step:3d}: 协作权重={weight:.3f}, KL惩罚={kl:.3f}")
    
    # 2. 课程学习演示
    print("\n【创新2】课程学习策略")
    print("-" * 40)
    curriculum = CurriculumLearningManager()
    
    test_questions = [
        ("5加3等于多少？", "8"),
        ("商店有50本书，卖出了23本，又进了15本，现在有多少本？", "42"),
        ("一个工厂第一季度生产了1200件产品，第二季度比第一季度多生产20%，请问两个季度共生产多少件？", "2640")
    ]
    
    for q, a in test_questions:
        diff = curriculum.assess_difficulty(q, a)
        print(f"  难度等级 {diff}: {q[:30]}...")
    
    # 3. 多轮改进演示
    print("\n【创新3】多轮迭代改进")
    print("-" * 40)
    multi_round = MultiRoundRefiner(max_rounds=3)
    
    qualities = [0.6, 0.75, 0.82]
    for i, q in enumerate(qualities):
        prev_q = qualities[i-1] if i > 0 else 0
        weight = multi_round.compute_round_weight(i)
        should_cont = multi_round.should_continue(q, prev_q, i)
        print(f"  轮次 {i+1}: 质量={q:.2f}, 权重={weight:.3f}, 继续={should_cont}")
    
    # 4. 增强奖励演示
    print("\n【创新4】增强奖励模型")
    print("-" * 40)
    enhanced_reward = EnhancedRewardModel()
    
    # 模拟多样性计算
    responses = [
        "答案是42，因为5乘以8等于40，加上2等于42",
        "计算过程：5×8+2=40+2=42，所以答案是42",
        "首先计算乘法5*8=40，然后加2得到最终答案42"
    ]
    
    diversity = enhanced_reward.compute_diversity_reward(responses)
    print(f"  三个响应的多样性分数: {diversity:.4f}")
    
    # 计算增强奖励
    total_reward = enhanced_reward.compute_enhanced_reward(
        task_reward=0.8,
        cooperation_reward=0.3,
        kl_divergence=0.1,
        responses=responses,
        round_num=0
    )
    print(f"  增强总奖励: {total_reward:.4f}")
    
    print("\n" + "=" * 60)
    print("创新拓展演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_innovations()
