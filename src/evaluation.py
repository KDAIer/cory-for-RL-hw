"""
CORY论文复现 - 评估模块
实现模型评估和结果可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class Evaluator:
    """
    CORY模型评估器
    计算各种评估指标并生成可视化结果
    """
    
    def __init__(
        self,
        output_dir: str = "./outputs/evaluation"
    ):
        """
        初始化评估器
        
        参数:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 评估结果存储
        self.results = {
            "accuracy": [],
            "pioneer_accuracy": [],
            "observer_accuracy": [],
            "improvement_rate": [],
            "cooperation_score": []
        }
    
    def extract_answer(self, text: str) -> Optional[float]:
        """
        从文本中提取数值答案
        
        参数:
            text: 响应文本
        
        返回:
            提取的数值答案，如果无法提取则返回None
        """
        # 尝试匹配 "#### number" 格式（GSM8K标准格式）
        match = re.search(r'####\s*(-?\d+\.?\d*)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # 尝试匹配最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def compute_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """
        计算准确率
        
        参数:
            predictions: 预测答案列表
            ground_truths: 真实答案列表
        
        返回:
            准确率（0-1之间）
        """
        correct = 0
        total = len(predictions)
        
        for pred, truth in zip(predictions, ground_truths):
            pred_answer = self.extract_answer(pred)
            truth_answer = self.extract_answer(truth)
            
            if pred_answer is not None and truth_answer is not None:
                if abs(pred_answer - truth_answer) < 1e-6:
                    correct += 1
        
        return correct / total if total > 0 else 0
    
    def evaluate_sequential(
        self,
        pioneer_responses: List[str],
        observer_responses: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        评估序贯协作的效果
        
        参数:
            pioneer_responses: Pioneer响应列表
            observer_responses: Observer响应列表
            ground_truths: 真实答案列表
        
        返回:
            包含各项指标的评估结果字典
        """
        # 计算Pioneer准确率
        pioneer_acc = self.compute_accuracy(pioneer_responses, ground_truths)
        
        # 计算Observer准确率
        observer_acc = self.compute_accuracy(observer_responses, ground_truths)
        
        # 计算改进率（Observer相比Pioneer的提升）
        if pioneer_acc > 0:
            improvement_rate = (observer_acc - pioneer_acc) / pioneer_acc
        else:
            improvement_rate = observer_acc if observer_acc > 0 else 0
        
        # 计算协作分数
        # 如果Observer改进了Pioneer的错误回答，给予额外分数
        cooperation_score = 0
        for pioneer_resp, observer_resp, truth in zip(
            pioneer_responses, observer_responses, ground_truths
        ):
            pioneer_correct = self._is_correct(pioneer_resp, truth)
            observer_correct = self._is_correct(observer_resp, truth)
            
            if not pioneer_correct and observer_correct:
                cooperation_score += 1.0  # Observer成功纠错
            elif pioneer_correct and observer_correct:
                cooperation_score += 0.5  # 保持正确
            elif pioneer_correct and not observer_correct:
                cooperation_score -= 0.5  # Observer引入错误
        
        cooperation_score /= len(ground_truths) if ground_truths else 1
        
        results = {
            "pioneer_accuracy": pioneer_acc,
            "observer_accuracy": observer_acc,
            "improvement_rate": improvement_rate,
            "cooperation_score": cooperation_score,
            "final_accuracy": observer_acc
        }
        
        # 保存结果
        for key, value in results.items():
            if key in self.results:
                self.results[key].append(value)
        
        return results
    
    def _is_correct(self, response: str, truth: str) -> bool:
        """
        判断响应是否正确
        
        参数:
            response: 响应文本
            truth: 真实答案
        
        返回:
            是否正确
        """
        pred_answer = self.extract_answer(response)
        truth_answer = self.extract_answer(truth)
        
        if pred_answer is not None and truth_answer is not None:
            return abs(pred_answer - truth_answer) < 1e-6
        
        return False
    
    def plot_training_curves(
        self,
        train_stats: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制训练曲线
        
        参数:
            train_stats: 训练统计数据字典
            save_path: 保存路径（如果为None则显示图像）
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 策略损失曲线
        if "policy_loss" in train_stats and train_stats["policy_loss"]:
            axes[0, 0].plot(train_stats["policy_loss"], 'b-', linewidth=1.5)
            axes[0, 0].set_title("策略损失 (Policy Loss)")
            axes[0, 0].set_xlabel("训练步骤")
            axes[0, 0].set_ylabel("损失值")
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 奖励曲线
        if "total_reward" in train_stats and train_stats["total_reward"]:
            axes[0, 1].plot(train_stats["total_reward"], 'g-', linewidth=1.5, label="总奖励")
            if "pioneer_reward" in train_stats:
                axes[0, 1].plot(train_stats["pioneer_reward"], 'r--', linewidth=1, label="Pioneer奖励")
            if "observer_reward" in train_stats:
                axes[0, 1].plot(train_stats["observer_reward"], 'b--', linewidth=1, label="Observer奖励")
            axes[0, 1].set_title("奖励曲线")
            axes[0, 1].set_xlabel("训练步骤")
            axes[0, 1].set_ylabel("奖励值")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 协作奖励曲线
        if "cooperation_bonus" in train_stats and train_stats["cooperation_bonus"]:
            axes[1, 0].plot(train_stats["cooperation_bonus"], 'purple', linewidth=1.5)
            axes[1, 0].set_title("协作奖励 (Cooperation Bonus)")
            axes[1, 0].set_xlabel("训练步骤")
            axes[1, 0].set_ylabel("协作奖励")
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 熵值曲线
        if "entropy" in train_stats and train_stats["entropy"]:
            axes[1, 1].plot(train_stats["entropy"], 'orange', linewidth=1.5)
            axes[1, 1].set_title("策略熵 (Entropy)")
            axes[1, 1].set_xlabel("训练步骤")
            axes[1, 1].set_ylabel("熵值")
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_accuracy_comparison(
        self,
        methods: List[str],
        accuracies: List[float],
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制准确率对比柱状图
        
        参数:
            methods: 方法名称列表
            accuracies: 对应准确率列表
            save_path: 保存路径
        """
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
        bars = ax.bar(methods, accuracies, color=colors[:len(methods)])
        
        # 在柱子上添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{acc:.2%}',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        ax.set_ylabel("准确率", fontsize=12)
        ax.set_title("不同方法准确率对比", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"准确率对比图已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_cooperation_analysis(
        self,
        pioneer_accs: List[float],
        observer_accs: List[float],
        epochs: List[int],
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制协作效果分析图
        
        参数:
            pioneer_accs: Pioneer各epoch准确率
            observer_accs: Observer各epoch准确率
            epochs: epoch列表
            save_path: 保存路径
        """
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, pioneer_accs, 'ro-', linewidth=2, markersize=8, label='Pioneer')
        ax.plot(epochs, observer_accs, 'bs-', linewidth=2, markersize=8, label='Observer')
        
        # 填充改进区域
        ax.fill_between(
            epochs,
            pioneer_accs,
            observer_accs,
            alpha=0.3,
            color='green',
            label='协作改进'
        )
        
        ax.set_xlabel("训练轮次 (Epoch)", fontsize=12)
        ax.set_ylabel("准确率", fontsize=12)
        ax.set_title("Pioneer vs Observer 协作效果分析", fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"协作分析图已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(
        self,
        train_stats: Dict[str, List[float]],
        eval_results: Dict[str, float],
        save_path: str
    ) -> None:
        """
        生成评估报告
        
        参数:
            train_stats: 训练统计数据
            eval_results: 评估结果
            save_path: 保存路径
        """
        report = {
            "summary": {
                "final_accuracy": eval_results.get("final_accuracy", 0),
                "pioneer_accuracy": eval_results.get("pioneer_accuracy", 0),
                "observer_accuracy": eval_results.get("observer_accuracy", 0),
                "improvement_rate": eval_results.get("improvement_rate", 0),
                "cooperation_score": eval_results.get("cooperation_score", 0)
            },
            "training_stats": {
                "total_steps": len(train_stats.get("policy_loss", [])),
                "final_loss": train_stats["policy_loss"][-1] if train_stats.get("policy_loss") else 0,
                "avg_reward": np.mean(train_stats.get("total_reward", [0])),
                "avg_cooperation_bonus": np.mean(train_stats.get("cooperation_bonus", [0]))
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"评估报告已保存至: {save_path}")
        
        # 打印摘要
        print("\n" + "=" * 50)
        print("评估结果摘要")
        print("=" * 50)
        print(f"最终准确率: {eval_results.get('final_accuracy', 0):.2%}")
        print(f"Pioneer准确率: {eval_results.get('pioneer_accuracy', 0):.2%}")
        print(f"Observer准确率: {eval_results.get('observer_accuracy', 0):.2%}")
        print(f"改进率: {eval_results.get('improvement_rate', 0):.2%}")
        print(f"协作分数: {eval_results.get('cooperation_score', 0):.4f}")
        print("=" * 50)
