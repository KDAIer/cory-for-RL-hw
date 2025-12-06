"""
CORY论文复现 - 快速演示脚本
===============================
这个脚本提供了最简化的演示，不需要安装额外依赖即可运行。
用于快速验证代码结构和算法流程。

运行方式:
    python run_demo.py
"""

import os
import sys
import json
import random
import time
from datetime import datetime

# 设置输出目录
OUTPUT_DIR = "./demo_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_header(text):
    """打印带框的标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def generate_mock_data(num_samples=10):
    """
    生成模拟数学数据
    
    参数:
        num_samples: 生成的样本数量
    
    返回:
        包含问题和答案的数据列表
    """
    templates = [
        ("小明有{a}个苹果，小红给了他{b}个苹果，小明现在有多少个苹果？",
         lambda a, b: a + b),
        ("商店有{a}本书，卖出了{b}本，还剩多少本？",
         lambda a, b: a - b),
        ("每个盒子有{a}个球，共有{b}个盒子，一共有多少个球？",
         lambda a, b: a * b),
    ]
    
    data = []
    for _ in range(num_samples):
        template, op = random.choice(templates)
        a = random.randint(10, 50)
        b = random.randint(1, min(a, 20))
        
        question = template.format(a=a, b=b)
        answer = str(op(a, b))
        
        data.append({"question": question, "answer": answer})
    
    return data


class MockPioneerAgent:
    """
    模拟Pioneer智能体
    用于演示双智能体协作的概念
    """
    
    def __init__(self, accuracy=0.4):
        """
        初始化模拟Pioneer
        
        参数:
            accuracy: 模拟的准确率
        """
        self.accuracy = accuracy
        self.name = "Pioneer"
    
    def generate(self, question, ground_truth):
        """
        模拟生成响应
        
        参数:
            question: 问题文本
            ground_truth: 真实答案
        
        返回:
            模拟的响应
        """
        # 以一定概率返回正确答案
        if random.random() < self.accuracy:
            return f"让我来计算一下...答案是 {ground_truth}"
        else:
            # 返回错误答案
            wrong_answer = int(ground_truth) + random.randint(-10, 10)
            if wrong_answer == int(ground_truth):
                wrong_answer += 1
            return f"我认为答案是 {wrong_answer}"


class MockObserverAgent:
    """
    模拟Observer智能体
    能够观察Pioneer的输出并进行改进
    """
    
    def __init__(self, accuracy=0.6, improvement_rate=0.3):
        """
        初始化模拟Observer
        
        参数:
            accuracy: 基础准确率
            improvement_rate: 纠正Pioneer错误的概率
        """
        self.accuracy = accuracy
        self.improvement_rate = improvement_rate
        self.name = "Observer"
    
    def refine(self, question, pioneer_response, ground_truth):
        """
        基于Pioneer的响应进行改进
        
        参数:
            question: 原始问题
            pioneer_response: Pioneer的响应
            ground_truth: 真实答案
        
        返回:
            改进后的响应
        """
        # 检查Pioneer是否正确
        pioneer_correct = ground_truth in pioneer_response
        
        if pioneer_correct:
            # Pioneer正确，Observer保持或可能引入错误
            if random.random() < 0.9:  # 90%概率保持正确
                return f"经过验证，Pioneer的回答是正确的。最终答案是 {ground_truth}"
            else:
                wrong = int(ground_truth) + 1
                return f"我重新计算了一下，答案应该是 {wrong}"
        else:
            # Pioneer错误，Observer尝试纠正
            if random.random() < self.improvement_rate:
                return f"我发现Pioneer的回答有误。正确答案应该是 {ground_truth}"
            else:
                # 仍然返回错误答案
                wrong = int(ground_truth) + random.randint(-5, 5)
                if wrong == int(ground_truth):
                    wrong += 1
                return f"重新思考后，我认为答案是 {wrong}"


class MockCORYTrainer:
    """
    模拟CORY训练过程
    用于演示训练流程和统计数据
    """
    
    def __init__(self):
        """初始化训练器"""
        self.pioneer = MockPioneerAgent(accuracy=0.35)
        self.observer = MockObserverAgent(accuracy=0.45, improvement_rate=0.4)
        
        # 训练统计
        self.stats = {
            "epoch": [],
            "pioneer_accuracy": [],
            "observer_accuracy": [],
            "cooperation_score": [],
            "policy_loss": []
        }
    
    def train_epoch(self, data, epoch):
        """
        模拟训练一个epoch
        
        参数:
            data: 训练数据
            epoch: 当前epoch编号
        
        返回:
            本epoch的统计数据
        """
        pioneer_correct = 0
        observer_correct = 0
        cooperation_bonus = 0
        
        for item in data:
            question = item["question"]
            truth = item["answer"]
            
            # Pioneer生成
            pioneer_response = self.pioneer.generate(question, truth)
            pioneer_is_correct = truth in pioneer_response
            if pioneer_is_correct:
                pioneer_correct += 1
            
            # Observer改进
            observer_response = self.observer.refine(
                question, pioneer_response, truth
            )
            observer_is_correct = truth in observer_response
            if observer_is_correct:
                observer_correct += 1
            
            # 计算协作奖励
            if not pioneer_is_correct and observer_is_correct:
                cooperation_bonus += 1  # Observer成功纠错
        
        # 模拟训练改进效果
        # 随着epoch增加，准确率逐渐提高
        improvement = 0.05 * epoch
        
        self.pioneer.accuracy = min(0.65, 0.35 + improvement)
        self.observer.improvement_rate = min(0.7, 0.4 + improvement)
        
        # 记录统计
        pioneer_acc = pioneer_correct / len(data)
        observer_acc = observer_correct / len(data)
        coop_score = cooperation_bonus / len(data)
        
        # 模拟损失下降
        mock_loss = 0.5 - 0.08 * epoch + random.uniform(-0.02, 0.02)
        
        self.stats["epoch"].append(epoch + 1)
        self.stats["pioneer_accuracy"].append(pioneer_acc)
        self.stats["observer_accuracy"].append(observer_acc)
        self.stats["cooperation_score"].append(coop_score)
        self.stats["policy_loss"].append(mock_loss)
        
        return {
            "pioneer_accuracy": pioneer_acc,
            "observer_accuracy": observer_acc,
            "cooperation_score": coop_score,
            "policy_loss": mock_loss
        }


def create_text_chart(values, title, width=40, char='█'):
    """
    创建文本柱状图
    
    参数:
        values: 数值字典
        title: 图表标题
        width: 最大宽度
        char: 柱状图字符
    """
    print(f"\n{title}")
    print("-" * 50)
    
    max_val = max(values.values()) if values else 1
    
    for label, val in values.items():
        bar_length = int(val / max_val * width)
        bar = char * bar_length
        print(f"{label:20s} | {bar} {val:.2%}")


def run_demo():
    """
    运行完整演示
    """
    print_header("CORY论文复现 - 快速演示")
    
    print("""
论文: "Coevolving with the Other You: Fine-Tuning LLM with 
       Sequential Cooperative Multi-Agent Reinforcement Learning"

核心创新点:
1. 双智能体架构: Pioneer(先驱者) + Observer(观察者)
2. 序贯协作机制: 两个智能体交替生成和改进响应
3. 基于PPO的协作强化学习优化
    """)
    
    input("按Enter键开始演示...")
    
    # 1. 数据准备
    print_header("第一步: 数据准备")
    
    print("生成模拟数学推理数据...")
    train_data = generate_mock_data(50)
    eval_data = generate_mock_data(20)
    
    print(f"训练集: {len(train_data)} 条")
    print(f"评估集: {len(eval_data)} 条")
    
    print("\n数据样例:")
    for i, item in enumerate(train_data[:3]):
        print(f"  问题 {i+1}: {item['question']}")
        print(f"  答案 {i+1}: {item['answer']}")
        print()
    
    # 2. 模型初始化
    print_header("第二步: 模型初始化")
    
    print("初始化双智能体系统...")
    print("  - Pioneer (先驱者): 负责生成初始响应")
    print("  - Observer (观察者): 负责观察和改进Pioneer的输出")
    print("  - 参考模型: 用于计算KL散度，防止过度偏离")
    
    trainer = MockCORYTrainer()
    print("\n智能体初始化完成!")
    
    # 3. 训练过程
    print_header("第三步: 协作训练")
    
    print("开始CORY序贯协作训练...\n")
    
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # 训练
        stats = trainer.train_epoch(train_data, epoch)
        
        print(f"  Pioneer准确率: {stats['pioneer_accuracy']:.2%}")
        print(f"  Observer准确率: {stats['observer_accuracy']:.2%}")
        print(f"  协作纠错率: {stats['cooperation_score']:.2%}")
        print(f"  策略损失: {stats['policy_loss']:.4f}")
        print()
        
        time.sleep(0.5)  # 模拟训练时间
    
    # 4. 评估结果
    print_header("第四步: 评估结果")
    
    # 在评估集上测试
    final_stats = trainer.train_epoch(eval_data, num_epochs)
    
    print("最终评估结果:")
    create_text_chart({
        "基准模型": 0.32,
        "Pioneer (CORY)": final_stats["pioneer_accuracy"],
        "Observer (CORY)": final_stats["observer_accuracy"]
    }, "准确率对比")
    
    # 5. 协作效果分析
    print_header("第五步: 协作效果分析")
    
    print("\n训练过程中的准确率变化:")
    print("-" * 50)
    print(f"{'Epoch':^8} | {'Pioneer':^12} | {'Observer':^12} | {'改进幅度':^12}")
    print("-" * 50)
    
    for i in range(len(trainer.stats["epoch"])):
        p_acc = trainer.stats["pioneer_accuracy"][i]
        o_acc = trainer.stats["observer_accuracy"][i]
        improvement = o_acc - p_acc
        
        print(f"{trainer.stats['epoch'][i]:^8} | {p_acc:^12.2%} | {o_acc:^12.2%} | {improvement:^12.2%}")
    
    print("-" * 50)
    
    # 6. 保存结果
    print_header("第六步: 保存结果")
    
    results = {
        "experiment_name": "CORY演示",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_stats": trainer.stats,
        "final_results": {
            "pioneer_accuracy": final_stats["pioneer_accuracy"],
            "observer_accuracy": final_stats["observer_accuracy"],
            "cooperation_score": final_stats["cooperation_score"],
            "improvement_rate": (final_stats["observer_accuracy"] - final_stats["pioneer_accuracy"]) / max(final_stats["pioneer_accuracy"], 0.01)
        }
    }
    
    result_path = os.path.join(OUTPUT_DIR, "demo_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存至: {result_path}")
    
    # 7. 总结
    print_header("演示总结")
    
    print("""
CORY算法核心特点:

1. 双智能体协作:
   - Pioneer负责初步探索和生成
   - Observer负责观察、验证和改进
   - 通过序贯决策实现协同进化

2. 奖励机制:
   - 任务奖励: 衡量回答的正确性
   - 协作奖励: 奖励成功的协作纠错
   - KL惩罚: 防止策略偏离过远

3. 训练效果:
   - Observer通过观察Pioneer的行为学习改进
   - 两个智能体在协作中共同进化
   - 最终模型性能超过单智能体基准
    """)
    
    print(f"\n{'='*60}")
    print("演示完成! 完整训练请运行: python main.py --mode train")
    print(f"{'='*60}")


if __name__ == "__main__":
    random.seed(42)  # 固定随机种子保证结果可复现
    run_demo()
