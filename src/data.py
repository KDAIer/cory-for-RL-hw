"""
CORY论文复现 - 数据处理模块
实现数据集加载、预处理和批次生成
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import random


class MathReasoningDataset(Dataset):
    """
    数学推理数据集
    用于GSM8K等数学问答任务
    """
    
    def __init__(
        self,
        questions: List[str],
        answers: List[str],
        max_length: int = 512
    ):
        """
        初始化数据集
        
        参数:
            questions: 问题列表
            answers: 答案列表
            max_length: 最大序列长度
        """
        self.questions = questions
        self.answers = answers
        self.max_length = max_length
        
        assert len(questions) == len(answers), "问题和答案数量必须相同"
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        获取单个样本
        
        参数:
            idx: 样本索引
        
        返回:
            包含问题和答案的字典
        """
        return {
            "question": self.questions[idx],
            "answer": self.answers[idx]
        }


def load_gsm8k_data(
    max_train_samples: int = 1000,
    max_eval_samples: int = 200
) -> Tuple[List[Dict], List[Dict]]:
    """
    加载GSM8K数据集
    如果无法从HuggingFace加载，则使用模拟数据
    
    参数:
        max_train_samples: 最大训练样本数
        max_eval_samples: 最大评估样本数
    
    返回:
        训练数据和评估数据元组
    """
    try:
        from datasets import load_dataset
        
        print("正在从HuggingFace加载GSM8K数据集...")
        dataset = load_dataset("gsm8k", "main")
        
        train_data = []
        for i, item in enumerate(dataset["train"]):
            if i >= max_train_samples:
                break
            train_data.append({
                "question": item["question"],
                "answer": item["answer"]
            })
        
        eval_data = []
        for i, item in enumerate(dataset["test"]):
            if i >= max_eval_samples:
                break
            eval_data.append({
                "question": item["question"],
                "answer": item["answer"]
            })
        
        print(f"成功加载 {len(train_data)} 条训练数据和 {len(eval_data)} 条评估数据")
        return train_data, eval_data
        
    except Exception as e:
        print(f"无法加载GSM8K数据集: {e}")
        print("使用模拟数据进行演示...")
        return generate_mock_math_data(max_train_samples, max_eval_samples)


def generate_mock_math_data(
    num_train: int = 100,
    num_eval: int = 20
) -> Tuple[List[Dict], List[Dict]]:
    """
    生成模拟数学数据
    用于在无法加载真实数据集时进行演示
    
    参数:
        num_train: 训练样本数
        num_eval: 评估样本数
    
    返回:
        模拟训练数据和评估数据
    """
    # 数学问题模板
    templates = [
        # 加法问题
        {
            "template": "小明有{a}个苹果，小红给了他{b}个苹果，小明现在有多少个苹果？",
            "answer_template": "小明原来有{a}个苹果，小红给了他{b}个苹果，所以现在有 {a} + {b} = {result} 个苹果。\n#### {result}",
            "operation": lambda a, b: a + b
        },
        # 减法问题
        {
            "template": "商店有{a}本书，卖出了{b}本，还剩多少本？",
            "answer_template": "商店原来有{a}本书，卖出{b}本后，还剩 {a} - {b} = {result} 本。\n#### {result}",
            "operation": lambda a, b: a - b
        },
        # 乘法问题
        {
            "template": "每个盒子有{a}个球，共有{b}个盒子，一共有多少个球？",
            "answer_template": "每个盒子{a}个球，{b}个盒子，总共有 {a} × {b} = {result} 个球。\n#### {result}",
            "operation": lambda a, b: a * b
        },
        # 除法问题
        {
            "template": "{a}个糖果平均分给{b}个小朋友，每人分到多少个？",
            "answer_template": "{a}个糖果分给{b}个小朋友，每人分到 {a} ÷ {b} = {result} 个。\n#### {result}",
            "operation": lambda a, b: a // b if b != 0 else 0
        },
        # 两步问题
        {
            "template": "小明有{a}元钱，买了一本{b}元的书后，妈妈又给了他{c}元，他现在有多少钱？",
            "answer_template": "小明原来{a}元，买书花了{b}元，还剩{a}-{b}={d}元。妈妈给了{c}元后，共有{d}+{c}={result}元。\n#### {result}",
            "operation": lambda a, b, c=None: a - b + (c if c else 10)
        }
    ]
    
    train_data = []
    eval_data = []
    
    # 生成训练数据
    for i in range(num_train):
        template_info = random.choice(templates)
        
        if "c" in template_info["answer_template"]:
            a = random.randint(50, 100)
            b = random.randint(10, 30)
            c = random.randint(10, 50)
            d = a - b
            result = d + c
            
            question = template_info["template"].format(a=a, b=b, c=c)
            answer = template_info["answer_template"].format(
                a=a, b=b, c=c, d=d, result=result
            )
        else:
            a = random.randint(10, 100)
            b = random.randint(1, min(a, 20))  # 确保减法和除法有意义
            result = template_info["operation"](a, b)
            
            question = template_info["template"].format(a=a, b=b)
            answer = template_info["answer_template"].format(a=a, b=b, result=result)
        
        train_data.append({
            "question": question,
            "answer": answer
        })
    
    # 生成评估数据
    for i in range(num_eval):
        template_info = random.choice(templates)
        
        if "c" in template_info["answer_template"]:
            a = random.randint(50, 100)
            b = random.randint(10, 30)
            c = random.randint(10, 50)
            d = a - b
            result = d + c
            
            question = template_info["template"].format(a=a, b=b, c=c)
            answer = template_info["answer_template"].format(
                a=a, b=b, c=c, d=d, result=result
            )
        else:
            a = random.randint(10, 100)
            b = random.randint(1, min(a, 20))
            result = template_info["operation"](a, b)
            
            question = template_info["template"].format(a=a, b=b)
            answer = template_info["answer_template"].format(a=a, b=b, result=result)
        
        eval_data.append({
            "question": question,
            "answer": answer
        })
    
    print(f"生成了 {len(train_data)} 条模拟训练数据和 {len(eval_data)} 条模拟评估数据")
    return train_data, eval_data


def create_dataloader(
    data: List[Dict],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    创建数据加载器
    
    参数:
        data: 数据列表
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作线程数
    
    返回:
        DataLoader对象
    """
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    
    dataset = MathReasoningDataset(questions, answers)
    
    def collate_fn(batch):
        """自定义批次整理函数"""
        return {
            "question": [item["question"] for item in batch],
            "answer": [item["answer"] for item in batch]
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader


class TextPromptFormatter:
    """
    文本提示格式化器
    将原始问题转换为适合模型的提示格式
    """
    
    def __init__(self, template_type: str = "cot"):
        """
        初始化格式化器
        
        参数:
            template_type: 模板类型（"cot"链式思维, "direct"直接回答）
        """
        self.template_type = template_type
        
        # 定义不同的提示模板
        self.templates = {
            "cot": "问题: {question}\n\n请一步步思考并给出答案:\n",
            "direct": "问题: {question}\n答案: ",
            "detailed": """你是一个数学解题助手。请仔细阅读以下问题，并给出详细的解答步骤。

问题: {question}

请按以下格式回答:
1. 理解问题
2. 列出已知条件
3. 分步骤计算
4. 得出最终答案

解答:
"""
        }
    
    def format(self, question: str) -> str:
        """
        格式化问题为提示
        
        参数:
            question: 原始问题
        
        返回:
            格式化后的提示
        """
        template = self.templates.get(self.template_type, self.templates["cot"])
        return template.format(question=question)
    
    def batch_format(self, questions: List[str]) -> List[str]:
        """
        批量格式化问题
        
        参数:
            questions: 问题列表
        
        返回:
            格式化后的提示列表
        """
        return [self.format(q) for q in questions]
