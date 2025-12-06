# CORY论文复现项目

## Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning

---

## 项目简介

本项目是对CORY论文的完整复现，实现了基于序贯协作多智能体强化学习的LLM微调方法。

### 核心特性

- 完整的双智能体架构（Pioneer + Observer）
- 角色交换机制（Role Swap） - CORY核心创新
- 序贯协作生成机制
- 基于PPO的强化学习训练
- 多种奖励机制（任务奖励、协作奖励、KL惩罚）
- 支持LoRA高效微调
- 适配CPU/GPU运行
- 创新拓展实现

---

## 快速开始

### 0. 模型选择

**重要**：项目现已支持7B模型以获得真正的数学推理能力！

| 模型                 | 参数量 | 准确率预期       | GPU显存需求  | 适用场景                  |
| -------------------- | ------ | ---------------- | ------------ | ------------------------- |
| GPT-2                | 117M   | ~0%              | 2GB          | 快速测试、验证算法        |
| **Mistral-7B** | 7B     | **50-60%** | 16GB (8-bit) | **推荐 - 真实性能** |
| Llama-2-7B           | 7B     | 50-60%           | 16GB (8-bit) | 需HF token                |

**配置7B模型**：编辑 `configs/config.py`

```python
model_name = "mistralai/Mistral-7B-v0.1"  # 使用7B模型
load_in_8bit = True                        # 启用量化节省显存
```

### 1. 环境配置

```bash
# 克隆项目
git clone [您的GitHub链接]
cd CORY_Reproduction

# 创建虚拟环境（可选）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 快速演示（无需GPU）

```bash
# 方式1：运行简化演示脚本
python run_demo.py

# 方式2：使用主程序的演示模式
python main.py --mode demo
```

### 3. 完整训练

```bash
# 轻量级训练（推荐，适合普通电脑）
python main.py --mode lightweight

# 完整训练
python main.py --mode train

# 自定义参数
python main.py --mode train --epochs 3 --batch_size 4 --model gpt2
```

### 4. 评估模型

```bash
python main.py --mode eval
```

---

## 项目结构

```
CORY_Reproduction/
├── configs/                  # 配置文件目录
│   ├── __init__.py
│   └── config.py            # 超参数配置
│
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── model.py             # 双智能体模型实现
│   ├── reward.py            # 奖励模型
│   ├── trainer.py           # PPO训练器
│   ├── data.py              # 数据处理
│   ├── evaluation.py        # 评估和可视化
│   └── innovations.py       # 创新拓展
│
├── utils/                   # 工具函数目录
│   ├── __init__.py
│   └── helpers.py           # 辅助函数
│
├── outputs/                 # 输出目录（训练后生成）
│   ├── checkpoints/         # 模型检查点
│   ├── logs/                # 训练日志
│   └── results/             # 评估结果
│
├── main.py                  # 主程序入口
├── run_demo.py              # 快速演示脚本
├── requirements.txt         # 依赖列表
└── README.md                # 本文件
```

---

## 核心算法说明

### CORY双智能体架构

```text
           ┌─────────────┐
           │   输入问题   │
           └──────┬──────┘
                  │
                  ▼
        ┌─────────────────┐
        │    Pioneer      │
        │  (先驱者智能体)  │
        │  生成初始响应    │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │    Observer     │
        │  (观察者智能体)  │
        │  观察并改进响应  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   最终输出       │
        └─────────────────┘
```

### 角色交换机制（Role Swap）

CORY的核心创新之一是**角色交换**：两个智能体周期性交换参数，互换Pioneer和Observer角色。

```text
训练步骤 1-100:   Pioneer(θ_A) → Observer(θ_B)
    ↓ 角色交换
训练步骤 101-200: Pioneer(θ_B) → Observer(θ_A)
    ↓ 角色交换
训练步骤 201-300: Pioneer(θ_A) → Observer(θ_B)
    ...
```

**支持三种交换策略：**

- **硬交换（Hard）**：完全交换所有参数
- **软交换（Soft）**：参数插值混合，更平滑
- **LoRA交换**：仅交换LoRA适配器参数，最轻量

### 奖励函数

总奖励由三部分组成：

```text
R_total = α·R_task + β·R_coop - γ·D_KL

其中：
- R_task: 任务奖励（回答正确性）
- R_coop: 协作奖励（Observer改进Pioneer的效果）
- D_KL:   KL散度惩罚（防止策略偏离过远）
```

---

## 运行模式说明

| 模式        | 命令                   | 说明                   | 硬件要求          |
| ----------- | ---------------------- | ---------------------- | ----------------- |
| demo        | `--mode demo`        | 快速演示，使用模拟数据 | 4GB RAM           |
| lightweight | `--mode lightweight` | 轻量训练，减少计算量   | 8GB RAM           |
| train       | `--mode train`       | 完整训练               | 16GB RAM, GPU推荐 |
| eval        | `--mode eval`        | 评估已训练模型         | 4GB RAM           |

---

## 配置参数

主要参数可通过命令行或配置文件修改：

```python
# 模型配置
model_name = "gpt2"        # 基础模型
max_length = 512           # 最大序列长度
use_lora = True            # 使用LoRA
lora_r = 8                 # LoRA秩

# 训练配置
num_epochs = 3             # 训练轮数
batch_size = 4             # 批大小
learning_rate = 1e-5       # 学习率
clip_epsilon = 0.2         # PPO裁剪参数
cooperation_weight = 0.3   # 协作奖励权重
kl_penalty = 0.1           # KL惩罚系数

# 角色交换配置（CORY核心机制）
role_swap_enabled = True   # 启用角色交换
role_swap_frequency = 100  # 每100步交换一次
role_swap_type = "soft"    # 交换类型: hard/soft/lora
role_swap_factor = 0.3     # 软交换插值因子
```

---

## 创新拓展

本项目在原论文基础上实现了两项创新改进：

1. **三智能体架构（TriAgent）**

   - 引入Critic智能体进行质量评估
2. **动态协作权重**

   - 根据训练进度自适应调整协作强度
