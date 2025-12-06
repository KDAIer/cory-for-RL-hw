"""
CORY论文复现 - 主程序入口
===============================
论文: Coevolving with the Other You: Fine-Tuning LLM with Sequential 
      Cooperative Multi-Agent Reinforcement Learning

本程序实现了CORY算法的完整训练和评估流程。
CORY的核心思想是使用双智能体（Pioneer和Observer）通过序贯协作来微调LLM。

使用方法:
    python main.py --mode train          # 训练模式
    python main.py --mode eval           # 评估模式
    python main.py --mode demo           # 演示模式（快速测试）
    python main.py --mode lightweight    # 轻量级训练（适合普通电脑）
"""

import os
import sys
import argparse
import time
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_default_config, get_lightweight_config
from src.model import CORYDualAgent
from src.reward import RewardModel
from src.trainer import PPOTrainer
from src.data import load_gsm8k_data, create_dataloader, TextPromptFormatter
from src.evaluation import Evaluator
from utils.helpers import set_seed, create_experiment_dir, Logger, print_training_info


def parse_args():
    """
    解析命令行参数
    
    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="CORY论文复现 - 基于序贯协作多智能体强化学习的LLM微调"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["train", "eval", "demo", "lightweight"],
        help="运行模式: train(完整训练), eval(评估), demo(快速演示), lightweight(轻量训练)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数（覆盖配置文件设置）"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="批大小（覆盖配置文件设置）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称（覆盖配置文件设置）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="禁用CUDA，强制使用CPU"
    )
    
    return parser.parse_args()


def run_demo_mode(config, args):
    """
    运行演示模式
    使用模拟数据快速展示CORY算法的效果
    
    参数:
        config: 配置对象
        args: 命令行参数
    """
    print("\n" + "=" * 60)
    print("CORY论文复现 - 演示模式")
    print("=" * 60)
    print("本演示将展示CORY双智能体协作的基本效果")
    print("使用模拟数据，不需要GPU即可运行")
    print("=" * 60 + "\n")
    
    # 创建实验目录
    exp_dir = create_experiment_dir(args.output_dir, "demo")
    logger = Logger(os.path.join(exp_dir, "logs"))
    
    # 使用轻量配置
    config.model.model_name = "gpt2"
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.data.max_train_samples = 20
    config.data.max_eval_samples = 10
    
    logger.log("开始加载模拟数据...")
    
    # 加载模拟数据
    from src.data import generate_mock_math_data
    train_data, eval_data = generate_mock_math_data(
        num_train=config.data.max_train_samples,
        num_eval=config.data.max_eval_samples
    )
    
    # 展示一些数据样例
    logger.log("\n数据样例:")
    for i, item in enumerate(train_data[:3]):
        logger.log(f"\n样例 {i + 1}:")
        logger.log(f"问题: {item['question']}")
        logger.log(f"答案: {item['answer'][:100]}...")
    
    # 创建数据加载器
    train_loader = create_dataloader(train_data, batch_size=2)
    
    logger.log("\n开始模拟训练过程...")
    logger.log("(演示模式跳过实际模型训练，展示训练流程)")
    
    # 模拟训练统计数据
    mock_stats = {
        "policy_loss": [0.5 - 0.02 * i + 0.01 * (i % 3) for i in range(10)],
        "total_reward": [0.2 + 0.05 * i - 0.01 * (i % 2) for i in range(10)],
        "pioneer_reward": [0.1 + 0.04 * i for i in range(10)],
        "observer_reward": [0.15 + 0.05 * i for i in range(10)],
        "cooperation_bonus": [0.05 + 0.02 * i for i in range(10)],
        "entropy": [2.0 - 0.1 * i for i in range(10)]
    }
    
    # 创建评估器并生成可视化
    evaluator = Evaluator(output_dir=os.path.join(exp_dir, "evaluation"))
    
    # 生成训练曲线图
    logger.log("\n正在生成训练曲线图...")
    evaluator.plot_training_curves(
        mock_stats,
        save_path=os.path.join(exp_dir, "training_curves.png")
    )
    
    # 生成准确率对比图
    logger.log("正在生成准确率对比图...")
    methods = ["基准模型", "Pioneer (单智能体)", "Observer (单智能体)", "CORY (协作)"]
    accuracies = [0.35, 0.42, 0.48, 0.58]
    evaluator.plot_accuracy_comparison(
        methods,
        accuracies,
        save_path=os.path.join(exp_dir, "accuracy_comparison.png")
    )
    
    # 生成协作分析图
    logger.log("正在生成协作分析图...")
    epochs = list(range(1, 6))
    pioneer_accs = [0.35, 0.40, 0.44, 0.47, 0.49]
    observer_accs = [0.38, 0.45, 0.52, 0.56, 0.58]
    evaluator.plot_cooperation_analysis(
        pioneer_accs,
        observer_accs,
        epochs,
        save_path=os.path.join(exp_dir, "cooperation_analysis.png")
    )
    
    # 生成评估报告
    eval_results = {
        "final_accuracy": 0.58,
        "pioneer_accuracy": 0.49,
        "observer_accuracy": 0.58,
        "improvement_rate": 0.18,
        "cooperation_score": 0.72
    }
    evaluator.generate_report(
        mock_stats,
        eval_results,
        save_path=os.path.join(exp_dir, "evaluation_report.json")
    )
    
    logger.log("\n" + "=" * 60)
    logger.log("演示完成!")
    logger.log(f"所有结果已保存至: {exp_dir}")
    logger.log("=" * 60)


def run_train_mode(config, args):
    """
    运行完整训练模式
    
    参数:
        config: 配置对象
        args: 命令行参数
    """
    print("\n" + "=" * 60)
    print("CORY论文复现 - 训练模式")
    print("=" * 60)
    
    # 创建实验目录
    exp_dir = create_experiment_dir(args.output_dir, config.experiment_name)
    logger = Logger(os.path.join(exp_dir, "logs"))
    
    # 打印配置信息
    print_training_info(config)
    
    # 设置随机种子
    set_seed(config.training.seed)
    logger.log(f"随机种子设置为: {config.training.seed}")
    
    # 加载数据
    logger.log("正在加载数据集...")
    train_data, eval_data = load_gsm8k_data(
        max_train_samples=config.data.max_train_samples,
        max_eval_samples=config.data.max_eval_samples
    )
    
    train_loader = create_dataloader(train_data, batch_size=config.training.batch_size)
    eval_loader = create_dataloader(eval_data, batch_size=config.training.batch_size, shuffle=False)
    
    logger.log(f"训练集大小: {len(train_data)}, 评估集大小: {len(eval_data)}")
    
    # 初始化模型
    logger.log("正在初始化CORY双智能体系统...")
    
    lora_config = {
        "r": config.model.lora_r,
        "alpha": config.model.lora_alpha,
        "dropout": config.model.lora_dropout
    }
    
    dual_agent = CORYDualAgent(
        model_name=config.model.model_name,
        device=config.device,
        use_lora=config.model.use_lora,
        lora_config=lora_config,
        load_in_8bit=config.model.load_in_8bit,
        load_in_4bit=config.model.load_in_4bit
    )
    
    # 初始化奖励模型
    reward_model = RewardModel(
        cooperation_weight=config.training.cooperation_weight,
        kl_penalty=config.training.kl_penalty,
        device=config.device
    )
    
    # 初始化训练器
    trainer_config = {
        "learning_rate": config.training.learning_rate,
        "clip_epsilon": config.training.clip_epsilon,
        "ppo_epochs": config.training.ppo_epochs,
        "value_loss_coef": config.training.value_loss_coef,
        "entropy_coef": config.training.entropy_coef,
        "max_grad_norm": config.training.max_grad_norm,
        "gamma": config.training.gamma,
        "gae_lambda": config.training.gae_lambda,
        "weight_decay": config.training.weight_decay,
        # 角色交换配置（CORY核心机制）
        "role_swap_enabled": config.training.role_swap_enabled,
        "role_swap_frequency": config.training.role_swap_frequency,
        "role_swap_type": config.training.role_swap_type,
        "role_swap_factor": config.training.role_swap_factor,
    }
    
    logger.log(f"角色交换配置: 启用={config.training.role_swap_enabled}, "
               f"频率=每{config.training.role_swap_frequency}步, "
               f"类型={config.training.role_swap_type}")
    
    trainer = PPOTrainer(
        dual_agent=dual_agent,
        reward_model=reward_model,
        config=trainer_config,
        device=config.device
    )
    
    # 初始化评估器
    evaluator = Evaluator(output_dir=os.path.join(exp_dir, "evaluation"))
    
    # 开始训练
    logger.log("\n开始训练...")
    start_time = time.time()
    
    best_accuracy = 0
    for epoch in range(config.training.num_epochs):
        epoch_start = time.time()
        
        # 训练一个epoch
        epoch_stats = trainer.train_epoch(train_loader, epoch)
        
        epoch_time = time.time() - epoch_start
        
        logger.log(f"\nEpoch {epoch + 1}/{config.training.num_epochs} 完成")
        logger.log(f"耗时: {epoch_time:.2f}秒")
        logger.log(f"策略损失: {epoch_stats['policy_loss']:.4f}")
        logger.log(f"总奖励: {epoch_stats['total_reward']:.4f}")
        logger.log(f"协作奖励: {epoch_stats['cooperation_bonus']:.4f}")
        
        # 评估
        if (epoch + 1) % 1 == 0:  # 每个epoch评估一次
            logger.log("\n进行评估...")
            
            # 生成评估样本的响应
            pioneer_responses = []
            observer_responses = []
            ground_truths = []
            
            dual_agent.eval()
            with torch.no_grad():
                for batch in eval_loader:
                    prompts = batch["question"]
                    truths = batch["answer"]
                    
                    # 序贯生成
                    results = dual_agent.sequential_generation(
                        prompts,
                        max_new_tokens=128,
                        num_iterations=1
                    )
                    
                    pioneer_responses.extend(results["pioneer_responses"])
                    observer_responses.extend(results["final_responses"])
                    ground_truths.extend(truths)
            
            # 计算评估指标
            eval_results = evaluator.evaluate_sequential(
                pioneer_responses,
                observer_responses,
                ground_truths
            )
            
            logger.log(f"Pioneer准确率: {eval_results['pioneer_accuracy']:.2%}")
            logger.log(f"Observer准确率: {eval_results['observer_accuracy']:.2%}")
            logger.log(f"改进率: {eval_results['improvement_rate']:.2%}")
            
            # 保存最佳模型
            if eval_results['final_accuracy'] > best_accuracy:
                best_accuracy = eval_results['final_accuracy']
                trainer.save_checkpoint(
                    os.path.join(exp_dir, "checkpoints", "best_model.pt")
                )
                logger.log(f"新的最佳模型! 准确率: {best_accuracy:.2%}")
        
        # 定期保存检查点
        if (epoch + 1) % config.training.save_steps == 0:
            trainer.save_checkpoint(
                os.path.join(exp_dir, "checkpoints", f"checkpoint_epoch_{epoch + 1}.pt")
            )
    
    total_time = time.time() - start_time
    logger.log(f"\n训练完成! 总耗时: {total_time:.2f}秒")
    
    # 生成最终可视化和报告
    evaluator.plot_training_curves(
        trainer.train_stats,
        save_path=os.path.join(exp_dir, "training_curves.png")
    )
    
    evaluator.generate_report(
        trainer.train_stats,
        eval_results,
        save_path=os.path.join(exp_dir, "final_report.json")
    )
    
    logger.log(f"\n所有结果已保存至: {exp_dir}")


def run_lightweight_mode(config, args):
    """
    运行轻量级训练模式
    适合普通电脑运行，减少资源消耗
    
    参数:
        config: 配置对象
        args: 命令行参数
    """
    print("\n" + "=" * 60)
    print("CORY论文复现 - 轻量级训练模式")
    print("=" * 60)
    print("使用简化配置，适合普通电脑运行")
    print("=" * 60 + "\n")
    
    # 使用轻量配置
    config = get_lightweight_config()
    
    # 覆盖命令行参数
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.no_cuda:
        config.device = "cpu"
    
    # 运行训练
    run_train_mode(config, args)


def run_eval_mode(config, args):
    """
    运行评估模式
    
    参数:
        config: 配置对象
        args: 命令行参数
    """
    print("\n" + "=" * 60)
    print("CORY论文复现 - 评估模式")
    print("=" * 60)
    
    # 这里可以加载已训练的模型进行评估
    # 目前显示如何使用评估器
    
    evaluator = Evaluator(output_dir=os.path.join(args.output_dir, "evaluation"))
    
    # 示例：评估已有结果
    print("评估模式需要加载已训练的模型检查点")
    print("请使用 --checkpoint 参数指定检查点路径")
    print("或先运行训练模式: python main.py --mode train")


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 获取配置
    if args.mode == "lightweight":
        config = get_lightweight_config()
    else:
        config = get_default_config()
    
    # 覆盖配置
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.model:
        config.model.model_name = args.model
    if args.no_cuda:
        config.device = "cpu"
    
    config.training.seed = args.seed
    config.training.output_dir = args.output_dir
    
    # 根据模式运行
    if args.mode == "demo":
        run_demo_mode(config, args)
    elif args.mode == "train":
        run_train_mode(config, args)
    elif args.mode == "lightweight":
        run_lightweight_mode(config, args)
    elif args.mode == "eval":
        run_eval_mode(config, args)
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
