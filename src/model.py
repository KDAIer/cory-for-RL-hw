"""
CORY论文复现 - 双智能体模型模块
实现CORY的核心双智能体架构（Pioneer和Observer）
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer


class CORYAgent(nn.Module):
    """
    CORY智能体类
    封装单个LLM智能体，用于生成响应
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        初始化CORY智能体
        
        参数:
            model_name: 预训练模型名称
            device: 计算设备
            use_lora: 是否使用LoRA进行高效微调
            lora_config: LoRA配置参数
            load_in_8bit: 是否使用8位量化（节省显存）
            load_in_4bit: 是否使用4位量化（更省显存）
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.use_lora = use_lora
        
        # 准备模型加载参数
        load_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True
        }
        
        # 配置量化参数（用于7B模型节省显存）
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
            print(f"使用8位量化加载 {model_name}...")
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
            print(f"使用4位量化加载 {model_name}...")
        else:
            load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            load_kwargs["device_map"] = device if device == "cuda" and torch.cuda.is_available() else None
        
        # 加载基础模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            print(f"成功加载模型: {model_name}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("回退到GPT-2模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float32,
                device_map=device if device == "cuda" else None
            )
        
        # 如果使用LoRA，应用LoRA适配器
        if use_lora and lora_config:
            self._apply_lora(lora_config)
        
        # 移动模型到指定设备（非量化情况）
        if not (load_in_8bit or load_in_4bit):
            if device != "cuda" or not torch.cuda.is_available():
                self.model = self.model.to(device)
    
    def _apply_lora(self, lora_config: Dict) -> None:
        """
        应用LoRA适配器以实现高效微调
        
        参数:
            lora_config: LoRA配置字典
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # 创建LoRA配置
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.1),
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj",
                               "c_attn", "c_proj"]  # GPT-2兼容
            )
            
            # 应用LoRA
            self.model = get_peft_model(self.model, peft_config)
            print(f"LoRA已应用，可训练参数数量: {self.model.print_trainable_parameters()}")
            
        except ImportError:
            print("警告: peft库未安装，将使用全量微调")
            self.use_lora = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            labels: 标签（用于计算损失）
        
        返回:
            包含损失和logits的字典
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        生成响应
        
        参数:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            do_sample: 是否使用采样
        
        返回:
            生成的token ID
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.model.config.eos_token_id
            )
        
        return outputs
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算给定序列的对数概率
        
        参数:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
        
        返回:
            对数概率张量
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 计算对数概率
        logits = outputs.logits[:, :-1, :]  # 移除最后一个位置
        labels = input_ids[:, 1:]  # 移除第一个位置
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # 获取对应token的对数概率
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return gathered_log_probs


class CORYDualAgent(nn.Module):
    """
    CORY双智能体系统
    实现Pioneer（先驱者）和Observer（观察者）的协作机制
    
    核心思想：
    - Pioneer: 主动探索，生成初始响应
    - Observer: 观察Pioneer的输出，提供改进建议
    - 两者通过序贯决策进行协作优化
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
        share_backbone: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        初始化双智能体系统
        
        参数:
            model_name: 预训练模型名称
            device: 计算设备
            use_lora: 是否使用LoRA
            lora_config: LoRA配置
            share_backbone: 是否共享骨干网络
            load_in_8bit: 是否使用8位量化
            load_in_4bit: 是否使用4位量化
        """
        super().__init__()
        
        self.device = device
        self.share_backbone = share_backbone
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 创建Pioneer智能体
        print("正在初始化Pioneer智能体...")
        self.pioneer = CORYAgent(
            model_name=model_name,
            device=device,
            use_lora=use_lora,
            lora_config=lora_config,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        
        # 创建Observer智能体
        # 如果共享骨干网络，Observer使用相同的模型但不同的LoRA参数
        if share_backbone and use_lora:
            print("正在初始化Observer智能体（共享骨干）...")
            self.observer = CORYAgent(
                model_name=model_name,
                device=device,
                use_lora=use_lora,
                lora_config=lora_config,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit
            )
        else:
            print("正在初始化Observer智能体（独立模型）...")
            self.observer = CORYAgent(
                model_name=model_name,
                device=device,
                use_lora=use_lora,
                lora_config=lora_config,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit
            )
        
        # 创建参考模型（用于KL散度计算）
        print("正在初始化参考模型...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        if device != "cuda" or not torch.cuda.is_available():
            self.ref_model = self.ref_model.to(device)
    
    def pioneer_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Pioneer智能体生成初始响应
        
        参数:
            prompts: 输入提示列表
            max_new_tokens: 最大生成token数
        
        返回:
            生成的响应列表和对应的token ID
        """
        # 编码输入
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        # Pioneer生成
        output_ids = self.pioneer.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens
        )
        
        # 解码输出
        responses = self.tokenizer.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return responses, output_ids
    
    def observer_refine(
        self,
        prompts: List[str],
        pioneer_responses: List[str],
        max_new_tokens: int = 128
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Observer智能体基于Pioneer的输出进行改进
        
        参数:
            prompts: 原始提示列表
            pioneer_responses: Pioneer的响应列表
            max_new_tokens: 最大生成token数
        
        返回:
            改进后的响应列表和对应的token ID
        """
        # 构建Observer的输入（包含Pioneer的响应作为上下文）
        observer_prompts = []
        for prompt, pioneer_resp in zip(prompts, pioneer_responses):
            observer_prompt = f"{prompt}\n\n初始回答: {pioneer_resp}\n\n请基于以上回答进行改进和补充:\n"
            observer_prompts.append(observer_prompt)
        
        # 编码输入
        inputs = self.tokenizer(
            observer_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384
        ).to(self.device)
        
        # Observer生成改进响应
        output_ids = self.observer.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens
        )
        
        # 解码输出
        refined_responses = self.tokenizer.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return refined_responses, output_ids
    
    def sequential_generation(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        num_iterations: int = 1
    ) -> Dict[str, List[str]]:
        """
        序贯协作生成
        Pioneer和Observer交替生成并相互改进
        
        参数:
            prompts: 输入提示列表
            max_new_tokens: 最大生成token数
            num_iterations: 协作迭代次数
        
        返回:
            包含各阶段生成结果的字典
        """
        results = {
            "prompts": prompts,
            "pioneer_responses": [],
            "observer_responses": [],
            "final_responses": []
        }
        
        # Pioneer生成初始响应
        pioneer_responses, _ = self.pioneer_generate(prompts, max_new_tokens)
        results["pioneer_responses"] = pioneer_responses
        
        current_responses = pioneer_responses
        
        # 序贯协作迭代
        for i in range(num_iterations):
            # Observer改进
            observer_responses, _ = self.observer_refine(
                prompts, current_responses, max_new_tokens
            )
            results["observer_responses"].append(observer_responses)
            current_responses = observer_responses
        
        results["final_responses"] = current_responses
        
        return results
    
    def compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        agent: str = "pioneer"
    ) -> torch.Tensor:
        """
        计算当前策略与参考策略之间的KL散度
        
        参数:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            agent: 使用哪个智能体（"pioneer"或"observer"）
        
        返回:
            KL散度值
        """
        # 选择智能体
        current_agent = self.pioneer if agent == "pioneer" else self.observer
        
        # 获取当前策略的logits
        with torch.no_grad():
            current_outputs = current_agent.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            current_logits = current_outputs.logits
            
            # 获取参考策略的logits
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            ref_logits = ref_outputs.logits
        
        # 计算概率分布
        current_probs = torch.nn.functional.softmax(current_logits, dim=-1)
        ref_probs = torch.nn.functional.softmax(ref_logits, dim=-1)
        
        # 计算KL散度
        kl_div = torch.nn.functional.kl_div(
            current_probs.log(),
            ref_probs,
            reduction="batchmean"
        )
        
        return kl_div
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        获取所有可训练参数
        
        返回:
            可训练参数列表
        """
        params = []
        params.extend([p for p in self.pioneer.parameters() if p.requires_grad])
        params.extend([p for p in self.observer.parameters() if p.requires_grad])
        return params

    def swap_roles(self) -> None:
        """
        角色交换（Role Swap）- CORY论文核心机制
        
        交换Pioneer和Observer的LoRA参数，使两个智能体互换角色。
        这是CORY的关键创新：通过周期性角色交换，两个智能体可以
        从对方的角度学习，避免陷入固定模式，促进更好的协作。
        
        原理:
        - Pioneer学习如何生成好的初始响应
        - Observer学习如何改进响应
        - 通过交换，Pioneer可以学习Observer的改进能力
        - Observer可以学习Pioneer的探索能力
        - 最终两者都能胜任两种角色，提升整体性能
        """
        print("执行角色交换: Pioneer <-> Observer")
        
        # 获取Pioneer和Observer的状态字典
        pioneer_state = self.pioneer.model.state_dict()
        observer_state = self.observer.model.state_dict()
        
        # 交换参数
        self.pioneer.model.load_state_dict(observer_state)
        self.observer.model.load_state_dict(pioneer_state)
        
        print("角色交换完成")
    
    def swap_lora_parameters(self) -> None:
        """
        仅交换LoRA参数（更轻量的角色交换）
        
        只交换LoRA适配器的参数，保持基础模型不变。
        这种方式更高效，因为LoRA参数量远小于完整模型。
        """
        print("执行LoRA参数交换: Pioneer <-> Observer")
        
        try:
            # 获取LoRA参数名称（包含"lora"的参数）
            pioneer_lora_state = {}
            observer_lora_state = {}
            
            for name, param in self.pioneer.model.named_parameters():
                if "lora" in name.lower():
                    pioneer_lora_state[name] = param.data.clone()
            
            for name, param in self.observer.model.named_parameters():
                if "lora" in name.lower():
                    observer_lora_state[name] = param.data.clone()
            
            # 交换LoRA参数
            for name, param in self.pioneer.model.named_parameters():
                if name in observer_lora_state:
                    param.data.copy_(observer_lora_state[name])
            
            for name, param in self.observer.model.named_parameters():
                if name in pioneer_lora_state:
                    param.data.copy_(pioneer_lora_state[name])
            
            print(f"LoRA参数交换完成，交换了 {len(pioneer_lora_state)} 个参数")
            
        except Exception as e:
            print(f"LoRA参数交换失败，回退到完整参数交换: {e}")
            self.swap_roles()
    
    def soft_role_swap(self, interpolation_factor: float = 0.5) -> None:
        """
        软角色交换（Soft Role Swap）
        
        不是完全交换参数，而是将两个智能体的参数进行插值混合。
        这种方式更加平滑，可以逐渐让两个智能体互相学习。
        
        参数:
            interpolation_factor: 插值因子，0.5表示完全平均
                - 0.0: 保持原样，不交换
                - 0.5: 两者参数取平均
                - 1.0: 完全交换
        """
        print(f"执行软角色交换，插值因子: {interpolation_factor}")
        
        pioneer_state = self.pioneer.model.state_dict()
        observer_state = self.observer.model.state_dict()
        
        # 计算插值后的参数
        new_pioneer_state = {}
        new_observer_state = {}
        
        for key in pioneer_state.keys():
            if key in observer_state:
                # Pioneer新参数 = (1-α)*Pioneer + α*Observer
                new_pioneer_state[key] = (
                    (1 - interpolation_factor) * pioneer_state[key] +
                    interpolation_factor * observer_state[key]
                )
                # Observer新参数 = α*Pioneer + (1-α)*Observer
                new_observer_state[key] = (
                    interpolation_factor * pioneer_state[key] +
                    (1 - interpolation_factor) * observer_state[key]
                )
            else:
                new_pioneer_state[key] = pioneer_state[key]
        
        for key in observer_state.keys():
            if key not in new_observer_state:
                new_observer_state[key] = observer_state[key]
        
        # 加载新参数
        self.pioneer.model.load_state_dict(new_pioneer_state)
        self.observer.model.load_state_dict(new_observer_state)
        
        print("软角色交换完成")
    
    def get_role_similarity(self) -> float:
        """
        计算两个智能体参数的相似度
        
        返回:
            余弦相似度，范围[-1, 1]，1表示完全相同
        """
        pioneer_params = []
        observer_params = []
        
        for (name_p, param_p), (name_o, param_o) in zip(
            self.pioneer.model.named_parameters(),
            self.observer.model.named_parameters()
        ):
            if "lora" in name_p.lower():
                pioneer_params.append(param_p.data.flatten())
                observer_params.append(param_o.data.flatten())
        
        if not pioneer_params:
            return 0.0
        
        pioneer_vec = torch.cat(pioneer_params)
        observer_vec = torch.cat(observer_params)
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            pioneer_vec.unsqueeze(0),
            observer_vec.unsqueeze(0)
        ).item()
        
        return similarity
