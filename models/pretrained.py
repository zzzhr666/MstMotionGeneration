import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List, Union
import os
import numpy as np

# 导入MDM适配器
from .mdm_adapter import MDMAdapter

class PretrainedMotionGenerator(nn.Module):
    """
    封装预训练运动生成模型，提供简化接口
    支持简化版MDM或真实MDM
    """
    
    def __init__(self, 
                 motion_types: List[str],
                 output_dim: int,
                 latent_dim: int = 128,
                 hidden_dim: int = 512,
                 seq_len: int = 60,
                 use_pretrained: bool = True,
                 pretrained_type: str = 'simplified',  # 'simplified' 或 'mdm'
                 pretrained_path: Optional[str] = None):
        """
        初始化预训练运动生成模型
        
        Args:
            motion_types: 支持的动作类型列表
            output_dim: 输出运动数据的维度
            latent_dim: 潜在向量维度
            hidden_dim: 隐藏层维度
            seq_len: 序列长度
            use_pretrained: 是否使用预训练模型
            pretrained_type: 预训练模型类型 ('simplified' 或 'mdm')
            pretrained_path: 预训练模型路径
        """
        super().__init__()
        self.motion_types = motion_types
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pretrained_type = pretrained_type
        
        self.use_simplified_mdm(motion_types, output_dim, hidden_dim, seq_len)
       
    
    def use_simplified_mdm(self, motion_types, output_dim, hidden_dim, seq_len):
        """配置使用简化的MDM"""
        # 动作类型嵌入
        self.motion_type_embedding = nn.Embedding(
            num_embeddings=len(motion_types),
            embedding_dim=hidden_dim
        )
        
        # 基础扩散模型 (简化版MDM)
        self.diffusion_model = SimplifiedMDM(
            input_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len
        )
        
        self.is_real_mdm = False
        
    def use_real_mdm(self, pretrained_path, motion_types):
        """配置使用真实的MDM"""
        # 创建MDM适配器
        self.mdm_adapter = MDMAdapter(
            model_path=pretrained_path,
            num_frames=self.seq_len
        )
        
        # 设置支持的动作类型
        self.mdm_adapter.setup_motion_types(motion_types)
        
        self.is_real_mdm = True
    
    def forward(self, motion_type_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成指定类型的运动序列
        
        Args:
            motion_type_indices: 动作类型索引 [batch_size]
            
        Returns:
            tuple: (motion_sequences, latent_vectors)
                motion_sequences: 生成的运动序列 [batch_size, seq_len, output_dim]
                latent_vectors: 潜在向量 [batch_size, latent_dim]
        """
        batch_size = motion_type_indices.shape[0]
        
        if self.is_real_mdm:
            # 使用真实MDM生成运动
            with torch.no_grad():
                motion_sequences, latent_vectors = self.mdm_adapter.generate(
                    [self.motion_types[i] for i in motion_type_indices],
                    num_samples=batch_size
                )
        else:
            # 使用简化MDM生成运动
            # 获取动作类型嵌入
            motion_embeds = self.motion_type_embedding(motion_type_indices)  # [batch_size, hidden_dim]
            
            # 生成运动序列
            motion_sequences = self.diffusion_model(motion_embeds)  # [batch_size, seq_len, output_dim]
            
            # 生成潜在向量表示
            latent_vectors = self.latent_proj(motion_embeds)  # [batch_size, latent_dim]
        
        return motion_sequences, latent_vectors
    
    def generate(self, 
                motion_types: Union[str, List[str]], 
                device: torch.device = torch.device('cpu')) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成指定类型的运动序列
        
        Args:
            motion_types: 动作类型或类型列表
            device: 计算设备
            
        Returns:
            tuple: (motion_sequences, latent_vectors)
                motion_sequences: 生成的运动序列 [batch_size, seq_len, output_dim]
                latent_vectors: 潜在向量 [batch_size, latent_dim]
        """
        if isinstance(motion_types, str):
            motion_types = [motion_types]
        
        batch_size = len(motion_types)
        indices = []
        
        for motion_type in motion_types:
            if motion_type in self.motion_types:
                index = self.motion_types.index(motion_type)
            else:
                # 如果不支持请求的动作类型，使用随机类型
                index = np.random.randint(0, len(self.motion_types))
                print(f"Warning: Motion type {motion_type} not supported. Using random type instead.")
            indices.append(index)
        
        motion_type_indices = torch.tensor(indices, device=device)
        
        # 生成运动序列
        self.eval()
        with torch.no_grad():
            motion_sequences, latent_vectors = self.forward(motion_type_indices)
        
        return motion_sequences.cpu().numpy(), latent_vectors.cpu().numpy()


class SimplifiedMDM(nn.Module):
    """
    简化版的Motion Diffusion Model (MDM)
    用于生成运动序列
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 hidden_dim: int = 512,
                 seq_len: int = 60,
                 num_layers: int = 4):
        """
        初始化简化版MDM
        
        Args:
            input_dim: 输入条件维度
            output_dim: 输出运动数据的维度
            hidden_dim: 隐藏层维度
            seq_len: 序列长度
            num_layers: Transformer层数
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 将条件投影到序列初始状态
        self.condition_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 时间步嵌入
        self.time_embedding = nn.Embedding(seq_len, hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            condition: 条件嵌入 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 生成的运动序列 [batch_size, seq_len, output_dim]
        """
        batch_size = condition.shape[0]
        
        # 投影条件到序列初始状态
        initial_state = self.condition_proj(condition)  # [batch_size, hidden_dim]
        
        # 创建输入序列 - 重复初始状态
        x = initial_state.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 添加时间步嵌入
        time_indices = torch.arange(self.seq_len, device=condition.device)
        time_embeddings = self.time_embedding(time_indices)  # [seq_len, hidden_dim]
        time_embeddings = time_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, seq_len, hidden_dim]
        
        x = x + time_embeddings  # 添加时间位置编码
        
        # Transformer编码
        mask = None  # 允许全局注意力
        x = self.transformer(x, mask)  # [batch_size, seq_len, hidden_dim]
        
        # 投影到输出空间
        motion = self.output_proj(x)  # [batch_size, seq_len, output_dim]
        
        return motion