import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List, Any

from .encoder import MotionEncoder, StyleEncoder
from .decoder import MotionDecoder
from .style_module import StyleModulator
from .pretrained import PretrainedMotionGenerator

class STMModel(nn.Module):
    """
    完整的风格迁移模型(STM)，结合内容编码器、风格编码器、风格调制器和解码器
    """
    
    def __init__(self, 
                 motion_dim: int,
                 latent_dim: int = 128,
                 style_dim: int = 64,
                 hidden_dim: int = 256,
                 seq_len: int = 60,
                 dropout: float = 0.1,
                 use_pretrained: bool = True,
                 motion_types: Optional[List[str]] = None,
                 pretrained_path: Optional[str] = None,**kwargs):
        """
        初始化风格迁移模型
        
        Args:
            motion_dim: 运动数据的特征维度
            latent_dim: 潜在表示维度
            style_dim: 风格特征维度
            hidden_dim: 隐藏层维度
            seq_len: 序列长度
            dropout: Dropout比率
            use_pretrained: 是否使用预训练运动生成模型
            motion_types: 支持的动作类型列表
            pretrained_path: 预训练模型路径
        """
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 内容编码器
        self.content_encoder = MotionEncoder(
            input_dim=motion_dim, 
            hidden_dim=hidden_dim, 
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # 风格编码器
        self.style_encoder = StyleEncoder(
            input_dim=motion_dim, 
            hidden_dim=hidden_dim, 
            style_dim=style_dim,
            dropout=dropout
        )
        
        # 风格调制器
        self.style_modulator = StyleModulator(
            content_dim=latent_dim,
            style_dim=style_dim, 
            hidden_dim=hidden_dim
        )
        
        # 解码器
        self.decoder = MotionDecoder(
            latent_dim=latent_dim, 
            output_dim=motion_dim, 
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            dropout=dropout
        )
        
        # 预训练运动生成模型(可选)
        self.use_pretrained = use_pretrained
        if use_pretrained:
            pretrained_type = kwargs.get('pretrained_type', 'mdm')
            print(f"Using pretrained type: {pretrained_type}")
            print(f"Loading pretrained model from: {pretrained_path}")
            self.motion_generator = PretrainedMotionGenerator(
                motion_types=motion_types,
                output_dim=motion_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                use_pretrained=use_pretrained,
                pretrained_type=pretrained_type,
                pretrained_path=pretrained_path
            )
        else:
            self.motion_generator = None
    
    def encode_content(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码内容特征"""
        return self.content_encoder(x)
    
    def encode_style(self, x: torch.Tensor) -> torch.Tensor:
        """编码风格特征"""
        return self.style_encoder(x)
    
    def transfer_style(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """迁移风格"""
        return self.style_modulator(content, style)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码运动序列"""
        return self.decoder(z)
    
    def forward(self, 
                content_motion: torch.Tensor, 
                style_motion: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 风格迁移
        
        Args:
            content_motion: 内容运动序列 [batch_size, seq_len, motion_dim]
            style_motion: 风格运动序列 [batch_size, seq_len, motion_dim]
            
        Returns:
            dict: 包含各种中间结果和输出的字典
        """
        # 编码内容
        content_mu, content_logvar, content_z = self.encode_content(content_motion)
        
        # 编码风格
        style_z = self.encode_style(style_motion)
        
        # 迁移风格
        transferred_z = self.transfer_style(content_z, style_z)
        
        # 解码生成风格化运动
        transferred_motion = self.decode(transferred_z)
        
        # 重建原始内容(用于训练)
        reconstructed_motion = self.decode(content_z)
        
        return {
            'content_mu': content_mu,
            'content_logvar': content_logvar,
            'content_z': content_z,
            'style_z': style_z,
            'transferred_z': transferred_z,
            'transferred_motion': transferred_motion,
            'reconstructed_motion': reconstructed_motion
        }
    
    def generate_and_transfer(self, 
                             motion_type_indices: torch.Tensor,
                             style_motion: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        使用预训练模型生成内容运动并迁移风格
        
        Args:
            motion_type_indices: 动作类型索引 [batch_size]
            style_motion: 风格运动序列 [batch_size, seq_len, motion_dim]
            
        Returns:
            dict: 包含生成和迁移结果的字典
        """
        if not self.use_pretrained or self.motion_generator is None:
            raise ValueError("Pretrained motion generator is not available")
        
        # 生成内容运动
        content_motion, content_z = self.motion_generator(motion_type_indices)
        
        # 编码风格
        style_z = self.encode_style(style_motion)
        
        # 迁移风格
        transferred_z = self.transfer_style(content_z, style_z)
        
        # 解码生成风格化运动
        transferred_motion = self.decode(transferred_z)
        
        return {
            'content_motion': content_motion,
            'content_z': content_z,
            'style_z': style_z,
            'transferred_z': transferred_z,
            'transferred_motion': transferred_motion
        }
    
    def generate_from_style_params(self,
                                  motion_type_indices: torch.Tensor,
                                  age_group: str,
                                  gender: str,
                                  age_mapper: Dict[str, int],
                                  gender_mapper: Dict[str, int]) -> torch.Tensor:
        """
        根据风格参数生成风格化运动
        
        Args:
            motion_type_indices: 动作类型索引 [batch_size]
            age_group: 年龄组
            gender: 性别
            age_mapper: 年龄组到索引的映射
            gender_mapper: 性别到索引的映射
            
        Returns:
            torch.Tensor: 生成的风格化运动 [batch_size, seq_len, motion_dim]
        """
        raise NotImplementedError("To be implemented in future versions")