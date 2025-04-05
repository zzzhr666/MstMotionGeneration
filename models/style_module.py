import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class StyleModulator(nn.Module):
    """风格调制模块，将内容特征和风格特征融合"""
    
    def __init__(self, 
                 content_dim: int, 
                 style_dim: int,
                 hidden_dim: int = 256):
        """
        初始化风格调制模块
        
        Args:
            content_dim: 内容特征维度
            style_dim: 风格特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        
        # 投影层
        self.content_proj = nn.Linear(content_dim, hidden_dim)
        self.style_proj = nn.Linear(style_dim, hidden_dim)
        
        # 风格调制参数生成器
        self.gamma_generator = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, content_dim)
        )
        
        self.beta_generator = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, content_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, content_dim)
        )
    
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            content: 内容特征 [batch_size, content_dim]
            style: 风格特征 [batch_size, style_dim]

        Returns:
            torch.Tensor: 调制后的特征 [batch_size, content_dim]
        """
        # 生成缩放和偏移参数
        gamma = self.gamma_generator(style)  # [batch_size, content_dim]
        beta = self.beta_generator(style)    # [batch_size, content_dim]

        # 使用LayerNorm代替InstanceNorm
        layer_norm = nn.LayerNorm(content.size(-1), elementwise_affine=False).to(content.device)
        normalized_content = layer_norm(content)

        # 应用风格迁移
        modulated = gamma * normalized_content + beta

        # 投影到共同空间
        content_proj = self.content_proj(content)  # [batch_size, hidden_dim]
        style_proj = self.style_proj(style)        # [batch_size, hidden_dim]

        # 残差连接
        fused = content_proj + style_proj  # 简单相加融合
        fused = self.fusion(fused)  # [batch_size, content_dim]

        # 残差连接
        output = modulated + fused

        return output