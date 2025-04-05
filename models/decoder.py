import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class MotionDecoder(nn.Module):
    """
    运动解码器，将潜在表示解码为运动序列
    """
    
    def __init__(self, 
                 latent_dim: int, 
                 output_dim: int,
                 hidden_dim: int = 256,
                 seq_len: int = 60,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        初始化运动解码器
        
        Args:
            latent_dim: 潜在表示维度
            output_dim: 输出维度(运动数据的特征数)
            hidden_dim: 隐藏层维度
            seq_len: 输出序列长度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 潜在向量投影层
        self.fc_init = nn.Linear(latent_dim, hidden_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出投影层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 潜在表示 [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: 重建的运动序列 [batch_size, seq_len, output_dim]
        """
        batch_size = z.shape[0]
        
        # 初始化隐藏状态
        h0 = self.fc_init(z).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim]
        c0 = torch.zeros_like(h0)  # [num_layers, batch_size, hidden_dim]
        
        # 创建输入序列 - 重复z
        x = self.fc_init(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM前向传播
        output, _ = self.lstm(x, (h0, c0))  # [batch_size, seq_len, hidden_dim]
        
        # 投影到输出空间
        motion = self.fc_out(output)  # [batch_size, seq_len, output_dim]
        
        return motion