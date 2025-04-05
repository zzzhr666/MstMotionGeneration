import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class MotionEncoder(nn.Module):
    """运动编码器，提取运动序列的潜在表示"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 256, 
                 latent_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        初始化运动编码器
        
        Args:
            input_dim: 输入维度(运动数据的特征数)
            hidden_dim: 隐藏层维度
            latent_dim: 潜在表示维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 时间卷积层：捕获局部时间模式
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # 双向LSTM：捕获长期时间依赖
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 自注意力层：捕获序列中的关键部分
        self.attention = TemporalSelfAttention(hidden_dim * 2)  # 双向LSTM输出维度是hidden_dim*2
        
        # 输出投影层
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入运动序列 [batch_size, seq_len, input_dim]
            
        Returns:
            tuple: (mu, logvar, z)
                mu: 均值向量 [batch_size, latent_dim]
                logvar: 对数方差向量 [batch_size, latent_dim]
                z: 采样的潜在向量 [batch_size, latent_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 时间卷积处理
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM处理
        x, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # 自注意力处理
        x, attention_weights = self.attention(x)  # [batch_size, hidden_dim*2]
        
        # 计算均值和对数方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # 重参数化技巧采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return mu, logvar, z


class StyleEncoder(nn.Module):
    """风格编码器，提取运动序列的风格特征"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 256, 
                 style_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        初始化风格编码器
        
        Args:
            input_dim: 输入维度(运动数据的特征数)
            hidden_dim: 隐藏层维度
            style_dim: 风格特征维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        
        # 时间卷积层：捕获局部时间模式
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # LSTM：捕获长期时间依赖
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全局平均池化 + 最大池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        fc_input_dim = hidden_dim * 2  # 双向LSTM
        self.fc1 = nn.Linear(fc_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, style_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入运动序列 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 风格特征 [batch_size, style_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 时间卷积处理
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM处理
        x, (h_n, _) = self.lstm(x)  # x: [batch_size, seq_len, hidden_dim*2]
        
        # 获取最后一层的隐藏状态
        h_n = h_n.view(2, -1, batch_size, self.hidden_dim)  # [num_directions, num_layers, batch_size, hidden_dim]
        h_n = h_n[-2:, -1]  # 取最后一层 [2, batch_size, hidden_dim]
        h_n = h_n.permute(1, 0, 2).contiguous()  # [batch_size, 2, hidden_dim]
        h_n = h_n.view(batch_size, -1)  # [batch_size, hidden_dim*2]
        
        # 全连接层处理
        x = F.relu(self.fc1(h_n))
        x = self.dropout(x)
        style = self.fc2(x)
        
        return style


class TemporalSelfAttention(nn.Module):
    """时间自注意力机制，捕获序列中的关键时间步"""
    
    def __init__(self, input_dim: int, attention_dim: int = 128):
        """
        初始化时间自注意力模块
        
        Args:
            input_dim: 输入特征维度
            attention_dim: 注意力机制的内部维度
        """
        super().__init__()
        self.attention_dim = attention_dim
        
        # 注意力查询、键、值投影
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        
        # 输出投影
        self.fc_out = nn.Linear(attention_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            tuple: (context, attention_weights)
                context: 注意力加权的上下文向量 [batch_size, input_dim]
                attention_weights: 注意力权重 [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算查询、键、值
        q = self.query(x)  # [batch_size, seq_len, attention_dim]
        k = self.key(x)    # [batch_size, seq_len, attention_dim]
        v = self.value(x)  # [batch_size, seq_len, attention_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attention_dim ** 0.5)  # [batch_size, seq_len, seq_len]
        
        # 应用softmax获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # 计算加权和
        context = torch.matmul(attention_weights, v)  # [batch_size, seq_len, attention_dim]
        
        # 输出投影
        context = self.fc_out(context)  # [batch_size, seq_len, input_dim]
        
        # 计算序列的全局表示 - 取平均
        context_avg = torch.mean(context, dim=1)  # [batch_size, input_dim]
        
        # 计算全局注意力权重
        global_weights = torch.mean(attention_weights, dim=1)  # [batch_size, seq_len]
        
        return context_avg, global_weights