import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union

def compute_reconstruction_error(original: Union[np.ndarray, torch.Tensor], 
                                reconstructed: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算重建误差 (MSE)
    
    Args:
        original: 原始运动序列
        reconstructed: 重建的运动序列
        
    Returns:
        error: 重建误差
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    return float(np.mean((original - reconstructed) ** 2))

def compute_style_retention(reference_style: Union[np.ndarray, torch.Tensor],
                           transferred_style: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算风格保留度量
    
    Args:
        reference_style: 参考风格特征
        transferred_style: 迁移后的风格特征
        
    Returns:
        retention: 风格保留度量 (0-1, 越高越好)
    """
    if isinstance(reference_style, torch.Tensor):
        reference_style = reference_style.detach().cpu().numpy()
    if isinstance(transferred_style, torch.Tensor):
        transferred_style = transferred_style.detach().cpu().numpy()
    
    # 计算余弦相似度
    dot_product = np.sum(reference_style * transferred_style)
    norm_product = np.linalg.norm(reference_style) * np.linalg.norm(transferred_style)
    
    if norm_product < 1e-8:
        return 0.0
    
    similarity = dot_product / norm_product
    # 将相似度映射到0-1范围
    retention = (similarity + 1) / 2
    
    return float(retention)

def compute_content_preservation(original_content: Union[np.ndarray, torch.Tensor],
                                transferred_content: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算内容保留度量
    
    Args:
        original_content: 原始内容特征
        transferred_content: 迁移后的内容特征
        
    Returns:
        preservation: 内容保留度量 (0-1, 越高越好)
    """
    if isinstance(original_content, torch.Tensor):
        original_content = original_content.detach().cpu().numpy()
    if isinstance(transferred_content, torch.Tensor):
        transferred_content = transferred_content.detach().cpu().numpy()
    
    # 计算余弦相似度
    dot_product = np.sum(original_content * transferred_content)
    norm_product = np.linalg.norm(original_content) * np.linalg.norm(transferred_content)
    
    if norm_product < 1e-8:
        return 0.0
    
    similarity = dot_product / norm_product
    # 将相似度映射到0-1范围
    preservation = (similarity + 1) / 2
    
    return float(preservation)

def compute_jerk(motion: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算运动的抖动度(jerk)，衡量运动的平滑性
    
    Args:
        motion: 运动序列 [seq_len, motion_dim]
        
    Returns:
        jerk: 抖动度量
    """
    if isinstance(motion, torch.Tensor):
        motion = motion.detach().cpu().numpy()
    
    # 计算速度、加速度和抖动
    velocity = np.diff(motion, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    
    # 计算平均抖动大小
    mean_jerk = np.mean(np.abs(jerk))
    
    return float(mean_jerk)

def compute_foot_sliding(original_motion: Union[np.ndarray, torch.Tensor],
                        transferred_motion: Union[np.ndarray, torch.Tensor],
                        foot_joint_indices: List[int]) -> float:
    """
    计算脚步滑动度量
    
    Args:
        original_motion: 原始运动序列 [seq_len, motion_dim]
        transferred_motion: 迁移后的运动序列 [seq_len, motion_dim] 
        foot_joint_indices: 脚部关节的索引列表
        
    Returns:
        sliding: 脚步滑动度量 (越低越好)
    """
    if isinstance(original_motion, torch.Tensor):
        original_motion = original_motion.detach().cpu().numpy()
    if isinstance(transferred_motion, torch.Tensor):
        transferred_motion = transferred_motion.detach().cpu().numpy()
    
    # 提取脚部关节位置
    foot_pos_original = []
    foot_pos_transferred = []
    
    for idx in foot_joint_indices:
        # 假设每个关节有3个坐标 (x, y, z)
        joint_idx = idx * 3
        foot_pos_original.append(original_motion[:, joint_idx:joint_idx+3])
        foot_pos_transferred.append(transferred_motion[:, joint_idx:joint_idx+3])
    
    # 计算脚部速度
    foot_vel_original = []
    foot_vel_transferred = []
    
    for pos in foot_pos_original:
        foot_vel_original.append(np.diff(pos, axis=0))
    
    for pos in foot_pos_transferred:
        foot_vel_transferred.append(np.diff(pos, axis=0))
    
    # 计算滑动差异
    sliding_diff = 0.0
    for i in range(len(foot_vel_original)):
        # 计算脚部应该静止但在迁移后运动的情况
        # 假设速度小于阈值表示静止
        threshold = 0.01
        original_static = np.linalg.norm(foot_vel_original[i], axis=1) < threshold
        transferred_vel = np.linalg.norm(foot_vel_transferred[i], axis=1)
        
        # 计算在原始静止但迁移后运动的帧的平均速度
        sliding = np.mean(transferred_vel[original_static]) if np.any(original_static) else 0.0
        sliding_diff += sliding
    
    # 平均每个脚部关节的滑动差异
    if foot_joint_indices:
        sliding_diff /= len(foot_joint_indices)
    
    return float(sliding_diff)

def compute_metrics(outputs: Dict[str, torch.Tensor], 
                   content_motion: torch.Tensor, 
                   style_motion: torch.Tensor) -> Dict[str, float]:
    """
    计算各种评估指标
    
    Args:
        outputs: 模型输出字典
        content_motion: 内容运动序列
        style_motion: 风格运动序列
        
    Returns:
        metrics: 指标字典
    """
    # 提取关键输出
    reconstructed_motion = outputs['reconstructed_motion']
    transferred_motion = outputs['transferred_motion']
    content_z = outputs['content_z']
    transferred_z = outputs['transferred_z']
    
    # 计算各种指标
    metrics = {
        'reconstruction_error': compute_reconstruction_error(content_motion, reconstructed_motion),
        'content_preservation': compute_content_preservation(content_z, transferred_z),
        'jerk': compute_jerk(transferred_motion),
    }
    
    return metrics