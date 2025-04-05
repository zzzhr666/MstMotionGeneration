import numpy as np
from typing import Dict, List, Optional, Tuple, Union

def extract_velocity(positions: np.ndarray) -> np.ndarray:
    """
    计算运动序列的速度
    
    Args:
        positions: 位置序列 [seq_len, dim]
        
    Returns:
        velocity: 速度序列 [seq_len, dim]
    """
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]
    velocity[0] = velocity[1]  # 复制第二帧的速度给第一帧
    return velocity

def extract_acceleration(velocity: np.ndarray) -> np.ndarray:
    """
    计算运动序列的加速度
    
    Args:
        velocity: 速度序列 [seq_len, dim]
        
    Returns:
        acceleration: 加速度序列 [seq_len, dim]
    """
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    acceleration[0] = acceleration[1]  # 复制第二帧的加速度给第一帧
    return acceleration

def extract_motion_range(positions: np.ndarray) -> np.ndarray:
    """
    计算运动范围
    
    Args:
        positions: 位置序列 [seq_len, dim]
        
    Returns:
        range: 运动范围 [dim]
    """
    return np.max(positions, axis=0) - np.min(positions, axis=0)

def extract_foot_contacts(positions: np.ndarray, foot_joints: List[int], 
                          threshold: float = 0.05) -> np.ndarray:
    """
    估计脚部接触点
    
    Args:
        positions: 位置序列 [seq_len, n_joints * 3]
        foot_joints: 脚部关节的索引列表
        threshold: 接触判定的阈值
        
    Returns:
        contacts: 接触状态序列 [seq_len, len(foot_joints)]
    """
    seq_len = positions.shape[0]
    n_foot_joints = len(foot_joints)
    contacts = np.zeros((seq_len, n_foot_joints), dtype=np.float32)
    
    for i, joint_idx in enumerate(foot_joints):
        # 提取特定关节的y坐标(假设y是高度)
        joint_pos = positions[:, joint_idx * 3 + 1]  # Y坐标
        
        # 计算速度
        joint_vel = np.zeros_like(joint_pos)
        joint_vel[1:] = joint_pos[1:] - joint_pos[:-1]
        
        # 高度小于阈值且速度较小表示接触
        contacts[:, i] = (joint_pos < threshold) & (np.abs(joint_vel) < threshold)
    
    return contacts

def extract_style_features(motion_data: np.ndarray) -> Dict[str, float]:
    """
    提取运动的风格特征
    
    Args:
        motion_data: 运动数据 [seq_len, motion_dim]
        
    Returns:
        features: 包含各种风格特征的字典
    """
    # 计算速度和加速度
    velocity = extract_velocity(motion_data)
    acceleration = extract_acceleration(velocity)
    
    # 计算运动范围
    motion_range = extract_motion_range(motion_data)
    
    # 计算各种统计量
    avg_velocity = np.mean(np.abs(velocity))
    avg_acceleration = np.mean(np.abs(acceleration))
    max_velocity = np.max(np.abs(velocity))
    avg_range = np.mean(motion_range)
    
    # 估计运动节奏(使用速度变化的频率)
    zero_crossings = np.sum(np.diff(np.signbit(velocity)).astype(bool), axis=0)
    rhythm = np.mean(zero_crossings) / motion_data.shape[0]
    
    # 返回风格特征
    features = {
        'avg_velocity': float(avg_velocity),
        'avg_acceleration': float(avg_acceleration),
        'max_velocity': float(max_velocity),
        'avg_range': float(avg_range),
        'rhythm': float(rhythm)
    }
    
    return features

def normalize_features(features_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    标准化风格特征列表
    
    Args:
        features_list: 风格特征字典的列表
        
    Returns:
        normalized_features: 标准化后的风格特征字典列表
    """
    # 收集所有特征值
    all_features = {}
    for feature_dict in features_list:
        for key, value in feature_dict.items():
            if key not in all_features:
                all_features[key] = []
            all_features[key].append(value)
    
    # 计算均值和标准差
    stats = {}
    for key, values in all_features.items():
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values) if np.std(values) > 1e-6 else 1.0
        }
    
    # 标准化特征
    normalized_features = []
    for feature_dict in features_list:
        normalized_dict = {}
        for key, value in feature_dict.items():
            normalized_dict[key] = (value - stats[key]['mean']) / stats[key]['std']
        normalized_features.append(normalized_dict)
    
    return normalized_features

def compute_style_difference(features1: Dict[str, float], 
                             features2: Dict[str, float]) -> float:
    """
    计算两个风格特征之间的距离
    
    Args:
        features1: 第一个风格特征字典
        features2: 第二个风格特征字典
        
    Returns:
        distance: 风格差异度量
    """
    # 确保两个字典有相同的键
    common_keys = set(features1.keys()).intersection(set(features2.keys()))
    
    if not common_keys:
        return float('inf')
    
    # 计算欧几里得距离
    squared_diff_sum = 0.0
    for key in common_keys:
        squared_diff_sum += (features1[key] - features2[key]) ** 2
    
    return np.sqrt(squared_diff_sum / len(common_keys))