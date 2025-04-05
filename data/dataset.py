import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import random

class MotionDataset(Dataset):
    """运动数据集类，用于训练和评估模型"""
    
    def __init__(self, 
                 data_list: List[Dict], 
                 seq_len: int = 60,
                 style_conditioning: bool = True,
                 motion_type_conditioning: bool = True):
        """
        初始化数据集
        
        Args:
            data_list: 预处理后的数据列表
            seq_len: 序列长度
            style_conditioning: 是否使用风格条件
            motion_type_conditioning: 是否使用动作类型条件
        """
        self.data_list = data_list
        self.seq_len = seq_len
        self.style_conditioning = style_conditioning
        self.motion_type_conditioning = motion_type_conditioning
        
        # 映射风格标签到索引
        self.age_groups = sorted(list(set([d['age_group'] for d in data_list if d['age_group'] != 'Unknown'])))
        self.genders = sorted(list(set([d['gender'] for d in data_list if d['gender'] != 'Unknown'])))
        self.motion_types = sorted(list(set([d['motion_type'] for d in data_list if d['motion_type'] != 'Unknown'])))
        
        self.age_to_idx = {age: i for i, age in enumerate(self.age_groups)}
        self.gender_to_idx = {gender: i for i, gender in enumerate(self.genders)}
        self.motion_to_idx = {motion: i for i, motion in enumerate(self.motion_types)}
        
        # 创建样本索引，将数据划分为固定长度序列
        self.indices = []
        for i, data in enumerate(data_list):
            motion = data['normalized_motion']
            n_frames = motion.shape[0]
            
            if n_frames >= seq_len:
                # 对于长序列，创建多个重叠片段
                for start in range(0, n_frames - seq_len + 1, seq_len // 2):
                    self.indices.append((i, start, start + seq_len))
            else:
                # 对于短序列，填充至所需长度
                self.indices.append((i, 0, n_frames))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_idx, start_frame, end_frame = self.indices[idx]
        data = self.data_list[data_idx]
        
        # 提取运动序列
        motion = data['normalized_motion'][start_frame:end_frame]
        
        # 处理序列长度
        if motion.shape[0] < self.seq_len:
            # 填充短序列
            pad_len = self.seq_len - motion.shape[0]
            padding = np.zeros((pad_len, motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        # 转换为张量
        motion_tensor = torch.from_numpy(motion).float()
        
        # 创建风格标签(one-hot编码)
        age_label = torch.zeros(len(self.age_groups))
        if data['age_group'] in self.age_to_idx:
            age_idx = self.age_to_idx[data['age_group']]
            age_label[age_idx] = 1.0
        
        gender_label = torch.zeros(len(self.genders))
        if data['gender'] in self.gender_to_idx:
            gender_idx = self.gender_to_idx[data['gender']]
            gender_label[gender_idx] = 1.0
        
        # 创建动作类型标签(one-hot编码)
        motion_label = torch.zeros(len(self.motion_types))
        if data['motion_type'] in self.motion_to_idx:
            motion_idx = self.motion_to_idx[data['motion_type']]
            motion_label[motion_idx] = 1.0
        
        result = {
            'motion': motion_tensor,
            'age_label': age_label,
            'gender_label': gender_label,
            'motion_label': motion_label,
            'age_group': data['age_group'],
            'gender': data['gender'],
            'motion_type': data['motion_type']
        }
        
        return result
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=4,
            pin_memory=True
        )
    
    def get_style_dims(self) -> Tuple[int, int]:
        """返回风格标签的维度"""
        return len(self.age_groups), len(self.genders)
    
    def get_motion_type_dims(self) -> int:
        """返回动作类型标签的维度"""
        return len(self.motion_types)


class PairedMotionDataset(Dataset):
    """
    成对运动数据集，用于风格迁移训练
    创建内容运动和风格运动的配对
    """
    
    def __init__(self, dataset: MotionDataset, pairs_per_sample: int = 4):
        """
        初始化成对数据集
        
        Args:
            dataset: 基础运动数据集
            pairs_per_sample: 每个样本生成的配对数量
        """
        self.dataset = dataset
        self.pairs_per_sample = pairs_per_sample
        
        # 按风格和动作类型对数据集进行索引
        self.style_indices = {}
        self.motion_type_indices = {}
        
        for i in range(len(dataset)):
            sample = dataset[i]
            
            style_key = (sample['age_group'], sample['gender'])
            if style_key not in self.style_indices:
                self.style_indices[style_key] = []
            self.style_indices[style_key].append(i)
            
            motion_key = sample['motion_type']
            if motion_key not in self.motion_type_indices:
                self.motion_type_indices[motion_key] = []
            self.motion_type_indices[motion_key].append(i)
    
    def __len__(self) -> int:
        return len(self.dataset) * self.pairs_per_sample
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 确定内容样本索引
        content_idx = idx // self.pairs_per_sample
        content_sample = self.dataset[content_idx]
        
        # 选择风格样本 - 随机选择不同风格的样本
        content_style = (content_sample['age_group'], content_sample['gender'])
        content_motion = content_sample['motion_type']
        
        # 找到相同动作类型但不同风格的样本
        valid_styles = [s for s in self.style_indices.keys() if s != content_style]
        if not valid_styles:  # 如果没有其他风格，使用相同风格
            valid_styles = [content_style]
        
        target_style = random.choice(valid_styles)
        style_candidates = self.style_indices[target_style]
        
        # 尝试找到相同动作类型的样本
        same_motion_candidates = [i for i in style_candidates 
                                if self.dataset[i]['motion_type'] == content_motion]
        
        if same_motion_candidates:
            style_idx = random.choice(same_motion_candidates)
        else:
            # 如果没有相同动作类型，则随机选择
            style_idx = random.choice(style_candidates)
        
        style_sample = self.dataset[style_idx]
        
        # 创建配对样本
        result = {
            'content_motion': content_sample['motion'],
            'style_motion': style_sample['motion'],
            'content_age_label': content_sample['age_label'],
            'content_gender_label': content_sample['gender_label'],
            'style_age_label': style_sample['age_label'],
            'style_gender_label': style_sample['gender_label'],
            'motion_label': content_sample['motion_label'],
            'content_age_group': content_sample['age_group'],
            'content_gender': content_sample['gender'],
            'style_age_group': style_sample['age_group'],
            'style_gender': style_sample['gender'],
            'motion_type': content_sample['motion_type']
        }
        
        return result