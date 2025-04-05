import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import torch

class BVHProcessor:
    """BVH文件处理器，负责读取和预处理BVH文件"""
    
    def __init__(self, root_dir: str, fps: int = 30):
        """
        初始化BVH处理器
        
        Args:
            root_dir: BVH文件根目录
            fps: 帧率，默认30fps
        """
        self.root_dir = root_dir
        self.fps = fps
        self.joint_names = []
        self.age_groups = ['Child', 'Youth', 'Old']
        self.genders = ['Male', 'Female']
        
    def parse_bvh(self, file_path: str) -> Tuple[Dict, np.ndarray]:
        """解析单个BVH文件，提取骨架结构和运动数据"""
        with open(file_path, 'r') as f:
            content = f.read()
            
        # 分离骨架结构和运动数据
        hierarchy_part = content.split('MOTION')[0]
        motion_part = 'MOTION' + content.split('MOTION')[1]
        
        # 解析骨架结构
        joints = {}
        joint_stack = []
        
        for line in hierarchy_part.split('\n'):
            line = line.strip()
            
            if 'ROOT' in line:
                joint_name = line.split('ROOT')[-1].strip()
                joints[joint_name] = {'parent': None, 'children': [], 'offset': None, 'channels': []}
                joint_stack.append(joint_name)
                if not self.joint_names:
                    self.joint_names.append(joint_name)
                    
            elif 'JOINT' in line:
                joint_name = line.split('JOINT')[-1].strip()
                if joint_stack:
                    parent = joint_stack[-1]
                    joints[joint_name] = {'parent': parent, 'children': [], 'offset': None, 'channels': []}
                    joints[parent]['children'].append(joint_name)
                    joint_stack.append(joint_name)
                    if joint_name not in self.joint_names:
                        self.joint_names.append(joint_name)
            
            elif '}' in line and joint_stack:
                joint_stack.pop()
                
            elif 'OFFSET' in line and joint_stack:
                offset = [float(x) for x in line.split('OFFSET')[-1].strip().split()]
                joints[joint_stack[-1]]['offset'] = offset
                
            elif 'CHANNELS' in line and joint_stack:
                channels = line.split('CHANNELS')[-1].strip().split()
                channels_count = int(channels[0])
                channels_names = channels[1:1+channels_count]
                joints[joint_stack[-1]]['channels'] = channels_names
        
        # 解析运动数据
        frames = 0
        frame_time = 0
        motion_data = []
        
        for line in motion_part.split('\n'):
            line = line.strip()
            
            if 'Frames:' in line:
                frames = int(line.split('Frames:')[-1].strip())
            elif 'Frame Time:' in line:
                frame_time = float(line.split('Frame Time:')[-1].strip())
            elif line and not line.startswith(('MOTION', 'Frames:', 'Frame Time:')):
                try:
                    values = [float(x) for x in line.split()]
                    if values:
                        motion_data.append(values)
                except ValueError:
                    continue
        
        motion_data = np.array(motion_data)
        
        return joints, motion_data
    
    def extract_features(self, motion_data: np.ndarray) -> Dict[str, np.ndarray]:
        """从运动数据中提取关键特征"""
        # 计算速度、加速度等特征
        velocity = np.diff(motion_data, axis=0)
        acceleration = np.diff(velocity, axis=0) if len(velocity) > 1 else np.zeros_like(velocity)
        
        # 计算运动范围
        motion_range = np.ptp(motion_data, axis=0)
        
        features = {
            'raw_motion': motion_data,
            'velocity': np.vstack([velocity, velocity[-1]]),  # 补齐维度
            'acceleration': np.vstack([acceleration, acceleration[-1], acceleration[-1]]),  # 补齐维度
            'motion_range': motion_range
        }
        
        return features
    
    def get_style_label(self, file_path: str) -> Tuple[str, str]:
        """从文件路径中提取风格标签(年龄组和性别)"""
        rel_path = os.path.normpath(file_path)
        parts = rel_path.split(os.sep)
        
        age_group = 'Unknown'
        gender = 'Unknown'
        
        for part in parts:
            if part in self.age_groups:
                age_group = part
            if part in self.genders:
                gender = part
        
        return age_group, gender
    
    def get_motion_type(self, file_path: str) -> str:
        """从文件名中提取动作类型"""
        filename = os.path.basename(file_path)
        if '_' in filename:
            return filename.split('_')[0]
        return 'Unknown'
    
    def process_file(self, file_path: str) -> Dict:
        """处理单个BVH文件，返回处理后的数据"""
        joints, motion_data = self.parse_bvh(file_path)
        features = self.extract_features(motion_data)
        age_group, gender = self.get_style_label(file_path)
        motion_type = self.get_motion_type(file_path)
        
        return {
            'motion_data': motion_data,
            'features': features,
            'age_group': age_group,
            'gender': gender,
            'motion_type': motion_type,
            'joints': joints,
            'file_path': file_path
        }
    
    def process_directory(self) -> List[Dict]:
        """处理整个数据集目录，返回所有处理后的数据"""
        processed_data = []
        
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.bvh'):
                    file_path = os.path.join(root, file)
                    try:
                        data = self.process_file(file_path)
                        processed_data.append(data)
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {str(e)}")
        
        return processed_data
    
    def normalize_data(self, data_list: List[Dict]) -> List[Dict]:
        """标准化数据，便于训练"""
        # 收集所有数据的均值和标准差
        all_motion = np.concatenate([d['motion_data'] for d in data_list])
        mean = np.mean(all_motion, axis=0)
        std = np.std(all_motion, axis=0)
        std[std < 1e-5] = 1.0  # 避免除以零
        
        # 标准化每个数据点
        for data in data_list:
            data['normalized_motion'] = (data['motion_data'] - mean) / std
            
        return data_list