import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

def parse_bvh_hierarchy(file_path: str) -> Dict:
    """
    解析BVH文件的层次结构
    
    Args:
        file_path: BVH文件路径
        
    Returns:
        hierarchy: 包含骨架层次结构信息的字典
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 分离HIERARCHY和MOTION部分
    hierarchy_part = content.split('MOTION')[0] if 'MOTION' in content else content
    
    # 解析关节结构
    joints = {}
    joint_stack = []
    
    for line in hierarchy_part.split('\n'):
        line = line.strip()
        
        if not line:
            continue
        
        if 'ROOT' in line:
            joint_name = line.split('ROOT')[-1].strip()
            joints[joint_name] = {'parent': None, 'children': [], 'offset': None, 'channels': []}
            joint_stack.append(joint_name)
        elif 'JOINT' in line:
            joint_name = line.split('JOINT')[-1].strip()
            if joint_stack:
                parent = joint_stack[-1]
                joints[joint_name] = {'parent': parent, 'children': [], 'offset': None, 'channels': []}
                joints[parent]['children'].append(joint_name)
                joint_stack.append(joint_name)
        elif 'End Site' in line:
            if joint_stack:
                end_name = f"{joint_stack[-1]}_end"
                joints[end_name] = {'parent': joint_stack[-1], 'children': [], 'offset': None, 'channels': []}
                joints[joint_stack[-1]]['children'].append(end_name)
                joint_stack.append(end_name)
        elif '{' in line:
            pass  # 忽略开括号
        elif '}' in line:
            if joint_stack:
                joint_stack.pop()
        elif 'OFFSET' in line:
            if joint_stack:
                offset = [float(x) for x in line.split('OFFSET')[-1].strip().split()]
                joints[joint_stack[-1]]['offset'] = offset
        elif 'CHANNELS' in line:
            if joint_stack and "End Site" not in joints.get(joint_stack[-1], {}).get('type', ''):
                channels = line.split('CHANNELS')[-1].strip().split()
                # 第一个数字是通道数量
                channels_count = int(channels[0])
                channels_names = channels[1:1+channels_count]
                joints[joint_stack[-1]]['channels'] = channels_names
    
    return joints

def numpy_to_bvh(motion_data: np.ndarray, 
                reference_bvh_path: str, 
                output_path: str,
                frame_time: float = 0.033333) -> None:
    """
    将NumPy数组转换为BVH文件
    
    Args:
        motion_data: 运动数据 [seq_len, motion_dim]
        reference_bvh_path: 参考BVH文件路径，用于获取骨架结构
        output_path: 输出BVH文件路径
        frame_time: 帧时间间隔
    """
    # 解析参考BVH文件获取骨架结构
    with open(reference_bvh_path, 'r') as f:
        content = f.read()
    
    # 分离HIERARCHY和MOTION部分
    hierarchy_part = content.split('MOTION')[0] if 'MOTION' in content else content
    
    # 创建新的BVH内容
    bvh_content = hierarchy_part + "MOTION\n"
    bvh_content += f"Frames: {motion_data.shape[0]}\n"
    bvh_content += f"Frame Time: {frame_time}\n"
    
    # 添加运动数据
    for frame in motion_data:
        frame_str = ' '.join([f"{value:.6f}" for value in frame])
        bvh_content += frame_str + "\n"
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write(bvh_content)

def save_bvh(motion_data: np.ndarray, 
             reference_file: str, 
             output_file: str, 
             frame_time: float = 0.033333) -> None:
    """
    保存运动数据为BVH文件
    
    Args:
        motion_data: 运动数据 [seq_len, motion_dim]
        reference_file: 参考BVH文件
        output_file: 输出BVH文件路径
        frame_time: 帧时间间隔
    """
    numpy_to_bvh(motion_data, reference_file, output_file, frame_time)

def blend_motions(motion1: np.ndarray, 
                 motion2: np.ndarray, 
                 weight: float = 0.5) -> np.ndarray:
    """
    混合两个运动序列
    
    Args:
        motion1: 第一个运动序列 [seq_len, motion_dim]
        motion2: 第二个运动序列 [seq_len, motion_dim]
        weight: 混合权重 (0.0 - 1.0)
        
    Returns:
        blended_motion: 混合后的运动序列 [seq_len, motion_dim]
    """
    # 确保两个序列长度相同
    seq_len = min(motion1.shape[0], motion2.shape[0])
    motion1 = motion1[:seq_len]
    motion2 = motion2[:seq_len]
    
    # 线性混合
    blended_motion = (1 - weight) * motion1 + weight * motion2
    
    return blended_motion

def get_joint_positions(motion_data: np.ndarray, 
                       joint_offsets: Dict[str, List[float]], 
                       joint_parents: Dict[str, Optional[str]],
                       joint_channels: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    从运动数据中计算关节的全局位置
    
    Args:
        motion_data: 运动数据 [seq_len, motion_dim]
        joint_offsets: 关节偏移字典
        joint_parents: 关节父节点字典
        joint_channels: 关节通道字典
        
    Returns:
        joint_positions: 关节位置字典 {joint_name: positions [seq_len, 3]}
    """
    # 待实现...
    # 注意：这个函数比较复杂，需要根据BVH格式的具体规则实现
    # 涉及到旋转矩阵计算等
    
    # 简化版作为占位符
    joint_positions = {}
    for joint in joint_offsets:
        joint_positions[joint] = np.zeros((motion_data.shape[0], 3))
    
    return joint_positions