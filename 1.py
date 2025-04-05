import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# 更健壮的BVH解析函数
def parse_bvh(file_path):
    """解析BVH文件，提取基本信息和统计数据"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 检查文件内容是否包含必要的部分
        if 'HIERARCHY' not in content or 'MOTION' not in content:
            print(f"文件格式不正确: {file_path}")
            return None, None
        
        # 分离动作数据部分
        motion_part = content.split('MOTION')[1] if 'MOTION' in content else ""
        
        # 提取帧数和帧时间
        frames = None
        frame_time = None
        frame_data = []
        
        for line in motion_part.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if 'Frames:' in line:
                try:
                    frames = int(line.split('Frames:')[-1].strip())
                except:
                    pass
            elif 'Frame Time:' in line:
                try:
                    frame_time = float(line.split('Frame Time:')[-1].strip())
                except:
                    pass
            else:
                # 尝试解析动作数据行
                try:
                    data = [float(x) for x in line.split()]
                    if data:
                        frame_data.append(data)
                except:
                    pass
        
        # 转换为numpy数组进行高效处理
        frame_data = np.array(frame_data) if frame_data else None
        
        # 简化的结构分析 - 仅提取关键信息
        joint_count = 0
        for line in content.split('\n'):
            if 'JOINT' in line or 'ROOT' in line:
                joint_count += 1
        
        # 返回简化的信息
        motion_info = {
            'frames': frames,
            'frame_time': frame_time,
            'data': frame_data,
            'joint_count': joint_count,
            'data_dimensions': frame_data.shape if frame_data is not None else None
        }
        
        return None, motion_info
    
    except Exception as e:
        print(f"解析文件时出错 {file_path}: {str(e)}")
        return None, None

# 分析不同风格的运动特征
def analyze_style_features(bvh_files, base_path):
    """计算不同风格组合(年龄+性别)的运动特征"""
    style_features = defaultdict(lambda: defaultdict(list))
    motion_type_features = defaultdict(lambda: defaultdict(list))
    
    processed_count = 0
    total_files = len(bvh_files)
    
    for file_info in bvh_files:
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"处理进度: {processed_count}/{total_files}")
        
        try:
            _, motion = parse_bvh(file_info['path'])
            
            if motion and motion['data'] is not None and motion['frames'] and motion['frame_time']:
                # 提取风格和动作类型的键
                style_key = (file_info['age_group'], file_info['gender'])
                motion_key = file_info['motion_type']
                
                # 计算速度和加速度
                velocity = np.diff(motion['data'], axis=0)
                acceleration = np.diff(velocity, axis=0) if len(velocity) > 1 else np.zeros_like(velocity)
                
                # 计算特征
                avg_velocity = np.mean(np.abs(velocity)) if velocity.size > 0 else 0
                avg_acceleration = np.mean(np.abs(acceleration)) if acceleration.size > 0 else 0
                motion_range = np.ptp(motion['data'], axis=0).mean() if motion['data'].size > 0 else 0
                
                # 使用根关节位置计算移动距离
                if motion['data'].shape[1] >= 3:  # 假设前三列是根关节位置
                    root_positions = motion['data'][:, :3]
                    total_distance = np.sum(np.sqrt(np.sum(np.diff(root_positions, axis=0)**2, axis=1)))
                else:
                    total_distance = 0
                
                duration = motion['frames'] * motion['frame_time']
                avg_speed = total_distance / duration if duration > 0 else 0
                
                # 存储风格特征
                style_features[style_key]['avg_velocity'].append(avg_velocity)
                style_features[style_key]['avg_acceleration'].append(avg_acceleration)
                style_features[style_key]['motion_range'].append(motion_range)
                style_features[style_key]['avg_speed'].append(avg_speed)
                style_features[style_key]['duration'].append(duration)
                
                # 存储动作类型特征
                motion_type_features[motion_key]['avg_velocity'].append(avg_velocity)
                motion_type_features[motion_key]['avg_acceleration'].append(avg_acceleration)
                motion_type_features[motion_key]['motion_range'].append(motion_range)
                motion_type_features[motion_key]['avg_speed'].append(avg_speed)
        except Exception as e:
            print(f"分析文件时出错 {file_info['path']}: {str(e)}")
    
    # 计算风格的均值特征
    style_summary = {}
    for style, features in style_features.items():
        style_summary[style] = {
            feature: np.mean(values) for feature, values in features.items() if values
        }
        style_summary[style]['sample_count'] = len(next(iter(features.values())))
    
    # 计算动作类型的均值特征
    motion_summary = {}
    for motion, features in motion_type_features.items():
        motion_summary[motion] = {
            feature: np.mean(values) for feature, values in features.items() if values
        }
        motion_summary[motion]['sample_count'] = len(next(iter(features.values())))
    
    return style_summary, motion_summary

# 分析不同年龄组和性别间的差异
def analyze_differences(style_summary):
    """分析不同风格组合间的运动特征差异"""
    # 按年龄组分组
    age_groups = defaultdict(list)
    for (age, gender), stats in style_summary.items():
        age_groups[age].append((gender, stats))
    
    # 按性别分组
    gender_groups = defaultdict(list)
    for (age, gender), stats in style_summary.items():
        gender_groups[gender].append((age, stats))
    
    # 输出年龄组间的差异
    print("\n年龄组间的差异:")
    for age, gender_stats in age_groups.items():
        print(f"\n{age}组:")
        for gender, stats in gender_stats:
            print(f"  {gender}:")
            for feature, value in stats.items():
                if feature != 'sample_count':
                    print(f"    {feature}: {value:.4f}")
            print(f"    样本数: {stats.get('sample_count', 0)}")
    
    # 输出性别间的差异
    print("\n性别间的差异:")
    for gender, age_stats in gender_groups.items():
        print(f"\n{gender}:")
        for age, stats in age_stats:
            print(f"  {age}:")
            for feature, value in stats.items():
                if feature != 'sample_count':
                    print(f"    {feature}: {value:.4f}")
            print(f"    样本数: {stats.get('sample_count', 0)}")
    
    return age_groups, gender_groups

# 数据集统计分析
def analyze_dataset_stats(bvh_files):
    """对数据集进行统计分析"""
    # 计算每个组合的样本数
    style_counts = defaultdict(int)
    motion_counts = defaultdict(int)
    style_motion_counts = defaultdict(lambda: defaultdict(int))
    
    for file_info in bvh_files:
        style_key = (file_info['age_group'], file_info['gender'])
        motion_key = file_info['motion_type']
        
        style_counts[style_key] += 1
        motion_counts[motion_key] += 1
        style_motion_counts[style_key][motion_key] += 1
    
    # 输出统计信息
    print("\n数据集统计:")
    print(f"总文件数: {len(bvh_files)}")
    
    print("\n风格组合统计:")
    for (age, gender), count in sorted(style_counts.items()):
        print(f"{age}-{gender}: {count}个文件")
    
    print("\n动作类型统计:")
    for motion, count in sorted(motion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{motion}: {count}个文件")
    
    # 查找每种风格最常见的动作类型
    print("\n每种风格最常见的动作类型:")
    for style, motion_dict in style_motion_counts.items():
        top_motions = sorted(motion_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"{style[0]}-{style[1]}:")
        for motion, count in top_motions:
            print(f"  {motion}: {count}个文件")
    
    return style_counts, motion_counts, style_motion_counts

# 主函数
def main():
    base_path = './BVH'  # 数据集根目录
    print(f"分析数据集: {base_path}")
    
    # 检查路径是否存在
    if not os.path.exists(base_path):
        print(f"错误: 路径 {base_path} 不存在")
        return
    
    # 获取所有BVH文件
    bvh_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.bvh'):
                # 从路径中提取年龄组和性别信息
                path_parts = os.path.normpath(root).split(os.sep)
                age_group = next((p for p in path_parts if p in ['Child', 'Youth', 'Old']), 'Unknown')
                gender = next((p for p in path_parts if p in ['Male', 'Female']), 'Unknown')
                
                # 从文件名提取动作类型
                motion_type = file.split('_')[0] if '_' in file else 'Unknown'
                
                bvh_files.append({
                    'path': os.path.join(root, file),
                    'age_group': age_group,
                    'gender': gender,
                    'motion_type': motion_type,
                    'filename': file
                })
    
    print(f"找到 {len(bvh_files)} 个BVH文件")
    
    # 进行数据集统计分析
    style_counts, motion_counts, style_motion_counts = analyze_dataset_stats(bvh_files)
    
    # 分析不同风格的运动特征
    print("\n分析不同风格的运动特征...")
    style_summary, motion_summary = analyze_style_features(bvh_files, base_path)
    
    # 分析风格差异
    age_groups, gender_groups = analyze_differences(style_summary)
    
    # 输出动作类型的特征
    print("\n常见动作类型的特征:")
    for motion, stats in sorted(motion_summary.items(), key=lambda x: x[1].get('sample_count', 0), reverse=True)[:10]:
        print(f"\n{motion} (样本数: {stats.get('sample_count', 0)}):")
        for feature, value in stats.items():
            if feature != 'sample_count':
                print(f"  {feature}: {value:.4f}")
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()