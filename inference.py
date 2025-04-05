import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from data.preprocess import BVHProcessor
from data.dataset import MotionDataset
from models.stm_model import STMModel
from utils.viz import visualize_motion, create_animation
from utils.bvh_utils import numpy_to_bvh, save_bvh

def generate(args):
    """生成风格化运动序列"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据预处理 - 仅用于获取数据格式和映射
    print("Loading data for format reference...")
    processor = BVHProcessor(args.data_dir)
    processed_data = processor.process_directory()
    normalized_data = processor.normalize_data(processed_data)
    
    # 创建数据集对象(用于获取数据格式和标签映射)
    dataset = MotionDataset(
        normalized_data,
        seq_len=args.seq_len,
        style_conditioning=True,
        motion_type_conditioning=True
    )
    
    # 获取数据维度和映射
    motion_dim = dataset.data_list[0]['normalized_motion'].shape[1]
    motion_types = dataset.motion_types
    age_groups = dataset.age_groups
    genders = dataset.genders
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model = STMModel(
        motion_dim=motion_dim,
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        dropout=0.0,  # 推理时不需要dropout
        use_pretrained=args.use_pretrained,
        motion_types=motion_types
    )
    
    # 加载预训练权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 准备生成
    print("Generating styled motions...")
    
    # 加载内容运动(用于提取内容)
    content_index = args.content_index
    if content_index >= len(dataset):
        print(f"Content index {content_index} out of range. Using index 0.")
        content_index = 0
    
    content_sample = dataset[content_index]
    content_motion = content_sample['motion'].unsqueeze(0).to(device)  # 添加批次维度
    content_motion_type = content_sample['motion_type']
    
    # 加载风格运动(用于提取风格)
    style_indices = args.style_indices
    if not style_indices:
        # 默认使用一个来自每个年龄-性别组合的样本
        style_indices = []
        for age_idx, age in enumerate(age_groups):
            for gender_idx, gender in enumerate(genders):
                # 找到符合条件的第一个样本
                for i in range(len(dataset)):
                    sample = dataset[i]
                    if sample['age_group'] == age and sample['gender'] == gender:
                        style_indices.append(i)
                        break
    
    results = []
    
    for style_idx in style_indices:
        if style_idx >= len(dataset):
            print(f"Style index {style_idx} out of range. Skipping.")
            continue
        
        style_sample = dataset[style_idx]
        style_motion = style_sample['motion'].unsqueeze(0).to(device)  # 添加批次维度
        style_age = style_sample['age_group']
        style_gender = style_sample['gender']
        
        print(f"Transferring {content_motion_type} motion to {style_age}-{style_gender} style...")
        
        # 生成风格迁移结果
        with torch.no_grad():
            outputs = model(content_motion, style_motion)
            transferred_motion = outputs['transferred_motion']
        
        # 转换为numpy数组
        content_motion_np = content_motion.cpu().numpy()[0]  # 移除批次维度
        style_motion_np = style_motion.cpu().numpy()[0]  # 移除批次维度
        transferred_motion_np = transferred_motion.cpu().numpy()[0]  # 移除批次维度
        
        # 可视化结果
        fig_path = os.path.join(args.output_dir, f"transfer_{content_motion_type}_{style_age}_{style_gender}.png")
        visualize_motion(
            content_motion_np, 
            style_motion_np, 
            transferred_motion_np,
            content_style=f"{content_sample['age_group']}-{content_sample['gender']}",
            target_style=f"{style_age}-{style_gender}",
            motion_type=content_motion_type,
            save_path=fig_path
        )
        
        # 为风格迁移结果创建动画
        anim_path = os.path.join(args.output_dir, f"animation_{content_motion_type}_{style_age}_{style_gender}.gif")
        create_animation(
            transferred_motion_np,
            fps=30,
            title=f"{content_motion_type} in {style_age}-{style_gender} style",
            save_path=anim_path
        )
        
        # 保存结果
        results.append({
            'content_motion': content_motion_np,
            'style_motion': style_motion_np,
            'transferred_motion': transferred_motion_np,
            'content_style': f"{content_sample['age_group']}-{content_sample['gender']}",
            'target_style': f"{style_age}-{style_gender}",
            'motion_type': content_motion_type
        })
        
        print(f"Saved visualization to {fig_path} and animation to {anim_path}")
    
    print(f"Generated {len(results)} styled motions.")
    return results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Generate styled motions")
    
    # 数据和模型参数
    parser.add_argument('--data_dir', type=str, default='./BVH', help='BVH data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./generated', help='Output directory')
    
    # 生成参数
    parser.add_argument('--content_index', type=int, default=0, help='Index of content motion sample')
    parser.add_argument('--style_indices', type=int, nargs='+', default=None, help='Indices of style motion samples')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained motion generator')
    
    # 模型配置
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--style_dim', type=int, default=64, help='Style dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    generate(args)