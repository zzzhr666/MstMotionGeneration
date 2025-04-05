import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.preprocess import BVHProcessor
from data.dataset import MotionDataset, PairedMotionDataset
from models.stm_model import STMModel
from utils.viz import visualize_motion, create_animation
from utils.metrics import compute_metrics, compute_reconstruction_error, compute_content_preservation, compute_jerk

def test(args):
    """测试训练好的模型"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # 数据预处理
    print("Processing BVH data...")
    processor = BVHProcessor(args.data_dir)
    processed_data = processor.process_directory()
    normalized_data = processor.normalize_data(processed_data)
    
    # 创建数据集
    print(f"Creating dataset with {len(normalized_data)} samples...")
    dataset = MotionDataset(
        normalized_data,
        seq_len=args.seq_len,
        style_conditioning=True,
        motion_type_conditioning=True
    )
    
    # 创建测试集
    test_paired_dataset = PairedMotionDataset(dataset, pairs_per_sample=1)
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_paired_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 获取数据维度
    motion_dim = dataset.data_list[0]['normalized_motion'].shape[1]
    motion_types = dataset.motion_types
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model = STMModel(
        motion_dim=motion_dim,
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        dropout=0.0,  # 测试时不需要dropout
        use_pretrained=args.use_pretrained,
        motion_types=motion_types
    )
    
    # 加载预训练权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 评估模型
    print("Evaluating model...")
    metrics = {
        'reconstruction_error': [],
        'content_preservation': [],
        'jerk': []
    }
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            # 将数据移动到设备
            content_motion = batch['content_motion'].to(device)
            style_motion = batch['style_motion'].to(device)
            
            # 前向传播
            outputs = model(content_motion, style_motion)
            
            # 计算指标
            batch_metrics = compute_metrics(outputs, content_motion, style_motion)
            
            # 记录指标
            for key, value in batch_metrics.items():
                metrics[key].append(value)
            
            # 可视化一部分结果
            if idx < args.num_viz_samples:
                # 转为CPU并转换为numpy
                content_np = content_motion[0].cpu().numpy()
                style_np = style_motion[0].cpu().numpy()
                transferred_np = outputs['transferred_motion'][0].cpu().numpy()
                
                # 可视化
                fig_path = os.path.join(args.output_dir, 'visualizations', f'test_sample_{idx+1}.png')
                visualize_motion(
                    content_np, 
                    style_np, 
                    transferred_np,
                    content_style=f"{batch['content_age_group'][0]}-{batch['content_gender'][0]}",
                    target_style=f"{batch['style_age_group'][0]}-{batch['style_gender'][0]}",
                    motion_type=batch['motion_type'][0],
                    save_path=fig_path
                )
    
    # 计算平均指标
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    
    # 打印结果
    print("\nTest Results:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # 保存结果
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write("Test Results:\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print(f"Test results saved to {os.path.join(args.output_dir, 'test_results.txt')}")
    
    return avg_metrics

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Test STM model")
    
    # 数据和模型参数
    parser.add_argument('--data_dir', type=str, default='./BVH', help='BVH data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory')
    
    # 模型配置
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--style_dim', type=int, default=64, help='Style dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained motion generator')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--num_viz_samples', type=int, default=10, help='Number of samples to visualize')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    test(args)