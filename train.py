import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data.preprocess import BVHProcessor
from data.dataset import MotionDataset, PairedMotionDataset
from models.stm_model import STMModel
from utilsnew.viz import visualize_motion
from utilsnew.metrics import compute_metrics

def train(args):
    """主训练函数"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoint'), exist_ok=True)
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
    
    # 划分训练集和验证集
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建成对数据集(用于风格迁移训练)
    train_paired_dataset = PairedMotionDataset(train_dataset, pairs_per_sample=args.pairs_per_sample)
    val_paired_dataset = PairedMotionDataset(val_dataset, pairs_per_sample=1)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_paired_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_paired_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 获取数据维度
    motion_dim = dataset.data_list[0]['normalized_motion'].shape[1]
    motion_types = dataset.motion_types
    
    # 创建模型
    print("Creating model...")
    model = STMModel(
        motion_dim=motion_dim,
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        dropout=args.dropout,
        use_pretrained=args.use_pretrained,
        motion_types=motion_types,
        pretrained_path=args.pretrained_path
    )
    model = model.to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_metrics = {
            'reconstruction_loss': [],
            'style_loss': [],
            'kl_loss': [],
            'content_consistency_loss': []
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in progress_bar:
            # 将数据移动到设备
            content_motion = batch['content_motion'].to(device)
            style_motion = batch['style_motion'].to(device)
            
            # 前向传播
            outputs = model(content_motion, style_motion)
            
            # 计算损失
            loss, losses = compute_loss(
                outputs, 
                content_motion, 
                style_motion,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma
            )
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
            # 记录损失
            train_losses.append(loss.item())
            for key, value in losses.items():
                train_metrics[key].append(value)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': np.mean(train_losses[-100:]),
                'rec_loss': np.mean(train_metrics['reconstruction_loss'][-100:]),
                'style_loss': np.mean(train_metrics['style_loss'][-100:])
            })
        
        # 计算平均训练损失
        train_loss = np.mean(train_losses)
        train_metric_avgs = {key: np.mean(values) for key, values in train_metrics.items()}
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_metrics = {
            'reconstruction_loss': [],
            'style_loss': [],
            'kl_loss': [],
            'content_consistency_loss': []
        }
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in progress_bar:
                # 将数据移动到设备
                content_motion = batch['content_motion'].to(device)
                style_motion = batch['style_motion'].to(device)
                
                # 前向传播
                outputs = model(content_motion, style_motion)
                
                # 计算损失
                loss, losses = compute_loss(
                    outputs, 
                    content_motion, 
                    style_motion,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma
                )
                
                # 记录损失
                val_losses.append(loss.item())
                for key, value in losses.items():
                    val_metrics[key].append(value)
                
                progress_bar.set_postfix({
                    'loss': np.mean(val_losses[-10:]),
                    'rec_loss': np.mean(val_metrics['reconstruction_loss'][-10:]),
                    'style_loss': np.mean(val_metrics['style_loss'][-10:])
                })
        
        # 计算平均验证损失
        val_loss = np.mean(val_losses)
        val_metric_avgs = {key: np.mean(values) for key, values in val_metrics.items()}
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印训练统计
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        for key in train_metric_avgs:
            print(f"  Train {key}: {train_metric_avgs[key]:.6f}, Val {key}: {val_metric_avgs[key]:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best model saved to {checkpoint_path}")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
        
        # 可视化生成的样本
        if (epoch + 1) % args.viz_every == 0:
            visualize_epoch_samples(model, val_loader, device, epoch, args)
    
    print("Training complete!")
    return model

def compute_loss(outputs, content_motion, style_motion, alpha=1.0, beta=0.1, gamma=0.5):
    """计算风格迁移模型的损失"""
    # 重建损失
    reconstruction_loss = F.mse_loss(outputs['reconstructed_motion'], content_motion)
    
    # 风格迁移损失
    style_loss = F.mse_loss(outputs['transferred_motion'], content_motion)
    
    # KL散度损失(VAE部分)
    kl_loss = -0.5 * torch.sum(1 + outputs['content_logvar'] - outputs['content_mu'].pow(2) - outputs['content_logvar'].exp())
    kl_loss = kl_loss / content_motion.size(0)  # 归一化
    
    # 内容一致性损失 - 确保风格迁移后内容不变
    content_consistency_loss = F.mse_loss(outputs['transferred_z'], outputs['content_z'])
    
    # 总损失
    total_loss = reconstruction_loss + alpha * style_loss + beta * kl_loss + gamma * content_consistency_loss
    
    losses = {
        'reconstruction_loss': reconstruction_loss.item(),
        'style_loss': style_loss.item(),
        'kl_loss': kl_loss.item(),
        'content_consistency_loss': content_consistency_loss.item()
    }
    
    return total_loss, losses

def visualize_epoch_samples(model, dataloader, device, epoch, args):
    """可视化当前epoch的样本结果"""
    model.eval()
    
    # 获取一个批次用于可视化
    batch = next(iter(dataloader))
    content_motion = batch['content_motion'].to(device)
    style_motion = batch['style_motion'].to(device)
    
    # 限制可视化样本数量
    n_samples = min(4, content_motion.size(0))
    content_motion = content_motion[:n_samples]
    style_motion = style_motion[:n_samples]
    
    # 生成风格迁移结果
    with torch.no_grad():
        outputs = model(content_motion, style_motion)
    
    # 转移到CPU并转换为numpy数组
    content_motion = content_motion.cpu().numpy()
    style_motion = style_motion.cpu().numpy()
    transferred_motion = outputs['transferred_motion'].cpu().numpy()
    
    # 为每个样本创建可视化
    for i in range(n_samples):
        fig_path = os.path.join(args.output_dir, 'visualizations', f'epoch_{epoch+1}_sample_{i+1}.png')
        
        # 创建可视化(使用工具函数)
        visualize_motion(
            content_motion[i], 
            style_motion[i], 
            transferred_motion[i],
            content_style=f"{batch['content_age_group'][i]}-{batch['content_gender'][i]}",
            target_style=f"{batch['style_age_group'][i]}-{batch['style_gender'][i]}",
            motion_type=batch['motion_type'][i],
            save_path=fig_path
        )
    
    print(f"  Visualizations saved to {os.path.join(args.output_dir, 'visualizations')}")

def parse_args(args=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train STM model")
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./BVH', help='BVH data directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--pairs_per_sample', type=int, default=4, help='Number of style pairs per sample')
    
    # 模型参数
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--style_dim', type=int, default=64, help='Style dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained motion generator')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')
    
    # MDM相关参数
    parser.add_argument('--pretrained_type', type=str, default='simplified', choices=['simplified', 'mdm'],
                        help='Type of pretrained motion generator to use')
    parser.add_argument('--layers', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--cond_mask_prob', type=float, default=0.1, help='Conditioning mask probability')
    parser.add_argument('--mask_frames', action='store_true', help='Whether to mask frames')
    parser.add_argument('--diffusion_steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--text_encoder_type', type=str, default='clip', 
                        choices=['clip', 'bert'], help='Text encoder type')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--alpha', type=float, default=1.0, help='Style loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='KL loss weight')
    parser.add_argument('--gamma', type=float, default=0.5, help='Content consistency loss weight')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # 保存与可视化参数
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--viz_every', type=int, default=5, help='Visualize samples every N epochs')
    
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    train(args)