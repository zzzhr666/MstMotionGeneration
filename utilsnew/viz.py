import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Tuple, Optional

def visualize_motion(content_motion, style_motion, transferred_motion, 
                    content_style=None, target_style=None, motion_type=None,
                    save_path=None, fig_size=(15, 10)):
    """
    可视化原始内容、风格以及迁移后的运动
    
    Args:
        content_motion: 内容运动序列 [seq_len, motion_dim]
        style_motion: 风格运动序列 [seq_len, motion_dim]
        transferred_motion: 风格迁移后的运动序列 [seq_len, motion_dim]
        content_style: 内容风格标签
        target_style: 目标风格标签
        motion_type: 动作类型标签
        save_path: 保存路径
        fig_size: 图像大小
    """
    fig = plt.figure(figsize=fig_size)
    
    # 提取共同帧数
    seq_len = min(
        content_motion.shape[0],
        style_motion.shape[0],
        transferred_motion.shape[0]
    )
    
    # 创建标题
    title = "Motion Style Transfer"
    if motion_type:
        title += f" - {motion_type}"
    if content_style and target_style:
        title += f"\nContent Style: {content_style} → Target Style: {target_style}"
    
    plt.suptitle(title, fontsize=16)
    
    # 简化的3D运动可视化
    # 假设前三列是根关节位置，我们用它来显示轨迹
    
    # 创建三个子图
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    # 假设前三列是根关节位置 (x, y, z)
    # 如果不是，您需要根据实际数据格式调整
    if content_motion.shape[1] >= 3:
        x1, y1, z1 = content_motion[:seq_len, 0], content_motion[:seq_len, 1], content_motion[:seq_len, 2]
        ax1.plot(x1, z1, y1, 'b-', label='Root Joint Path')
        ax1.set_title('Content Motion')
        
        x2, y2, z2 = style_motion[:seq_len, 0], style_motion[:seq_len, 1], style_motion[:seq_len, 2]
        ax2.plot(x2, z2, y2, 'r-', label='Root Joint Path')
        ax2.set_title('Style Motion')
        
        x3, y3, z3 = transferred_motion[:seq_len, 0], transferred_motion[:seq_len, 1], transferred_motion[:seq_len, 2]
        ax3.plot(x3, z3, y3, 'g-', label='Root Joint Path')
        ax3.set_title('Transferred Motion')
    
    # 设置轴标签
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        
        # 设置相同的轴范围以便比较
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 2])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，为整体标题留出空间
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_animation(motion_sequence, fps=30, title=None, save_path=None):
    """
    创建3D运动序列的动画
    
    Args:
        motion_sequence: 运动序列 [seq_len, motion_dim]
        fps: 每秒帧数
        title: 动画标题
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if title:
        plt.title(title)
    
    # 假设前三列是根关节位置
    if motion_sequence.shape[1] >= 3:
        x = motion_sequence[:, 0]
        y = motion_sequence[:, 1]
        z = motion_sequence[:, 2]
        
        # 设置轴范围
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        
        # 添加一些余量
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        z_margin = (z_max - z_min) * 0.1
        
        ax.set_xlim([x_min - x_margin, x_max + x_margin])
        ax.set_ylim([y_min - y_margin, y_max + y_margin])
        ax.set_zlim([z_min - z_margin, z_max + z_margin])
        
        # 设置轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 创建线对象
        line, = ax.plot([], [], [], 'b-', label='Root Joint Path')
        point, = ax.plot([], [], [], 'ro', markersize=8)
        
        # 初始化函数
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        
        # 动画更新函数
        def update(frame):
            line.set_data(x[:frame], y[:frame])
            line.set_3d_properties(z[:frame])
            point.set_data([x[frame-1]], [y[frame-1]])
            point.set_3d_properties([z[frame-1]])
            return line, point
        
        # 创建动画
        ani = animation.FuncAnimation(
            fig, update, frames=len(x),
            init_func=init, blit=True, interval=1000/fps
        )
        
        if save_path:
            ani.save(save_path, writer='pillow', fps=fps)
            plt.close()
        else:
            plt.show()
    else:
        print("Motion sequence doesn't have enough dimensions for visualization")