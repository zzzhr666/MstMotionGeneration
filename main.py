import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from data.preprocess import BVHProcessor
from data.dataset import MotionDataset
from models.stm_model import STMModel
from utilsnew.viz import visualize_motion, create_animation
from utilsnew.bvh_utils import numpy_to_bvh, save_bvh

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from data.preprocess import BVHProcessor
from data.dataset import MotionDataset
from models.stm_model import STMModel
from utilsnew.viz import visualize_motion, create_animation
from utilsnew.bvh_utils import numpy_to_bvh, save_bvh

def main():
    """主程序入口函数"""
    parser = argparse.ArgumentParser(description="Style-aware Motion Generation")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'generate'], 
                        help='Program mode: train, test or generate')
    parser.add_argument('--config', type=str, default='config.py', help='Configuration file')
    
    # MDM相关参数
    parser.add_argument('--pretrained_type', type=str, default='simplified', choices=['simplified', 'mdm'],
                        help='Type of pretrained motion generator to use')
    parser.add_argument('--layers', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--cond_mask_prob', type=float, default=0.1, help='Conditioning mask probability')
    parser.add_argument('--mask_frames', action='store_true', help='Whether to mask frames')
    parser.add_argument('--diffusion_steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--text_encoder_type', type=str, default='clip', 
                        choices=['clip', 'bert'], help='Text encoder type')
    
    args, unknown = parser.parse_known_args()
    
    if args.mode == 'train':
        from train import train, parse_args
        # 将当前参数传递给train.py
        train_args = parse_args(unknown + ['--pretrained_type', args.pretrained_type,
                                          '--layers', str(args.layers),
                                          '--cond_mask_prob', str(args.cond_mask_prob),
                                          '--diffusion_steps', str(args.diffusion_steps),
                                          '--text_encoder_type', args.text_encoder_type])
        
        if args.mask_frames:
            train_args.mask_frames = True
            
        train(train_args)
    
    elif args.mode == 'test':
        from test import test, parse_args
        test_args = parse_args()
        test(test_args)
    
    elif args.mode == 'generate':
        from inference import generate, parse_args
        generate_args = parse_args()
        generate(generate_args)
    
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()