#!/usr/bin/env python3
# debug_mdm_adapter_fix.py
# 放在项目根目录执行

import os
import sys
import torch
import traceback

# 确保路径正确
sys.path.insert(0, os.getcwd())

print("===== MDM适配器调试与修复脚本 =====")
print(f"当前工作目录: {os.getcwd()}")

try:
    # 1. 导入必要的模块
    print("\n[1] 导入必要模块...")
    from model.mdm import MDM
    from diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
    from diffusion.respace import SpacedDiffusion, space_timesteps
    from models.mdm_adapter import MDMAdapter
    print("✓ 模块导入成功")

    # 2. 检查并修复MDMAdapter代码
    print("\n[2] 检查MDMAdapter实现...")
    
    # 获取MDMAdapter源文件路径
    adapter_path = sys.modules['models.mdm_adapter'].__file__
    print(f"MDMAdapter源文件路径: {adapter_path}")
    
    # 读取原始文件内容
    with open(adapter_path, 'r') as f:
        orig_code = f.read()
    
    # 检查init方法中的模型加载问题
    if "self.model, self.diffusion = self.load_mdm_model(model_path)" in orig_code:
        print("✓ MDMAdapter中已有正确的赋值语句")
    else:
        print("✗ MDMAdapter中存在赋值问题，需要修复")
        
        # 修复代码 - 查找并替换有问题的语句
        if "self.model = self.load_mdm_model(model_path)" in orig_code:
            fixed_code = orig_code.replace(
                "self.model = self.load_mdm_model(model_path)",
                "self.model, self.diffusion = self.load_mdm_model(model_path)"
            )
        else:
            # 寻找load_mdm_model的调用，并确保返回值正确赋值
            import re
            init_pattern = r"def __init__\(self[^)]*\):(.*?)def"
            init_match = re.search(init_pattern, orig_code, re.DOTALL)
            
            if init_match:
                init_code = init_match.group(1)
                
                # 查找load_mdm_model调用
                load_pattern = r"([^\n]*)load_mdm_model\(([^)]*)\)"
                load_match = re.search(load_pattern, init_code)
                
                if load_match:
                    load_line = load_match.group(0)
                    full_line = load_match.group(0)
                    args = load_match.group(2)
                    
                    # 构建正确的赋值语句
                    replacement = f"self.model, self.diffusion = self.load_mdm_model({args})"
                    
                    # 替换整行
                    fixed_code = orig_code.replace(full_line, replacement)
                    print(f"  正在替换: {full_line} -> {replacement}")
                else:
                    print("✗ 无法找到load_mdm_model调用")
                    fixed_code = orig_code
            else:
                print("✗ 无法找到__init__方法")
                fixed_code = orig_code
        
        # 备份原始文件
        backup_path = adapter_path + ".backup"
        with open(backup_path, 'w') as f:
            f.write(orig_code)
        print(f"✓ 原始文件已备份至 {backup_path}")
        
        # 写入修复后的代码
        with open(adapter_path, 'w') as f:
            f.write(fixed_code)
        print("✓ MDMAdapter已修复")
    
    # 3. 测试修复后的MDMAdapter
    print("\n[3] 测试修复后的MDMAdapter...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 预训练模型路径
    pretrained_path = "./humanml_enc_512_50steps/model000750000.pt"
    print(f"使用预训练模型: {pretrained_path}")
    
    # 创建MDMAdapter实例
    try:
        adapter = MDMAdapter(
            model_path=pretrained_path,
            device=device,
            num_frames=60,
            diffusion_steps=50
        )
        print("✓ MDMAdapter创建成功!")
        
        # 检查model和diffusion属性
        if adapter.model is not None:
            print("✓ adapter.model 正确加载")
        else:
            print("✗ adapter.model 仍然为None")
            
        if adapter.diffusion is not None:
            print("✓ adapter.diffusion 正确加载")
        else:
            print("✗ adapter.diffusion 仍然为None")
            
        # 设置动作类型
        adapter.setup_motion_types(["walk", "run", "jump"])
        print("✓ 动作类型设置成功")
        
        # 生成测试动作
        motion, latent = adapter.generate(["walk"], num_samples=1)
        print(f"✓ 动作生成成功，形状: {motion.shape}")
        
    except Exception as e:
        print(f"✗ MDMAdapter测试失败: {e}")
        traceback.print_exc()
    
    print("\n===== 调试和修复完成 =====")
    print("请再次运行你的训练脚本，问题应该已经解决。")
    
except Exception as e:
    print(f"调试过程中发生错误: {e}")
    traceback.print_exc()