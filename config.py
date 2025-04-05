"""
配置文件，用于保存各种默认参数
"""

# 数据配置
DATA_CONFIG = {
    'data_dir': './BVH',
    'output_dir': './output',
    'val_split': 0.1,
    'seq_len': 60,
    'pairs_per_sample': 4
}

# 模型配置
MODEL_CONFIG = {
    'latent_dim': 128,
    'style_dim': 64,
    'hidden_dim': 256,
    'dropout': 0.1,
    'use_pretrained': True,
    'pretrained_path': None
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'clip_grad': 1.0,
    'alpha': 1.0,  # 风格损失权重
    'beta': 0.1,   # KL损失权重
    'gamma': 0.5,  # 内容一致性损失权重
    'device': 'cuda',
    'num_workers': 4,
    'save_every': 10,
    'viz_every': 5
}

# 测试配置
TEST_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'num_viz_samples': 10,
    'device': 'cuda'
}

# 生成配置
GENERATE_CONFIG = {
    'device': 'cuda'
}

# 初始化函数
def get_config(config_type='train'):
    """获取指定类型的配置"""
    if config_type == 'train':
        return {**DATA_CONFIG, **MODEL_CONFIG, **TRAIN_CONFIG}
    elif config_type == 'test':
        return {**DATA_CONFIG, **MODEL_CONFIG, **TEST_CONFIG}
    elif config_type == 'generate':
        return {**DATA_CONFIG, **MODEL_CONFIG, **GENERATE_CONFIG}
    else:
        raise ValueError(f"Unknown config type: {config_type}")