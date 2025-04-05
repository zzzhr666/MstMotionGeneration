# 基于深度神经网络的风格化运动生成

该项目实现了一个基于深度神经网络的风格化运动生成系统，能够根据用户的年龄和性别特征生成流畅、自然且符合生物力学规律的风格化运动序列。

## 项目特点

- **风格迁移**：将一种运动的风格（如年龄、性别特征）迁移到另一种运动上
- **灵活架构**：支持多种运动生成方式，包括自定义模型和预训练模型集成
- **BVH格式支持**：直接处理BVH格式的运动捕捉数据

## 文件结构

```
project/
│
├── data/ 
│   ├── preprocess.py     # BVH文件预处理和数据加载
│   ├── dataset.py        # 数据集类定义
│   └── features.py       # 运动特征提取工具
│
├── models/
│   ├── encoder.py        # 内容和风格编码器
│   ├── decoder.py        # 运动解码器
│   ├── style_module.py   # 风格调制模块
│   ├── pretrained.py     # 预训练运动生成模型集成
│   ├── stm_model.py      # 完整风格迁移模型
│   └── mdm_adapter.py    # MDM模型适配器(可选)
│
├── utils/
│   ├── viz.py            # 可视化工具
│   ├── bvh_utils.py      # BVH文件操作工具
│   └── metrics.py        # 评估指标计算
│
├── BVH/                  # 数据集目录
│   ├── Child/
│   │   ├── Male/
│   │   └── Female/
│   ├── Youth/
│   │   ├── Male/
│   │   └── Female/
│   └── Old/
│       ├── Male/
│       └── Female/
│
├── train.py              # 模型训练脚本
├── test.py               # 模型测试脚本
├── inference.py          # 运动风格迁移推理
├── main.py               # 主程序入口
└── config.py             # 配置文件
```

## 安装依赖

```bash
pip install torch numpy matplotlib pandas tqdm imageio pyyaml
```

## 使用方法

### 训练模型

```bash
# 基础训练
python main.py --mode train --data_dir ./BVH --output_dir ./output

# 使用预训练运动生成模型训练
python main.py --mode train --data_dir ./BVH --output_dir ./output --use_pretrained --pretrained_type simplified

# 使用MDM预训练模型训练(需要先下载MDM预训练权重)
python main.py --mode train \
  --data_dir ./BVH \
  --output_dir ./output \
  --use_pretrained \
  --pretrained_type 'simplified' \
  --pretrained_path ./humanml_enc_512_50steps/model000750000.pt \
  --latent_dim 512 \
  --hidden_dim 512 \
  --layers 8 \
  --cond_mask_prob 0.1 \
  --lr 0.0001 \
  --mask_frames \
  --diffusion_steps 50
```

### 测试模型

```bash
python main.py --mode test --model_path ./output/checkpoints/best_model.pth --output_dir ./test_results
```

### 生成风格化运动

```bash
python main.py --mode generate --model_path ./output/checkpoints/best_model.pth --output_dir ./generated --content_motion Walk --style_age Youth --style_gender Female
```

## 参数说明

以下是主要命令行参数的说明：

- `--mode`: 运行模式，可选 'train', 'test', 'generate'
- `--data_dir`: BVH数据目录
- `--output_dir`: 输出目录
- `--model_path`: 模型路径（用于测试和生成）
- `--batch_size`: 批次大小，默认32
- `--epochs`: 训练轮数，默认100
- `--lr`: 学习率，默认0.001
- `--latent_dim`: 潜在空间维度，默认128
- `--style_dim`: 风格维度，默认64
- `--hidden_dim`: 隐藏层维度，默认256
- `--use_pretrained`: 是否使用预训练运动生成模型
- `--pretrained_type`: 预训练模型类型，可选 'simplified' 或 'mdm'
- `--pretrained_path`: 预训练模型路径

## 使用MDM预训练模型

如果要使用MDM预训练模型，需要先下载MDM预训练权重：

```bash
# 克隆MDM仓库
git clone https://github.com/GuyTevet/motion-diffusion-model.git
cd motion-diffusion-model

# 下载预训练模型
bash prepare/download_humanml3d_models.sh

# 将模型复制到项目目录
mkdir -p ../project/save
cp -r ./save/* ../project/save/
```

## 模型训练提示

1. **训练时间**：根据数据集大小和模型配置，训练可能需要几个小时到几天不等。

2. **早停**：当前实现没有早停机制，将完成所有设定的epoch。你可以通过观察验证损失手动停止训练。

3. **超参数调优**：
   - 增大`batch_size`可加速训练但可能影响精度
   - 调整`latent_dim`和`style_dim`可平衡模型容量和泛化能力
   - 降低`lr`可使训练更稳定但速度更慢

4. **显存不足**：如果遇到显存不足，可以尝试减小`batch_size`或模型维度

## 扩展功能

- **自定义数据集**：修改`data/preprocess.py`和`data/dataset.py`以支持其他格式的运动数据
- **新的风格特征**：扩展`models/style_module.py`以支持更多风格特征
- **集成更多预训练模型**：通过扩展`models/pretrained.py`支持更多类型的预训练模型

## 故障排除

- **维度不匹配错误**：检查数据预处理和模型架构中的张量维度
- **NaN损失**：检查学习率是否过高，或尝试使用梯度裁剪
- **显存溢出**：减小批次大小或模型维度

## 引用

如果您在研究中使用了本项目，请引用：

```
基于深度神经网络的风格化运动生成, 2025
```

## 许可证

MIT 许可证