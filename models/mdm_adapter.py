import os
import torch
import numpy as np
from copy import deepcopy

# 导入MDM相关组件
from diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.mdm import MDM

class MDMAdapter:
    """MDM模型的适配器，提供与STM项目兼容的接口"""
    
    def __init__(self, 
                model_path, 
                device='cuda',
                num_frames=60,
                diffusion_steps=50):
        """
        初始化MDM适配器
        
        Args:
            model_path: 预训练MDM模型路径
            device: 计算设备
            num_frames: 生成序列的帧数
            diffusion_steps: 扩散步数
        """
        self.device = device
        self.num_frames = num_frames
        self.diffusion_steps = diffusion_steps
        
        # 加载模型
        self.model, self.diffusion = self.load_mdm_model(model_path)
        self.model.eval()  # 设置为评估模式
        
        # 缓存用于生成不同动作类型的标签映射
        self._motion_types = None
        self._motion_type_to_idx = None
        
    def load_mdm_model(self, model_path):
        """加载预训练的MDM模型"""
        # 参数设置
        args = self._get_default_args()
        args.model_path = model_path
        
        # 创建模型
        model = self._create_model(args)
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        if 'model_avg' in state_dict.keys():
            state_dict = state_dict['model_avg']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
            
        # 删除位置编码信息，以避免大小不匹配
        if 'sequence_pos_encoder.pe' in state_dict:
            del state_dict['sequence_pos_encoder.pe']
        if 'embed_timestep.sequence_pos_encoder.pe' in state_dict:
            del state_dict['embed_timestep.sequence_pos_encoder.pe']
            
        # 加载模型权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # 创建扩散过程
        diffusion = self._create_diffusion(args)
        
        return model.to(self.device), diffusion
    
    def _get_default_args(self):
        """获取默认参数"""
        # 创建简单的参数对象
        class Args:
            def __init__(self):
                # 模型参数
                self.arch = 'trans_enc'
                self.layers = 8
                self.latent_dim = 512
                self.text_encoder_type = 'clip'
                self.cond_mask_prob = 0.1
                self.emb_trans_dec = False
                self.dataset = 'humanml'
                self.unconstrained = False
                self.pos_embed_max_len = 5000
                self.mask_frames = True
                
                # 扩散参数
                self.noise_schedule = 'cosine'
                self.diffusion_steps = 50
                self.sigma_small = True
                
                # 动作生成参数
                self.pred_len = 0
                self.context_len = 0
                self.lambda_vel = 0.0
                self.lambda_rcxyz = 0.0
                self.lambda_fc = 0.0
                self.lambda_target_loc = 0.0
                
        return Args()
    
    def _create_model(self, args):
        """创建MDM模型"""
        # 简化的模型参数
        model_args = {
            'modeltype': '',
            'njoints': 263,  # humanml3d
            'nfeats': 1,
            'num_actions': 1,
            'translation': True,
            'pose_rep': 'rot6d',
            'glob': True,
            'glob_rot': True,
            'latent_dim': args.latent_dim,
            'ff_size': 1024,
            'num_layers': args.layers,
            'num_heads': 4,
            'dropout': 0.1,
            'activation': "gelu",
            'data_rep': 'hml_vec',  # humanml3d
            'cond_mode': 'text',
            'cond_mask_prob': args.cond_mask_prob,
            'action_emb': 'tensor',
            'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec,
            'clip_version': 'ViT-B/32',
            'dataset': args.dataset,
            'text_encoder_type': args.text_encoder_type,
            'pos_embed_max_len': args.pos_embed_max_len,
            'mask_frames': args.mask_frames,
            'pred_len': args.pred_len,
            'context_len': args.context_len,
            'all_goal_joint_names': ['pelvis'] + ['ankle', 'wrist', 'elbow', 'knee', 'toe']
        }
        
        return MDM(**model_args)
    
    def _create_diffusion(self, args):
        """创建扩散过程"""
        # 获取扩散参数
        betas = self._get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
        
        # 创建扩散过程
        return SpacedDiffusion(
            use_timesteps=space_timesteps(args.diffusion_steps, [args.diffusion_steps]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL if args.sigma_small else ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
            lambda_vel=args.lambda_vel,
            lambda_rcxyz=args.lambda_rcxyz,
            lambda_fc=args.lambda_fc,
            lambda_target_loc=args.lambda_target_loc
        )
        
    def _get_named_beta_schedule(self, schedule_name, num_diffusion_timesteps, scale_beta=1.):
        """获取预定义的beta时间表"""
        if schedule_name == "linear":
            scale = scale_beta * 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        elif schedule_name == "cosine":
            return self._betas_for_alpha_bar(
                num_diffusion_timesteps,
                lambda t: np.cos((t + 0.008) / 1.008 * np.math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")
    
    def _betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """创建beta时间表"""
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)
    
    def setup_motion_types(self, motion_types):
        """设置支持的动作类型"""
        self._motion_types = motion_types
        self._motion_type_to_idx = {m: i for i, m in enumerate(motion_types)}
    
    def generate_motion(self, motion_type_indices, batch_size=4):
        """
        生成指定类型的运动
        
        Args:
            motion_type_indices: 动作类型索引 [batch_size]
            batch_size: 批次大小
            
        Returns:
            motion_sequences: 生成的运动序列 [batch_size, seq_len, output_dim]
        """
        model = self.model
        diffusion = self.diffusion
        
        # 确保输入是张量
        if not isinstance(motion_type_indices, torch.Tensor):
            motion_type_indices = torch.tensor(motion_type_indices, device=self.device)
            
        # 创建模型输入
        shape = (batch_size, model.njoints, model.nfeats, self.num_frames)
        
        # 创建空条件字典
        model_kwargs = {}
        model_kwargs['y'] = {'mask': torch.ones((batch_size, 1, 1, self.num_frames), device=self.device)}
        
        # 添加文本条件
        text_conditions = []
        for idx in motion_type_indices:
            if self._motion_types is not None and idx < len(self._motion_types):
                text_conditions.append(self._motion_types[idx])
            else:
                text_conditions.append("walking")  # 默认动作
                
        model_kwargs['y']['text'] = text_conditions
        
        # 无噪声采样
        noise = torch.randn(shape, device=self.device)
        
        # 执行采样
        with torch.no_grad():
            samples = diffusion.p_sample_loop(
                model, shape, noise=noise, clip_denoised=True, model_kwargs=model_kwargs,
                progress=False, device=self.device
            )
            
        return samples
    
    def convert_to_stm_format(self, mdm_motion):
        """
        将MDM生成的运动格式转换为STM使用的格式
        
        Args:
            mdm_motion: MDM生成的运动序列 [batch_size, njoints, nfeats, nframes]
            
        Returns:
            stm_motion: STM格式的运动序列 [batch_size, seq_len, output_dim]
        """
        # 重新排列维度为STM期望的格式
        # MDM: [batch_size, njoints, nfeats, nframes]
        # STM: [batch_size, seq_len, output_dim]
        
        batch_size, njoints, nfeats, nframes = mdm_motion.shape
        
        # 将MDM运动重新整形为STM格式
        stm_motion = mdm_motion.permute(0, 3, 1, 2)  # [batch_size, nframes, njoints, nfeats]
        stm_motion = stm_motion.reshape(batch_size, nframes, njoints * nfeats)  # [batch_size, nframes, njoints*nfeats]
        
        return stm_motion
    
    def generate(self, motion_types, num_samples=4):
        """
        生成指定类型的运动
        
        Args:
            motion_types: 动作类型列表或字符串
            num_samples: 每种类型生成的样本数
            
        Returns:
            tuple: (motion_sequences, latent_vectors)
                motion_sequences: 生成的运动序列 [batch_size, seq_len, output_dim]
                latent_vectors: 潜在向量 [batch_size, latent_dim]
        """
        # 确保motion_types是列表
        if isinstance(motion_types, str):
            motion_types = [motion_types]
            
        batch_size = len(motion_types)
        
        # 将动作类型转换为索引
        indices = []
        for motion_type in motion_types:
            if self._motion_type_to_idx and motion_type in self._motion_type_to_idx:
                index = self._motion_type_to_idx[motion_type]
            else:
                # 如果不支持请求的动作类型，使用随机类型
                index = np.random.randint(0, len(self._motion_types) if self._motion_types else 1)
            indices.append(index)
            
        # 生成动作
        motion_indices = torch.tensor(indices, device=self.device)
        mdm_motion = self.generate_motion(motion_indices, batch_size=batch_size)
        
        # 转换为STM格式
        stm_motion = self.convert_to_stm_format(mdm_motion)
        
        # 生成一个假的潜在向量
        latent_vectors = torch.randn(batch_size, self.model.latent_dim, device=self.device)
        
        return stm_motion, latent_vectors