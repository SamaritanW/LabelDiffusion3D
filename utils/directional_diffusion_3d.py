import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import math
from collections import namedtuple

# 引用 3D U-Net
from utils.model_3d_unet import Simple3DUNet

# ==========================================
# 1. 辅助函数 (保持原版逻辑)
# ==========================================

ModelPrediction = namedtuple('ModelPrediction', ['pred_res', 'pred_noise', 'pred_y0'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    """
    3D 适配版 extract:
    Input: a=(B,), t=(B,), x_shape=(B, C, D, H, W)
    Output: (B, 1, 1, 1, 1) 用于广播
    """
    b, *_ = t.shape
    out = a.gather(-1, t.to(a.device))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 原版系数生成逻辑 (完全复刻)
def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    assert (alphas.sum()-1).abs() < 1e-6

    return alphas*sum_scale

# ==========================================
# 2. 核心类 DirectionalDiffusion3D
# ==========================================

class DirectionalDiffusion3D(nn.Module):
    def __init__(self, 
                 num_timesteps=1000, 
                 img_size=64,
                 device='cuda', 
                 sum_scale=1.,
                 sampling_timesteps=10,
                 ddim_sampling_eta=0.,
                 objective='pred_noise'): # 默认 pred_noise
        
        super().__init__()
        self.device = device
        self.num_timesteps = num_timesteps
        self.img_size = img_size
        self.ddim_sampling_eta = ddim_sampling_eta
        self.objective = objective
        
        # === A. 模型设定 ===
        # 使用 3D U-Net 替代原版的 MLP
        self.model = Simple3DUNet(img_ch=1, mask_ch=1).to(device)
        
        # === B. 扩散系数 (完全复刻 __init__ 逻辑) ===
        alphas = gen_coefficients(self.num_timesteps, schedule="average", ratio=1)
        betas2 = gen_coefficients(self.num_timesteps, schedule="average", sum_scale=sum_scale, ratio=1)
        
        alphas = alphas.float().to(self.device)
        betas2 = betas2.float().to(self.device)

        alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
        betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        
        betas_cumsum = torch.sqrt(betas2_cumsum)
        
        # 后验方差计算 (用于 q_posterior，虽不常用但为了完整性保留)
        posterior_variance = betas2 * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0

        # 注册 Buffer
        def register_buffer(name, val): 
            self.register_buffer(name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        
        # DLD 特有的后验系数
        register_buffer('posterior_mean_coef1', betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 * alphas_cumsum_prev - betas2_cumsum_prev * alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        # 边界条件处理
        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6
        
        self.sampling_timesteps = default(sampling_timesteps, num_timesteps)

    # === C. 基础预测公式 ===
    
    def q_sample(self, y_0, y_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        # 原版 DLD 公式: y_0 + alpha_bar * y_res + beta_bar * noise
        return (y_0 + extract(self.alphas_cumsum, t, y_0.shape) * y_res +
            extract(self.betas_cumsum, t, y_0.shape) * noise
        )

    def predict_start_from_yinput_noise(self, y_t, t, y_input, noise):
        # 从 noise 反推 y0
        return (
            (y_t - extract(self.alphas_cumsum, t, y_t.shape) * y_input -
             extract(self.betas_cumsum, t, y_t.shape) * noise) / 
             extract(self.one_minus_alphas_cumsum, t, y_t.shape)
        )
    
    def predict_start_from_res_noise(self, y_t, t, y_res, noise):
        # 从 res 和 noise 反推 y0
        return (
            y_t - extract(self.alphas_cumsum, t, y_t.shape) * y_res -
            extract(self.betas_cumsum, t, y_t.shape) * noise
        )

    # === D. 完整复刻 model_predictions ===
    
    def model_predictions(self, y_input, y, x_batch, fp_x, t, clip_denoised=True):
        """
        这里 fp_x 保留参数位，但 3D U-Net 不需要它 (x_batch 本身就是 Feature)
        """
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_denoised else identity

        # Forward Pass
        # 注意: 3D U-Net 的 forward 签名是 (x, y, t)，忽略 fp_x
        model_output = self.model(x_batch, y, t)

        if self.objective == 'pred_res_noise':
            # 如果是双输出 (暂未实现 3D 版的双输出，这里留坑)
            # 这里的逻辑假设 model_output 是 (B, 2*C, ...)
            # 为了稳健，目前 3D 版推荐用 pred_noise
            # 如果强行要用，需要修改 Simple3DUNet 输出通道数
            pred_res, pred_noise = torch.split(model_output, 1, dim=1) # 假设 Channel=2
            pred_res = maybe_clip(pred_res)
            pred_y0 = self.predict_start_from_res_noise(y, t, pred_res, pred_noise)
            pred_y0 = maybe_clip(pred_y0)
            
        elif self.objective == "pred_noise":
            pred_noise = model_output
            # 核心: 根据预测的 noise 推导 y0
            pred_y0 = self.predict_start_from_yinput_noise(y, t, y_input, pred_noise)
            pred_y0 = maybe_clip(pred_y0)
            # 核心: 推导 res = input - y0
            pred_res = y_input - pred_y0
            pred_res = maybe_clip(pred_res)

        return ModelPrediction(pred_res, pred_noise, pred_y0)

    # === E. 完整复刻 ddim_sample (修复 eta/sigma/beta 逻辑) ===

    @torch.no_grad()
    def ddim_sample(self, x_batch, y_input=0, fp_x=None, last=True, stochastic=False):
        batch_size = x_batch.shape[0]
        device = self.device
        
        # 3D 噪声形状
        shape = (batch_size, 1, self.img_size, self.img_size, self.img_size)
        
        # 初始化 y_T
        # old version y_T = stochastic * torch.randn(shape, device=device) if stochastic else torch.randn(shape, device=device)
        noise = torch.randn_like(y_input)
        y_T = y_input + noise  # <--- 这行代码决定了能不能跑通！

        if isinstance(y_input, int) or isinstance(y_input, float):
             y_input = torch.full(shape, y_input, dtype=torch.float32, device=device)

        pred_y0 = None
        if not last:    
            y_list = []

        # 时间步列表构造
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        y_t = y_T
        
        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            
            # 预测
            preds = self.model_predictions(y_input, y_t, x_batch, fp_x, time_cond)
            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            pred_y0 = preds.pred_y0

            if time_next < 0:
                y_t = pred_y0
                if not last: y_list.append(y_t)
                continue

            # === 严格对齐原版的数学逻辑 ===
            
            # 1. 系数提取
            alpha_cumsum = self.alphas_cumsum[time]         # alpha_bar_t
            alpha_cumsum_next = self.alphas_cumsum[time_next] # alpha_bar_{t-1}
            alpha = alpha_cumsum - alpha_cumsum_next          # alpha_t (增量)

            betas2_cumsum = self.betas2_cumsum[time]        # beta^2_bar_t
            betas2_cumsum_next = self.betas2_cumsum[time_next] # beta^2_bar_{t-1}
            
            betas_cumsum = self.betas_cumsum[time]          # beta_bar_t
            
            # 2. Sigma 计算 (DDIM eta logic)
            # sigma2 = eta * (betas2_t * betas2_{t-1}_bar / betas2_t_bar)
            # 注意: 这里 betas2 = betas2_cumsum - betas2_cumsum_next
            betas2_t = betas2_cumsum - betas2_cumsum_next
            sigma2 = self.ddim_sampling_eta * (betas2_t * betas2_cumsum_next / betas2_cumsum)
            
            # 3. 噪声项计算
            # coeff = beta_bar_t - sqrt(beta^2_bar_{t-1} - sigma2)
            noise_coeff = betas_cumsum - (betas2_cumsum_next - sigma2).sqrt()
            
            # 4. 随机噪声
            noise = torch.randn_like(y_t) if self.ddim_sampling_eta != 0 else 0

            # 5. 更新公式 (Directional Update)
            if self.objective == "pred_noise" or self.objective == "pred_res_noise":
                y_t = y_t - alpha * pred_res - noise_coeff * pred_noise + sigma2.sqrt() * noise
                
            if not last:
                y_list.append(y_t)    

        if not last:
            return y_list
        else:
            return y_t

    # === F. 缺失方法补充: forward_t ===
    
    def forward_t(self, y_input, y_0, x_batch, t, fp_x, noise=None):
        """
        用于在特定时间步 t 进行单次前向传播测试
        """
        noise = default(noise, lambda: torch.randn_like(y_0))

        y_res = y_input - y_0
        y_t_batch = self.q_sample(y_0=y_0, y_res=y_res, t=t, noise=noise)

        # Forward
        model_out = self.model(x_batch, y_t_batch, t)

        target = []
        if self.objective == 'pred_res_noise':
            # 暂不支持，需改模型
            pass 
        elif self.objective == "pred_noise":
            target = noise

        return model_out, target    

    # === G. 缺失方法补充: load_diffusion_net ===

    def load_diffusion_net(self, net_state_dicts):
        """
        加载权重。兼容原版字典结构。
        """
        # 如果是单模型模式 (我们现在是)
        if 'model' in net_state_dicts:
            self.model.load_state_dict(net_state_dicts['model'])
        # 如果保存的是 model0/model1 结构，尝试加载到 model
        elif 'model0' in net_state_dicts:
            print("Warning: Loading model0 weights into single model.")
            self.model.load_state_dict(net_state_dicts['model0'])
            
    # === H. 训练 Forward ===
    
    def forward(self, y_input, y_0, x_batch, noise=None):
        b, c, d, h, w = y_0.shape
        device = self.device
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise = default(noise, lambda: torch.randn_like(y_0))

        y_res = y_input - y_0
        y_t = self.q_sample(y_0=y_0, y_res=y_res, t=t, noise=noise)

        # 这里 fp_x 传 None
        pred_noise = self.model(x_batch, y_t, t)
        
        return F.mse_loss(pred_noise, noise)