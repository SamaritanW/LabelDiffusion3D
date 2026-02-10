import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
from utils.directional_diffusion_3d import DirectionalDiffusion3D
from utils.ema_3d import EMA 

# --- 1. Dataset (修复参数传递) ---
class LIDCDataset(Dataset):
    def __init__(self, data_root, mean_mask_root, mode='train', specific_pid=None):
        self.files = []
        self.mean_mask_root = mean_mask_root # 保存路径
        
        all_files = [f for f in os.listdir(data_root) if f.endswith('_image.npy')]
        for f in all_files:
            pid = f.split('_nodule')[0]
            if specific_pid and pid != specific_pid: continue
            
            mask_name = f.replace('_image.npy', '_masks.npy')
            # 确保对应的 mean mask 也存在
            if os.path.exists(os.path.join(data_root, mask_name)) and \
               os.path.exists(os.path.join(mean_mask_root, mask_name)):
                self.files.append({
                    'img': os.path.join(data_root, f), 
                    'msk': os.path.join(data_root, mask_name),
                    'mean': os.path.join(mean_mask_root, mask_name) # 记录 mean mask 路径
                })
        print(f"[{mode}] Loaded {len(self.files)} samples.")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        # 1. 读取 CT 和 GT
        img = np.load(self.files[idx]['img'])
        masks = np.load(self.files[idx]['msk'])
        
        # 随机选一个医生作为 GT
        doctor_idx = np.random.randint(0, masks.shape[0])
        y0 = masks[doctor_idx][np.newaxis, ...] # (1, 64, 64, 64)
        
        # 2. 读取 Mean Mask (y_input)
        y_mean = np.load(self.files[idx]['mean'])[np.newaxis, ...]
        
        return torch.from_numpy(img).float(), \
               torch.from_numpy(y0).float(), \
               torch.from_numpy(y_mean).float() # 返回3个: CT, GT, Mean

# --- 2. 主程序 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/LIDC-IDRI/processed_check')
    parser.add_argument('--mean_mask_path', type=str, default='data/LIDC-IDRI/mean_masks', help='Path to mean masks')
    parser.add_argument('--pid', type=str, default='LIDC-IDRI-0078')
    parser.add_argument('--save_dir', type=str, default='results/lidc_dld_mean') # 改个名，区分实验
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 准备数据 (传入 mean_mask_path)
    dataset = LIDCDataset(args.data_path, args.mean_mask_path, specific_pid=args.pid)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 2. 模型初始化
    diffusion = DirectionalDiffusion3D(
        num_timesteps=1000, img_size=64, device=device,
        sampling_timesteps=50 
    )
    ema = EMA(diffusion, decay=0.995, update_every=10).to(device)
    optimizer = optim.AdamW(diffusion.parameters(), lr=1e-4)

    # === 🔒 关键修改：固定验证样本 (解包3个变量) ===
    fixed_iter = iter(dataloader)
    # 这里必须解包 3 个变量，否则报错
    fixed_x, fixed_y0, fixed_y_mean = next(fixed_iter) 
    
    fixed_x = fixed_x.to(device)
    fixed_y0 = fixed_y0.to(device)
    fixed_y_mean = fixed_y_mean.to(device) 
    
    # 保存参考图 (CT | GT | Mean Mask)
    z_slice = 32
    ct_show = fixed_x[0, 0, z_slice].cpu()
    gt_show = fixed_y0[0, 0, z_slice].cpu()
    mean_show = fixed_y_mean[0, 0, z_slice].cpu()
    
    # 归一化显示
    ref_img = torch.cat([ct_show, gt_show, mean_show], dim=1)
    ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
    save_image(ref_img, f"{args.save_dir}/reference_CT_GT_Mean.png")
    print(f"✅ Saved reference image (CT | GT | Mean). Start DLD Training...")

    
    for epoch in range(501):
        diffusion.train()
        pbar = tqdm(dataloader, desc=f"Ep {epoch}")
        
        # 训练循环 (解包3个变量)
        for x, y0, y_mean in pbar:
            x = x.to(device)
            y0 = y0.to(device)
            y_mean = y_mean.to(device)
            
            # === DLD 核心设定 ===
            y_input = y_mean # 使用 Mean Mask 作为起点 (Dirty Label)
            
            # Forward (直接调用 diffusion，它内部会自动算 q_sample 和 loss)
            loss = diffusion(y_input, y0, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()
            
            pbar.set_postfix({'L': f"{loss.item():.5f}"})

        # === 🔍 监控部分 (可视化精修过程) ===
        if epoch % 20 == 0: 
            with torch.no_grad():
                # 我们想看：模型能否把 fixed_y_mean 修成 fixed_y0 ?
                
                # 1. 构造验证输入
                # 验证时的 y_input 必须也是 Mean Mask (保持和训练一致)
                debug_y_input = fixed_y_mean
                
                # 构造 y_t (模拟 t=500 时刻的中间状态)
                debug_t = torch.full((fixed_x.shape[0],), 500, device=device).long()
                debug_noise = torch.randn_like(fixed_y0)
                
                # 注意：这里计算 residual 是为了生成正确的 y_t
                debug_y_res = debug_y_input - fixed_y0 
                debug_y_t = diffusion.q_sample(y_0=fixed_y0, y_res=debug_y_res, t=debug_t, noise=debug_noise)
                
                # 2. 预测
                pred_noise = diffusion.model(fixed_x, debug_y_t, debug_t)
                
                # 3. 倒推 y0 (去噪结果)
                # 关键：这里传入的 y_input 也是 Mean Mask
                pred_y0 = diffusion.predict_start_from_yinput_noise(debug_y_t, debug_t, debug_y_input, pred_noise)
                
                # 4. 可视化
                # Mean Mask (原始模糊输入)
                mean_view = fixed_y_mean[0, 0, z_slice].cpu()
                # Prediction (模型精修结果)
                pred_view = pred_y0[0, 0, z_slice].cpu()
                
                # 截断处理，保证显示清晰
                pred_view = torch.clamp(pred_view, 0, 1)
                
                # 拼接: [Mean Mask (Input) | Prediction (Output)]
                row = torch.cat([mean_view, pred_view], dim=1)
                
                save_image(row, f"{args.save_dir}/debug_epoch_{epoch}.png")
                # 你应该看到右边的图比左边的图更锐利，更接近 GT

if __name__ == "__main__":
    main()