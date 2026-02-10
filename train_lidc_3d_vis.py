import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
from utils.directional_diffusion_3d import DirectionalDiffusion3D
from utils.ema_3d import EMA 

# --- 0. 辅助函数：提取轮廓线 (用于画线) ---
def get_edges(mask):
    """
    输入: (1, H, W) 的二值 Mask (0或1)
    输出: (1, H, W) 的边缘 Mask (0或1)
    原理: 膨胀后的图像 - 原图像 = 边缘
    """
    # 增加 batch 和 channel 维以便卷积: (1, 1, H, W)
    mask_tensor = mask.unsqueeze(0) 
    # MaxPool2d with kernel 3, stride 1, padding 1 等效于 3x3 膨胀
    dilated = F.max_pool2d(mask_tensor, kernel_size=3, stride=1, padding=1)
    # 边缘 = 膨胀 - 原图
    edges = dilated - mask_tensor
    return edges.squeeze(0) # 还原回 (1, H, W)

# --- 1. Dataset (保持不变) ---
class LIDCDataset(Dataset):
    def __init__(self, data_root, mean_mask_root, mode='train', specific_pid=None):
        self.files = []
        self.mean_mask_root = mean_mask_root
        
        all_files = [f for f in os.listdir(data_root) if f.endswith('_image.npy')]
        for f in all_files:
            pid = f.split('_nodule')[0]
            if specific_pid and pid != specific_pid: continue
            
            mask_name = f.replace('_image.npy', '_masks.npy')
            if os.path.exists(os.path.join(data_root, mask_name)) and \
               os.path.exists(os.path.join(mean_mask_root, mask_name)):
                self.files.append({
                    'img': os.path.join(data_root, f), 
                    'msk': os.path.join(data_root, mask_name),
                    'mean': os.path.join(mean_mask_root, mask_name)
                })
        print(f"[{mode}] Loaded {len(self.files)} samples.")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx]['img'])
        masks = np.load(self.files[idx]['msk'])
        
        doctor_idx = np.random.randint(0, masks.shape[0])
        y0 = masks[doctor_idx][np.newaxis, ...] 
        y_mean = np.load(self.files[idx]['mean'])[np.newaxis, ...]
        
        return torch.from_numpy(img).float(), \
               torch.from_numpy(y0).float(), \
               torch.from_numpy(y_mean).float()

# --- 2. 主程序 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/LIDC-IDRI/processed_check')
    parser.add_argument('--mean_mask_path', type=str, default='data/LIDC-IDRI/mean_masks')
    # 依然先跑 0078 做验证
    parser.add_argument('--pid', type=str, default='LIDC-IDRI-0078') 
    # === 🛑 修改保存路径，避免污染 ===
    parser.add_argument('--save_dir', type=str, default='results/lidc_0078_vis_fix') 
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = LIDCDataset(args.data_path, args.mean_mask_path, specific_pid=args.pid)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    diffusion = DirectionalDiffusion3D(
        num_timesteps=1000, img_size=64, device=device,
        sampling_timesteps=50 
    )
    ema = EMA(diffusion, decay=0.995, update_every=10).to(device)
    optimizer = optim.AdamW(diffusion.parameters(), lr=1e-4)

    # === 固定验证样本 ===
    fixed_iter = iter(dataloader)
    fixed_x, fixed_y0, fixed_y_mean = next(fixed_iter)
    fixed_x = fixed_x.to(device)
    fixed_y0 = fixed_y0.to(device)
    fixed_y_mean = fixed_y_mean.to(device)
    
    # 保存参考图 (只是为了记录 GT 和 Mean 长啥样)
    # 我们这次不再存那个 reference_CT_GT_Mean.png 了，直接在后面生成漂亮的 Overlay
    z_slice = 32

    print(f"🚀 Start Training (Visual Fix Mode)... Check {args.save_dir} for results.")

    for epoch in range(501):
        diffusion.train()
        pbar = tqdm(dataloader, desc=f"Ep {epoch}")
        
        for x, y0, y_mean in pbar:
            x, y0, y_mean = x.to(device), y0.to(device), y_mean.to(device)
            y_input = y_mean 
            loss = diffusion(y_input, y0, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()
            pbar.set_postfix({'L': f"{loss.item():.5f}"})

        # === 🔍 升级版可视化 (每 20 epoch) ===
        if epoch % 20 == 0: 
            with torch.no_grad():
                # 1. 采样
                debug_t = torch.full((fixed_x.shape[0],), 500, device=device).long()
                debug_noise = torch.randn_like(fixed_y0)
                debug_y_res = fixed_y_mean - fixed_y0 
                debug_y_t = diffusion.q_sample(y_0=fixed_y0, y_res=debug_y_res, t=debug_t, noise=debug_noise)
                
                pred_noise = diffusion.model(fixed_x, debug_y_t, debug_t)
                pred_y0_raw = diffusion.predict_start_from_yinput_noise(debug_y_t, debug_t, fixed_y_mean, pred_noise)
                
                # 2. 准备切片数据
                ct_slice = fixed_x[0, 0, z_slice].cpu() # CT
                gt_slice = fixed_y0[0, 0, z_slice].cpu() # GT (0/1)
                mean_slice = fixed_y_mean[0, 0, z_slice].cpu() # Mean (0~1)
                
                # === 🛠️ 关键修复 1: 强行二值化，背景变纯黑 ===
                # 这里用了 > 0.5，你也可以调成 > 0.1 试试，取决于 DLD 收敛程度
                pred_slice_bin = (pred_y0_raw[0, 0, z_slice].cpu() > 0.5).float()

                # === 🛠️ 关键修复 2: 提取轮廓线 ===
                gt_edges = get_edges(gt_slice)       # 真值轮廓
                pred_edges = get_edges(pred_slice_bin) # 预测轮廓

                # === 3. 制作漂亮的 Overlay 图 ===
                # 归一化 CT 到 0-1 用于显示
                ct_show = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
                
                # 转成 RGB (3, H, W)
                rgb_img = torch.stack([ct_show, ct_show, ct_show], dim=0)

                # 叠加颜色:
                # 🟢 绿色通道 += GT 轮廓 (Green)
                rgb_img[1, :, :] = torch.clamp(rgb_img[1, :, :] + gt_edges, 0, 1)
                # 🔴 红色通道 += Pred 轮廓 (Red)
                rgb_img[0, :, :] = torch.clamp(rgb_img[0, :, :] + pred_edges, 0, 1)
                # (如果重叠，R+G=Yellow，正好表示重合部分)

                # 为了对比，我们把 "Input Mean Mask" 也画出来看看
                # 左边放 Mean Mask (云雾状)，右边放 最终结果 (红绿线)
                
                # 做一个 Input 的可视化 (Input 叠加在 CT 上，蓝色云雾)
                input_vis = torch.stack([ct_show, ct_show, ct_show], dim=0)
                input_vis[2, :, :] = torch.clamp(input_vis[2, :, :] + mean_slice, 0, 1) # Blue channel

                # 拼接: [Input(Mean) | Output(Contour)]
                final_vis = torch.cat([input_vis, rgb_img], dim=2) # 横向拼接

                save_path = f"{args.save_dir}/vis_epoch_{epoch}.png"
                save_image(final_vis, save_path)
                # print(f"Saved {save_path} (Left: Input Mean, Right: Red=Pred, Green=GT)")

if __name__ == "__main__":
    main()