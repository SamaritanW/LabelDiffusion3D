import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast
import numpy as np
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
import torch.nn.functional as F
from skimage import measure
import random

# 引入你的模型
from utils.directional_diffusion_3d import DirectionalDiffusion3D
from utils.ema_3d import EMA

# === 0. 辅助工具函数 ===
def get_edges(mask):
    """提取边缘用于可视化"""
    mask_tensor = mask.unsqueeze(0)
    dilated = F.max_pool2d(mask_tensor, kernel_size=3, stride=1, padding=1)
    edges = dilated - mask_tensor
    return edges.squeeze(0)

def compute_dice_3d(pred, target):
    """
    计算 3D Dice 系数 (Volumetric)
    pred, target: (D, H, W)
    """
    smooth = 1e-5
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def keep_largest_component_3d(mask):
    """
    3D 后处理：保留最大连通域
    mask: (D, H, W) Tensor (on GPU)
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    if mask_np.sum() == 0: return mask
    
    # 3D 连通域 (connectivity=2)
    labels = measure.label(mask_np, connectivity=2) 
    if labels.max() == 0: return mask
    
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    largest_cc = (labels == largest_label)
    
    return torch.from_numpy(largest_cc).float().to(mask.device)

# === 1. 数据集 (智能维度适配) ===
class LIDCDataset(Dataset):
    def __init__(self, data_root, mean_mask_root, mode='train', split_ratio=(0.8, 0.1, 0.1)):
        self.files = []
        all_files = sorted([f for f in os.listdir(data_root) if f.endswith('_image.npy')])
        
        # Patient-level Split
        patient_ids = sorted(list(set([f.split('_nodule')[0] for f in all_files])))
        random.seed(42) 
        random.shuffle(patient_ids)
        
        n_total = len(patient_ids)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        
        train_pids = set(patient_ids[:n_train])
        val_pids = set(patient_ids[n_train:n_train+n_val])
        test_pids = set(patient_ids[n_train+n_val:])
        
        if mode == 'train': target_pids = train_pids
        elif mode == 'val': target_pids = val_pids
        else: target_pids = test_pids
            
        for f in all_files:
            pid = f.split('_nodule')[0]
            if pid not in target_pids: continue
            
            mask_name = f.replace('_image.npy', '_masks.npy')
            img_path = os.path.join(data_root, f)
            msk_path = os.path.join(data_root, mask_name)
            mean_path = os.path.join(mean_mask_root, mask_name)
            
            if os.path.exists(msk_path) and os.path.exists(mean_path):
                self.files.append({'img': img_path, 'msk': msk_path, 'mean': mean_path})
                
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[{mode.upper()}] PIDs: {len(target_pids)} | Samples: {len(self.files)}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        # 1. 加载数据
        img = np.load(self.files[idx]['img']) # 可能是 (64,64,64) 或 (1,64,64,64)
        masks = np.load(self.files[idx]['msk']) 
        y_mean = np.load(self.files[idx]['mean'])

        # 2. 随机选一个医生作为 GT
        doctor_idx = np.random.randint(0, masks.shape[0])
        y0 = masks[doctor_idx]
        
        # 3. 🚨 智能维度修正 (防止 6D 错误)
        # 目标: 所有输出都必须是 4D -> (C, D, H, W) 即 (1, 64, 64, 64)
        
        # Image 处理
        if img.ndim == 3:
            img = img[np.newaxis, ...] # (64,64,64) -> (1,64,64,64)
        
        # GT Mask 处理
        if y0.ndim == 3:
            y0 = y0[np.newaxis, ...]
            
        # Mean Mask 处理
        if y_mean.ndim == 3:
            y_mean = y_mean[np.newaxis, ...]

        return torch.from_numpy(img).float(), \
               torch.from_numpy(y0).float(), \
               torch.from_numpy(y_mean).float()

# === 2. DDP 初始化 ===
def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    else:
        print('Not using Distributed Mode')
        return 0, 1, 0
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()
    return rank, world_size, gpu

def cleanup_ddp():
    dist.destroy_process_group()

# === 3. 验证逻辑 (3D Metrics) ===
def validate_epoch(model, ema_model, val_loader, device):
    """
    使用 ddim_sample 进行全量 3D 验证
    """
    # 切换模式
    model.eval()
    ema_model.eval()
    
    total_val_loss = 0.0
    total_val_dice = 0.0
    num_batches = 0
    
    # 只需要 Rank 0 跑验证
    with torch.no_grad():
        for x, y0, y_mean in tqdm(val_loader, desc="🔍 Validating (3D)", leave=False):
            x, y0, y_mean = x.to(device), y0.to(device), y_mean.to(device)
            
            # --- 1. 计算 Val Loss (3D MSE) ---
            # 使用 model.forward 计算去噪误差
            # 参数顺序: (y_input, y_0, x_batch)
            val_loss = model(y_mean, y0, x)
            total_val_loss += val_loss.item()
            
            # --- 2. 计算 3D Dice (DDIM Sampling) ---
            # 🚨 使用 ddim_sample (x, y_input)
            pred_y0 = ema_model.ddim_sample(x, y_input=y_mean)
            
            # 兼容性处理: 如果返回 list，取最后一个
            if isinstance(pred_y0, list): pred_y0 = pred_y0[-1]
            
            # 二值化
            pred_y0 = (pred_y0 > 0.5).float()
            
            # Batch 内逐个计算
            batch_dice = 0.0
            for i in range(x.shape[0]):
                mask_pred = pred_y0[i, 0] # (D, H, W)
                mask_gt = y0[i, 0]        # (D, H, W)
                
                # 3D 后处理
                clean_pred = keep_largest_component_3d(mask_pred)
                
                # Dice
                d = compute_dice_3d(clean_pred, mask_gt)
                batch_dice += d.item()
            
            total_val_dice += (batch_dice / x.shape[0])
            num_batches += 1
            
    avg_loss = total_val_loss / num_batches
    avg_dice = total_val_dice / num_batches
    
    return avg_loss, avg_dice

# === 4. 可视化 ===
def visualize(model, ema_model, fixed_data, epoch, save_dir, device):
    ema_model.eval()
    fixed_x, fixed_y0, fixed_y_mean = fixed_data
    z_slice = 32

    with torch.no_grad():
        # 🚨 使用 ddim_sample
        pred_y0 = ema_model.ddim_sample(fixed_x, y_input=fixed_y_mean)
        if isinstance(pred_y0, list): pred_y0 = pred_y0[-1]
        
        pred_slice = (pred_y0[0, 0, z_slice].cpu() > 0.5).float()
        gt_slice = fixed_y0[0, 0, z_slice].cpu()
        mean_slice = fixed_y_mean[0, 0, z_slice].cpu()
        ct_slice = fixed_x[0, 0, z_slice].cpu()

        # Visualization Logic
        ct_show = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
        rgb_img = torch.stack([ct_show, ct_show, ct_show], dim=0)
        
        rgb_img[1, :, :] = torch.clamp(rgb_img[1, :, :] + gt_slice, 0, 1) # Green
        rgb_img[0, :, :] = torch.clamp(rgb_img[0, :, :] + pred_slice, 0, 1) # Red
        
        input_vis = torch.stack([ct_show, ct_show, ct_show], dim=0)
        input_vis[2, :, :] = torch.clamp(input_vis[2, :, :] + mean_slice, 0, 1) 
        
        overlay_vis = torch.cat([input_vis, rgb_img], dim=2)
        save_image(overlay_vis, f"{save_dir}/vis_epoch_{epoch}.png")

# === 5. 主程序 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/users/baohengl/miccai/DLD/data/LIDC-IDRI/process')
    parser.add_argument('--mean_mask_path', type=str, default='/data/users/baohengl/miccai/DLD/data/LIDC-IDRI/mean_masks')
    parser.add_argument('--save_dir', type=str, default='results/lidc_final_v7')
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    rank, world_size, gpu = setup_ddp()
    device = torch.device(f"cuda:{gpu}")
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"🚀 Start Training V7: Fixed Dimensions & Logic")

    # 1. Dataset
    train_dataset = LIDCDataset(args.data_path, args.mean_mask_path, mode='train')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)

    val_loader = None
    fixed_data = None
    if rank == 0:
        val_dataset = LIDCDataset(args.data_path, args.mean_mask_path, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
        fixed_data = next(iter(val_loader))
        fixed_x, fixed_y0, fixed_y_mean = fixed_data
        fixed_data = (fixed_x.to(device), fixed_y0.to(device), fixed_y_mean.to(device))

    # 2. Model
    model = DirectionalDiffusion3D(num_timesteps=1000, img_size=64, device=device, sampling_timesteps=50).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[gpu], output_device=gpu)
    
    ema = EMA(model.module if dist.is_initialized() else model, decay=0.995, update_every=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda')

    best_dice = 0.0

    # 3. Loop
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        steps = 0
        
        if rank == 0: pbar = tqdm(train_loader, desc=f"Ep {epoch}")
        else: pbar = train_loader

        for step, (x, y0, y_mean) in enumerate(pbar):
            x, y0, y_mean = x.to(device), y0.to(device), y_mean.to(device)
            # x 和 y0, y_mean 现在都应该是 5D (B, 1, 64, 64, 64)

            optimizer.zero_grad()
            with autocast('cuda'):
                # forward(y_input, y_0, x_batch)
                loss = model(y_mean, y0, x)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            
            epoch_loss += loss.item()
            steps += 1
            if rank == 0: pbar.set_postfix({'MSE': f"{loss.item():.5f}"})

        # Rank 0: Validation
        if rank == 0:
            avg_train_loss = epoch_loss / steps
            
            checkpoint = {'model': ema.ema_model.state_dict(), 'epoch': epoch, 'best_dice': best_dice}
            torch.save(checkpoint, os.path.join(args.save_dir, "latest_model.pth"))

            if epoch % 50 == 0:
                raw_model = model.module if dist.is_initialized() else model
                
                visualize(raw_model, ema.ema_model, fixed_data, epoch, args.save_dir, device)
                
                # 计算 3D 指标
                val_loss, val_dice = validate_epoch(raw_model, ema.ema_model, val_loader, device)
                
                print(f"\n📊 [Epoch {epoch}] Tr Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | 3D Dice: {val_dice:.4f}")
                
                if val_dice > best_dice:
                    best_dice = val_dice
                    torch.save(ema.ema_model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
                    print(f"🏆 New Best Model! Dice: {best_dice:.4f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()