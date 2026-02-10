import torch
import torch.nn as nn
import numpy as np
import os
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm3d(out_ch)
        self.act1 = nn.SiLU()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.BatchNorm3d(out_ch)
        self.act2 = nn.SiLU()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.act1(self.norm1(h))
        # 注入时间信息
        time_emb = self.mlp(time_emb)
        time_emb = time_emb[(..., ) + (None, ) * 3]
        h = h + time_emb
        h = self.conv2(h)
        h = self.act2(self.norm2(h))
        return h

class Simple3DUNet(nn.Module):
    def __init__(self, img_ch=1, mask_ch=1, base_ch=32):
        super().__init__()
        # 时间编码
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        # 输入通道 = CT图像(1) + Noisy Mask(1) = 2
        self.inc = Block3D(img_ch + mask_ch, base_ch, time_dim)
        self.down1 = nn.Conv3d(base_ch, base_ch*2, 4, 2, 1)
        self.block1 = Block3D(base_ch*2, base_ch*2, time_dim)
        
        # Bottleneck
        self.bot = Block3D(base_ch*2, base_ch*2, time_dim)

        # Decoder
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 4, 2, 1)
        self.block2 = Block3D(base_ch*2, base_ch, time_dim) # Concat后通道翻倍
        
        # Output
        self.outc = nn.Conv3d(base_ch, mask_ch, 1)

    def forward(self, x, y_noisy, t):
        # 1. 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 2. 拼接输入 (CT + Noisy Mask)
        # x: [B, 1, 64, 64, 64], y_noisy: [B, 1, 64, 64, 64]
        x_in = torch.cat([x, y_noisy], dim=1) 
        
        # 3. 网络传播
        # Down
        x1 = self.inc(x_in, t_emb)      # -> 32通道
        x2 = self.down1(x1)             # -> 64x32x32x32
        x2 = self.block1(x2, t_emb)
        
        # Middle
        x_mid = self.bot(x2, t_emb)
        
        # Up
        x_up = self.up1(x_mid)          # -> 32x64x64x64
        x_up = torch.cat([x_up, x1], dim=1) # Skip Connection
        x_dec = self.block2(x_up, t_emb)
        
        # Out
        out = self.outc(x_dec)
        return out

# ==========================================
# 2. 测试流程
# ==========================================
def run_test():
    print("🚀 开始网络联通性测试...")
    
    # A. 路径配置 (使用你刚才生成的 0078 数据)
    # 假设你之前跑的是 preprocess_lidc.py 生成在 processed_check
    # 如果你全量跑了，路径可能是 data/LIDC-IDRI/processed
    data_dir = "data/LIDC-IDRI/processed_check" 
    pid = "LIDC-IDRI-0078"
    nodule_idx = 0
    
    img_path = os.path.join(data_dir, f"{pid}_nodule{nodule_idx}_image.npy")
    msk_path = os.path.join(data_dir, f"{pid}_nodule{nodule_idx}_masks.npy")
    
    if not os.path.exists(img_path):
        print(f"❌ 找不到文件: {img_path}，请先运行 preprocess_lidc.py")
        return

    # B. 加载数据
    # Image: (1, 64, 64, 64)
    np_img = np.load(img_path)
    # Mask: (4, 64, 64, 64) -> 我们只取第1个医生的标注来模拟一次训练
    np_msk = np.load(msk_path)[0:1] 
    
    print(f"✅ 数据加载成功:")
    print(f"   Image shape: {np_img.shape}")
    print(f"   Mask  shape: {np_msk.shape}")

    # C. 转为 Tensor 并增加 Batch 维度
    # Pytorch 需要 (Batch, Channel, D, H, W)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # [1, 64, 64, 64] -> [1, 1, 64, 64, 64] (Batch=1)
    t_img = torch.from_numpy(np_img).unsqueeze(0).float().to(device)
    t_msk = torch.from_numpy(np_msk).unsqueeze(0).float().to(device)
    
    # 构造一个随机时间步 t (Batch=1)
    t_time = torch.tensor([500]).to(device) # 假设 t=500

    # D. 初始化网络
    model = Simple3DUNet().to(device)
    print("✅ 网络初始化成功")

    # E. 喂入数据 (Forward Pass)
    try:
        print("⏳ 正在执行前向传播 (Feed Forward)...")
        output = model(t_img, t_msk, t_time)
        
        print("\n🎉🎉🎉 测试通过！🎉🎉🎉")
        print(f"输入尺寸: {t_img.shape}")
        print(f"输出尺寸: {output.shape}")
        print("结论: 这里的 .npy 数据可以完美被 3D U-Net 消化。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()