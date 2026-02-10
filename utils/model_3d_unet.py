import torch
import torch.nn as nn
import math

# ==========================================
# 1. 基础组件 (Positional Embedding & ResBlock)
# ==========================================

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
        
        # 时间步注入层 (MLP)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.BatchNorm3d(out_ch)
        self.act2 = nn.SiLU()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.act1(self.norm1(h))
        
        # 注入时间信息: (Batch, TimeDim) -> (Batch, OutCh) -> Broadcast to (Batch, OutCh, D, H, W)
        time_emb = self.mlp(time_emb)
        time_emb = time_emb[(..., ) + (None, ) * 3]
        h = h + time_emb
        
        h = self.conv2(h)
        h = self.act2(self.norm2(h))
        return h

# ==========================================
# 2. 核心网络 (Simple 3D U-Net)
# ==========================================

class Simple3DUNet(nn.Module):
    def __init__(self, img_ch=1, mask_ch=1, base_ch=32):
        super().__init__()
        # 时间编码层
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # === Encoder ===
        # 输入通道 = CT图像(img_ch) + Noisy Mask(mask_ch) = 2
        self.inc = Block3D(img_ch + mask_ch, base_ch, time_dim)
        
        self.down1 = nn.Conv3d(base_ch, base_ch*2, 4, 2, 1) # Downsample
        self.block1 = Block3D(base_ch*2, base_ch*2, time_dim)
        
        # === Bottleneck ===
        self.bot = Block3D(base_ch*2, base_ch*2, time_dim)

        # === Decoder ===
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 4, 2, 1) # Upsample
        # Skip Connection: concat 后通道数是 base_ch * 2
        self.block2 = Block3D(base_ch*2, base_ch, time_dim) 
        
        # === Output ===
        self.outc = nn.Conv3d(base_ch, mask_ch, 1)

    def forward(self, x, y_noisy, t):
        """
        x: CT Image [B, 1, 64, 64, 64]
        y_noisy: Noisy Mask [B, 1, 64, 64, 64]
        t: Time step [B]
        """
        # 1. 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 2. 拼接输入 (Conditional Input)
        # 将 CT 和 带噪Mask 在通道维度拼接 -> [B, 2, 64, 64, 64]
        x_in = torch.cat([x, y_noisy], dim=1) 
        
        # 3. 网络传播
        # Down
        x1 = self.inc(x_in, t_emb)      # -> [B, 32, 64, 64, 64]
        x2 = self.down1(x1)             # -> [B, 64, 32, 32, 32]
        x2 = self.block1(x2, t_emb)
        
        # Middle
        x_mid = self.bot(x2, t_emb)     # -> [B, 64, 32, 32, 32]
        
        # Up
        x_up = self.up1(x_mid)          # -> [B, 32, 64, 64, 64]
        
        # Skip Connection (x1 和 x_up 拼接)
        x_up = torch.cat([x_up, x1], dim=1) # -> [B, 64, 64, 64, 64]
        x_dec = self.block2(x_up, t_emb)    # -> [B, 32, 64, 64, 64]
        
        # Out
        out = self.outc(x_dec)          # -> [B, 1, 64, 64, 64]
        return out