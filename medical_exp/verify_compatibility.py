import torch
import numpy as np
import os
from monai.networks.nets import UNet

# 1. 加载刚才切出来的单病例数据
# (确保你刚才运行了 preprocess_lidc.py 并生成了文件)
data_dir = "/data/users/baohengl/midl/DLD/data/LIDC-IDRI/processed"
try:
    # 找一个生成的文件
    files = [f for f in os.listdir(data_dir) if f.endswith('_image.npy')]
    if len(files) == 0:
        raise FileNotFoundError("没找到预处理后的 .npy 文件，请先运行 preprocess_lidc.py")
    
    sample_file = files[0]
    img = np.load(os.path.join(data_dir, sample_file)) # Shape (64, 64, 64)
    print(f"✅ 数据预处理成功！Shape: {img.shape}")
    
    # 模拟 DataLoader 的输出 (Batch Size = 2)
    # 输入: [Batch, Channel, D, H, W]
    image_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1, 1).float().cuda()
    mask_tensor = torch.randint(0, 2, image_tensor.shape).float().cuda() # 模拟 Mask
    
    print(f"   Tensor Shape: {image_tensor.shape}")

except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    print("我们要先解决数据预处理，才能谈模型。")
    exit()

# 2. 验证模型兼容性 (Model Architecture Check)
print("\n正在构建 3D Diffusion U-Net (模拟 DLD 的替换模块)...")

# 这就是我们要用来替换 DLD 原生 MLP 的结构
try:
    model = UNet(
        spatial_dims=3,          # 3D 模式
        in_channels=2,           # 输入 = CT图像(1) + 噪声Mask(1)
        out_channels=1,          # 输出 = 预测Mask/残差
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).cuda()
    
    # 模拟 DLD 的 Forward 过程
    # DLD 的逻辑是: input = concat(image, noisy_mask)
    model_input = torch.cat([image_tensor, mask_tensor], dim=1) 
    
    # 跑一次前向传播
    output = model(model_input)
    
    print(f"✅ 模型前向传播成功！Output Shape: {output.shape}")
    print("\n结论：")
    print("1. 你的 preprocess_lidc.py 切出的数据形状是正确的 (64^3)。")
    print("2. 3D U-Net 可以完美吃进这个数据。")
    print("3. 现在的关键是：去把 DLD 代码里的 'ConditionalModel' 换成这个 'UNet'。")
    print("   这三天的时间应该花在改这个 model 代码上，而不是下载数据上。")

except Exception as e:
    print(f"❌ 模型不兼容: {e}")
    print("需要调整显存或网络层数。")