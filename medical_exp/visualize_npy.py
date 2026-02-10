import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 设置路径
data_dir = "/data/users/baohengl/midl/DLD/data/LIDC-IDRI/processed"
save_plot_dir = "vis_results" # 图片保存到这里，方便下载查看
os.makedirs(save_plot_dir, exist_ok=True)

# 2. 找一个文件
files = [f for f in os.listdir(data_dir) if f.endswith('_image.npy')]
if not files:
    print("没找到 npy 文件！")
    exit()

filename = files[0] # 取第一个结节
img_path = os.path.join(data_dir, filename)
print(f"正在可视化: {filename}")

# 3. 加载数据
# Shape: (64, 64, 64) -> (Depth, Height, Width)
vol = np.load(img_path) 
print(f"数据范围: Min={vol.min():.2f}, Max={vol.max():.2f}")

# 4. 归一化到 0-255 用于显示 (简单的窗宽窗位调整，模拟肺窗)
# 肺部 CT HU 值通常在 -1000 到 400 之间
def normalize(v):
    v = np.clip(v, -1000, 400)
    v = (v - (-1000)) / (400 - (-1000))
    return v

vol_norm = normalize(vol)

# 5. 取中间切片 (Axial, Sagittal, Coronal)
mid_d = vol.shape[0] // 2
mid_h = vol.shape[1] // 2
mid_w = vol.shape[2] // 2

slice_axial = vol_norm[mid_d, :, :]    # 横断面
slice_sagittal = vol_norm[:, :, mid_w] # 矢状面
slice_coronal = vol_norm[:, mid_h, :]  # 冠状面

# 6. 画图
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(slice_axial, cmap='gray')
ax[0].set_title(f"Axial (Z={mid_d})")
ax[1].imshow(slice_sagittal, cmap='gray')
ax[1].set_title(f"Sagittal (X={mid_w})")
ax[2].imshow(slice_coronal, cmap='gray')
ax[2].set_title(f"Coronal (Y={mid_h})")

save_path = os.path.join(save_plot_dir, f"{filename}.png")
plt.savefig(save_path)
print(f"✅ 可视化图片已保存到: {save_path}")
print("请把这张图下载下来看一眼，确认是不是一个肺结节（中间应该有个小白点）。")