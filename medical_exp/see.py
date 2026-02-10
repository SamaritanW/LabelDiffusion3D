import pylidc as pl
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
pid = 'LIDC-IDRI-0078'
scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
nods = scan.cluster_annotations(verbose=False)
vol = scan.to_volume() # 读取原始体积 (Z, Y, X)

# 2. 获取结节中心 (以第一个结节簇为例)
if len(nods) > 0:
    cluster = nods[0] # 取第一个结节
    centroids = [a.centroid for a in cluster]
    center = np.mean(centroids, axis=0)
    cz, cy, cx = int(center[0]), int(center[1]), int(center[2])
    print(f"结节中心坐标 (Z, Y, X): {cz}, {cy}, {cx}")
else:
    print("未找到结节！展示中心切片。")
    cz, cy, cx = vol.shape[0]//2, vol.shape[1]//2, vol.shape[2]//2

# 3. 计算纵横比 (Aspect Ratio) 用于修复显示变形
# Z轴的缩放比例 = 层厚 / 像素间距
aspect_ratio = scan.slice_thickness / scan.pixel_spacing

# 4. 画图
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# --- Axial (轴状面, XY平面, Z=固定) ---
# 这是最常见的医生看片视角
ax[0].imshow(vol[cz, :, :], cmap='gray', vmin=-1200, vmax=600)
ax[0].scatter(cx, cy, s=100, facecolors='none', edgecolors='r', label='Nodule') # 圈出结节
ax[0].set_title(f'Axial (Top-Down, Z={cz})')
ax[0].set_xlabel('X (Left-Right)')
ax[0].set_ylabel('Y (Anterior-Posterior)')

# --- Coronal (冠状面, XZ平面, Y=固定) ---
# 正面看肺，需要修复 Z 轴比例
ax[1].imshow(vol[:, cy, :], cmap='gray', vmin=-1200, vmax=600, aspect=aspect_ratio)
ax[1].scatter(cx, cz, s=100, facecolors='none', edgecolors='r')
ax[1].set_title(f'Coronal (Front-Back, Y={cy})')
ax[1].set_xlabel('X')
ax[1].set_ylabel('Z (Head-Feet)')
ax[1].invert_yaxis() # 图像坐标系原点在左上，医学习惯 Z 轴向上或向下，这里保持一致

# --- Sagittal (矢状面, YZ平面, X=固定) ---
# 侧面看肺，同样需要修复 Z 轴比例
ax[2].imshow(vol[:, :, cx], cmap='gray', vmin=-1200, vmax=600, aspect=aspect_ratio)
ax[2].scatter(cy, cz, s=100, facecolors='none', edgecolors='r')
ax[2].set_title(f'Sagittal (Side View, X={cx})')
ax[2].set_xlabel('Y')
ax[2].set_ylabel('Z')
ax[2].invert_yaxis()

plt.tight_layout()
plt.savefig(f"{pid}_full_view.png") # 保存图片
plt.show()