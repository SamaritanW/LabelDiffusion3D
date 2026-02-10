import sys
import os
import numpy as np
import pylidc as pl
import matplotlib.pyplot as plt

# === 1. 环境设置 ===
# 确保能找到 dld/nnunetv2
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    print("✅ 成功加载 nnU-Net 模块")
except ImportError:
    print("❌ 无法加载 nnunetv2，请检查路径或安装 batchgenerators")
    sys.exit(1)

# === 配置 ===
PID = 'LIDC-IDRI-0078'  # 指定病人 ID
OUTPUT_DIR = "data/LIDC-IDRI/processed_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SPACING = np.array([1.0, 1.0, 1.0]) # 目标间距 (mm)
CROP_SIZE = 64 # 切块大小

def process_and_visualize():
    print(f"🚀 开始处理病人: {PID}")
    
    # 1. 读取数据
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == PID).first()
    if not scan:
        print("❌ 未找到数据")
        return

    # 2. 图像转置 (Pylidc YXZ -> ZYX)
    vol = scan.to_volume() # (Y, X, Z)
    vol = vol.transpose(2, 0, 1) # -> (Z, Y, X)
    
    src_spacing = np.array([scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing])
    
    # 3. 归一化 (Windowing & Norm)
    vol = np.clip(vol, -1200, 600)
    vol = (vol - (-1200)) / 1800.0 # [0, 1]

    # 4. 计算重采样参数
    new_shape = np.round(vol.shape * src_spacing / TARGET_SPACING).astype(int)
    resize_factor = new_shape / vol.shape
    
    print(f"  原始形状(ZYX): {vol.shape}")
    print(f"  目标形状(ZYX): {new_shape}")

    # 5. 重采样图像 (Image)
    print("  ⏳ 正在重采样图像...")
    vol_resampled = resample_data_or_seg_to_shape(
        data=vol[None, ...].astype(float),
        new_shape=new_shape,
        current_spacing=src_spacing,
        new_spacing=TARGET_SPACING,
        is_seg=False, order=3, order_z=0, force_separate_z=False
    )[0]

    # 6. 处理结节
    nods = scan.cluster_annotations(verbose=False)
    
    for i, cluster in enumerate(nods):
        # 只要多人标注的 (>=3人)
        if len(cluster) < 3: continue
            
        print(f"\n  🔍 处理结节 {i} (共 {len(cluster)} 医生)...")

        # === A. 坐标修正 (核心!) ===
        # pylidc centroid 是 (Y, X, Z)
        centroids_yxz = [a.centroid for a in cluster]
        center_yxz = np.mean(centroids_yxz, axis=0)
        
        # 强制转换为 (Z, Y, X)
        center_zyx = np.array([center_yxz[2], center_yxz[0], center_yxz[1]])
        
        # 映射到新坐标系
        center_res = center_zyx * resize_factor
        cz, cy, cx = int(center_res[0]), int(center_res[1]), int(center_res[2])
        
        print(f"     中心坐标(ZYX): {cz}, {cy}, {cx}")

        # === B. 安全切片 (带 Padding) ===
        # 计算切框范围
        z_s, z_e = cz - CROP_SIZE//2, cz + CROP_SIZE//2
        y_s, y_e = cy - CROP_SIZE//2, cy + CROP_SIZE//2
        x_s, x_e = cx - CROP_SIZE//2, cx + CROP_SIZE//2
        
        # 提取图像 (利用 numpy 的 pad 避免越界)
        # 先计算需要 pad 多少
        pad_z = (max(0, -z_s), max(0, z_e - vol_resampled.shape[0]))
        pad_y = (max(0, -y_s), max(0, y_e - vol_resampled.shape[1]))
        pad_x = (max(0, -x_s), max(0, x_e - vol_resampled.shape[2]))
        
        # 修正切片索引到有效范围
        sl_z = slice(max(0, z_s), min(vol_resampled.shape[0], z_e))
        sl_y = slice(max(0, y_s), min(vol_resampled.shape[1], y_e))
        sl_x = slice(max(0, x_s), min(vol_resampled.shape[2], x_e))
        
        # 切取有效部分
        crop_valid = vol_resampled[sl_z, sl_y, sl_x]
        
        # 填充回 64x64x64
        img_crop = np.pad(crop_valid, (pad_z, pad_y, pad_x), mode='constant', constant_values=0)
        
        if img_crop.shape != (64, 64, 64):
            print(f"     ⚠️ 切片形状异常 {img_crop.shape}，跳过")
            continue

        # === C. 处理 Masks (核心修复!) ===
        masks_stack = []
        for ann in cluster:
            # 1. 获取局部 Mask (Y, X, Z) -> 转置 (Z, Y, X)
            m_local = ann.boolean_mask(pad=0).transpose(2, 0, 1)
            
            # 2. 还原到全图
            full_mask = np.zeros(vol.shape, dtype=np.uint8)
            bbox = ann.bbox() # (slice_y, slice_x, slice_z)
            
            # 赋值: full_mask[Z, Y, X]
            full_mask[bbox[2], bbox[0], bbox[1]] = m_local
            
            # 3. 重采样 Mask (最近邻)
            m_res = resample_data_or_seg_to_shape(
                data=full_mask[None, ...].astype(float),
                new_shape=new_shape,
                current_spacing=src_spacing,
                new_spacing=TARGET_SPACING,
                is_seg=True, order=0, order_z=0, force_separate_z=False
            )[0]
            
            # 4. 切取 Mask (同样的 Padding 逻辑)
            m_valid = m_res[sl_z, sl_y, sl_x]
            m_crop = np.pad(m_valid, (pad_z, pad_y, pad_x), mode='constant', constant_values=0)
            
            masks_stack.append(m_crop)
            
        masks_arr = np.stack(masks_stack).astype(np.uint8)

        # === D. 保存 ===
        name_base = f"{PID}_nodule{i}"
        # 增加 Channel 维给 Image: (1, 64, 64, 64)
        img_final = img_crop[None, ...].astype(np.float32)
        
        np.save(os.path.join(OUTPUT_DIR, f"{name_base}_image.npy"), img_final)
        np.save(os.path.join(OUTPUT_DIR, f"{name_base}_masks.npy"), masks_arr)
        print(f"     💾 保存完毕: {name_base}")

        # === E. 立即调用可视化验证 ===
        visualize_single_nodule(img_final[0], masks_arr, name_base)

def visualize_single_nodule(img, masks, title):
    """
    画出正中心切片的三视图，验证位置和对齐
    """
    cz, cy, cx = 32, 32, 32 # 既然 crop 是 64，中心就是 32
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    colors = ['red', 'blue', 'lime', 'yellow']
    
    # 1. Axial (Z=32)
    ax[0].imshow(img[cz, :, :], cmap='gray', vmin=0, vmax=1)
    for k in range(masks.shape[0]):
        ax[0].contour(masks[k, cz, :, :], levels=[0.5], colors=colors[k%4], linewidths=1)
    ax[0].set_title(f"{title}\nAxial (Z)")

    # 2. Coronal (Y=32)
    ax[1].imshow(img[:, cy, :], cmap='gray', vmin=0, vmax=1)
    for k in range(masks.shape[0]):
        ax[1].contour(masks[k, :, cy, :], levels=[0.5], colors=colors[k%4], linewidths=1)
    ax[1].set_title("Coronal (Y)")
    ax[1].invert_yaxis() # 修正医学坐标

    # 3. Sagittal (X=32)
    ax[2].imshow(img[:, :, cx], cmap='gray', vmin=0, vmax=1)
    for k in range(masks.shape[0]):
        ax[2].contour(masks[k, :, :, cx], levels=[0.5], colors=colors[k%4], linewidths=1)
    ax[2].set_title("Sagittal (X)")
    ax[2].invert_yaxis()

    save_path = os.path.join(OUTPUT_DIR, f"{title}_check.png")
    plt.savefig(save_path)
    plt.close()
    print(f"     🖼️  验证图已生成: {save_path}")

if __name__ == "__main__":
    process_and_visualize()