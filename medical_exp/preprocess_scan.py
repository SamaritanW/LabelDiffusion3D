import os
import numpy as np
import pylidc as pl
from scipy.ndimage.interpolation import zoom

# === 1. 搬运 DeepLung 的核心工具函数 (来源: 行 18 和 行 79) ===

def resample(imgs, spacing, new_spacing, order=2):
    """
    来源: DeepLung preprocess.py Line 18
    强制将图像缩放到 new_spacing (通常是 1mm)
    Order=2 或 1 用于图像(线性插值), Order=0 用于 Mask(最近邻)
    """
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        # mode='nearest' 是处理边界填充，order 是插值方式
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, resize_factor
    else:
        raise ValueError('Shape mismatch')

def lumTrans(img):
    """
    来源: DeepLung preprocess.py Line 79
    窗口化: [-1200, 600] -> [0, 255] (uint8)
    """
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg

# === 2. 你的主处理逻辑 (魔改版) ===

project_root = "/data/users/baohengl/midl/DLD"
processed_dir = os.path.join(project_root, "data/LIDC-IDRI/processed")
os.makedirs(processed_dir, exist_ok=True)
CROP_SIZE = 64
TARGET_SPACING = np.array([1.0, 1.0, 1.0]) # 来源: DeepLung Line 254 (resolution)

def process_scan(scan):
    print(f"处理: {scan.patient_id}")
    
    # 1. 获取原始数据
    try:
        vol = scan.to_volume() # 读取原始 HU 值
    except: return
    
    # 获取原始 Spacing (Z, Y, X)
    # pylidc 的 spacing 顺序是 (slice_thickness, pixel_spacing_x, pixel_spacing_y)
    spacing = np.array([scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing], dtype=np.float32)

    # === 改动 A: 先做 LumTrans (参考 DeepLung Line 306) ===
    # DeepLung 是先 mask 再 lumTrans 再 resample，但我们这里直接对全图操作
    # 必须先处理成 uint8，否则 zoom 插值可能会产生奇怪的 HU 值
    vol_uint8 = lumTrans(vol)

    # === 改动 B: 重采样 Volume (参考 DeepLung Line 310) ===
    # order=1 (线性插值) 用于图像，和 DeepLung 保持一致
    vol_resampled, resize_factor = resample(vol_uint8, spacing, TARGET_SPACING, order=1)

    nods = scan.cluster_annotations(verbose=False)
    for i, cluster in enumerate(nods):
        if len(cluster) < 3: continue
        
        # === 改动 C: 坐标变换 (参考 DeepLung Line 344) ===
        centroids = [a.centroid for a in cluster]
        center_orig = np.mean(centroids, axis=0)
        
        # 核心：原坐标 * 缩放因子 = 新坐标
        center_resampled = center_orig * resize_factor
        cz, cy, cx = int(center_resampled[0]), int(center_resampled[1]), int(center_resampled[2])

        # 定义 Crop 区域 (在重采样后的图上切)
        c_size = CROP_SIZE
        z_s, z_e = max(0, cz-c_size//2), min(vol_resampled.shape[0], cz+c_size//2)
        y_s, y_e = max(0, cy-c_size//2), min(vol_resampled.shape[1], cy+c_size//2)
        x_s, x_e = max(0, cx-c_size//2), min(vol_resampled.shape[2], cx+c_size//2)

        # 切 Image
        img_crop = vol_resampled[z_s:z_e, y_s:y_e, x_s:x_e]
        
        # Padding (如果切到边界)
        if img_crop.shape != (CROP_SIZE, CROP_SIZE, CROP_SIZE):
            pad = ((0, c_size-img_crop.shape[0]), (0, c_size-img_crop.shape[1]), (0, c_size-img_crop.shape[2]))
            # DeepLung 用 170 (灰度) 填充背景 (Line 304 pad_value=170)
            img_crop = np.pad(img_crop, pad, 'constant', constant_values=170) 

        # === 改动 D: Mask 的重采样与对齐 ===
        masks_stack = []
        for ann in cluster:
            # 1. 拿到该 Annotation 的局部 Mask (原始分辨率)
            m_bbox = ann.boolean_mask(pad=0) 
            bbox = ann.bbox()
            
            # 2. 为了准确重采样，必须把局部 Mask 放回全图尺寸，或者计算局部缩放
            # 最稳妥的方法：构建一个和原图一样大的稀疏 Mask，缩放后再切
            # (虽然慢一点，但绝对对齐，DeepLung 是处理整个 Lung Mask 的)
            full_mask = np.zeros(vol.shape, dtype=bool)
            full_mask[bbox] = m_bbox
            
            # 3. 重采样 Mask (必须用 order=0Nearest Neighbor，否则 0/1 变小数)
            full_mask_resampled, _ = resample(full_mask, spacing, TARGET_SPACING, order=0)
            
            # 4. 在新坐标上切
            m_crop = full_mask_resampled[z_s:z_e, y_s:y_e, x_s:x_e]
            
            # Padding
            if m_crop.shape != (CROP_SIZE, CROP_SIZE, CROP_SIZE):
                pad = ((0, c_size-m_crop.shape[0]), (0, c_size-m_crop.shape[1]), (0, c_size-m_crop.shape[2]))
                m_crop = np.pad(m_crop, pad, 'constant', constant_values=0)
            
            masks_stack.append(m_crop)

        if len(masks_stack) < 3: continue
        masks_array = np.stack(masks_stack).astype(np.uint8)

        # 保存 (注意：img_crop 已经是 uint8 [0-255])
        name = f"{scan.patient_id}_nodule{i}"
        np.save(os.path.join(processed_dir, f"{name}_image.npy"), img_crop)
        np.save(os.path.join(processed_dir, f"{name}_masks.npy"), masks_array)
        print(f"  -> 保存: {name}, Image {img_crop.shape}")

# Run
pid = 'LIDC-IDRI-0078'
scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid)
if scans.count() > 0: process_scan(scans.first())