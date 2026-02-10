import sys
import os
import numpy as np
import pylidc as pl
import matplotlib.pyplot as plt
from tqdm import tqdm # 引入进度条

# === 1. 环境设置 ===
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 尝试加载 nnU-Net 重采样工具 (如果没有安装，可以用 scipy.ndimage zoom 代替，但 nnU-Net 更好)
try:
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    print("✅ 成功加载 nnU-Net 模块")
    USE_NNUNET = True
except ImportError:
    print("⚠️ 无法加载 nnunetv2，将使用 scipy.ndimage (可能稍慢)")
    from scipy.ndimage import zoom
    USE_NNUNET = False

# === 配置 ===
OUTPUT_DIR = "data/LIDC-IDRI/process" # 改成你全量数据的存放目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SPACING = np.array([1.0, 1.0, 1.0]) # 目标间距 (mm)
CROP_SIZE = 64 # 切块大小

# === 辅助函数: 重采样 (兼容 nnU-Net 和 Scipy) ===
def resample_volume(volume, current_spacing, target_spacing, is_seg=False):
    new_shape = np.round(volume.shape * current_spacing / target_spacing).astype(int)
    
    if USE_NNUNET:
        # 使用 nnU-Net 的高精度重采样
        # nnU-Net 要求 input 是 (c, z, y, x)
        data = volume[None, ...].astype(float)
        order = 0 if is_seg else 3
        resampled = resample_data_or_seg_to_shape(
            data=data,
            new_shape=new_shape,
            current_spacing=current_spacing,
            new_spacing=target_spacing,
            is_seg=is_seg, order=order, order_z=0, force_separate_z=False
        )[0]
        return resampled, new_shape
    else:
        # 使用 scipy.ndimage (备选方案)
        resize_factor = target_spacing / current_spacing
        order = 0 if is_seg else 3
        resampled = zoom(volume, resize_factor, order=order, mode='nearest')
        return resampled, resampled.shape

# === 核心处理逻辑 ===
def process_all_patients():
    # 1. 查询所有扫描
    print("🔍正在查询数据库中的所有病人...")
    scans = pl.query(pl.Scan).all()
    print(f"🚀 共找到 {len(scans)} 个扫描，开始全量处理...")
    
    # 2. 遍历处理 (带进度条)
    for scan in tqdm(scans, desc="Processing Patients"):
        pid = scan.patient_id
        
        try:
            process_single_scan(scan, pid)
        except Exception as e:
            print(f"\n❌ 处理病人 {pid} 时出错: {str(e)}")
            continue # 跳过出错的病人，继续下一个

def process_single_scan(scan, pid):
    # 1. 读取 CT (转为 Z, Y, X)
    try:
        vol = scan.to_volume(verbose=False) # (Y, X, Z)
        vol = vol.transpose(2, 0, 1) # -> (Z, Y, X)
    except Exception as e:
        # 有些 DICOM 文件可能损坏
        print(f"  ⚠️ {pid}: 无法读取 DICOM")
        return

    src_spacing = np.array([scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing])
    
    # 2. 归一化 (Windowing -1200 ~ 600)
    vol = np.clip(vol, -1200, 600)
    vol = (vol - (-1200)) / 1800.0 # [0, 1]

    # 3. 重采样图像
    vol_resampled, new_shape = resample_volume(vol, src_spacing, TARGET_SPACING, is_seg=False)
    
    # 计算缩放因子 (用于坐标映射)
    resize_factor = new_shape / vol.shape

    # 4. 聚类结节
    nods = scan.cluster_annotations(verbose=False)
    
    for i, cluster in enumerate(nods):
        # 只要 >= 3 人标注的强共识结节
        if len(cluster) < 3: continue
            
        # === A. 计算中心坐标 ===
        # pylidc centroid 是 (Y, X, Z)
        centroids_yxz = [a.centroid for a in cluster]
        center_yxz = np.mean(centroids_yxz, axis=0)
        # 转为 (Z, Y, X)
        center_zyx = np.array([center_yxz[2], center_yxz[0], center_yxz[1]])
        
        # 映射到重采样后的坐标
        center_res = center_zyx * resize_factor
        cz, cy, cx = int(center_res[0]), int(center_res[1]), int(center_res[2])

        # === B. 切片 (Padding 逻辑) ===
        # Z轴范围
        z_s, z_e = cz - CROP_SIZE//2, cz + CROP_SIZE//2
        # Y轴范围
        y_s, y_e = cy - CROP_SIZE//2, cy + CROP_SIZE//2
        # X轴范围
        x_s, x_e = cx - CROP_SIZE//2, cx + CROP_SIZE//2
        
        # 检查是否越界严重
        if z_s < -CROP_SIZE or z_e > vol_resampled.shape[0] + CROP_SIZE:
             continue # 这种通常是坐标算飞了
             
        # 安全提取 CT 切块
        img_crop = safe_crop(vol_resampled, z_s, z_e, y_s, y_e, x_s, x_e)
        
        if img_crop.shape != (CROP_SIZE, CROP_SIZE, CROP_SIZE):
            continue

        # === C. 处理 Masks (最耗时的一步) ===
        masks_stack = []
        for ann in cluster:
            # 1. 获取局部 Mask (Y, X, Z) -> (Z, Y, X)
            m_local = ann.boolean_mask(pad=0).transpose(2, 0, 1)
            bbox = ann.bbox() # (slice_y, slice_x, slice_z)
            
            # 2. 还原到全图尺寸 (必须这样做才能准确对齐)
            full_mask = np.zeros(vol.shape, dtype=np.uint8)
            # 注意 bbox 顺序是 y, x, z，对应 full_mask 的 Z, Y, X 轴
            full_mask[bbox[2], bbox[0], bbox[1]] = m_local
            
            # 3. 重采样 Mask (最近邻插值)
            m_res, _ = resample_volume(full_mask, src_spacing, TARGET_SPACING, is_seg=True)
            
            # 4. 切取 Mask
            m_crop = safe_crop(m_res, z_s, z_e, y_s, y_e, x_s, x_e)
            masks_stack.append(m_crop)
            
        masks_arr = np.stack(masks_stack).astype(np.uint8)

        # === D. 保存 ===
        # 格式: LIDC-IDRI-0078_nodule0_image.npy
        name_base = f"{pid}_nodule{i}"
        
        # Image 增加 Channel 维 -> (1, 64, 64, 64)
        # Masks 保持 -> (N, 64, 64, 64)
        np.save(os.path.join(OUTPUT_DIR, f"{name_base}_image.npy"), img_crop[None, ...].astype(np.float32))
        np.save(os.path.join(OUTPUT_DIR, f"{name_base}_masks.npy"), masks_arr)

def safe_crop(volume, z_s, z_e, y_s, y_e, x_s, x_e):
    """
    带 Padding 的安全切片函数
    """
    # 原始尺寸
    D, H, W = volume.shape
    
    # 计算有效区域 (Intersection)
    vz_s, vz_e = max(0, z_s), min(D, z_e)
    vy_s, vy_e = max(0, y_s), min(H, y_e)
    vx_s, vx_e = max(0, x_s), min(W, x_e)
    
    # 提取有效区域
    crop = volume[vz_s:vz_e, vy_s:vy_e, vx_s:vx_e]
    
    # 计算需要 Pad 多少
    # 前面补: max(0, -z_s)
    # 后面补: max(0, z_e - D)
    pad_z = (max(0, -z_s), max(0, z_e - D))
    pad_y = (max(0, -y_s), max(0, y_e - H))
    pad_x = (max(0, -x_s), max(0, x_e - W))
    
    return np.pad(crop, (pad_z, pad_y, pad_x), mode='constant', constant_values=0)

if __name__ == "__main__":
    process_all_patients()