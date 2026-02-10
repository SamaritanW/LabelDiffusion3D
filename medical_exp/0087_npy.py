import os
import sys
import numpy as np
import pylidc as pl

# 确保能找到 nnunetv2
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape

# === 配置 ===
PID = 'LIDC-IDRI-0087' # 🎯 指定 0087
OUTPUT_DIR = "data/LIDC-IDRI/processed_test" # 这里的测试数据单独放一个文件夹，方便看
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_SPACING = np.array([1.0, 1.0, 1.0]) 
CROP_SIZE = 64 

def process_single_case():
    print(f"🚀 开始处理: {PID}")
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == PID).first()
    if not scan: return print("❌ 没找到数据，请检查 ID")

    # 1. 读取 & 转置 (Z, Y, X)
    vol = scan.to_volume().transpose(2, 0, 1)
    src_spacing = np.array([scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing])
    
    # 2. 归一化 [-1200, 600] -> [0, 1]
    vol = np.clip(vol, -1200, 600)
    vol = (vol - (-1200)) / 1800.0

    # 3. 重采样 Image
    new_shape = np.round(vol.shape * src_spacing / TARGET_SPACING).astype(int)
    vol_resampled = resample_data_or_seg_to_shape(
        data=vol[None, ...].astype(float),
        new_shape=new_shape,
        current_spacing=src_spacing,
        new_spacing=TARGET_SPACING,
        is_seg=False, order=3, order_z=0, force_separate_z=False
    )[0]

    # 4. 切结节
    nods = scan.cluster_annotations(verbose=False)
    resize_factor = new_shape / vol.shape
    
    for i, cluster in enumerate(nods):
        if len(cluster) < 3: continue # 只要多人标注的
        
        # 算中心 (Z, Y, X)
        center = np.mean([a.centroid for a in cluster], axis=0) * resize_factor
        cz, cy, cx = map(int, center)
        
        # 算切框
        z_s, y_s, x_s = cz - 32, cy - 32, cx - 32
        z_e, y_e, x_e = z_s + 64, y_s + 64, x_s + 64
        
        # 简单的切片与 Padding 逻辑
        # (这里为了代码短，省略了边界检查的 Padding 代码，假设结节不在边缘)
        # 实际使用请复用之前完整的 preprocess_lidc.py
        img_crop = vol_resampled[z_s:z_e, y_s:y_e, x_s:x_e]
        
        # 处理 Masks
        masks = []
        for ann in cluster:
            m_bbox = ann.boolean_mask(pad=0).transpose(2, 0, 1)
            full_mask = np.zeros(vol.shape, dtype=np.uint8)
            bbox = ann.bbox()
            full_mask[bbox[2], bbox[0], bbox[1]] = m_bbox
            
            # 重采样 Mask
            m_res = resample_data_or_seg_to_shape(
                data=full_mask[None, ...].astype(float),
                new_shape=new_shape,
                current_spacing=src_spacing,
                new_spacing=TARGET_SPACING,
                is_seg=True, order=0, order_z=0, force_separate_z=False
            )[0]
            masks.append(m_res[z_s:z_e, y_s:y_e, x_s:x_e])
            
        # 保存
        masks_arr = np.stack(masks).astype(np.uint8)
        img_save = img_crop[None, ...].astype(np.float32) # (1, 64, 64, 64)
        
        name = f"{PID}_nodule{i}"
        np.save(f"{OUTPUT_DIR}/{name}_image.npy", img_save)
        np.save(f"{OUTPUT_DIR}/{name}_masks.npy", masks_arr)
        print(f"✅ 生成: {name} | Img: {img_save.shape} | Mask: {masks_arr.shape}")

process_single_case()