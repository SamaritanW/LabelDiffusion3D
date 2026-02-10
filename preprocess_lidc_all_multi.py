import sys
import os
import numpy as np
import pylidc as pl
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# === 1. 环境与配置 ===
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 尝试加载 nnU-Net
try:
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    USE_NNUNET = True
    print("✅ [Main] 成功加载 nnU-Net 模块")
except ImportError:
    from scipy.ndimage import zoom
    USE_NNUNET = False
    print("⚠️ [Main] 使用 scipy.ndimage")

# === 🚨 关键配置: 输出目录 ===
# 建议写绝对路径，防止找不到
OUTPUT_DIR = "/data/users/baohengl/midl/DLD/data/LIDC-IDRI/process" 

TARGET_SPACING = np.array([1.0, 1.0, 1.0])
CROP_SIZE = 64

# === 2. 核心逻辑函数 (Worker) ===
def process_single_patient(pid):
    """
    单个病人处理逻辑，将被多进程调用
    """
    # 重新连接数据库
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    if not scan:
        return f"Skip: {pid}"

    try:
        # --- A. 读取数据 ---
        # pylidc 会自动去 ~/.pylidcrc 配置的路径找 DICOM
        vol = scan.to_volume(verbose=False)
        vol = vol.transpose(2, 0, 1) # (Z, Y, X)
        
        src_spacing = np.array([scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing])
        
        # --- B. 归一化 ---
        vol = np.clip(vol, -1200, 600)
        vol = (vol - (-1200)) / 1800.0

        # --- C. 重采样 (Image) ---
        new_shape = np.round(vol.shape * src_spacing / TARGET_SPACING).astype(int)
        
        if USE_NNUNET:
            vol_resampled = resample_data_or_seg_to_shape(
                data=vol[None, ...].astype(float),
                new_shape=new_shape,
                current_spacing=src_spacing,
                new_spacing=TARGET_SPACING,
                is_seg=False, order=3, order_z=0, force_separate_z=False
            )[0]
        else:
            resize_factor = TARGET_SPACING / src_spacing
            vol_resampled = zoom(vol, resize_factor, order=3)

        resize_factor = new_shape / vol.shape 

        # --- D. 处理结节 ---
        nods = scan.cluster_annotations(verbose=False)
        
        save_count = 0
        for i, cluster in enumerate(nods):
            # 过滤: 只保留 >= 3 人标注的
            if len(cluster) < 3: continue 
            
            # 1. 坐标修正
            centroids_yxz = [a.centroid for a in cluster]
            center_yxz = np.mean(centroids_yxz, axis=0)
            center_zyx = np.array([center_yxz[2], center_yxz[0], center_yxz[1]])
            
            center_res = center_zyx * resize_factor
            cz, cy, cx = int(center_res[0]), int(center_res[1]), int(center_res[2])

            # 2. 切片范围
            z_s, z_e = cz - CROP_SIZE//2, cz + CROP_SIZE//2
            y_s, y_e = cy - CROP_SIZE//2, cy + CROP_SIZE//2
            x_s, x_e = cx - CROP_SIZE//2, cx + CROP_SIZE//2
            
            # 越界检查
            if z_s < -CROP_SIZE or z_e > vol_resampled.shape[0] + CROP_SIZE: continue
            
            # 3. 安全切片 (Image)
            pad_z = (max(0, -z_s), max(0, z_e - vol_resampled.shape[0]))
            pad_y = (max(0, -y_s), max(0, y_e - vol_resampled.shape[1]))
            pad_x = (max(0, -x_s), max(0, x_e - vol_resampled.shape[2]))
            
            sl_z = slice(max(0, z_s), min(vol_resampled.shape[0], z_e))
            sl_y = slice(max(0, y_s), min(vol_resampled.shape[1], y_e))
            sl_x = slice(max(0, x_s), min(vol_resampled.shape[2], x_e))
            
            crop_valid = vol_resampled[sl_z, sl_y, sl_x]
            img_crop = np.pad(crop_valid, (pad_z, pad_y, pad_x), mode='constant', constant_values=0)
            
            if img_crop.shape != (CROP_SIZE, CROP_SIZE, CROP_SIZE): continue

            # 4. 处理 Masks
            masks_stack = []
            for ann in cluster:
                m_local = ann.boolean_mask(pad=0).transpose(2, 0, 1)
                bbox = ann.bbox()
                
                full_mask = np.zeros(vol.shape, dtype=np.uint8)
                full_mask[bbox[2], bbox[0], bbox[1]] = m_local
                
                # 重采样 Mask
                if USE_NNUNET:
                    m_res = resample_data_or_seg_to_shape(
                        data=full_mask[None, ...].astype(float),
                        new_shape=new_shape,
                        current_spacing=src_spacing,
                        new_spacing=TARGET_SPACING,
                        is_seg=True, order=0, order_z=0, force_separate_z=False
                    )[0]
                else:
                     m_res = zoom(full_mask, TARGET_SPACING / src_spacing, order=0)

                # 切片 Mask
                m_valid = m_res[sl_z, sl_y, sl_x]
                m_crop = np.pad(m_valid, (pad_z, pad_y, pad_x), mode='constant', constant_values=0)
                masks_stack.append(m_crop)

            masks_arr = np.stack(masks_stack).astype(np.uint8)

            # --- E. 保存 ---
            name_base = f"{pid}_nodule{i}"
            np.save(os.path.join(OUTPUT_DIR, f"{name_base}_image.npy"), img_crop[None, ...].astype(np.float32))
            np.save(os.path.join(OUTPUT_DIR, f"{name_base}_masks.npy"), masks_arr)
            save_count += 1
            
        return f"OK: {pid} ({save_count} nodules)"

    except Exception as e:
        return f"Error {pid}: {str(e)}"

# === 3. 多进程启动入口 ===
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🔍 查询数据库中所有病人...")
    scans = pl.query(pl.Scan).all()
    # 拿到所有病人 ID (去重)
    all_pids = sorted(list(set([s.patient_id for s in scans])))
    print(f"🚀 准备处理 {len(all_pids)} 个病人...")
    print(f"💾 输出目录: {OUTPUT_DIR}")
    
    # 启动多进程
    # 留 2 个核给系统，剩下的全部用来跑数据
    workers = max(1, cpu_count() - 2)
    print(f"🔥 火力全开: 启动 {workers} 个进程并行处理...")
    
    with Pool(workers) as pool:
        # 使用 tqdm 显示进度
        results = list(tqdm(pool.imap_unordered(process_single_patient, all_pids), total=len(all_pids)))
    
    # 统计一下
    errors = [r for r in results if r.startswith("Error")]
    success = [r for r in results if r.startswith("OK")]
    
    print("\n" + "="*30)
    print(f"✅ 完成处理: {len(success)}")
    print(f"❌ 失败/跳过: {len(results) - len(success)}")
    if errors:
        print(f"⚠️ 错误样本 ({len(errors)}):")
        for e in errors[:5]: print(e)
    print("="*30)