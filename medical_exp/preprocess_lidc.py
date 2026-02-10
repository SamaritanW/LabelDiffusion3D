import os
import numpy as np
import pylidc as pl

project_root = "/data/users/baohengl/midl/DLD"
processed_dir = os.path.join(project_root, "data/LIDC-IDRI/processed")
os.makedirs(processed_dir, exist_ok=True)
CROP_SIZE = 64 

def process_scan(scan):
    print(f"处理: {scan.patient_id}")
    nods = scan.cluster_annotations(verbose=False)
    try:
        vol = scan.to_volume()
    except: return

    for i, cluster in enumerate(nods):
        if len(cluster) < 3: continue
        
        # 计算中心
        centroids = [a.centroid for a in cluster]
        center = np.mean(centroids, axis=0)
        cz, cy, cx = int(center[0]), int(center[1]), int(center[2])

        # 定义 Crop 区域
        c_size = CROP_SIZE
        z_s, z_e = max(0, cz-c_size//2), min(vol.shape[0], cz+c_size//2)
        y_s, y_e = max(0, cy-c_size//2), min(vol.shape[1], cy+c_size//2)
        x_s, x_e = max(0, cx-c_size//2), min(vol.shape[2], cx+c_size//2)

        # 切 Image
        img_crop = vol[z_s:z_e, y_s:y_e, x_s:x_e]
        # Pad Image if needed (code omitted for brevity, assume center is safe for demo)
        # 为演示简便，如果尺寸不对直接跳过
        if img_crop.shape != (CROP_SIZE, CROP_SIZE, CROP_SIZE):
            # 简单的 Padding 逻辑
            pad = ((0, c_size-img_crop.shape[0]), (0, c_size-img_crop.shape[1]), (0, c_size-img_crop.shape[2]))
            img_crop = np.pad(img_crop, pad, 'constant', constant_values=-1000)

        # === 核心：切 Mask ===
        masks_stack = []
        for ann in cluster:
            # 1. 生成该标注的局部 Boolean Mask
            m_bbox = ann.boolean_mask(pad=0)
            bbox = ann.bbox()
            
            # 2. 创建一个空的 Crop 大小的 Mask
            m_crop = np.zeros((CROP_SIZE, CROP_SIZE, CROP_SIZE), dtype=bool)
            
            # 3. 计算对齐坐标
            # Mask 在全局的起始点
            bz, by, bx = bbox[0].start, bbox[1].start, bbox[2].start
            
            # 计算 Mask 相对于 Crop 原点的偏移
            oz, oy, ox = bz - z_s, by - y_s, bx - x_s
            
            # 4. 填充 (需要处理重叠区域)
            # 计算 m_bbox 和 m_crop 的重叠区域
            # Mask 内部区间
            m_z_s = max(0, -oz)
            m_z_e = min(m_bbox.shape[0], CROP_SIZE - oz)
            m_y_s = max(0, -oy)
            m_y_e = min(m_bbox.shape[1], CROP_SIZE - oy)
            m_x_s = max(0, -ox)
            m_x_e = min(m_bbox.shape[2], CROP_SIZE - ox)
            
            # Crop 内部区间
            c_z_s = max(0, oz)
            c_z_e = min(CROP_SIZE, oz + m_bbox.shape[0])
            c_y_s = max(0, oy)
            c_y_e = min(CROP_SIZE, oy + m_bbox.shape[1])
            c_x_s = max(0, ox)
            c_x_e = min(CROP_SIZE, ox + m_bbox.shape[2])
            
            try:
                m_crop[c_z_s:c_z_e, c_y_s:c_y_e, c_x_s:c_x_e] = \
                    m_bbox[m_z_s:m_z_e, m_y_s:m_y_e, m_x_s:m_x_e]
                masks_stack.append(m_crop)
            except:
                print("Mask 对齐失败，跳过该 Rater")
                continue

        if len(masks_stack) < 3: continue

        # 堆叠 Masks: (N_Rater, 64, 64, 64)
        masks_array = np.stack(masks_stack).astype(np.uint8)

        # 保存
        name = f"{scan.patient_id}_nodule{i}"
        np.save(os.path.join(processed_dir, f"{name}_image.npy"), img_crop)
        np.save(os.path.join(processed_dir, f"{name}_masks.npy"), masks_array)
        print(f"  -> 保存成功: Image {img_crop.shape}, Masks {masks_array.shape}")

# Run
pid = 'LIDC-IDRI-0078'
scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid)
if scans.count() > 0: process_scan(scans.first())