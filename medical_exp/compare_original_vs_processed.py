import sys
import os
import numpy as np
import pylidc as pl
import matplotlib.pyplot as plt

# === 配置 ===
PID = 'LIDC-IDRI-0078'
NODULE_IDX = 0          
ORIGINAL_Z_TO_VIEW = 26 
PROCESSED_DIR = "data/LIDC-IDRI/processed_check" 

# === 绘图风格设置 ===
# 肺窗设置 (Lung Window)
# 也是 ITK-SNAP 等软件默认的肺部显示范围
VMIN, VMAX = -1200, 600 

def compare_views():
    # ==========================
    # 1. 准备数据
    # ==========================
    print(f"🔄 读取原始数据: {PID}...")
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == PID).first()
    vol_orig = scan.to_volume().transpose(2, 0, 1) # (Z, Y, X)
    
    # 获取标注
    cluster = scan.cluster_annotations(verbose=False)[NODULE_IDX]
    c_yxz = np.mean([a.centroid for a in cluster], axis=0)
    c_orig_z, c_orig_y, c_orig_x = int(c_yxz[2]), int(c_yxz[0]), int(c_yxz[1])

    # 计算 物理纵横比 (Aspect Ratio)
    # 这就是让图片不再“扁”的关键
    # 比如层厚 3.0mm, 像素 0.65mm -> ratio ≈ 4.6
    aspect_ratio = scan.slice_thickness / scan.pixel_spacing
    print(f"📐 计算显示比例 (Aspect Ratio): {aspect_ratio:.2f}")

    # 加载预处理数据
    npy_img_path = os.path.join(PROCESSED_DIR, f"{PID}_nodule{NODULE_IDX}_image.npy")
    npy_msk_path = os.path.join(PROCESSED_DIR, f"{PID}_nodule{NODULE_IDX}_masks.npy")
    
    if not os.path.exists(npy_img_path):
        print("❌ 找不到预处理文件，请先运行上一步脚本！")
        return

    vol_proc = np.load(npy_img_path)[0]
    msk_proc = np.load(npy_msk_path)
    c_proc = 32 # 预处理切片的中心

    # ==========================
    # 2. 开始绘图 (高清版)
    # ==========================
    # dpi=300 保证文字和线条清晰，适合发论文
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    colors = ['red', 'blue', 'lime', 'yellow']
    
    # 通用美化函数
    def clean_ax(ax, title):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off') # 隐藏坐标轴刻度，更像专业软件截图

    # --- 第一排：原始数据 (Original) ---
    
    # A. Axial (横断面) - 比例 1:1
    ax = axes[0, 0]
    # interpolation='bicubic' 会让图像看起来更平滑，像医院胶片一样
    ax.imshow(vol_orig[ORIGINAL_Z_TO_VIEW], cmap='gray', vmin=VMIN, vmax=VMAX, interpolation='bicubic')
    
    # 画 Mask (细致处理)
    for i, ann in enumerate(cluster):
        bbox = ann.bbox()
        if ORIGINAL_Z_TO_VIEW in range(bbox[2].start, bbox[2].stop):
            m = ann.boolean_mask(pad=0)[:, :, ORIGINAL_Z_TO_VIEW - bbox[2].start]
            full_m = np.zeros(vol_orig.shape[1:], dtype=bool)
            full_m[bbox[0], bbox[1]] = m
            # linewidths=1.5 稍微加粗一点点，更清晰
            ax.contour(full_m, levels=[0.5], colors=colors[i%4], linewidths=1.5)
    clean_ax(ax, f"Original Axial (Z={ORIGINAL_Z_TO_VIEW})\nRaw CT")

    # B. Coronal (冠状面) - 🚨 关键：使用 aspect_ratio 拉伸
    ax = axes[0, 1]
    ax.imshow(vol_orig[:, c_orig_y, :], cmap='gray', vmin=VMIN, vmax=VMAX, 
              aspect=aspect_ratio, interpolation='bicubic') # <--- 这里加了 aspect
    
    for i, ann in enumerate(cluster):
        bbox = ann.bbox()
        if c_orig_y in range(bbox[0].start, bbox[0].stop):
            m = ann.boolean_mask(pad=0)[c_orig_y - bbox[0].start, :, :].T 
            full_m = np.zeros((vol_orig.shape[0], vol_orig.shape[2]), dtype=bool)
            full_m[bbox[2], bbox[1]] = m
            ax.contour(full_m, levels=[0.5], colors=colors[i%4], linewidths=1.5)
    ax.invert_yaxis()
    clean_ax(ax, f"Original Coronal (Y={c_orig_y})\nAspect Corrected")

    # C. Sagittal (矢状面) - 🚨 关键：使用 aspect_ratio 拉伸
    ax = axes[0, 2]
    ax.imshow(vol_orig[:, :, c_orig_x], cmap='gray', vmin=VMIN, vmax=VMAX, 
              aspect=aspect_ratio, interpolation='bicubic') # <--- 这里加了 aspect
    
    for i, ann in enumerate(cluster):
        bbox = ann.bbox()
        if c_orig_x in range(bbox[1].start, bbox[1].stop):
            m = ann.boolean_mask(pad=0)[:, c_orig_x - bbox[1].start, :].T
            full_m = np.zeros((vol_orig.shape[0], vol_orig.shape[1]), dtype=bool)
            full_m[bbox[2], bbox[0]] = m
            ax.contour(full_m, levels=[0.5], colors=colors[i%4], linewidths=1.5)
    ax.invert_yaxis()
    clean_ax(ax, f"Original Sagittal (X={c_orig_x})\nAspect Corrected")


    # --- 第二排：预处理后 (Processed) ---
    # 这里的 aspect 默认是 1 (因为我们已经重采样到 1mm x 1mm x 1mm 了)
    
    # A. Axial
    ax = axes[1, 0]
    ax.imshow(vol_proc[c_proc], cmap='gray', vmin=0, vmax=1, interpolation='bicubic')
    for i in range(msk_proc.shape[0]):
        ax.contour(msk_proc[i, c_proc, :, :], levels=[0.5], colors=colors[i%4], linewidths=1.5)
    clean_ax(ax, f"Processed Input (Z=32)\nIsotropic 1mm")

    # B. Coronal
    ax = axes[1, 1]
    ax.imshow(vol_proc[:, c_proc, :], cmap='gray', vmin=0, vmax=1, interpolation='bicubic')
    for i in range(msk_proc.shape[0]):
        ax.contour(msk_proc[i, :, c_proc, :], levels=[0.5], colors=colors[i%4], linewidths=1.5)
    ax.invert_yaxis()
    clean_ax(ax, "Processed Coronal")

    # C. Sagittal
    ax = axes[1, 2]
    ax.imshow(vol_proc[:, :, c_proc], cmap='gray', vmin=0, vmax=1, interpolation='bicubic')
    for i in range(msk_proc.shape[0]):
        ax.contour(msk_proc[i, :, :, c_proc], levels=[0.5], colors=colors[i%4], linewidths=1.5)
    ax.invert_yaxis()
    clean_ax(ax, "Processed Sagittal")

    # 保存高清图
    plt.tight_layout()
    save_path = "compare_result_hd.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # 300 DPI 印刷级清晰度
    print(f"\n✅ 高清对比图已生成: {save_path}")
    print("💡 技巧: 下排图片会显得比较'大颗粒'(模糊)，这是正常的。")
    print("   因为上排是 0.65mm 高分辨率显示，下排是 1.0mm 重采样后被放大显示的。")

if __name__ == "__main__":
    compare_views()