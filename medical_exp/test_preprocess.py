import sys
import os
import numpy as np
import pylidc as pl
import matplotlib.pyplot as plt

# 把 dld 目录加入系统路径（这一步没变，是为了找到 nnunetv2 文件夹）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ❌ 之前是 from nnunet_core ...
# ✅ 现在改回:
try:
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization
    print("✅ 成功加载 nnU-Net 模块！")
except ImportError as e:
    print(f"❌ 加载失败: {e}")
    # 这里加个提示，如果还没装 batchgenerators
    if 'batchgenerators' in str(e):
        print("提示: 你可能还没安装 batchgenerators，请运行 pip install batchgenerators")
    exit()

# === 配置 ===
PID = 'LIDC-IDRI-0078'  # 测试病人
TARGET_SPACING = np.array([1.0, 1.0, 1.0])

def test_pipeline():
    # 1. 读取数据 (Pylidc)
    print(f"正在读取 {PID}...")
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == PID).first()
    if not scan:
        print("未找到数据，请检查 pylidc 配置。")
        return

    vol = scan.to_volume()
    vol = vol.transpose(2, 0, 1)
    
    # Pylidc Spacing: (SliceThickness, PixelSpacing, PixelSpacing)
    src_spacing = np.array([scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing])
    
    print(f"原始形状: {vol.shape}")
    print(f"原始间距: {src_spacing}")

    # 2. 计算目标形状
    new_shape = np.round(vol.shape * src_spacing / TARGET_SPACING).astype(int)
    print(f"目标形状: {new_shape}")

    # 3. 执行 nnU-Net 重采样
    # 构造输入: (Batch/Channel, Z, Y, X) -> nnU-Net 需要 Channel 维度
    data_input = vol[np.newaxis, ...].astype(float)
    
    print("正在运行 nnU-Net 重采样 (这可能需要几秒钟)...")
    try:
        vol_resampled = resample_data_or_seg_to_shape(
            data=data_input,
            new_shape=new_shape,
            current_spacing=src_spacing,
            new_spacing=TARGET_SPACING,
            is_seg=False,   # 这是图像，不是 Mask
            order=3,        # 三次样条插值 (高质量)
            order_z=0,      # 有些时候为了速度选0，但3更好
            force_separate_z=False
        )
        print("✅ 重采样成功！")
        print(f"输出形状: {vol_resampled.shape}")
    except Exception as e:
        print(f"❌ 重采样出错: {e}")
        return

    # 4. 可视化对比 (保存图片)
    # 取中间切片
    mid_z_src = vol.shape[0] // 2
    mid_z_dst = vol_resampled.shape[1] // 2 # 注意 channel 维度在 0

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(vol[mid_z_src, :, :], cmap='gray')
    ax[0].set_title(f"Original\n{vol.shape}\n{src_spacing}")
    
    # vol_resampled 是 (1, D, H, W)
    ax[1].imshow(vol_resampled[0, mid_z_dst, :, :], cmap='gray')
    ax[1].set_title(f"Resampled (nnUNet)\n{vol_resampled.shape[1:]}\n{TARGET_SPACING}")
    
    plt.savefig("nnunet_test_result.png")
    print("✅ 结果已保存为 nnunet_test_result.png")

if __name__ == "__main__":
    test_pipeline()