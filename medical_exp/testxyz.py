import pylidc as pl
import numpy as np

pid = 'LIDC-IDRI-0078'
scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()

print(f"=== 维度验证: {pid} ===")

# 1. 打印 pylidc 提供的属性
# 注意：scan.pixel_spacing 在 pylidc 里通常是一个 float (假设正方形像素)
# 或者是 [RowSpacing, ColSpacing]
print(f"scan.pixel_spacing: {scan.pixel_spacing}")
print(f"scan.slice_thickness: {scan.slice_thickness}")

# 2. 如果你想看最原始的 DICOM 定义 (Row vs Col)
# 我们读取第一层 DICOM 的详细头信息
try:
    # 获取第一张切片的 DICOM 数据集
    # pylidc 内部会加载 dicom，我们可以通过这种hack方式拿一个来看看
    import pydicom
    import os
    dicom_path = scan.get_path_to_dicom_files()[0]
    dcm = pydicom.dcmread(dicom_path)
    
    print("\n=== DICOM Header (原始证据) ===")
    print(f"Rows (Y轴像素数): {dcm.Rows}")
    print(f"Columns (X轴像素数): {dcm.Columns}")
    # Pixel Spacing 顺序标准是: [RowSpacing, ColSpacing] -> [Y间距, X间距]
    print(f"Pixel Spacing (DICOM tag): {dcm.PixelSpacing}  <-- [Y_spacing, X_spacing]")
    
except Exception as e:
    print(f"\n无法读取原始 DICOM: {e}")

# 3. 验证 Volume 形状
vol = scan.to_volume()
print(f"\n=== Numpy Volume Shape ===")
print(f"vol.shape: {vol.shape}") 
print(f"对应关系: (Row={vol.shape[0]}, Col={vol.shape[1]}, Slice={vol.shape[2]})")
print(f"对应轴向: (Y, X, Z)")