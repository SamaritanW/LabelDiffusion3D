import pylidc as pl
import os

# 随便查一个病人
scan = pl.query(pl.Scan).first()
print(f"当前病人 ID: {scan.patient_id}")

# 看看 pylidc 认为文件在哪里
dicom_path = scan.get_path_to_dicom_files()
print(f"pylidc 认为 DICOM 在: {dicom_path}")

if os.path.exists(dicom_path):
    print("✅ 路径存在！")
    try:
        vol = scan.to_volume()
        print(f"✅ 成功读取 Volume，形状: {vol.shape}")
    except Exception as e:
        print(f"❌ 读取 Volume 失败: {e}")
else:
    print(f"❌ 路径不存在！请检查 ~/.pylidcrc 中的 path 配置是否正确。")
    print("注意: path 应该指向包含 LIDC-IDRI-xxxx 文件夹的那个父目录。")