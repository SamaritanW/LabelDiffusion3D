import os


import pylidc as pl

# ==========================================
# 纯查询逻辑
# ==========================================
print("正在连接数据库查询病例...")
try:
    # 过滤条件：层厚小于 2.5mm 的扫描 (只选高质量CT)
    scans = pl.query(pl.Scan).filter(pl.Scan.pixel_spacing <= 2.5)
    count = scans.count()
    print(f"查询成功！数据库中共有 {count} 个符合条件的扫描。")
    
    if count > 0:
        # 打印第一个病例的信息，验证配置是否生效
        scan = scans.first()
        print(f"\n[验证] 示例病例 ID: {scan.patient_id}")
        print(f"[验证] 数据的下载/读取路径指向: {scan.get_path_to_dicom_files()}")
        print("配置正常，可以开始编写 DataLoader 了。")
    else:
        print("警告：没有找到符合条件的扫描，请检查过滤条件。")

except Exception as e:
    print(f"发生错误: {e}")
    print("可能是数据库连接失败，或者路径配置有误。")
