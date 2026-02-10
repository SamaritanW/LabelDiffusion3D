import os
import requests
import zipfile
import io
import pylidc as pl

# ================= 配置区域 =================
# 1. 数据保存路径 
# (必须和你 ~/.pylidcrc 里配置的 path 保持一致，否则 pylidc 读不到下载好的文件)
data_root = "/data/users/baohengl/midl/DLD/data/LIDC-IDRI/raw"

# 2. 目标病例
patient_id = "LIDC-IDRI-0078"
# ===========================================

print(f"正在查询病例 {patient_id} 的 Series UID...")
# pylidc 会自动读取 ~/.pylidcrc
scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id)

if scans.count() == 0:
    print("错误：pylidc 未找到该病例元数据。请检查 ~/.pylidcrc 配置是否正确连接了数据库。")
    exit()

# 获取 UID
target_scan = scans.first()
series_uid = target_scan.series_instance_uid
print(f"获取成功！Series UID: {series_uid}")

# 构造下载路径：data_root/LIDC-IDRI-0078
save_dir = os.path.join(data_root, patient_id)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print(f"正在通过 NBIA API 下载数据 (不依赖 tcia_utils)...")
print(f"目标目录: {save_dir}")

# === 核心：直接调用 API 下载 (稳健版) ===
url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
params = {
    "SeriesInstanceUID": series_uid
}

try:
    # 发起请求
    response = requests.get(url, params=params, stream=True)
    
    if response.status_code == 200:
        print("下载成功 (200 OK)！正在解压...")
        # API 返回的是 ZIP 流，直接在内存中解压
        try:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(save_dir)
            print(f"✅ 解压完成！文件已保存在 {save_dir}")
            
            # 验证环节
            print("-" * 30)
            print("正在验证 pylidc 是否能读取本地文件...")
            images = target_scan.load_all_dicom_images()
            print(f"✅ 验证通过！pylidc 成功加载了 {len(images)} 张切片。")
            print("数据管道已打通！你可以开始写 DataLoader 了。")
            print("-" * 30)
            
        except zipfile.BadZipFile:
            print("❌ 解压失败：下载的文件不是有效的 ZIP。可能 API 返回了错误信息。")
    else:
        print(f"❌ 下载请求失败，状态码: {response.status_code}")
        print(f"服务器响应: {response.text}")

except Exception as e:
    print(f"❌ 发生异常: {e}")