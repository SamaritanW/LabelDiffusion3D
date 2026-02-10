import os
import requests
import zipfile
import io
import pylidc as pl
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# ================= 配置区域 =================
# 必须和你 ~/.pylidcrc 配置的 path 一致
DATA_ROOT = "/data/users/baohengl/midl/DLD/data/LIDC-IDRI/raw" 
# ===========================================

def download_patient(pid):
    """
    单个病人下载逻辑 (Worker函数)
    """
    # 1. 检查是否已存在 (断点续传)
    save_dir = os.path.join(DATA_ROOT, pid)
    
    # 简单判断: 如果文件夹存在且里面有文件，默认跳过
    # (如果想强制重新下载，可以删掉这个判断)
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        return f"Skip (Exists): {pid}"

    # 2. 查询 Series UID
    # 注意: 这里需要重新建立连接，不能用全局变量
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid)
    if scans.count() == 0:
        return f"Error: {pid} metadata not found in pylidc"
    
    # LIDC 每个病人可能有多个 Scan (CT/DX)，通常我们取第一个 CT
    # pylidc 默认过滤的通常就是主要的 CT Scan
    scan = scans.first()
    series_uid = scan.series_instance_uid
    
    # 3. 下载
    url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
    params = {
        "SeriesInstanceUID": series_uid
    }
    
    try:
        # 设置超时，防止卡死
        response = requests.get(url, params=params, stream=True, timeout=60)
        
        if response.status_code == 200:
            os.makedirs(save_dir, exist_ok=True)
            try:
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall(save_dir)
                return f"Success: {pid}"
            except zipfile.BadZipFile:
                return f"Error: {pid} - Bad Zip File"
        else:
            return f"Error: {pid} - HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error: {pid} - {str(e)}"

def main():
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)
        print(f"📁 创建目录: {DATA_ROOT}")

    # 1. 获取所有病人 ID
    print("🔍 查询所有病人列表...")
    scans = pl.query(pl.Scan).all()
    # 去重，只拿 ID
    all_pids = sorted(list(set([s.patient_id for s in scans])))
    print(f"🚀 准备下载 {len(all_pids)} 个病人数据...")
    print(f"💾 数据将保存在: {DATA_ROOT}")
    print("☕ 这可能需要几个小时 (约120GB)，建议挂后台运行...")

    # 2. 多进程下载
    # 建议进程数不要太多，以免被 TCIA 服务器封 IP，推荐 4-8 进程
    num_workers = min(8, cpu_count())
    
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(download_patient, all_pids), total=len(all_pids)))

    # 3. 统计结果
    success = [r for r in results if r.startswith("Success")]
    skipped = [r for r in results if r.startswith("Skip")]
    errors = [r for r in results if r.startswith("Error")]

    print("\n" + "="*30)
    print(f"✅ 下载完成: {len(success)}")
    print(f"⏩ 跳过已存在: {len(skipped)}")
    print(f"❌ 失败: {len(errors)}")
    
    if len(errors) > 0:
        print("\n失败列表 (建议重跑脚本以重试):")
        for e in errors:
            print(e)
    print("="*30)

if __name__ == "__main__":
    main()