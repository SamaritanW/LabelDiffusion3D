import os
import numpy as np
import pylidc as pl
from tqdm import tqdm

# 配置
LIDC_DIR = "/data/users/baohengl/miccai/DLD/data/LIDC-IDRI/process"
OUTPUT_DIR = "/data/users/baohengl/miccai/DLD/data/LIDC-IDRI/mean_masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_mean_masks():
    files = [f for f in os.listdir(LIDC_DIR) if f.endswith('_masks.npy')]
    
    print(f"🚀 开始生成 Mean Masks，共 {len(files)} 个文件...")
    
    for f in tqdm(files):
        # f: LIDC-IDRI-0078_nodule0_masks.npy
        path = os.path.join(LIDC_DIR, f)
        
        # Load Masks: (N_doctors, 64, 64, 64)
        masks = np.load(path)
        
        # 计算平均 (Float 0~1)
        # axis=0 表示在医生维度求平均
        mean_mask = np.mean(masks, axis=0) 
        
        # 保存为同名文件
        save_path = os.path.join(OUTPUT_DIR, f)
        np.save(save_path, mean_mask)

if __name__ == "__main__":
    generate_mean_masks()