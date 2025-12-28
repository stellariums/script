import os
import argparse
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

"""
extract_kappa.py
================

此脚本用于批量提取 GPUMD 计算生成的热导率数据 (kappa.out)，并汇总为 CSV 文件。

功能描述：
1. 遍历指定文件夹及其所有子目录，自动查找 `kappa.out` 文件。
2. 读取每个文件中的热流数据，计算热导率的运行平均值 (Running Average)。
3. 提取最终时刻的 x 和 y 方向热导率。
4. 将提取结果（包括文件夹名、子文件名、热导率数据）保存为 `kappa.csv`。

依赖库：
- numpy: 用于数值计算
- pandas: 用于数据整理和导出 CSV
- scipy: 用于积分计算 (cumulative_trapezoid)
- tqdm: 用于显示进度条

用法：
    python extract_kappa.py [root_dir] [--direction {x,y,both}]

参数：
    root_dir : 必选参数，指定要搜索的根目录路径。
    --direction : 可选参数，指定提取方向。
                  'x'   : 仅提取 x 方向热导率
                  'y'   : 仅提取 y 方向热导率
                  'both': 提取 x 和 y 方向热导率（默认值）

输出：
    在脚本运行目录下生成 `kappa.csv` 文件。
"""

def running_ave(y, x):
    """
    计算运行平均值 (Running Average)。
    
    原理：
    使用累积梯形积分法计算 $\int_0^t y(\tau) d\tau$，然后除以时间 $t$。
    
    参数：
    y : array-like, 被积函数值（如热流自相关函数）
    x : array-like, 自变量值（如相关时间）
    
    返回：
    运行平均值数组。
    """
    # Avoid division by zero if x[0] is 0, though here x starts at 0.001
    return cumulative_trapezoid(y, x, initial=0) / x

def process_kappa_file(file_path):
    """
    处理单个 kappa.out 文件，提取最终的 kx 和 ky 值。
    
    参数：
    file_path : str, kappa.out 文件的完整路径
    
    返回：
    tuple : (kx_final, ky_final) 如果处理成功
    None  : 如果处理出错或数据无效
    """
    try:
        # Load data
        # kappa.out format based on plt_kappa.py:
        # columns: kxi, kxo, kyi, kyo, kz
        data = np.loadtxt(file_path)
        
        if data.ndim != 2 or data.shape[1] < 4:
            # Handle cases where data might be empty or malformed
            return None

        # Time array (ns)
        # matches plt_kappa.py: t = np.arange(1, shape[0]+1)*0.001
        t = np.arange(1, data.shape[0] + 1) * 0.001
        
        # Extract components
        kxi = data[:, 0]
        kxo = data[:, 1]
        kyi = data[:, 2]
        kyo = data[:, 3]
        # kz = data[:, 4] # Not needed for x/y request, but available
        
        # Calculate running averages
        kxi_ra = running_ave(kxi, t)
        kxo_ra = running_ave(kxo, t)
        kyi_ra = running_ave(kyi, t)
        kyo_ra = running_ave(kyo, t)
        
        # Sum components
        kx_ra = kxi_ra + kxo_ra
        ky_ra = kyi_ra + kyo_ra
        
        # Get final values
        return kx_ra[-1], ky_ra[-1]
        
    except Exception as e:
        print(f"\nError processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract thermal conductivity data from kappa.out files.')
    parser.add_argument('root_dir', help='Root directory to search for kappa.out files')
    parser.add_argument('--direction', default='both', choices=['x', 'y', 'both'],
                        help='Direction to extract: x, y, or both (default: both)')
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.root_dir)
    
    if not os.path.isdir(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    # 1. Find all kappa.out files
    kappa_files = []
    print(f"Scanning '{root_dir}' for kappa.out files...")
    for dirpath, _, filenames in os.walk(root_dir):
        if 'kappa.out' in filenames:
            kappa_files.append(os.path.join(dirpath, 'kappa.out'))
    
    if not kappa_files:
        print("No kappa.out files found.")
        return

    print(f"Found {len(kappa_files)} files. Processing...")

    results = []
    
    # 2. Process files with progress bar
    for file_path in tqdm(kappa_files, unit="file"):
        vals = process_kappa_file(file_path)
        
        if vals is None:
            continue
            
        kx_val, ky_val = vals
        
        # Extract folder names for CSV
        # Path structure assumed: .../Folder Name/Subfile Name/kappa.out
        # e.g. .../1000k-100/a.100/kappa.out
        
        parent_dir = os.path.dirname(file_path)
        subfile_name = os.path.basename(parent_dir) # e.g., a.100
        folder_name = os.path.basename(os.path.dirname(parent_dir)) # e.g., 1000k-100
        
        # Prepare row data
        row = {
            'Folder Name': folder_name,
            'Subfile Name': subfile_name
        }
        
        if args.direction in ['x', 'both']:
            row['kx (W/m/K)'] = kx_val
        if args.direction in ['y', 'both']:
            row['ky (W/m/K)'] = ky_val
            
        results.append(row)

    # 3. Save to CSV
    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns to match requirement: Folder Name, Subfile Name, then data
        cols = ['Folder Name', 'Subfile Name']
        if args.direction in ['x', 'both']:
            cols.append('kx (W/m/K)')
        if args.direction in ['y', 'both']:
            cols.append('ky (W/m/K)')
            
        # Ensure only existing columns are selected (in case logic changes)
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        output_file = os.path.join(os.getcwd(), 'kappa.csv')
        try:
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\nSuccessfully saved data to '{output_file}'")
            print(f"Total records: {len(df)}")
        except Exception as e:
            print(f"\nError saving CSV: {e}")
    else:
        print("\nNo valid data extracted.")

if __name__ == "__main__":
    main()
