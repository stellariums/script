import os
import shutil
import pandas as pd
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re

"""
txt2csv.py
==========

此脚本用于将多个包含环统计信息的文本文件 (.txt) 解析并合并到现有的 `kappa.csv` 文件中。

功能描述：
1. 读取指定的 TXT 文件列表，提取环统计数据（如 Total_Rings, 5-ring(%), 等）。
2. 解析 TXT 文件名作为匹配键（去除 .xyz 后缀），并从 TXT 文件名解析温度生成 `Folder Name`。
3. 读取现有的 `kappa.csv`，并根据 `Folder Name` + `Subfile Name` 与 TXT 数据进行左连接 (Left Join) 合并。
4. 如果 CSV 中已存在相关统计列，脚本会自动覆盖更新。
5. 在修改前自动备份原 CSV 文件 (.bak)。
6. 支持并行读取文件以提高性能，并提供日志记录和进度条。

依赖库：
- pandas: 用于数据处理和 CSV 读写
- logging: 用于记录运行日志
- tqdm: 用于显示处理进度条
- concurrent.futures: 用于并行处理文件读取任务

输入文件：
- TXT_FILES 列表中指定的多个 .txt 文件（包含环统计信息）
- CSV_FILE 指定的 kappa.csv 文件（包含热导率数据）

输出：
- 更新后的 kappa.csv 文件（新增了环统计数据列）
- txt2csv.log 日志文件
"""

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("txt2csv.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 目标文件列表
TXT_FILES = [
    r"c:\Users\USTC\Desktop\monolayer\ML\1000.txt",
    r"c:\Users\USTC\Desktop\monolayer\ML\1500.txt",
    r"c:\Users\USTC\Desktop\monolayer\ML\1700.txt",
    r"c:\Users\USTC\Desktop\monolayer\ML\1800.txt",
    r"c:\Users\USTC\Desktop\monolayer\ML\1900.txt",
    r"c:\Users\USTC\Desktop\monolayer\ML\2000.txt"
]

CSV_FILE = r"c:\Users\USTC\Desktop\monolayer\ML\kappa.csv"

def read_txt_file(file_path):
    """
    读取并解析单个txt文件，返回DataFrame。
    处理列名映射，并去掉文件名的 .xyz 后缀。
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None

        # 手动指定列名，根据用户提供的文件头
        # # 文件名 总环数 5-ring(count) 6-ring(count) 7-ring(count) 8-ring(count) 9-ring(count) 5-ring(%) 6-ring(%) 7-ring(%) 8-ring(%) 9-ring(%)
        # 注意：pandas read_csv(sep='\s+') 会把 # 后的内容当作第一列数据如果 header=None 且第一行是注释
        # 实际上 Read 工具显示第一行是 # 文件名 ...
        # 我们重新读取，跳过第一行注释，自己赋列名
        
        cols = [
            "Filename", "Total_Rings",
            "5-ring(count)", "6-ring(count)", "7-ring(count)", "8-ring(count)", "9-ring(count)",
            "5-ring(%)", "6-ring(%)", "7-ring(%)", "8-ring(%)", "9-ring(%)"
        ]

        usecols = [
            "Filename",
            "5-ring(%)", "6-ring(%)", "7-ring(%)", "8-ring(%)", "9-ring(%)"
        ]
        
        # 重新读取，跳过第一行（它是header但带#号可能解析有问题，我们手动指定）
        df = pd.read_csv(file_path, sep=r'\s+', names=cols, skiprows=1, usecols=usecols)

        # 去除 .xyz 后缀以匹配 Subfile Name
        df['Subfile Name'] = df['Filename'].astype(str).str.replace('.xyz', '', regex=False)

        base = os.path.splitext(os.path.basename(file_path))[0]
        m = re.search(r'\d+', base)
        if not m:
            logging.error(f"Failed to parse temperature from TXT filename: {file_path}")
            return None
        df['Folder Name'] = f"{m.group(0)}k-100"
        
        logging.info(f"Successfully parsed {file_path}, rows: {len(df)}")
        return df
    
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

def process_files():
    # 1. 备份 CSV 文件
    if os.path.exists(CSV_FILE):
        backup_file = CSV_FILE + ".bak"
        try:
            shutil.copy2(CSV_FILE, backup_file)
            logging.info(f"Backup created: {backup_file}")
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            return
    else:
        logging.error(f"Target CSV file not found: {CSV_FILE}")
        return

    # 2. 读取 CSV 文件
    try:
        kappa_df = pd.read_csv(CSV_FILE)
        logging.info(f"Loaded kappa.csv with {len(kappa_df)} rows")
    except Exception as e:
        logging.error(f"Failed to read kappa.csv: {e}")
        return

    if 'Folder Name' not in kappa_df.columns or 'Subfile Name' not in kappa_df.columns:
        logging.error("kappa.csv must contain 'Folder Name' and 'Subfile Name' columns.")
        return

    dup_count = int(kappa_df.duplicated(subset=['Folder Name', 'Subfile Name']).sum())
    if dup_count:
        logging.warning(f"Found {dup_count} duplicate rows in kappa.csv by (Folder Name, Subfile Name). Keeping first occurrence.")
        kappa_df = kappa_df.drop_duplicates(subset=['Folder Name', 'Subfile Name'], keep='first').reset_index(drop=True)

    ring_count_cols = [
        "Total_Rings",
        "5-ring(count)", "6-ring(count)", "7-ring(count)", "8-ring(count)", "9-ring(count)"
    ]
    cols_present = [c for c in ring_count_cols if c in kappa_df.columns]
    if cols_present:
        logging.info(f"Dropping ring count columns from kappa.csv: {cols_present}")
        kappa_df = kappa_df.drop(columns=cols_present)

    # 3. 并行读取 TXT 文件
    all_txt_data = []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(read_txt_file, TXT_FILES), total=len(TXT_FILES), desc="Reading TXT files"))
        
    for res in results:
        if res is not None:
            all_txt_data.append(res)
    
    if not all_txt_data:
        logging.error("No data extracted from TXT files.")
        return

    # 合并所有 TXT 数据
    combined_txt_df = pd.concat(all_txt_data, ignore_index=True)

    if 'Folder Name' not in combined_txt_df.columns or 'Subfile Name' not in combined_txt_df.columns:
        logging.error("Parsed TXT data must contain 'Folder Name' and 'Subfile Name' columns.")
        return

    txt_dup_count = int(combined_txt_df.duplicated(subset=['Folder Name', 'Subfile Name']).sum())
    if txt_dup_count:
        logging.warning(f"Found {txt_dup_count} duplicate rows in TXT data by (Folder Name, Subfile Name). Keeping last occurrence.")
        combined_txt_df = combined_txt_df.drop_duplicates(subset=['Folder Name', 'Subfile Name'], keep='last').reset_index(drop=True)
    
    # 4. 数据合并
    # 确保 Subfile Name 列类型一致
    kappa_df['Subfile Name'] = kappa_df['Subfile Name'].astype(str)
    combined_txt_df['Subfile Name'] = combined_txt_df['Subfile Name'].astype(str)
    kappa_df['Folder Name'] = kappa_df['Folder Name'].astype(str)
    combined_txt_df['Folder Name'] = combined_txt_df['Folder Name'].astype(str)
    
    # 检查是否有重复的列名（除了关联键），如果有，先在 kappa_df 中删除，避免重复添加
    cols_to_add = [c for c in combined_txt_df.columns if c not in ['Filename', 'Folder Name', 'Subfile Name']]
    
    # 如果 kappa.csv 已经包含了这些列，我们可以选择更新或者保留
    # 这里我们选择更新：先删除旧列，再合并新列
    existing_cols = [c for c in cols_to_add if c in kappa_df.columns]
    if existing_cols:
        logging.warning(f"Overwriting existing columns in kappa.csv: {existing_cols}")
        kappa_df = kappa_df.drop(columns=existing_cols)

    # 执行左连接 (Left Join) 以保留 kappa.csv 的所有行
    merged_df = pd.merge(
        kappa_df,
        combined_txt_df[['Folder Name', 'Subfile Name'] + cols_to_add],
        on=['Folder Name', 'Subfile Name'],
        how='left',
        validate='one_to_one'
    )
    
    # 5. 保存结果
    try:
        merged_df.to_csv(CSV_FILE, index=False, encoding='utf-8')
        logging.info(f"Successfully updated {CSV_FILE}")
        logging.info(f"Final columns: {list(merged_df.columns)}")
    except Exception as e:
        logging.error(f"Failed to write to kappa.csv: {e}")

if __name__ == "__main__":
    process_files()
