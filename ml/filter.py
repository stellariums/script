"""
脚本用途：
- 批量读取当前目录下的 .xyz 结构文件
- 过滤出 z 坐标绝对值不超过 100.0 的原子
- 将过滤后的结构写入当前目录下的 filtered 子目录

使用方式：
- 将本文件与待处理的 .xyz 文件放在同一目录下
- 在该目录中运行：python filter.py
- 过滤后的结果保存在 ./filtered/ 目录中，文件名与原文件保持一致
"""

import os
from ase.io import read, write
import numpy as np

base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, "filtered")
os.makedirs(output_dir, exist_ok=True)

for name in os.listdir(base_dir):
    if not name.lower().endswith(".xyz"):
        continue
    if "_filtered" in name:
        continue
    input_path = os.path.join(base_dir, name)
    atoms = read(input_path)
    z = atoms.positions[:, 2]
    mask = np.abs(z) <= 100.0
    filtered_atoms = atoms[mask]
    output_path = os.path.join(output_dir, name)
    write(output_path, filtered_atoms)
