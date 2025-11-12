from ase.io import read
from matscipy.rings import ring_statistics
import numpy as np

# 读取 XYZ 文件
atoms = read('restart.xyz')  # 这里替换为你的 XYZ 文件路径

# 设置分析参数
cutoff = 1.85  # 环的最大切割距离（1.85 Å 为你要求的距离）
maxlength = 9  # 最大环长度设置为9元环

# 计算环统计信息
ring_stats = ring_statistics(atoms, cutoff=cutoff, maxlength=maxlength)

# 输出 5，6，7，8，9 元环的数量
ring_counts = {i: 0 for i in range(4, 10)}

# 遍历所有环的统计，计算每种环的数量
for ring in ring_stats:
    if isinstance(ring, np.ndarray):  # 确保ring是一个包含原子索引的数组
        ring_length = len(ring)  # 计算环中原子的数量
        if 4 <= ring_length <= 9:  # 只统计 5 元到 9 元的环
            ring_counts[ring_length] += 1

# 输出结果
for ring_size, count in ring_counts.items():
    print(f"{ring_size}-元环的数量: {count}")

#python test.py restart.xyz --cutoff 1.85 --min-size 5 --max-size 8 --wrap 