#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计 EXTXYZ/XYZ 文件中的 5–9 元环数量（基于 matscipy 的最短路径环算法）
- 成键阈值：默认 1.85 Å（--cutoff 可调）
- 报告范围：--min-size 到 --max-size（默认 5–9）
- 自动处理某些版本 matscipy 的 ring_statistics 形状不匹配问题：
  若传入 maxlength 报 ValueError，就回退为不传 maxlength 再计算
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from matscipy.rings import ring_statistics

def safe_ring_statistics(atoms, cutoff, maxlength):
    """
    调用 matscipy.rings.ring_statistics，若遇到某些版本的
    broadcast 形状不匹配错误，则回退为不传 maxlength 再计算。
    """
    try:
        return ring_statistics(atoms, cutoff=float(cutoff), maxlength=int(maxlength))
    except ValueError as e:
        msg = str(e)
        # 典型报错：operands could not be broadcast together with shapes (9,) (10,) (9,)
        if "could not be broadcast together" in msg or "broadcast" in msg:
            print("[WARN] 捕获到 ring_statistics 的形状不匹配问题，"
                  "改为不传 maxlength（全范围）重新计算，以绕过该版本 Bug。")
            return ring_statistics(atoms, cutoff=float(cutoff))  # 不传 maxlength（默认 -1）
        raise  # 其他错误照常抛出

def main():
    parser = argparse.ArgumentParser(
        description="Count specified n-member rings in an EXTXYZ/XYZ file using matscipy.rings.ring_statistics."
    )
    parser.add_argument("xyz", type=str, help="Path to EXTXYZ/XYZ file, e.g., restart2.xyz")
    parser.add_argument("--cutoff", type=float, default=1.85, help="Bond cutoff in Å (default: 1.85)")
    parser.add_argument("--min-size", type=int, default=5, help="Minimum ring size to report (default: 5)")
    parser.add_argument("--max-size", type=int, default=9, help="Maximum ring size to search/report (default: 9)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index if multiple frames exist (default: 0)")
    parser.add_argument("--wrap", action="store_true", help="Wrap atoms into cell before analysis")
    args = parser.parse_args()

    xyz_path = Path(args.xyz)
    if not xyz_path.exists():
        print(f"ERROR: File not found: {xyz_path}")
        sys.exit(1)

    if args.min_size < 3:
        print("WARNING: ring size < 3 不具物理意义，已将 --min-size 调整为 3。")
        args.min_size = 3
    if args.max_size < args.min_size:
        print("ERROR: --max-size 必须 ≥ --min-size")
        sys.exit(1)

    # 优先用 extxyz 解析（能读出 Lattice/pbc/Properties）
    try:
        atoms = read(str(xyz_path), index=args.frame, format="extxyz")
    except Exception:
        atoms = read(str(xyz_path), index=args.frame)

    natoms = len(atoms)
    cell = atoms.cell.array if atoms.cell is not None else None
    pbc = tuple(bool(x) for x in getattr(atoms, "pbc", (False, False, False)))

    if args.wrap and cell is not None and any(pbc):
        atoms.wrap(eps=1e-12)

    # 打印输入摘要
    print("=== Input summary ===")
    print(f"File: {xyz_path.name}")
    print(f"Atoms: {natoms}")
    if cell is not None:
        a_len, b_len, c_len = (np.linalg.norm(cell[i]) for i in range(3))
        print(f"Cell lengths (Å): a={a_len:.3f}, b={b_len:.3f}, c={c_len:.3f}")
    print(f"PBC: {pbc}")
    print(f"Cutoff (Å): {args.cutoff}")
    print(f"Ring size range: {args.min_size}–{args.max_size}")
    print("=====================\n")

    # 先按用户要求的上限尝试（可加速）；若触发版本 Bug，则自动回退为“无上限”再算
    ring_hist = safe_ring_statistics(atoms, cutoff=args.cutoff, maxlength=args.max_size)

    # 转成 numpy 数组，便于切片与越界保护
    ring_hist = np.asarray(ring_hist, dtype=np.int64)
    print(f"[INFO] 得到的直方图长度：{len(ring_hist)}（索引即环的原子数：r[n] = n 元环的个数）")

    # 报告指定范围（默认 5–9）
    any_reported = False
    for n in range(args.min_size, args.max_size + 1):
        count = int(ring_hist[n]) if n < len(ring_hist) else 0
        print(f"{n}-元环的数量: {count}")
        any_reported = True

    if not any_reported:
        print("没有可报告的环统计（检查 --max-size、输入结构或 cutoff）。")

    print("\n=== Tips ===")
    print("* 若仍为 0：")
    print("  - 确认 cutoff=1.85 Å 是否适合你的体系；可在 1.6–2.0 Å 范围做灵敏度分析；")
    print("  - 确认 PBC 设置与你体系一致（你给的示例为 pbc=(T,T,F)）；")
    print("  - 可先截取子块测试流程无误，再在全系统上运行；")
    print("  - 升级 matscipy 到最新版本，旧版存在 ring_statistics 的 off-by-one 广播问题。")
    print("* 文档：ring_statistics 返回“按环大小的计数数组”，maxlength 用于加速但非必需。")

if __name__ == "__main__":
    main()
