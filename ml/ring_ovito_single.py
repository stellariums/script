#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 OVITO 的单文件环统计脚本（FindRingsModifier 版）。

功能概述
- 读取一个扩展 XYZ（extxyz）文件，支持单帧或多帧
- 自动建立键（范德华半径 vdw 或统一截断 cutoff）
- 使用 OVITO 的 FindRingsModifier 查找 N 元环（N = min_ring..max_ring）
- 对每一帧的 N 元环做“忽略起点/忽略方向”的去重计数
- 将每一帧的原子数、键数、各 N 元环数以及总环数逐行写入文本文件

依赖环境
- Python 3
- OVITO（建议 ovito >= 3.14，才有 FindRingsModifier）

用法示例
1) 使用范德华半径建键，统计 4–9 元环，处理所有帧:
    python ring_ovito_single.py -i sample.extxyz -o rings.txt --min_ring 4 --max_ring 9 --bond_mode vdw --frames all

2) 使用统一截断建键，只统计前 100 帧:
    python ring_ovito_single.py -i sample.extxyz -o rings_cutoff.txt --bond_mode cutoff --cutoff 1.85 --frames 0:100

参数说明
- -i / --input: 输入 extxyz/xyz 文件路径，例如 graphite.xyz
- -o / --output: 输出 txt 文件路径，例如 rings.txt
- --min_ring / --max_ring: 统计的最小、最大环尺寸（默认 4 和 9）
- --bond_mode: 建键方式，vdw 使用范德华半径，cutoff 使用统一截断半径
- --cutoff: 当 bond_mode=cutoff 时使用的截断距离，单位与坐标相同
- --frames: 要处理的帧，all 或 Python 切片语法（如 "0:100:5"）
- --print_check: 在控制台打印每帧的原子数、键数和各环数，便于自检

输出格式
- 文件头包含 OVITO 版本、输入文件路径、建键模式和环尺寸范围等元信息
- 随后一行表头:
    # Columns: frame  particles  bonds  4-ring  5-ring  6-ring  7-ring  8-ring  9-ring  total_rings
- 每一帧一行，依次给出帧号、原子数、键数、各尺寸环数以及该帧总环数
- 文件末尾附加 “Summary” 汇总各尺寸环数在全部处理帧上的总和

去重策略
- 使用 canonical_cycle 对 OVITO FindRingsModifier 输出的 N-rings 数据进行规范化
- 忽略环的起点和方向（顺/逆时针视为相同），避免同一物理环被重复计数
"""

import argparse
import sys


def parse_frames(frames_arg: str, nframes: int):
    """
    解析 --frames:
      all        -> 全部帧
      0:100      -> [0, 1, ..., 99]
      0:100:5    -> [0, 5, 10, ...]
    """
    s = (frames_arg or "all").strip().lower()
    if s == "all":
        return list(range(nframes))

    parts = s.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("Invalid --frames format. Use 'all' or 'start:stop' or 'start:stop:step'.")

    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if (len(parts) == 3 and parts[2]) else None

    return list(range(nframes))[slice(start, stop, step)]


def canonical_cycle(seq):
    """
    把一个环（索引序列）规范化：
    - 忽略起点（循环移位等价）
    - 忽略方向（顺时针/逆时针等价）
    返回字典序最小的表示，用于 set 去重。
    """
    s = [int(x) for x in seq]
    n = len(s)
    rots_fwd = [tuple(s[i:] + s[:i]) for i in range(n)]
    rs = list(reversed(s))
    rots_rev = [tuple(rs[i:] + rs[:i]) for i in range(n)]
    return min(rots_fwd + rots_rev)


def ring_count_dedup(data, n: int) -> int:
    """
    ✅ 去重计数：
    FindRingsModifier 输出名为 'N-rings' 的 DataTable。
    该表通常只有一个多分量列：每行存一个 N 元环的原子索引（长度为 N）。
    某些情况下同一环可能以不同起点或顺/逆方向出现多次；这里做循环去重后计数。
    """
    key = f"{n}-rings"
    if key not in data.tables:
        return 0

    table = data.tables[key]
    # 取第一个（通常也是唯一的）列名
    col = next(iter(table.keys()))
    arr = table[col]  # 形如 (num_rows, n)

    uniq = {canonical_cycle(row) for row in arr}
    return len(uniq)


def main():
    parser = argparse.ArgumentParser(
        description="OVITO: count 4-9 membered rings (and total rings) from an extended XYZ (with dedup)."
    )
    parser.add_argument("-i", "--input", required=True, help="输入 extxyz 文件路径，例如 graphite.xyz")
    parser.add_argument("-o", "--output", required=True, help="输出 txt 文件路径，例如 rings.txt")

    parser.add_argument("--min_ring", type=int, default=4, help="最小环尺寸 (default: 4)")
    parser.add_argument("--max_ring", type=int, default=9, help="最大环尺寸 (default: 9)")

    parser.add_argument(
        "--bond_mode",
        choices=["vdw", "cutoff"],
        default="vdw",
        help="建键方式：vdw(默认) 或 cutoff(统一截断半径)"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="统一截断半径（当 --bond_mode cutoff 时必填），单位同坐标单位"
    )

    parser.add_argument(
        "--frames",
        default="all",
        help="处理帧：all(默认) 或 0:100 或 0:100:5（Python slice 语法）"
    )

    parser.add_argument(
        "--print_check",
        action="store_true",
        help="在控制台打印每帧的原子数/键数/环数自检信息"
    )

    args = parser.parse_args()

    # 导入 OVITO
    try:
        import ovito
        from ovito.io import import_file
        from ovito.modifiers import CreateBondsModifier, FindRingsModifier
    except Exception:
        print("ERROR: 无法导入 ovito。请先安装/升级：pip install -U ovito", file=sys.stderr)
        raise

    # 读入文件
    pipeline = import_file(args.input, multiple_frames=True)
    nframes = pipeline.source.num_frames
    frame_indices = parse_frames(args.frames, nframes)
    if not frame_indices:
        raise RuntimeError("没有选中任何帧要处理，请检查 --frames 参数。")

    # 1) 建键（cutoff 用 Uniform）
    if args.bond_mode == "cutoff":
        if args.cutoff is None or args.cutoff <= 0:
            raise ValueError("你选择了 --bond_mode cutoff，但没有提供正的 --cutoff 数值。")
        create_bonds = CreateBondsModifier(cutoff=float(args.cutoff))  # Uniform cutoff
    else:
        create_bonds = CreateBondsModifier(mode=CreateBondsModifier.Mode.VdWRadius)

    pipeline.modifiers.append(create_bonds)

    # 2) 找环
    pipeline.modifiers.append(
        FindRingsModifier(minimum_ring_size=args.min_ring, maximum_ring_size=args.max_ring)
    )

    # 逐帧统计 + 累积
    total_counts = {n: 0 for n in range(args.min_ring, args.max_ring + 1)}
    total_rings_all_frames = 0

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# OVITO version: {getattr(ovito, 'version', 'unknown')}\n")
        f.write(f"# Input: {args.input}\n")
        if args.bond_mode == "cutoff":
            f.write(f"# Bond mode: cutoff (Uniform), cutoff={args.cutoff}\n")
        else:
            f.write("# Bond mode: vdw\n")
        f.write(f"# Ring size range: {args.min_ring}..{args.max_ring}\n")
        f.write("# NOTE: Ring counting uses cyclic + reverse-direction deduplication (canonical cycle).\n")
        f.write("# Columns: frame  particles  bonds  4-ring  5-ring  6-ring  7-ring  8-ring  9-ring  total_rings\n")

        for frame in frame_indices:
            data = pipeline.compute(frame)

            # 自检：原子数与键数
            n_particles = data.particles.count
            n_bonds = data.particles.bonds.count if data.particles.bonds is not None else 0

            # 环统计（去重后）
            counts = {n: ring_count_dedup(data, n) for n in range(args.min_ring, args.max_ring + 1)}
            frame_total = sum(counts.values())

            for n, c in counts.items():
                total_counts[n] += c
            total_rings_all_frames += frame_total

            # 输出 4..9（缺的按 0）
            c4 = counts.get(4, 0)
            c5 = counts.get(5, 0)
            c6 = counts.get(6, 0)
            c7 = counts.get(7, 0)
            c8 = counts.get(8, 0)
            c9 = counts.get(9, 0)

            f.write(
                f"{frame:6d}  {n_particles:9d}  {n_bonds:6d}  "
                f"{c4:6d}  {c5:6d}  {c6:6d}  {c7:6d}  {c8:6d}  {c9:6d}  {frame_total:11d}\n"
            )

            if args.print_check:
                print(f"[frame {frame}] particles={n_particles}, bonds={n_bonds}, "
                      f"4={c4},5={c5},6={c6},7={c7},8={c8},9={c9}, total={frame_total}")

        # 汇总
        f.write("\n# Summary (sum over processed frames)\n")
        f.write(f"# frames: {frame_indices}\n")
        f.write(f"# 4-ring: {total_counts.get(4, 0)}\n")
        f.write(f"# 5-ring: {total_counts.get(5, 0)}\n")
        f.write(f"# 6-ring: {total_counts.get(6, 0)}\n")
        f.write(f"# 7-ring: {total_counts.get(7, 0)}\n")
        f.write(f"# 8-ring: {total_counts.get(8, 0)}\n")
        f.write(f"# 9-ring: {total_counts.get(9, 0)}\n")
        f.write(f"# total_rings: {total_rings_all_frames}\n")

    print(f"Done. Wrote ring counts to: {args.output}")


if __name__ == "__main__":
    main()
