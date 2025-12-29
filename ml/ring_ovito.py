#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 OVITO 的环统计脚本（FindRingsModifier 版）。

功能概述
- 支持对单个扩展 XYZ（extxyz）文件或整个目录中的多个文件进行环统计
- 自动建立键（范德华半径 vdw 或统一截断 cutoff）
- 对每个文件在选定帧集合上的 N 元环（N = min_ring..max_ring）做去重计数
- 将每个文件的总环数、各 N 元环计数以及占总环数的百分比写入一个制表符分隔的文本文件
- 输出格式与 ring_batch.py 生成的 1000.txt 一致，便于对比和后处理

依赖环境
- Python 3
- OVITO（建议 ovito >= 3.14，才有 FindRingsModifier）

用法示例
1) 单文件模式（类似 ring_batch.py 的 --single-file）:
    python ring_ovito.py -i a.100.xyz -o 1000_ovito.txt --min_ring 5 --max_ring 9 --bond_mode vdw --frames all

2) 目录批量模式（类似 ring_batch.py 的 --input-dir）:
    python ring_ovito.py --input-dir ./xyz_files --extxyz-pattern "a.*.xyz" -o 1000_ovito.txt --min_ring 5 --max_ring 9 --bond_mode vdw --frames all

参数说明
- -i / --input: 单个 extxyz/xyz 文件路径
- --input-dir: 包含多个 extxyz/xyz 文件的目录路径，批量模式
- --extxyz-pattern: 在 --input-dir 下匹配文件的 glob 模式（默认: "*.xyz"）
- -o / --output: 输出结果 txt 文件路径
- --min_ring / --max_ring: 统计的最小、最大环尺寸（默认 4 和 9）
- --bond_mode: 建键方式，vdw 使用范德华半径，cutoff 使用统一截断半径
- --cutoff: 当 bond_mode=cutoff 时使用的截断距离，单位与坐标相同
- --frames: 要处理的帧，all 或 Python 切片语法（如 "0:100:5"）
- --print_check: 打印每一帧的总环数供自检

输出格式
- 首行表头（制表符分隔）:
    # 文件名\t总环数\tN-ring(count)...\tN-ring(%)
- 随后每一行对应一个输入文件:
    文件名  该文件所有选定帧的总环数  各 N 元环计数  各 N 元环在总环数中的百分比

去重策略
- 使用 canonical_cycle 函数对 OVITO FindRingsModifier 输出的 N-rings 表逐环做规范化
- 忽略环的起点和方向（顺时针/逆时针视为同一个环），保证同一物理环只计一次
"""

import argparse
import sys
from pathlib import Path


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
    parser.add_argument("-i", "--input", required=False, help="输入 extxyz 文件路径，例如 graphite.xyz")
    parser.add_argument("--input-dir", type=str, help="包含多个 extxyz 文件的文件夹路径，批量模式。")
    parser.add_argument("--extxyz-pattern", type=str, default="*.xyz", help="批量模式下匹配的文件模式，例如 *.xyz")
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

    if args.input is None and args.input_dir is None:
        parser.error("必须指定 --input 或 --input-dir 之一。")
    if args.input is not None and args.input_dir is not None:
        parser.error("--input 与 --input-dir 不能同时使用。")

    try:
        import ovito
        from ovito.io import import_file
        from ovito.modifiers import CreateBondsModifier, FindRingsModifier
    except Exception:
        print("ERROR: 无法导入 ovito。请先安装/升级：pip install -U ovito", file=sys.stderr)
        raise

    header_sizes = list(range(args.min_ring, args.max_ring + 1))

    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise RuntimeError(f"输入目录不存在或不是文件夹：{input_dir}")

        files = sorted(input_dir.rglob(args.extxyz_pattern))
        if not files:
            raise RuntimeError(f"在目录中没有找到匹配的文件（pattern={args.extxyz_pattern}）：{input_dir}")

        with open(args.output, "w", encoding="utf-8") as f:
            count_cols = "\t".join([f"{n}-ring(count)" for n in header_sizes])
            pct_cols = "\t".join([f"{n}-ring(%)" for n in header_sizes])
            f.write("# 文件名\t总环数\t" + count_cols + "\t" + pct_cols + "\n")

            for path in files:
                pipeline = import_file(str(path), multiple_frames=True)
                nframes = pipeline.source.num_frames
                frame_indices = parse_frames(args.frames, nframes)
                if not frame_indices:
                    continue

                if args.bond_mode == "cutoff":
                    if args.cutoff is None or args.cutoff <= 0:
                        raise ValueError("你选择了 --bond_mode cutoff，但没有提供正的 --cutoff 数值。")
                    create_bonds = CreateBondsModifier(cutoff=float(args.cutoff))
                else:
                    create_bonds = CreateBondsModifier(mode=CreateBondsModifier.Mode.VdWRadius)

                pipeline.modifiers.clear()
                pipeline.modifiers.append(create_bonds)
                pipeline.modifiers.append(
                    FindRingsModifier(minimum_ring_size=args.min_ring, maximum_ring_size=args.max_ring)
                )

                total_counts = {n: 0 for n in header_sizes}
                total_rings = 0

                for frame in frame_indices:
                    data = pipeline.compute(frame)
                    counts = {n: ring_count_dedup(data, n) for n in header_sizes}
                    frame_total = sum(counts.values())

                    for n, c in counts.items():
                        total_counts[n] += c
                    total_rings += frame_total

                    if args.print_check:
                        print(
                            f"[{path.name} frame {frame}] total_rings_frame={frame_total}"
                        )

                counts_list = [total_counts.get(n, 0) for n in header_sizes]
                if total_rings > 0:
                    percents = [100.0 * c / float(total_rings) for c in counts_list]
                else:
                    percents = [0.0 for _ in header_sizes]

                line = (
                    f"{path.name}\t{total_rings}\t"
                    + "\t".join(str(c) for c in counts_list) + "\t"
                    + "\t".join(f"{p:.6f}" for p in percents) + "\n"
                )
                f.write(line)

        print(f"Done. Wrote ring summary to: {args.output}")
    else:
        pipeline = import_file(args.input, multiple_frames=True)
        nframes = pipeline.source.num_frames
        frame_indices = parse_frames(args.frames, nframes)
        if not frame_indices:
            raise RuntimeError("没有选中任何帧要处理，请检查 --frames 参数。")

        if args.bond_mode == "cutoff":
            if args.cutoff is None or args.cutoff <= 0:
                raise ValueError("你选择了 --bond_mode cutoff，但没有提供正的 --cutoff 数值。")
            create_bonds = CreateBondsModifier(cutoff=float(args.cutoff))
        else:
            create_bonds = CreateBondsModifier(mode=CreateBondsModifier.Mode.VdWRadius)

        pipeline.modifiers.append(create_bonds)
        pipeline.modifiers.append(
            FindRingsModifier(minimum_ring_size=args.min_ring, maximum_ring_size=args.max_ring)
        )

        total_counts = {n: 0 for n in header_sizes}
        total_rings = 0

        for frame in frame_indices:
            data = pipeline.compute(frame)
            counts = {n: ring_count_dedup(data, n) for n in header_sizes}
            frame_total = sum(counts.values())

            for n, c in counts.items():
                total_counts[n] += c
            total_rings += frame_total

            if args.print_check:
                print(
                    f"[frame {frame}] total_rings_frame={frame_total}"
                )

        counts_list = [total_counts.get(n, 0) for n in header_sizes]
        if total_rings > 0:
            percents = [100.0 * c / float(total_rings) for c in counts_list]
        else:
            percents = [0.0 for _ in header_sizes]

        with open(args.output, "w", encoding="utf-8") as f:
            count_cols = "\t".join([f"{n}-ring(count)" for n in header_sizes])
            pct_cols = "\t".join([f"{n}-ring(%)" for n in header_sizes])
            f.write("# 文件名\t总环数\t" + count_cols + "\t" + pct_cols + "\n")

            line = (
                f"{Path(args.input).name}\t{total_rings}\t"
                + "\t".join(str(c) for c in counts_list) + "\t"
                + "\t".join(f"{p:.6f}" for p in percents) + "\n"
            )
            f.write(line)

        print(f"Done. Wrote ring summary to: {args.output}")


if __name__ == "__main__":
    main()
