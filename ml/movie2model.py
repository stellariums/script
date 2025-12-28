# -*- coding: utf-8 -*-
"""
movie2model.py

将多帧结构文件中的指定帧范围拆分为单独的 XYZ 文件。
支持进度条显示以及使用多进程进行 CPU 并行处理。
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from ase.io import iread, read, write


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    必要参数：
        -i / --input  输入结构文件路径
        -s / --start  起始帧编号（从 0 开始）
        -e / --end    结束帧编号（从 0 开始，包含）
    可选参数：
        -p / --processes  并行进程数（默认使用所有可用 CPU 核心）
    """
    parser = argparse.ArgumentParser(
        description=(
            "从多帧原子结构文件中提取指定范围的帧，并将每一帧单独保存为 XYZ 文件。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "使用示例:\n"
            "  python movie2model.py -i c:\\Users\\USTC\\Desktop\\500k-100\\movie.xyz -s 10 -e 20\n"
            "\n"
            "参数说明:\n"
            "  -i, --input      多帧结构输入文件路径\n"
            "  -s, --start      起始帧编号（从 0 开始）\n"
            "  -e, --end        结束帧编号（从 0 开始，包含）\n"
            "  -p, --processes  并行进程数（可选，默认使用全部 CPU 核心）\n"
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="输入多帧结构文件路径（支持 XYZ、EXTXYZ 等常见格式）。",
    )
    parser.add_argument(
        "-s",
        "--start",
        required=True,
        type=int,
        help="起始帧编号（从 0 开始）。",
    )
    parser.add_argument(
        "-e",
        "--end",
        required=True,
        type=int,
        help="结束帧编号（从 0 开始，包含）。",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=None,
        help="并行进程数（默认使用所有可用 CPU 核心数）。",
    )
    return parser.parse_args()


def print_progress(done: int, total: int) -> None:
    """
    在终端中打印简单的文本进度条。

    为了避免影响正常标准输出，进度条打印到标准错误输出（stderr）。
    """
    if total <= 0:
        return
    bar_width = 30
    ratio = done / total
    if ratio < 0:
        ratio = 0
    if ratio > 1:
        ratio = 1
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = int(ratio * 100)
    print(
        f"\r进度: [{bar}] {done}/{total} ({percent}%)",
        end="",
        file=sys.stderr,
        flush=True,
    )


def process_single_frame(
    input_path: str,
    base_name: str,
    output_dir: str,
    frame_index: int,
    num_width: int,
) -> tuple[int, str | None]:
    """
    工作进程函数：读取单个帧并写出到 XYZ 文件。

    返回值:
        (frame_index, error_message)
        error_message 为 None 表示成功，否则为错误信息字符串。
    """
    try:
        atoms = read(input_path, index=frame_index)
        file_name = f"{base_name}_frame_{frame_index:0{num_width}d}.xyz"
        output_path = os.path.join(output_dir, file_name)
        write(output_path, atoms, format="extxyz")
        return frame_index, None
    except Exception as exc:  # noqa: BLE001
        return frame_index, str(exc)


def main() -> None:
    """
    主函数：
        1. 解析参数并检查输入合法性；
        2. 统计总帧数，验证帧范围；
        3. 使用多进程并行处理指定帧范围；
        4. 显示进度条并汇总写出结果。
    """
    args = parse_args()

    input_path = os.path.abspath(args.input)
    start = args.start
    end = args.end
    n_procs = args.processes

    # 检查输入文件是否存在
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 检查帧编号是否合法
    if start < 0 or end < 0:
        print("Error: frame indices must be non-negative integers.", file=sys.stderr)
        sys.exit(1)

    if end < start:
        print(
            "Error: end frame index must be greater than or equal to start frame index.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 首先统计总帧数，用于检查范围是否合法
    try:
        total_frames = 0
        for _ in iread(input_path, index=":"):
            total_frames += 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: failed to read input file '{input_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    if total_frames == 0:
        print(f"Error: no frames found in input file '{input_path}'.", file=sys.stderr)
        sys.exit(1)

    if start >= total_frames or end >= total_frames:
        print(
            f"Error: frame range [{start}, {end}] is outside available frames 0..{total_frames - 1}.",
            file=sys.stderr,
        )
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path)

    # 输出文件名中的序号宽度至少为 4 位
    num_width = max(4, len(str(end)))
    indices = list(range(start, end + 1))
    total_to_write = len(indices)

    # 计算并行进程数，默认使用全部 CPU 核心
    if n_procs is None or n_procs <= 0:
        n_procs = os.cpu_count() or 1

    print(
        f"Preparing to write {total_to_write} frame(s) "
        f"from [{start}, {end}] using {n_procs} process(es)..."
    )

    written = 0
    errors: list[tuple[int, str]] = []

    # 使用多进程并行处理每一帧的读写
    try:
        with ProcessPoolExecutor(max_workers=n_procs) as executor:
            future_to_index = {
                executor.submit(
                    process_single_frame,
                    input_path,
                    base_name,
                    output_dir,
                    frame_index,
                    num_width,
                ): frame_index
                for frame_index in indices
            }

            for i, future in enumerate(as_completed(future_to_index), start=1):
                frame_index = future_to_index[future]
                try:
                    frame_idx_returned, err = future.result()
                    if err is None:
                        written += 1
                    else:
                        errors.append((frame_idx_returned, err))
                except Exception as exc:  # noqa: BLE001
                    errors.append((frame_index, str(exc)))

                print_progress(i, total_to_write)

        # 打印进度条结束换行
        print("", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: failed while writing frames: {exc}", file=sys.stderr)
        sys.exit(1)

    # 汇总结果输出
    if errors and written == 0:
        print("Error: all frame writes failed.", file=sys.stderr)
        for frame_idx, msg in errors:
            print(f"  frame {frame_idx}: {msg}", file=sys.stderr)
        sys.exit(1)

    if errors:
        print(
            f"Completed with warnings: {written} frame(s) written, "
            f"{len(errors)} frame(s) failed.",
            file=sys.stderr,
        )
        for frame_idx, msg in errors:
            print(f"  frame {frame_idx}: {msg}", file=sys.stderr)
    else:
        print(
            f"Successfully wrote {written} frame(s) from [{start}, {end}] "
            f"to directory '{output_dir}'."
        )


if __name__ == "__main__":
    main()

