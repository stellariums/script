#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 ABACUS 的 SCF 计算输出批量转换为 NEP 训练可用的多帧 extxyz 文件。

脚本用途：
1. 递归遍历一个根目录下的多个 ABACUS 计算子目录。
2. 自动识别并读取收敛的 SCF 输出结果。
3. 从输出中提取每个结构的：
   - 总能量 energy
   - 晶格 Lattice
   - 原子坐标 positions
   - 原子受力 forces
   - 可选 virial（若输出中包含 stress）
4. 将所有收敛结构汇总写入一个多帧 xyz/extxyz 文件，供 NEP 或其他机器学习势训练使用。

支持的输入类型：
1. `running_scf.log`
   通过 `ase.io.read(..., format="abacus-out")` 解析。
   这一分支通常依赖 ASE 对 ABACUS 输出格式的支持，必要时需要额外安装
   对应插件，例如 `ase-abacus`。
2. `abacus.json`
   当目录中没有 `running_scf.log` 但存在 `abacus.json` 时，脚本会改为从
   JSON 文件中提取数据。

输入目录要求：
- 你提供的 `root_dir` 应是包含多个 ABACUS 子任务目录的根目录。
- 每个子目录中通常应包含：
  - `INPUT`
  - `running_scf.log` 或 `abacus.json`
- `INPUT` 中如果定义了 `scf_nmax`，脚本会读取它，用于判断 SCF 是否收敛；
  若未找到，则默认使用 `100`。

收敛判断逻辑：
- 对 `running_scf.log`：
  统计日志中 `ALGORITHM` 的出现次数，并检查是否存在 `Total  Time`。
  若 `ALGORITHM` 次数达到 `scf_nmax`，通常视为未在最大步数内收敛，因此跳过。
- 对 `abacus.json`：
  读取 `output[0]["scf"]` 的长度。
  若 SCF 步数大于等于 `scf_nmax`，则视为未收敛并跳过。

输出内容：
- 输出为一个多帧 xyz/extxyz 文件。
- 每一帧包含：
  1. 第一行：原子数
  2. 第二行：注释行，含以下字段：
     - `energy=...`
     - `Lattice="..."`
     - `Virial="..."`（如果可用）
     - `config_type="..."`
     - `Properties=species:S:1:pos:R:3:forces:R:3`
  3. 后续行为各原子的元素、坐标和力

注意事项：
- `running_scf.log` 分支依赖 `ASE` 和 ABACUS 输出解析支持。
- `abacus.json` 的元素信息在不同版本中字段名可能略有差异，
  脚本已尽量兼容常见格式，但如果你的 JSON 结构特别特殊，可能需要再做适配。
- 若输出中没有 stress，脚本不会报错，只是省略 `Virial` 字段。

命令行用法：
    python abacus_scfoutput2nepxyz.py <root_dir> <output_xyz>

参数说明：
- `root_dir`：包含多个 ABACUS 计算子目录的根目录
- `output_xyz`：输出的 NEP/extxyz 文件路径

示例：
    python abacus_scfoutput2nepxyz.py ./jobs ./train.xyz
    python abacus_scfoutput2nepxyz.py /path/to/abacus_runs /path/to/train.xyz
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read

KBAR_PER_EV_ANG3 = 1602.1766208


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect converged ABACUS SCF outputs into one NEP/extxyz file."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing ABACUS calculation subdirectories.",
    )
    parser.add_argument(
        "output_xyz",
        help="Output xyz/extxyz file path.",
    )
    return parser.parse_args()


def strip_comment(line):
    return line.split("#", 1)[0].strip()


def get_scf_nmax(root):
    input_file = Path(root) / "INPUT"
    if not input_file.exists():
        return 100

    with input_file.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = strip_comment(raw_line)
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "scf_nmax":
                return int(parts[1])

    return 100


def is_converged_log(log_file, scf_nmax):
    scf_count = 0
    total_time_count = 0

    with Path(log_file).open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            scf_count += line.count("ALGORITHM")
            total_time_count += line.count("Total  Time")

    return total_time_count > 0 and scf_count < scf_nmax


def stress6_to_virial(stress6, volume):
    xx, yy, zz, yz, xz, xy = map(float, stress6)
    stress_matrix = np.array(
        [
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ],
        dtype=float,
    )
    return (-stress_matrix * float(volume)).reshape(-1)


def json_stress_to_virial(stress_data, volume):
    stress = np.asarray(stress_data, dtype=float)

    if stress.shape == (3, 3):
        stress_matrix = stress
    elif stress.size == 9:
        stress_matrix = stress.reshape(3, 3)
    else:
        raise ValueError(f"Unsupported stress shape in abacus.json: {stress.shape}")

    # abacus.json stress is commonly stored in kbar.
    return (-stress_matrix * (float(volume) / KBAR_PER_EV_ANG3)).reshape(-1)


def flatten_cell(cell):
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"Cell shape must be (3, 3), got {cell.shape}")
    return cell.reshape(-1)


def infer_json_symbols(init_data, natoms):
    per_atom_keys = (
        "symbols",
        "atom_symbols",
        "atom_symbol",
        "elements",
        "species",
        "atom_labels",
        "labels",
    )
    for key in per_atom_keys:
        value = init_data.get(key)
        if isinstance(value, list) and len(value) == natoms:
            return [str(x) for x in value]

    labels = init_data.get("label")
    if isinstance(labels, str):
        labels = [labels]
    elif labels is None:
        labels = []
    else:
        labels = list(labels)

    if len(labels) == natoms:
        return [str(x) for x in labels]

    count_keys = (
        "atom_numbs",
        "natom_each_type",
        "atom_num",
        "type_num",
        "count",
        "counts",
    )
    for count_key in count_keys:
        counts = init_data.get(count_key)
        if not isinstance(counts, list):
            continue
        if len(labels) != len(counts):
            continue
        if sum(int(x) for x in counts) != natoms:
            continue

        symbols = []
        for label, count in zip(labels, counts):
            symbols.extend([str(label)] * int(count))
        return symbols

    raise ValueError(
        "Unable to infer per-atom symbols from abacus.json. "
        "Expected either per-atom labels or species labels with counts."
    )


def relative_config_type(root_dir, workdir):
    root_dir = Path(root_dir).resolve()
    workdir = Path(workdir).resolve()
    try:
        return str(workdir.relative_to(root_dir))
    except ValueError:
        return str(workdir)


def find_text_output_dir(workdir):
    candidates = (Path(workdir), Path(workdir) / "OUT.ABACUS")
    for candidate in candidates:
        if (candidate / "running_scf.log").is_file():
            return candidate
    return None


def find_json_output_dir(workdir):
    candidates = (Path(workdir), Path(workdir) / "OUT.ABACUS")
    for candidate in candidates:
        if (candidate / "abacus.json").is_file():
            return candidate
    return None


def extract_from_log(root_dir, workdir):
    output_dir = find_text_output_dir(workdir)
    if output_dir is None:
        return None

    log_file = output_dir / "running_scf.log"
    scf_nmax = get_scf_nmax(workdir)
    if not is_converged_log(log_file, scf_nmax):
        print(f"Skip {workdir}: calculation is incomplete or unconverged")
        return None

    try:
        atoms = read(str(log_file), format="abacus-out")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse {log_file}. "
            "This path depends on ASE support for 'abacus-out' "
            "(for example via ase-abacus)."
        ) from exc

    virial = None
    try:
        stress6 = atoms.calc.get_stress()
    except Exception:
        stress6 = None

    if stress6 is not None:
        virial = stress6_to_virial(stress6, atoms.get_volume())
    else:
        print(f"Warning: stress is missing in {workdir}, Virial will be omitted")

    return {
        "natoms": len(atoms),
        "cell": flatten_cell(atoms.get_cell()),
        "energy": float(atoms.get_potential_energy()),
        "symbols": atoms.get_chemical_symbols(),
        "positions": np.asarray(atoms.get_positions(), dtype=float),
        "forces": np.asarray(atoms.get_forces(), dtype=float),
        "virial": virial,
        "config_type": relative_config_type(root_dir, workdir),
    }


def extract_from_json(root_dir, workdir):
    output_dir = find_json_output_dir(workdir)
    if output_dir is None:
        return None

    json_file = output_dir / "abacus.json"
    scf_nmax = get_scf_nmax(workdir)

    with json_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    output0 = data["output"][0]
    scf_steps = len(output0.get("scf", []))
    if scf_steps >= scf_nmax:
        print(f"Skip {workdir}: calculation is incomplete or unconverged")
        return None

    natoms = int(data["init"]["natom"])
    cell_matrix = np.asarray(output0["cell"], dtype=float)
    positions = np.asarray(output0["coordinate"], dtype=float)
    forces = np.asarray(output0["force"], dtype=float)

    if positions.shape != (natoms, 3):
        raise ValueError(
            f"Unexpected coordinate shape in {json_file}: {positions.shape}, "
            f"expected ({natoms}, 3)"
        )
    if forces.shape != (natoms, 3):
        raise ValueError(
            f"Unexpected force shape in {json_file}: {forces.shape}, "
            f"expected ({natoms}, 3)"
        )

    virial = None
    if "stress" in output0 and output0["stress"] is not None:
        volume = abs(np.dot(cell_matrix[0], np.cross(cell_matrix[1], cell_matrix[2])))
        virial = json_stress_to_virial(output0["stress"], volume)
    else:
        print(f"Warning: stress is missing in {workdir}, Virial will be omitted")

    return {
        "natoms": natoms,
        "cell": flatten_cell(cell_matrix),
        "energy": float(output0["energy"]),
        "symbols": infer_json_symbols(data["init"], natoms),
        "positions": positions,
        "forces": forces,
        "virial": virial,
        "config_type": relative_config_type(root_dir, workdir),
    }


def iter_frames(root_dir):
    root_dir = Path(root_dir).resolve()
    processed_dirs = set()

    for current_root, dirs, files in os.walk(root_dir):
        dirs.sort()
        dirs[:] = [d for d in dirs if d != "OUT.ABACUS"]
        files = set(files)
        workdir = Path(current_root)

        if workdir.name == "OUT.ABACUS":
            continue

        if workdir in processed_dirs:
            continue

        has_log = "running_scf.log" in files or (workdir / "OUT.ABACUS" / "running_scf.log").is_file()
        has_json = "abacus.json" in files or (workdir / "OUT.ABACUS" / "abacus.json").is_file()

        if has_log:
            frame = extract_from_log(root_dir, workdir)
        elif has_json:
            frame = extract_from_json(root_dir, workdir)
        else:
            if "device.log" in files or "devie.log" in files:
                print(f"Skip {workdir}: output files are missing")
            continue

        if frame is not None:
            processed_dirs.add(workdir)
            yield frame


def format_header(frame):
    lattice_str = " ".join(f"{x:.10f}" for x in frame["cell"])
    parts = [
        f'energy={frame["energy"]:.16f}',
        f'Lattice="{lattice_str}"',
    ]

    if frame["virial"] is not None:
        virial_str = " ".join(f"{x:.10f}" for x in frame["virial"])
        parts.append(f'Virial="{virial_str}"')

    parts.append(f'config_type="{frame["config_type"]}"')
    parts.append("Properties=species:S:1:pos:R:3:forces:R:3")
    return " ".join(parts)


def write_xyz(frames, output_xyz):
    output_xyz = Path(output_xyz)
    output_xyz.parent.mkdir(parents=True, exist_ok=True)

    with output_xyz.open("w", encoding="utf-8", newline="\n") as fh:
        for frame in frames:
            fh.write(f'{frame["natoms"]}\n')
            fh.write(format_header(frame) + "\n")
            for symbol, pos, force in zip(
                frame["symbols"], frame["positions"], frame["forces"]
            ):
                fh.write(
                    f"{symbol:<6}"
                    f"{pos[0]:20.10f}{pos[1]:20.10f}{pos[2]:20.10f}"
                    f"{force[0]:20.10f}{force[1]:20.10f}{force[2]:20.10f}\n"
                )


def main():
    args = parse_args()
    frames = list(iter_frames(args.root_dir))

    if not frames:
        raise SystemExit("No converged ABACUS structures were found.")

    write_xyz(frames, args.output_xyz)
    print(f'Wrote {len(frames)} frame(s) to "{args.output_xyz}"')


if __name__ == "__main__":
    main()
