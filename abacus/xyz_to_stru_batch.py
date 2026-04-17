#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量将 extxyz 轨迹中的多帧结构转换为 ABACUS 的 STRU 输入文件。

脚本用途：
1. 读取一个包含多帧结构的 extxyz 文件。
2. 读取一个模板 STRU 文件，保留其中的元素信息、轨道信息、磁矩设置以及原子行附加字段。
3. 对每一帧结构：
   - 从 extxyz 注释行中读取 Lattice 信息；
   - 从笛卡尔坐标转换为分数坐标；
   - 按模板中的元素顺序和每种元素的原子数写出新的 STRU 文件。
4. 最终批量生成 `STRU-1`、`STRU-2` 等文件，可用于后续 ABACUS 计算。

输入要求：
- xyz 文件必须是 extxyz 格式，每一帧的注释行中必须包含 `Lattice="..."`，且共有 9 个晶格分量。
- xyz 文件中的原子顺序必须与模板 STRU 中的元素分组顺序一致。
- 模板 STRU 中应包含：
  `ATOMIC_SPECIES`、`LATTICE_CONSTANT`、`ATOMIC_POSITIONS`，
  可选包含 `NUMERICAL_ORBITAL`。

命令行用法：
    python xyz_to_stru_batch.py [xyz_file] [template_stru] [nout]
                                [--outdir OUTDIR] [--prefix PREFIX]
                                [--job-dirs | --flat-output]
                                [--input-template INPUT]
                                [--kpt-template KPT]

参数说明：
- `xyz_file`：输入 extxyz 文件，默认是 `pertube.xyz`
- `template_stru`：模板 STRU 文件，默认是 `STRU`
- `nout`：输出前多少帧，默认是 `50`
- `--outdir`：输出目录，默认是当前目录
- `--prefix`：平铺输出模式下的文件名前缀，默认是 `STRU-`
- `--job-dirs`：按编号目录输出，例如 `1/`、`2/`、`3/`，每个目录内写入 `STRU`、`INPUT`、`KPT`；这是默认模式
- `--flat-output`：关闭编号目录模式，直接在输出目录下生成 `STRU-1`、`STRU-2` 这类文件
- `--input-template`：`--job-dirs` 模式下使用的 INPUT 模板文件，默认是 `INPUT`
- `--kpt-template`：`--job-dirs` 模式下使用的 KPT 模板文件，默认是 `KPT`

示例：
    python xyz_to_stru_batch.py train.xyz STRU 50
    python xyz_to_stru_batch.py train.xyz STRU 100 --outdir out
    python xyz_to_stru_batch.py train.xyz STRU 20 --outdir out --prefix sample-
    python xyz_to_stru_batch.py train.xyz STRU 100 --outdir jobs --job-dirs
    python xyz_to_stru_batch.py train.xyz STRU 20 --outdir out --flat-output
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np

BOHR_TO_ANG = 0.529177210903


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def find_section(lines, name):
    for i, line in enumerate(lines):
        if strip_comment(line).upper() == name.upper():
            return i
    raise ValueError(f"找不到 section: {name}")


def collect_block(lines, start_name, stop_names):
    i = find_section(lines, start_name) + 1
    out = []
    while i < len(lines):
        s = strip_comment(lines[i])
        if s.upper() in stop_names:
            break
        out.append(lines[i])
        i += 1
    return out


def parse_template_stru(template_path):
    lines = read_lines(template_path)

    atomic_species_block = collect_block(
        lines,
        "ATOMIC_SPECIES",
        {
            "NUMERICAL_ORBITAL",
            "LATTICE_CONSTANT",
            "LATTICE_VECTORS",
            "LATTICE_PARAMETERS",
            "ATOMIC_POSITIONS",
        },
    )

    try:
        numerical_orbital_block = collect_block(
            lines,
            "NUMERICAL_ORBITAL",
            {
                "LATTICE_CONSTANT",
                "LATTICE_VECTORS",
                "LATTICE_PARAMETERS",
                "ATOMIC_POSITIONS",
            },
        )
    except Exception:
        numerical_orbital_block = []

    i_lc = find_section(lines, "LATTICE_CONSTANT") + 1
    while i_lc < len(lines) and not strip_comment(lines[i_lc]):
        i_lc += 1
    lattice_constant_bohr = float(strip_comment(lines[i_lc]).split()[0])

    i_pos = find_section(lines, "ATOMIC_POSITIONS") + 1
    while i_pos < len(lines) and not strip_comment(lines[i_pos]):
        i_pos += 1
    i_pos += 1  # 跳过 Direct/Cartesian 这一行

    species_names = []
    for line in atomic_species_block:
        s = strip_comment(line)
        if s:
            species_names.append(s.split()[0])
    species_set = set(species_names)

    species_order = []
    mag_lines = {}
    natoms_per_species = {}
    atom_suffix_per_species = {}

    i = i_pos
    while i < len(lines):
        while i < len(lines) and not strip_comment(lines[i]):
            i += 1
        if i >= len(lines):
            break

        label = strip_comment(lines[i]).split()[0]
        if label not in species_set:
            break

        species_order.append(label)
        i += 1

        while i < len(lines) and not strip_comment(lines[i]):
            i += 1
        mag_lines[label] = lines[i]
        i += 1

        while i < len(lines) and not strip_comment(lines[i]):
            i += 1
        nat = int(strip_comment(lines[i]).split()[0])
        natoms_per_species[label] = nat
        i += 1

        suffixes = []
        for _ in range(nat):
            while i < len(lines) and not strip_comment(lines[i]):
                i += 1
            parts = strip_comment(lines[i]).split()
            if len(parts) < 3:
                raise ValueError(f"模板原子行格式错误: {lines[i]}")
            suffix = " ".join(parts[3:]) if len(parts) > 3 else ""
            suffixes.append(suffix)
            i += 1
        atom_suffix_per_species[label] = suffixes

    return {
        "atomic_species_block": atomic_species_block,
        "numerical_orbital_block": numerical_orbital_block,
        "lattice_constant_bohr": lattice_constant_bohr,
        "species_order": species_order,
        "mag_lines": mag_lines,
        "natoms_per_species": natoms_per_species,
        "atom_suffix_per_species": atom_suffix_per_species,
    }


def parse_extxyz_frames(xyz_path):
    lines = Path(xyz_path).read_text(encoding="utf-8").splitlines()
    frames = []

    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        natoms = int(lines[i].strip())
        if i + 1 >= len(lines):
            raise ValueError("xyz 文件格式不完整：缺少注释行")

        comment = lines[i + 1].strip()
        m = re.search(r'Lattice="([^"]+)"', comment)
        if not m:
            raise ValueError(f"第 {len(frames)+1} 帧缺少 Lattice 信息")

        lattice_vals = [float(x) for x in m.group(1).split()]
        if len(lattice_vals) != 9:
            raise ValueError(f"第 {len(frames)+1} 帧的 Lattice 不是 9 个数")

        lattice_ang = np.array(
            [
                lattice_vals[0:3],
                lattice_vals[3:6],
                lattice_vals[6:9],
            ],
            dtype=float,
        )

        atoms = []
        start = i + 2
        end = start + natoms
        if end > len(lines):
            raise ValueError(f"第 {len(frames)+1} 帧原子数不足")

        for line in lines[start:end]:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"原子行格式错误: {line}")
            elem = parts[0]
            x, y, z = map(float, parts[1:4])
            atoms.append((elem, x, y, z))

        frames.append({
            "natoms": natoms,
            "lattice_ang": lattice_ang,
            "atoms": atoms,
        })

        i = end

    return frames


def group_atoms_by_species(frame_atoms, species_order, natoms_per_species):
    out = {}
    idx = 0
    for sp in species_order:
        n = natoms_per_species[sp]
        chunk = frame_atoms[idx:idx + n]
        if len(chunk) != n:
            raise ValueError(f"{sp} 原子数不足，期望 {n}，实际 {len(chunk)}")
        for atom in chunk:
            if atom[0] != sp:
                raise ValueError(
                    f"原子顺序和模板不一致：期望 {sp}，实际读到 {atom[0]}"
                )
        out[sp] = chunk
        idx += n

    if idx != len(frame_atoms):
        raise ValueError(
            f"模板原子总数与 xyz 不一致：模板累计 {idx}，xyz 实际 {len(frame_atoms)}"
        )

    return out


def lattice_ang_to_scaled_vectors(lattice_ang, lattice_constant_bohr):
    scale = lattice_constant_bohr * BOHR_TO_ANG
    return lattice_ang / scale


def cart_to_frac(cart_xyz, lattice_ang):
    # 约定：分数坐标按 row-vector 右乘晶格矩阵
    # cart = frac @ lattice
    # 所以 frac = cart @ inv(lattice)
    inv_lat = np.linalg.inv(lattice_ang)
    frac = np.array(cart_xyz, dtype=float) @ inv_lat
    return frac


def wrap_frac(frac, tol=1e-10):
    frac = np.mod(frac, 1.0)
    frac[np.isclose(frac, 1.0, atol=tol)] = 0.0
    frac[np.isclose(frac, 0.0, atol=tol)] = 0.0
    return frac


def write_stru(outpath, template_info, frame):
    atomic_species_block = template_info["atomic_species_block"]
    numerical_orbital_block = template_info["numerical_orbital_block"]
    lattice_constant_bohr = template_info["lattice_constant_bohr"]
    species_order = template_info["species_order"]
    mag_lines = template_info["mag_lines"]
    natoms_per_species = template_info["natoms_per_species"]
    atom_suffix_per_species = template_info["atom_suffix_per_species"]

    grouped = group_atoms_by_species(frame["atoms"], species_order, natoms_per_species)
    lattice_ang = frame["lattice_ang"]
    scaled_vecs = lattice_ang_to_scaled_vectors(lattice_ang, lattice_constant_bohr)

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("ATOMIC_SPECIES\n")
        for line in atomic_species_block:
            f.write(line.rstrip() + "\n")
        f.write("\n")

        if numerical_orbital_block:
            f.write("NUMERICAL_ORBITAL\n")
            for line in numerical_orbital_block:
                f.write(line.rstrip() + "\n")
            f.write("\n")

        f.write("LATTICE_CONSTANT\n")
        f.write(f"{lattice_constant_bohr:.10f}\n\n")

        f.write("LATTICE_VECTORS\n")
        for row in scaled_vecs:
            f.write(f"{row[0]:18.10f} {row[1]:18.10f} {row[2]:18.10f}\n")
        f.write("\n")

        f.write("ATOMIC_POSITIONS\n")
        f.write("Direct\n\n")

        for sp in species_order:
            f.write(f"{sp} #label\n")
            f.write(mag_lines[sp].rstrip() + "\n")
            f.write(f"{natoms_per_species[sp]} #number of atoms\n")

            suffixes = atom_suffix_per_species[sp]
            atoms = grouped[sp]

            if len(suffixes) != len(atoms):
                raise ValueError(f"{sp} 的模板磁矩信息数目与 xyz 原子数不一致")

            for (_, x, y, z), suffix in zip(atoms, suffixes):
                frac = cart_to_frac([x, y, z], lattice_ang)
                frac = wrap_frac(frac)
                if suffix:
                    f.write(
                        f"{frac[0]:18.10f} {frac[1]:18.10f} {frac[2]:18.10f} {suffix}\n"
                    )
                else:
                    f.write(
                        f"{frac[0]:18.10f} {frac[1]:18.10f} {frac[2]:18.10f}\n"
                    )
            f.write("\n")


def copy_support_file(src, dst):
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"Required support file not found: {src}")
    shutil.copy2(src, dst)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Batch-convert frames from an extxyz file into ABACUS STRU files."
    )
    parser.add_argument(
        "xyz_file",
        nargs="?",
        default="pertube.xyz",
        help="Input extxyz file. Default: %(default)s",
    )
    parser.add_argument(
        "template_stru",
        nargs="?",
        default="STRU",
        help="Template STRU file. Default: %(default)s",
    )
    parser.add_argument(
        "nout",
        nargs="?",
        type=int,
        default=50,
        help="Number of frames to write. Default: %(default)s",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory for generated STRU files. Default: current directory",
    )
    parser.add_argument(
        "--prefix",
        default="STRU-",
        help="Prefix for generated file names. Default: %(default)s",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--job-dirs",
        dest="job_dirs",
        action="store_true",
        help=(
            "Create numbered directories (1, 2, 3, ...) under outdir. "
            "Each directory will contain STRU, INPUT, and KPT. This is the default."
        ),
    )
    mode_group.add_argument(
        "--flat-output",
        dest="job_dirs",
        action="store_false",
        help="Write prefixed STRU files directly under outdir instead of numbered directories.",
    )
    parser.set_defaults(job_dirs=True)
    parser.add_argument(
        "--input-template",
        default="INPUT",
        help="Template INPUT file used with --job-dirs. Default: %(default)s",
    )
    parser.add_argument(
        "--kpt-template",
        default="KPT",
        help="Template KPT file used with --job-dirs. Default: %(default)s",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    xyz_file = Path(args.xyz_file)
    template_stru = Path(args.template_stru)
    nout = args.nout
    outdir = Path(args.outdir)
    input_template = Path(args.input_template)
    kpt_template = Path(args.kpt_template)

    if nout <= 0:
        raise ValueError(f"nout must be positive, got {nout}")

    outdir.mkdir(parents=True, exist_ok=True)

    template_info = parse_template_stru(template_stru)
    frames = parse_extxyz_frames(xyz_file)

    if len(frames) < nout:
        raise ValueError(f"xyz 里只有 {len(frames)} 帧，不足 {nout} 帧")

    for i in range(nout):
        if args.job_dirs:
            job_dir = outdir / f"{i+1}"
            job_dir.mkdir(parents=True, exist_ok=True)
            write_stru(job_dir / "STRU", template_info, frames[i])
            copy_support_file(input_template, job_dir / "INPUT")
            copy_support_file(kpt_template, job_dir / "KPT")
        else:
            outname = outdir / f"{args.prefix}{i+1}"
            write_stru(outname, template_info, frames[i])

    if args.job_dirs:
        print(f"已在 {outdir} 下生成 1 到 {nout} 号目录，每个目录包含 STRU、INPUT、KPT")
    else:
        print(f"已生成 {args.prefix}1 到 {args.prefix}{nout}")


if __name__ == "__main__":
    main()
