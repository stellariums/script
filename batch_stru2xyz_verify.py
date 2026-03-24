#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOHR_TO_ANG = 0.529177210903
STRU_NAME_PATTERN = re.compile(r"STRU-\d+$")
LATTICE_PATTERN = re.compile(r'Lattice="([^"]+)"')


@dataclass
class Structure:
    lattice: np.ndarray
    atoms: list[tuple[str, np.ndarray]]


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def read_lines(filename: Path) -> list[str]:
    with filename.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def find_section(lines: list[str], name: str) -> int:
    for index, line in enumerate(lines):
        if strip_comment(line).upper() == name.upper():
            return index
    raise ValueError(f"找不到 section: {name}")


def parse_atomic_species(lines: list[str]) -> list[str]:
    index = find_section(lines, "ATOMIC_SPECIES") + 1
    species = []
    stop_sections = {
        "NUMERICAL_ORBITAL",
        "LATTICE_CONSTANT",
        "LATTICE_VECTORS",
        "LATTICE_PARAMETERS",
        "ATOMIC_POSITIONS",
        "NUMERICAL_DESCRIPTOR",
        "PAW_FILES",
    }

    while index < len(lines):
        stripped = strip_comment(lines[index])
        if not stripped:
            index += 1
            continue
        if stripped.upper() in stop_sections:
            break
        parts = stripped.split()
        if parts:
            species.append(parts[0])
        index += 1

    if not species:
        raise ValueError("ATOMIC_SPECIES 为空")
    return species


def parse_lattice_constant(lines: list[str]) -> float:
    index = find_section(lines, "LATTICE_CONSTANT") + 1
    while index < len(lines) and not strip_comment(lines[index]):
        index += 1
    return float(strip_comment(lines[index]).split()[0])


def parse_lattice(lines: list[str]) -> np.ndarray:
    lattice_constant_bohr = parse_lattice_constant(lines)
    index = find_section(lines, "LATTICE_VECTORS") + 1
    vectors = []

    while index < len(lines) and len(vectors) < 3:
        stripped = strip_comment(lines[index])
        if stripped:
            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError("LATTICE_VECTORS 格式错误")
            vectors.append([float(parts[0]), float(parts[1]), float(parts[2])])
        index += 1

    if len(vectors) != 3:
        raise ValueError("LATTICE_VECTORS 不完整")

    return np.array(vectors, dtype=float) * lattice_constant_bohr * BOHR_TO_ANG


def parse_atomic_positions(
    lines: list[str], species_list: list[str], lattice: np.ndarray
) -> list[tuple[str, np.ndarray]]:
    index = find_section(lines, "ATOMIC_POSITIONS") + 1

    while index < len(lines) and not strip_comment(lines[index]):
        index += 1
    coord_type = strip_comment(lines[index]).lower()
    index += 1

    atoms = []
    species_set = set(species_list)
    lattice_constant_bohr = parse_lattice_constant(lines)

    while index < len(lines):
        while index < len(lines) and not strip_comment(lines[index]):
            index += 1
        if index >= len(lines):
            break

        label = strip_comment(lines[index]).split()[0]
        if label not in species_set:
            break
        element = label
        index += 1

        while index < len(lines) and not strip_comment(lines[index]):
            index += 1
        index += 1

        while index < len(lines) and not strip_comment(lines[index]):
            index += 1
        atom_count = int(strip_comment(lines[index]).split()[0])
        index += 1

        for _ in range(atom_count):
            while index < len(lines) and not strip_comment(lines[index]):
                index += 1
            if index >= len(lines):
                raise ValueError("ATOMIC_POSITIONS 提前结束")

            parts = strip_comment(lines[index]).split()
            if len(parts) < 3:
                raise ValueError(f"坐标行格式错误: {lines[index]}")

            coords = np.array(list(map(float, parts[:3])), dtype=float)
            if coord_type == "direct":
                cartesian = coords @ lattice
            elif coord_type == "cartesian":
                cartesian = coords * lattice_constant_bohr * BOHR_TO_ANG
            elif coord_type == "cartesian_angstrom":
                cartesian = coords
            elif coord_type == "cartesian_au":
                cartesian = coords * BOHR_TO_ANG
            else:
                raise ValueError(f"暂不支持坐标类型: {coord_type}")

            atoms.append((element, cartesian))
            index += 1

    return atoms


def parse_stru_file(path: Path) -> Structure:
    lines = read_lines(path)
    species_list = parse_atomic_species(lines)
    lattice = parse_lattice(lines)
    atoms = parse_atomic_positions(lines, species_list, lattice)
    return Structure(lattice=lattice, atoms=atoms)


def format_lattice_line(lattice: np.ndarray) -> str:
    lattice_str = " ".join(f"{value:.8f}" for value in lattice.reshape(-1))
    return f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3'


def write_extxyz(output_path: Path, structure: Structure) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(structure.atoms)}\n")
        handle.write(format_lattice_line(structure.lattice) + "\n")
        for element, coords in structure.atoms:
            handle.write(
                f"{element} {coords[0]:.10f} {coords[1]:.10f} {coords[2]:.10f}\n"
            )


def parse_extxyz(path: Path) -> Structure:
    lines = read_lines(path)
    if len(lines) < 2:
        raise ValueError("extxyz 文件内容不完整")

    atom_count = int(lines[0].strip())
    lattice_match = LATTICE_PATTERN.search(lines[1])
    if lattice_match is None:
        raise ValueError("extxyz 缺少 Lattice 信息")

    lattice_values = [float(value) for value in lattice_match.group(1).split()]
    if len(lattice_values) != 9:
        raise ValueError("Lattice 不是 3x3 矩阵")

    atoms = []
    for line in lines[2 : 2 + atom_count]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"原子行格式错误: {line}")
        atoms.append((parts[0], np.array(list(map(float, parts[1:4])), dtype=float)))

    if len(atoms) != atom_count:
        raise ValueError("extxyz 原子数与头信息不一致")

    return Structure(lattice=np.array(lattice_values, dtype=float).reshape(3, 3), atoms=atoms)


def compare_structures(
    original: Structure, converted: Structure, atol: float
) -> tuple[bool, str]:
    if len(original.atoms) != len(converted.atoms):
        return False, f"原子数不一致: {len(original.atoms)} != {len(converted.atoms)}"

    if not np.allclose(original.lattice, converted.lattice, atol=atol, rtol=0.0):
        max_diff = np.abs(original.lattice - converted.lattice).max()
        return False, f"晶格不一致，最大偏差 {max_diff:.3e}"

    for index, ((orig_elem, orig_pos), (conv_elem, conv_pos)) in enumerate(
        zip(original.atoms, converted.atoms), start=1
    ):
        if orig_elem != conv_elem:
            return False, f"第 {index} 个原子元素不一致: {orig_elem} != {conv_elem}"
        if not np.allclose(orig_pos, conv_pos, atol=atol, rtol=0.0):
            max_diff = np.abs(orig_pos - conv_pos).max()
            return False, f"第 {index} 个原子坐标不一致，最大偏差 {max_diff:.3e}"

    return True, "OK"


def iter_stru_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir() if path.is_file() and STRU_NAME_PATTERN.fullmatch(path.name)
    )


def convert_and_verify(root: Path, output_dir: Path, atol: float) -> int:
    stru_files = iter_stru_files(root)
    if not stru_files:
        raise FileNotFoundError(f"在 {root} 下未找到 STRU-xxxxx 文件")

    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_cases = []

    for stru_path in stru_files:
        xyz_path = output_dir / f"{stru_path.name}.xyz"
        try:
            original = parse_stru_file(stru_path)
            write_extxyz(xyz_path, original)
            converted = parse_extxyz(xyz_path)
            matched, message = compare_structures(original, converted, atol=atol)
            if not matched:
                failed_cases.append((stru_path.name, message))
                continue

            success_count += 1
            print(f"[OK] {stru_path.name} -> {xyz_path.name}")
        except Exception as exc:
            failed_cases.append((stru_path.name, str(exc)))

    print()
    print(f"总文件数: {len(stru_files)}")
    print(f"转换并校验成功: {success_count}")
    print(f"失败: {len(failed_cases)}")

    if failed_cases:
        print("失败详情:")
        for name, reason in failed_cases:
            print(f"  - {name}: {reason}")
        return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="遍历目录中的 STRU 文件，批量转为 extxyz，并校验转换前后结构是否一致。"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="STRUs 所在目录，默认当前目录",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="xyz_outputs",
        help="输出 xyz 文件的目录，默认 ./xyz_outputs",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="结构对比的绝对容差，默认 1e-6",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return convert_and_verify(root, output_dir, atol=args.atol)


if __name__ == "__main__":
    raise SystemExit(main())