#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

BOHR_TO_ANG = 0.529177210903


def strip_comment(line: str) -> str:
    # 去掉 # 注释
    return line.split("#", 1)[0].strip()


def read_lines(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def find_section(lines, name):
    for i, line in enumerate(lines):
        if strip_comment(line).upper() == name.upper():
            return i
    raise ValueError(f"找不到 section: {name}")


def parse_atomic_species(lines):
    i = find_section(lines, "ATOMIC_SPECIES") + 1
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

    while i < len(lines):
        s = strip_comment(lines[i])
        if not s:
            i += 1
            continue

        if s.upper() in stop_sections:
            break

        parts = s.split()
        if parts:
            species.append(parts[0])
        i += 1

    if not species:
        raise ValueError("ATOMIC_SPECIES 为空")
    return species


def parse_lattice(lines):
    # LATTICE_CONSTANT
    i = find_section(lines, "LATTICE_CONSTANT") + 1
    while i < len(lines) and not strip_comment(lines[i]):
        i += 1
    lattice_constant_bohr = float(strip_comment(lines[i]).split()[0])

    # LATTICE_VECTORS
    j = find_section(lines, "LATTICE_VECTORS") + 1
    vecs = []
    while j < len(lines) and len(vecs) < 3:
        s = strip_comment(lines[j])
        if s:
            parts = s.split()
            if len(parts) < 3:
                raise ValueError("LATTICE_VECTORS 格式错误")
            vecs.append([float(parts[0]), float(parts[1]), float(parts[2])])
        j += 1

    if len(vecs) != 3:
        raise ValueError("LATTICE_VECTORS 不完整")

    # ABACUS:
    # lattice(Angstrom) = lattice_constant(Bohr) * vectors * BOHR_TO_ANG
    lattice = np.array(vecs, dtype=float) * lattice_constant_bohr * BOHR_TO_ANG
    return lattice


def parse_atomic_positions(lines, species_list, lattice):
    i = find_section(lines, "ATOMIC_POSITIONS") + 1

    while i < len(lines) and not strip_comment(lines[i]):
        i += 1

    coord_type = strip_comment(lines[i]).lower()
    i += 1

    atoms = []
    species_set = set(species_list)

    while i < len(lines):
        while i < len(lines) and not strip_comment(lines[i]):
            i += 1
        if i >= len(lines):
            break

        s = strip_comment(lines[i])
        label = s.split()[0]

        if label not in species_set:
            break

        elem = label
        i += 1

        # 跳过 magnetism 行
        while i < len(lines) and not strip_comment(lines[i]):
            i += 1
        i += 1

        # 原子个数
        while i < len(lines) and not strip_comment(lines[i]):
            i += 1
        natom = int(strip_comment(lines[i]).split()[0])
        i += 1

        for _ in range(natom):
            while i < len(lines) and not strip_comment(lines[i]):
                i += 1
            if i >= len(lines):
                raise ValueError("ATOMIC_POSITIONS 提前结束")

            parts = strip_comment(lines[i]).split()
            if len(parts) < 3:
                raise ValueError(f"坐标行格式错误: {lines[i]}")

            xyz = np.array(list(map(float, parts[:3])), dtype=float)

            if coord_type == "direct":
                # 分数坐标 -> 笛卡尔坐标
                cart = xyz @ lattice
            elif coord_type == "cartesian":
                # 以 lattice_constant 为单位
                lc_ang = np.linalg.norm(lattice[0]) / np.linalg.norm(
                    np.array([1.0, 0.0, 0.0])
                )
                # 更稳妥：直接从 lattice_constant 重新算
                # 但这里 cartesian 少见，这里仍按 ABACUS 定义转
                # xyz * lattice_constant(Bohr) * BOHR_TO_ANG
                i_lc = find_section(lines, "LATTICE_CONSTANT") + 1
                while i_lc < len(lines) and not strip_comment(lines[i_lc]):
                    i_lc += 1
                lattice_constant_bohr = float(strip_comment(lines[i_lc]).split()[0])
                cart = xyz * lattice_constant_bohr * BOHR_TO_ANG
            elif coord_type == "cartesian_angstrom":
                cart = xyz
            elif coord_type == "cartesian_au":
                cart = xyz * BOHR_TO_ANG
            else:
                raise ValueError(f"暂不支持坐标类型: {coord_type}")

            atoms.append((elem, cart[0], cart[1], cart[2]))
            i += 1

    return atoms


def format_lattice_line(lattice):
    # extxyz 常见写法：按 3x3 顺序展开
    flat = lattice.reshape(-1)
    lattice_str = " ".join(f"{x:.8f}" for x in flat)
    return f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3'


def write_extxyz(outfile, lattice, atoms):
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(format_lattice_line(lattice) + "\n")
        for elem, x, y, z in atoms:
            f.write(f"{elem} {x:.10f} {y:.10f} {z:.10f}\n")


def main():
    if len(sys.argv) < 2:
        print("用法: python stru2extxyz.py STRU [output.xyz]")
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) > 2 else "output.xyz"

    lines = read_lines(infile)
    species_list = parse_atomic_species(lines)
    lattice = parse_lattice(lines)
    atoms = parse_atomic_positions(lines, species_list, lattice)

    write_extxyz(outfile, lattice, atoms)
    print(f"已生成: {outfile}")


if __name__ == "__main__":
    main()
