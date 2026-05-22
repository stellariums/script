#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np

BOHR_TO_ANG = 0.529177210903

DEFAULT_ATOM_TAIL = "m 1 1 1"

KNOWN_SECTIONS = {
    "ATOMIC_SPECIES",
    "NUMERICAL_ORBITAL",
    "NUMERICAL_DESCRIPTOR",
    "LATTICE_CONSTANT",
    "LATTICE_VECTORS",
    "ATOMIC_POSITIONS",
}


def remove_comment(line):
    return line.split("#")[0].strip()


def is_section(line):
    s = line.strip().upper()
    return s in KNOWN_SECTIONS


def next_nonempty(lines, i):
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i


def read_section_lines(lines, section_name):
    section_name = section_name.upper()
    start = None

    for i, line in enumerate(lines):
        if line.strip().upper() == section_name:
            start = i + 1
            break

    if start is None:
        return []

    out = []
    for line in lines[start:]:
        if is_section(line):
            break
        if line.strip():
            out.append(line.rstrip())

    return out


def read_template_stru(path):
    """
    从模板 STRU 中读取：
    1. ATOMIC_SPECIES
    2. NUMERICAL_ORBITAL
    3. LATTICE_CONSTANT
    4. 每个元素的 magnetism 行
    5. 每个元素原子坐标后的尾巴，比如 m 1 1 1
    """
    lines = open(path, "r", encoding="utf-8").read().splitlines()

    atomic_species_lines = read_section_lines(lines, "ATOMIC_SPECIES")
    numerical_orbital_lines = read_section_lines(lines, "NUMERICAL_ORBITAL")

    lattice_constant_lines = read_section_lines(lines, "LATTICE_CONSTANT")
    if not lattice_constant_lines:
        raise RuntimeError(f"Cannot find LATTICE_CONSTANT in template STRU: {path}")

    lattice_constant = float(remove_comment(lattice_constant_lines[0]).split()[0])

    mag_lines = {}
    atom_tails = {}
    atom_counts = {}

    pos_header = None
    for i, line in enumerate(lines):
        if line.strip().upper() == "ATOMIC_POSITIONS":
            pos_header = i
            break

    if pos_header is None:
        raise RuntimeError(f"Cannot find ATOMIC_POSITIONS in template STRU: {path}")

    i = next_nonempty(lines, pos_header + 1)

    # 坐标类型行，例如 Direct / Cartesian
    if i >= len(lines):
        raise RuntimeError("ATOMIC_POSITIONS section is incomplete.")
    coord_type = lines[i].strip()
    i += 1

    while i < len(lines):
        i = next_nonempty(lines, i)
        if i >= len(lines) or is_section(lines[i]):
            break

        # 元素标签行，例如：U #label
        elem = remove_comment(lines[i]).split()[0]
        i += 1

        i = next_nonempty(lines, i)
        if i >= len(lines):
            raise RuntimeError(f"Missing magnetism line for element {elem}")
        mag_line = lines[i].strip()
        i += 1

        i = next_nonempty(lines, i)
        if i >= len(lines):
            raise RuntimeError(f"Missing atom number line for element {elem}")
        count = int(remove_comment(lines[i]).split()[0])
        i += 1

        tails = []
        for _ in range(count):
            i = next_nonempty(lines, i)
            if i >= len(lines):
                raise RuntimeError(f"Missing atomic coordinate lines for element {elem}")

            parts = lines[i].split()
            if len(parts) > 3:
                tail = " ".join(parts[3:])
            else:
                tail = DEFAULT_ATOM_TAIL

            tails.append(tail)
            i += 1

        mag_lines[elem] = mag_line
        atom_tails[elem] = tails
        atom_counts[elem] = count

    return {
        "atomic_species_lines": atomic_species_lines,
        "numerical_orbital_lines": numerical_orbital_lines,
        "lattice_constant": lattice_constant,
        "mag_lines": mag_lines,
        "atom_tails": atom_tails,
        "atom_counts": atom_counts,
        "coord_type": coord_type,
    }


def read_poscar(path):
    lines = [line.strip() for line in open(path, "r", encoding="utf-8") if line.strip()]

    scale = float(lines[1].split()[0])

    cell = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)],
        dtype=float,
    ) * scale

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    natom = sum(counts)

    idx = 7

    # Selective dynamics
    if lines[idx].lower().startswith("s"):
        idx += 1

    coord_type = lines[idx].lower()
    pos_start = idx + 1

    pos = np.array(
        [[float(x) for x in lines[pos_start + i].split()[:3]] for i in range(natom)],
        dtype=float,
    )

    if coord_type.startswith("d"):
        direct = pos
    elif coord_type.startswith("c") or coord_type.startswith("k"):
        direct = pos @ np.linalg.inv(cell)
    else:
        raise ValueError(f"Unknown coordinate type in {path}: {coord_type}")

    direct = direct - np.floor(direct)

    direct[np.isclose(direct, 0.0, atol=1e-10)] = 0.0
    direct[np.isclose(direct, 1.0, atol=1e-10)] = 0.0
    direct[np.isclose(direct, 0.5, atol=1e-10)] = 0.5

    return cell, species, counts, direct


def format_mag_line(mag_line):
    """
    保留模板 STRU 里的磁矩行。
    如果模板里只有数字，例如 3.0000，则自动补 #magnetism。
    """
    s = mag_line.strip()
    if "#magnetism" not in s:
        s = f"{s}   #magnetism"
    return s


def get_atom_tail(template, elem, atom_index):
    """
    如果模板 STRU 中每个原子坐标后面有额外信息，例如：
        m 1 1 1
    或者将来你写了逐原子的 mag 信息，也会按元素内原子顺序循环复制。

    对 FM/nonmag：通常所有 U 尾巴一样，没问题。
    对 AFM：如果你的模板里通过逐原子 tail 区分磁矩，这里会按原始模式重复。
    """
    tails = template["atom_tails"].get(elem, [])
    if not tails:
        return DEFAULT_ATOM_TAIL

    return tails[atom_index % len(tails)]


def write_stru(path, cell_angstrom, species, counts, direct, template):
    lattice_constant = template["lattice_constant"]

    # ABACUS 的 LATTICE_CONSTANT 单位是 Bohr。
    # POSCAR 读出来的 cell 是 Angstrom。
    # 所以这里要把 Angstrom cell 转成 ABACUS 写法下的无量纲 LATTICE_VECTORS。
    scale_angstrom = lattice_constant * BOHR_TO_ANG
    cell_to_write = cell_angstrom / scale_angstrom

    with open(path, "w", encoding="utf-8") as f:
        f.write("ATOMIC_SPECIES\n")
        for line in template["atomic_species_lines"]:
            f.write(f"{line}\n")

        f.write("\nNUMERICAL_ORBITAL\n")
        for line in template["numerical_orbital_lines"]:
            f.write(f"{line}\n")

        f.write("\nLATTICE_CONSTANT\n")
        f.write(f"{lattice_constant:.10f}\n")

        f.write("\nLATTICE_VECTORS\n")
        for v in cell_to_write:
            f.write(f"{v[0]:20.10f} {v[1]:20.10f} {v[2]:20.10f}\n")

        f.write("\nATOMIC_POSITIONS\n")
        f.write("Direct\n\n")

        atom_i = 0

        for elem, count in zip(species, counts):
            f.write(f"{elem} #label\n")

            if elem in template["mag_lines"]:
                f.write(f"{format_mag_line(template['mag_lines'][elem])}\n")
            else:
                print(f"[Warning] Element {elem} not found in template STRU magnetism. Use 0.0000.")
                f.write("0.0000   #magnetism\n")

            f.write(f"{count} #number of atoms\n")

            for j in range(count):
                x, y, z = direct[atom_i]
                tail = get_atom_tail(template, elem, j)
                f.write(f"{x:20.10f} {y:20.10f} {z:20.10f} {tail}\n")
                atom_i += 1

            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3RD.POSCAR.* files to ABACUS STRU files using magnetism information from template STRU."
    )

    parser.add_argument(
        "--template",
        default="STRU",
        help="Template STRU file containing ATOMIC_SPECIES, NUMERICAL_ORBITAL and magnetism information. Default: STRU",
    )

    parser.add_argument(
        "--glob",
        default="3RD.POSCAR.*",
        help="Input POSCAR file pattern. Default: 3RD.POSCAR.*",
    )

    parser.add_argument(
        "--out-prefix",
        default="STRU_",
        help="Output STRU prefix. Default: STRU_",
    )

    args = parser.parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        raise RuntimeError(f"Template STRU not found: {template_path}")

    template = read_template_stru(template_path)

    print(f"Template STRU : {template_path}")
    print("Magnetism read from template:")
    for elem, mag in template["mag_lines"].items():
        print(f"  {elem}: {mag}")

    files = sorted(Path(".").glob(args.glob))

    if not files:
        raise RuntimeError(f"No files found with pattern: {args.glob}")

    print(f"\nFound {len(files)} POSCAR files.")

    for p in files:
        suffix = p.name.split(".")[-1]
        out = Path(f"{args.out_prefix}{suffix}")

        cell, species, counts, direct = read_poscar(p)
        write_stru(out, cell, species, counts, direct, template)

        print(f"{p.name} -> {out.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()