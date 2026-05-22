#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert ABACUS STRU to VASP POSCAR.

Usage:
    python stru2pos.py STRU POSCAR

If no arguments are given:
    python stru2pos.py
will read ./STRU and write ./POSCAR.
"""

import sys
import numpy as np
from pathlib import Path

BOHR_TO_ANG = 0.529177210903


def clean_line(line: str) -> str:
    """Remove comments and strip."""
    for mark in ["#", "//"]:
        if mark in line:
            line = line.split(mark, 1)[0]
    return line.strip()


def read_nonempty_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = clean_line(raw)
            if s:
                lines.append(s)
    return lines


def find_section(lines, key):
    key = key.upper()
    for i, line in enumerate(lines):
        if line.upper() == key:
            return i
    return -1


def parse_lattice_constant(lines):
    idx = find_section(lines, "LATTICE_CONSTANT")
    if idx < 0:
        raise ValueError("Cannot find LATTICE_CONSTANT in STRU.")
    return float(lines[idx + 1].split()[0])


def parse_lattice_vectors(lines, lattice_constant_bohr):
    idx = find_section(lines, "LATTICE_VECTORS")
    if idx < 0:
        raise ValueError(
            "Cannot find LATTICE_VECTORS in STRU. "
            "This script does not handle latname-generated lattice vectors."
        )

    vecs = []
    for k in range(3):
        vecs.append([float(x) for x in lines[idx + 1 + k].split()[:3]])

    # ABACUS: lattice vectors are scaled by LATTICE_CONSTANT, whose unit is Bohr.
    cell_angstrom = np.array(vecs, dtype=float) * lattice_constant_bohr * BOHR_TO_ANG
    return cell_angstrom


def parse_atomic_positions(lines, lattice_constant_bohr, cell_angstrom):
    idx = find_section(lines, "ATOMIC_POSITIONS")
    if idx < 0:
        raise ValueError("Cannot find ATOMIC_POSITIONS in STRU.")

    coord_type = lines[idx + 1].split()[0].lower()

    species = []
    counts = []
    positions = []

    i = idx + 2
    known_sections = {
        "ATOMIC_SPECIES",
        "NUMERICAL_ORBITAL",
        "LATTICE_CONSTANT",
        "LATTICE_VECTORS",
        "LATTICE_PARAMETERS",
        "ATOMIC_POSITIONS",
    }

    while i < len(lines):
        token = lines[i].split()[0]
        if token.upper() in known_sections:
            break

        elem = token
        mag_line = lines[i + 1]       # not used here
        natom = int(lines[i + 2].split()[0])

        species.append(elem)
        counts.append(natom)

        for j in range(natom):
            parts = lines[i + 3 + j].split()
            xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
            positions.append(xyz)

        i += 3 + natom

    positions = np.array(positions, dtype=float)

    if coord_type.startswith("direct"):
        pos_mode = "Direct"
        pos_out = positions

    elif coord_type == "cartesian":
        # ABACUS Cartesian: coordinates are in units of LATTICE_CONSTANT.
        pos_mode = "Cartesian"
        pos_out = positions * lattice_constant_bohr * BOHR_TO_ANG

    elif coord_type == "cartesian_au":
        pos_mode = "Cartesian"
        pos_out = positions * BOHR_TO_ANG

    elif coord_type == "cartesian_angstrom":
        pos_mode = "Cartesian"
        pos_out = positions

    else:
        raise ValueError(f"Unsupported ATOMIC_POSITIONS coordinate type: {coord_type}")

    return species, counts, pos_mode, pos_out


def write_poscar(path, cell, species, counts, pos_mode, positions):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Converted from ABACUS STRU\n")
        f.write("1.0\n")

        for v in cell:
            f.write(f"{v[0]:20.12f} {v[1]:20.12f} {v[2]:20.12f}\n")

        f.write(" ".join(species) + "\n")
        f.write(" ".join(str(x) for x in counts) + "\n")
        f.write(pos_mode + "\n")

        for p in positions:
            f.write(f"{p[0]:20.12f} {p[1]:20.12f} {p[2]:20.12f}\n")


def main():
    in_file = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("STRU")
    out_file = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("POSCAR")

    lines = read_nonempty_lines(in_file)

    lattice_constant_bohr = parse_lattice_constant(lines)
    cell_angstrom = parse_lattice_vectors(lines, lattice_constant_bohr)
    species, counts, pos_mode, positions = parse_atomic_positions(
        lines, lattice_constant_bohr, cell_angstrom
    )

    if sum(counts) != len(positions):
        raise RuntimeError("Atom count mismatch while parsing STRU.")

    write_poscar(out_file, cell_angstrom, species, counts, pos_mode, positions)

    print(f"Read  : {in_file}")
    print(f"Write : {out_file}")
    print(f"Atoms : {sum(counts)}")
    print(f"Species: {species}")
    print(f"Counts : {counts}")
    print(f"Coordinate mode in POSCAR: {pos_mode}")


if __name__ == "__main__":
    main()