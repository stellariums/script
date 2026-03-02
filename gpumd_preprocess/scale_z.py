from ase.io import read, write
import numpy as np
import argparse


def scale_z(input_path: str, output_path: str, new_c: float) -> None:
    atoms = read(input_path)

    cell = atoms.cell.array.copy()
    old_c = cell[2, 2]
    if abs(old_c) < 1e-12:
        old_c = atoms.cell.lengths()[2]

    scale = new_c / old_c

    cell[2, :] *= scale
    atoms.set_cell(cell, scale_atoms=False)

    pos = atoms.get_positions()
    pos[:, 2] *= scale
    atoms.set_positions(pos)

    atoms.pbc = (bool(atoms.pbc[0]), bool(atoms.pbc[1]), False)

    write(output_path, atoms, format="extxyz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--new-c", type=float, default=3.38)
    args = parser.parse_args()
    scale_z(args.input, args.output, args.new_c)


if __name__ == "__main__":
    main()