from pathlib import Path

import numpy as np
from ase.build import make_supercell
from ase.io import read, write


INPUT_FILE = Path("model.xyz")
OUTPUT_FILE = Path("orthorhombic_model.xyz")


def main() -> None:
    atoms = read(INPUT_FILE)
    atoms.set_pbc((True, True, True))

    # For a hexagonal cell with a = b and gamma = 60 deg, this integer
    # transformation generates an orthorhombic supercell:
    # A = a - b, B = a + b, C = c.
    transform = np.array(
        [
            [1, -1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]
    )

    orthorhombic = make_supercell(atoms, transform)
    scaled_positions = orthorhombic.get_scaled_positions(wrap=True)
    lengths = orthorhombic.cell.lengths()

    # Rewrite the already-orthogonal cell in axis-aligned form so that
    # a, b, c lie exactly along x, y, z.
    orthorhombic.set_cell(np.diag(lengths), scale_atoms=False)
    orthorhombic.set_scaled_positions(scaled_positions)
    orthorhombic.set_pbc((True, True, True))
    orthorhombic.wrap()

    write(OUTPUT_FILE, orthorhombic, format="extxyz")

    lengths = orthorhombic.cell.lengths()
    angles = orthorhombic.cell.angles()

    print(f"Input file:  {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Atoms:       {len(atoms)} -> {len(orthorhombic)}")
    print(
        "Lengths (A): "
        f"a={lengths[0]:.8f}, b={lengths[1]:.8f}, c={lengths[2]:.8f}"
    )
    print(
        "Angles (deg): "
        f"alpha={angles[0]:.8f}, beta={angles[1]:.8f}, gamma={angles[2]:.8f}"
    )
    print("Cell vectors (A):")
    for vector in orthorhombic.cell:
        print("  " + "  ".join(f"{value: .10f}" for value in vector))


if __name__ == "__main__":
    main()
