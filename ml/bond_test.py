"""
Compute average bond length with OVITO Python.

Assumptions:
- Input is an extxyz file (e.g., model.xyz) containing Lattice and (optionally) pbc="T T F".
- Bonds are defined by a simple distance cutoff (in Å).
- Works well for large systems because OVITO builds bonds efficiently.

Install (one option):
    pip install ovito

Run:
    python avg_bond_length_ovito.py --input model.xyz --cutoff 1.85
Optional (only count C-C bonds):
    python avg_bond_length_ovito.py --input model.xyz --cutoff 1.85 --pair C C
"""

import argparse
import numpy as np

from ovito.io import import_file
from ovito.modifiers import (
    PythonScriptModifier,
    CreateBondsModifier,
    ComputePropertyModifier,
)


def avg_bond_length_ovito(path: str, cutoff: float, pair: tuple[str, str] | None = None,
                         force_pbc: tuple[bool, bool, bool] | None = (True, True, False)) -> dict:
    """
    Parameters
    ----------
    path : str
        extxyz file path
    cutoff : float
        bond cutoff distance in Å (e.g., 1.75~1.90 for sp2 C-C first neighbors)
    pair : (str, str) or None
        If provided, only count bonds whose endpoint element names match this pair (order-insensitive).
        Example: ("C","C")
    force_pbc : (bool,bool,bool) or None
        If not None, force the cell periodicity flags before creating bonds. For 2D: (True, True, False)

    Returns
    -------
    dict with keys: count, mean, std, min, max
    """

    pipeline = import_file(path)

    # (Optional but recommended) Force periodicity flags before building bonds
    if force_pbc is not None:
        def _set_pbc(frame, data):
            # data.cell is a SimulationCell
            data.cell.pbc = force_pbc

        pipeline.modifiers.append(PythonScriptModifier(function=_set_pbc))

    # 1) Create bonds by uniform distance cutoff
    pipeline.modifiers.append(CreateBondsModifier(cutoff=cutoff))

    # 2) Compute per-bond length into a bond property called "Length"
    pipeline.modifiers.append(
        ComputePropertyModifier(
            operate_on="bonds",
            output_property="Length",
            expressions=["BondLength"],  # OVITO expression for bond length
        )
    )

    data = pipeline.compute()

    bonds = data.particles.bonds
    if bonds is None or bonds.count == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    lengths = np.asarray(bonds["Length"], dtype=float)

    # Optional: filter by element pair (e.g., C-C only)
    if pair is not None:
        pair_set = {pair[0], pair[1]}

        # Bond topology: (M,2) particle indices per bond
        topo = np.asarray(bonds.topology, dtype=int)

        # Particle type ids per atom
        # This property is present if the file provides species/types (extxyz does).
        ptype_ids = np.asarray(data.particles["Particle Type"], dtype=int)

        # Map OVITO ParticleType.id -> ParticleType.name (element string)
        id_to_name = {t.id: t.name for t in data.particles.particle_types}

        mask = np.zeros(len(lengths), dtype=bool)
        for k, (i, j) in enumerate(topo):
            a = id_to_name.get(int(ptype_ids[i]), None)
            b = id_to_name.get(int(ptype_ids[j]), None)
            if a is None or b is None:
                continue
            if {a, b} == pair_set:
                mask[k] = True

        lengths = lengths[mask]

    if len(lengths) == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    return {
        "count": int(len(lengths)),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "min": float(lengths.min()),
        "max": float(lengths.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to extxyz file, e.g., model.xyz")
    ap.add_argument("--cutoff", type=float, default=1.85, help="Bond cutoff in Å (default: 1.85)")
    ap.add_argument("--pair", nargs=2, metavar=("E1", "E2"),
                    help="Only count bonds between E1 and E2, e.g., --pair C C")
    ap.add_argument("--no-force-pbc", action="store_true",
                    help="Do not force PBC flags; rely on file metadata.")
    args = ap.parse_args()

    force_pbc = None if args.no_force_pbc else (True, True, False)
    pair = tuple(args.pair) if args.pair else None

    stats = avg_bond_length_ovito(args.input, args.cutoff, pair=pair, force_pbc=force_pbc)

    print(f"Input: {args.input}")
    print(f"Cutoff (Å): {args.cutoff}")
    if pair:
        print(f"Pair filter: {pair[0]}-{pair[1]}")
    print(f"Bond count: {stats['count']}")
    print(f"Avg bond length (Å): {stats['mean']}")
    print(f"Std (Å): {stats['std']}")
    print(f"Min/Max (Å): {stats['min']} / {stats['max']}")

    # Quick sanity hint for sp2 carbon:
    # If max approaches ~2.4-2.5 Å, your cutoff may be including 2nd neighbors.
    if np.isfinite(stats["max"]) and stats["max"] > 2.2:
        print("Note: max bond length is relatively large; consider reducing cutoff if you only want 1st neighbors.")


if __name__ == "__main__":
    main()
