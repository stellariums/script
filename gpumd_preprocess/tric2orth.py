"""
用途
----
把六方（或一般的非正交）晶胞通过 ASE 的超胞变换转成正交晶胞（alpha=beta=gamma=90°），
同时保持原子之间的相对几何关系不变（只做整数晶胞变换 + 刚体旋转，不做形变/松弛）。

说明
----
- 该转换通常会把原始晶胞扩成一个等价的正交超胞，因此原子数可能会增加（例如六方常见会变为 2 倍）。
- 适合后续需要正交晶胞的可视化、网格划分、某些程序输入等场景。

用法
----
在脚本所在目录运行：

    python .\\tric2orth.py -i .\\COF.xyz -o .\\COF_orth.xyz

参数
----
-i/--input   输入结构文件（默认 COF.xyz）
-o/--output  输出结构文件（默认 COF_orth.xyz）
--no-align-axes         不把晶胞旋转对齐到 x/y/z 轴
--no-orthogonalize-xy   只做超胞变换，不强制输出为对角形式晶胞
"""

import argparse
from pathlib import Path

import numpy as np
from ase.build import make_supercell
from ase.io import read, write


def _cellpar(atoms):
    a, b, c = atoms.cell.lengths()
    alpha, beta, gamma = atoms.cell.angles()
    return float(a), float(b), float(c), float(alpha), float(beta), float(gamma)


def _rotation_matrix_from_cell_rows(v1, v2, v3):
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    u3 = np.cross(u1, u2)
    u3 = u3 / np.linalg.norm(u3)

    if np.dot(u3, v3) < 0.0:
        u2 = -u2
        u3 = -u3

    return np.vstack([u1, u2, u3])


def hex2orth(atoms, orthogonalize_xy=True, align_axes=True):
    p = np.array(
        [
            [1, 1, 0],
            [1, -1, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )

    sc = make_supercell(atoms, p)
    sc.pbc = atoms.pbc

    if not (orthogonalize_xy or align_axes):
        sc.wrap()
        return sc

    cell = sc.cell.array
    r = _rotation_matrix_from_cell_rows(cell[0], cell[1], cell[2])

    sc.positions = sc.positions @ r.T
    sc.set_cell(cell @ r.T, scale_atoms=False)

    if orthogonalize_xy:
        lengths = np.linalg.norm(sc.cell.array, axis=1)
        sc.set_cell(np.diag(lengths), scale_atoms=False)

    sc.wrap()
    return sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default=str(Path(__file__).with_name("COF.xyz")),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(Path(__file__).with_name("COF_orth.xyz")),
    )
    parser.add_argument(
        "--no-align-axes",
        action="store_true",
        help="Keep orthogonal vectors but do not rotate to x/y/z axes.",
    )
    parser.add_argument(
        "--no-orthogonalize-xy",
        action="store_true",
        help="Only build the supercell; do not enforce a diagonal cell matrix.",
    )
    args = parser.parse_args()

    atoms = read(args.input)
    a, b, c, alpha, beta, gamma = _cellpar(atoms)
    print(f"Input:  a={a:.6f} b={b:.6f} c={c:.6f}  alpha={alpha:.6f} beta={beta:.6f} gamma={gamma:.6f}")

    out = hex2orth(
        atoms,
        orthogonalize_xy=not args.no_orthogonalize_xy,
        align_axes=not args.no_align_axes,
    )
    a, b, c, alpha, beta, gamma = _cellpar(out)
    print(f"Output: a={a:.6f} b={b:.6f} c={c:.6f}  alpha={alpha:.6f} beta={beta:.6f} gamma={gamma:.6f}")

    write(args.output, out)
    print(f"Wrote:  {args.output}")


if __name__ == "__main__":
    main()
