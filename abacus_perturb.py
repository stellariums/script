#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


# -----------------------------
# JSON config
# -----------------------------
def load_perturb_config(json_path: str | Path) -> Dict[str, Any]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    try:
        cfg = data['prepare']['pert_stru']
    except KeyError as e:
        raise KeyError(f'Cannot find prepare/pert_stru in {json_path}') from e

    required = [
        'pert_number',
        'cell_pert_frac',
        'atom_pert_dist',
        'mag_rotate_angle',
        'mag_tilt_angle',
        'mag_norm_dist',
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f'Missing keys in perturb config: {missing}')
    return cfg


# -----------------------------
# STRU parser for the uploaded format
# -----------------------------
def parse_stru(stru_path: str | Path) -> Dict[str, Any]:
    with open(stru_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lattice_constant_line = None
    lattice_vectors_line = None
    atomic_positions_line = None

    for i, line in enumerate(lines):
        tag = line.strip().upper()
        if tag == 'LATTICE_CONSTANT':
            lattice_constant_line = i
        elif tag == 'LATTICE_VECTORS':
            lattice_vectors_line = i
        elif tag == 'ATOMIC_POSITIONS':
            atomic_positions_line = i

    if lattice_constant_line is None:
        raise ValueError('STRU missing LATTICE_CONSTANT')
    if lattice_vectors_line is None:
        raise ValueError('STRU missing LATTICE_VECTORS')
    if atomic_positions_line is None:
        raise ValueError('STRU missing ATOMIC_POSITIONS')

    lattice_constant = float(lines[lattice_constant_line + 1].split()[0])

    lattice = np.array(
        [[float(x) for x in lines[lattice_vectors_line + 1 + r].split()[:3]] for r in range(3)],
        dtype=float,
    )

    coord_type_line = atomic_positions_line + 1
    coord_type_raw = lines[coord_type_line].strip()
    coord_type = coord_type_raw.split('#')[0].strip().lower()
    if coord_type not in ('cartesian', 'direct'):
        raise ValueError(f'Unsupported coordinate type: {coord_type_raw}')

    species_blocks = []
    i = coord_type_line + 2
    n = len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue

        species_name = lines[i].strip()
        if i + 2 >= n:
            raise ValueError(f'Incomplete species block starting at line {i + 1}')

        species_param_line = i + 1
        natom_line = i + 2
        try:
            natom = int(lines[natom_line].split()[0])
        except Exception as e:
            raise ValueError(f'Cannot parse atom count for species block {species_name!r} at line {natom_line + 1}') from e

        atom_start = i + 3
        atom_end = atom_start + natom
        if atom_end > n:
            raise ValueError(f'Species block {species_name!r} atom list exceeds file length')

        atoms = [parse_atom_line(lines[j], coord_type) for j in range(atom_start, atom_end)]
        species_blocks.append(
            {
                'species_name': species_name,
                'species_name_line': i,
                'species_param_line': species_param_line,
                'natom_line': natom_line,
                'atom_start': atom_start,
                'atom_end': atom_end,
                'atoms': atoms,
            }
        )
        i = atom_end

    return {
        'lines': lines,
        'lattice_constant': lattice_constant,
        'lattice_constant_line': lattice_constant_line,
        'lattice_vectors_line': lattice_vectors_line,
        'lattice': lattice,
        'atomic_positions_line': atomic_positions_line,
        'coord_type_line': coord_type_line,
        'coord_type': coord_type,
        'species_blocks': species_blocks,
    }


def parse_atom_line(line: str, coord_type: str) -> Dict[str, Any]:
    parts = line.split()
    if len(parts) < 3:
        raise ValueError(f'Atom line too short: {line.rstrip()}')

    coord = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)

    mag_index = None
    for idx, token in enumerate(parts):
        if token.lower() == 'mag':
            mag_index = idx
            break

    if mag_index is None:
        raise ValueError(f'Cannot find mag in atom line: {line.rstrip()}')
    if mag_index + 3 >= len(parts):
        raise ValueError(f'Incomplete magnetic moment after mag in line: {line.rstrip()}')

    middle_tokens = parts[3:mag_index]
    mag_vec = np.array([float(parts[mag_index + 1]), float(parts[mag_index + 2]), float(parts[mag_index + 3])], dtype=float)
    tail_tokens = parts[mag_index + 4:]

    return {
        'coord': coord,
        'middle_tokens': middle_tokens,
        'mag_vec': mag_vec,
        'tail_tokens': tail_tokens,
        'coord_type': coord_type,
    }


def format_atom_line(atom: Dict[str, Any]) -> str:
    xyz = atom['coord']
    middle = atom['middle_tokens']
    mag = atom['mag_vec']
    tail = atom['tail_tokens']

    tokens = [
        f'{xyz[0]: .12f}',
        f'{xyz[1]: .12f}',
        f'{xyz[2]: .12f}',
        *middle,
        'mag',
        f'{mag[0]: .12f}',
        f'{mag[1]: .12f}',
        f'{mag[2]: .12f}',
        *tail,
    ]
    return ' '.join(tokens) + '\n'


# -----------------------------
# Perturbations
# -----------------------------
def perturb_cell(lattice: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    eps = rng.uniform(-frac, frac, size=(3, 3))
    return (np.eye(3) + eps) @ lattice


def direct_to_cart(frac_coords: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return frac_coords @ lattice


def cart_to_direct(cart_coords: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return cart_coords @ np.linalg.inv(lattice)


def random_displacement_cart(max_dist: float, rng: np.random.Generator) -> np.ndarray:
    # Use isotropic random direction and radius ~ U(0, max_dist)
    v = rng.normal(size=3)
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return np.zeros(3)
    direction = v / norm
    radius = rng.uniform(0.0, max_dist)
    return direction * radius


def perturb_position(coord: np.ndarray, coord_type: str, lattice_for_conversion: np.ndarray,
                     atom_pert_dist: float, rng: np.random.Generator) -> np.ndarray:
    disp_cart = random_displacement_cart(atom_pert_dist, rng)
    if coord_type == 'cartesian':
        return coord + disp_cart
    if coord_type == 'direct':
        cart = direct_to_cart(coord.reshape(1, 3), lattice_for_conversion).reshape(3)
        cart_new = cart + disp_cart
        frac_new = cart_to_direct(cart_new.reshape(1, 3), lattice_for_conversion).reshape(3)
        return np.mod(frac_new, 1.0)
    raise ValueError(f'Unsupported coordinate type: {coord_type}')


def unit_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def orthonormal_basis_from_axis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis = unit_vector(axis)
    if abs(axis[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = np.cross(axis, ref)
    e1 = unit_vector(e1)
    e2 = np.cross(axis, e1)
    e2 = unit_vector(e2)
    return e1, e2


def perturb_magnetic_moment(mag_vec: np.ndarray, rotate_angle_deg: float, tilt_angle_deg: float,
                            norm_dist: float, rng: np.random.Generator) -> np.ndarray:
    mag_norm = np.linalg.norm(mag_vec)
    if mag_norm < 1e-15:
        return mag_vec.copy()

    axis = unit_vector(mag_vec)
    e1, e2 = orthonormal_basis_from_axis(axis)

    phi_max = math.radians(max(0.0, rotate_angle_deg))
    phi = rng.uniform(0.0, phi_max if phi_max > 0 else 0.0)

    theta_max = math.radians(max(0.0, tilt_angle_deg))
    theta = rng.uniform(0.0, theta_max if theta_max > 0 else 0.0)

    transverse_dir = math.cos(phi) * e1 + math.sin(phi) * e2
    new_dir = math.cos(theta) * axis + math.sin(theta) * transverse_dir
    new_dir = unit_vector(new_dir)

    new_norm = mag_norm + rng.uniform(-norm_dist, norm_dist)
    if new_norm < 0.0:
        new_norm = 0.0

    return new_norm * new_dir


# -----------------------------
# File generation
# -----------------------------
def make_single_perturbed_lines(parsed: Dict[str, Any], cfg: Dict[str, Any], rng: np.random.Generator) -> List[str]:
    new_lines = parsed['lines'][:]
    old_lattice = parsed['lattice']
    new_lattice = perturb_cell(old_lattice, float(cfg['cell_pert_frac']), rng)

    lv_line = parsed['lattice_vectors_line']
    for i in range(3):
        new_lines[lv_line + 1 + i] = (
            f'{new_lattice[i, 0]: .15f} {new_lattice[i, 1]: .15f} {new_lattice[i, 2]: .15f}\n'
        )

    coord_type = parsed['coord_type']
    for block in parsed['species_blocks']:
        for offset, atom in enumerate(block['atoms']):
            atom_new = {
                'coord': perturb_position(
                    atom['coord'],
                    coord_type,
                    new_lattice,
                    float(cfg['atom_pert_dist']),
                    rng,
                ),
                'middle_tokens': atom['middle_tokens'][:],
                'mag_vec': perturb_magnetic_moment(
                    atom['mag_vec'],
                    float(cfg['mag_rotate_angle']),
                    float(cfg['mag_tilt_angle']),
                    float(cfg['mag_norm_dist']),
                    rng,
                ),
                'tail_tokens': atom['tail_tokens'][:],
                'coord_type': coord_type,
            }
            line_idx = block['atom_start'] + offset
            new_lines[line_idx] = format_atom_line(atom_new)

    return new_lines


def generate_perturbed_structures(stru_path: str | Path, json_path: str | Path, out_dir: str | Path,
                                  seed: int | None = None, prefix: str = 'STRU_') -> None:
    cfg = load_perturb_config(json_path)
    parsed = parse_stru(stru_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    pert_number = int(cfg['pert_number'])

    for idx in range(pert_number):
        lines = make_single_perturbed_lines(parsed, cfg, rng)
        out_path = out_dir / f'{prefix}{idx:04d}'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Apply DP-GEN-style structure and magnetic perturbations to an ABACUS STRU file.'
    )
    p.add_argument('--stru', required=True, help='Path to input ABACUS STRU file')
    p.add_argument('--config', required=True, help='Path to perturbation JSON file')
    p.add_argument('--out-dir', default='perturbed_stru', help='Output directory')
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    p.add_argument('--prefix', default='STRU_', help='Output filename prefix')
    return p


def main() -> None:
    args = build_argparser().parse_args()
    generate_perturbed_structures(
        stru_path=args.stru,
        json_path=args.config,
        out_dir=args.out_dir,
        seed=args.seed,
        prefix=args.prefix,
    )
    print(f'Done. Perturbed structures written to: {args.out_dir}')


if __name__ == '__main__':
    main()
