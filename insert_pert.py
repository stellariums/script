from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def is_main_section_header(line: str) -> bool:
    s = line.strip()
    return bool(s) and s.isupper() and " " not in s


def flip_magnetic_moment(line: str) -> str:
    parts = line.split()
    if not parts:
        raise ValueError("empty atom line")
    if "mag" in parts:
        i = parts.index("mag") + 1
        for j in range(i, min(i + 3, len(parts))):
            parts[j] = f"{-float(parts[j]):.10f}"
    return " ".join(parts)


def ensure_sc_suffix(line: str) -> str:
    parts = line.split()
    if not parts:
        return line
    if "sc" in parts:
        return " ".join(parts)
    return " ".join(parts + ["sc", "1", "1", "1"])


def insert_cr_in_one_file(path: Path, rng: random.Random) -> tuple[str, bool]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if "ATOMIC_POSITIONS" not in lines:
        raise ValueError(f"{path.name}: missing ATOMIC_POSITIONS")
    pos_idx = lines.index("ATOMIC_POSITIONS")
    cursor = pos_idx + 1
    while cursor < len(lines) and lines[cursor].strip() == "":
        cursor += 1
    if cursor >= len(lines):
        raise ValueError(f"{path.name}: invalid ATOMIC_POSITIONS block")
    coord_type_idx = cursor
    cursor += 1
    species_blocks: list[dict[str, Any]] = []
    while cursor < len(lines):
        while cursor < len(lines) and lines[cursor].strip() == "":
            cursor += 1
        if cursor >= len(lines):
            break
        if is_main_section_header(lines[cursor]):
            break
        specie = lines[cursor].strip()
        if cursor + 2 >= len(lines):
            raise ValueError(f"{path.name}: incomplete species block")
        param_idx = cursor + 1
        count_idx = cursor + 2
        count = int(lines[count_idx].strip())
        atom_start = cursor + 3
        atom_end = atom_start + count
        if atom_end > len(lines):
            raise ValueError(f"{path.name}: atom count overflow")
        atom_lines = [ensure_sc_suffix(x) for x in lines[atom_start:atom_end]]
        species_blocks.append(
            {
                "specie": specie,
                "param_idx": param_idx,
                "count": count,
                "atom_lines": atom_lines,
            }
        )
        cursor = atom_end
    fe_block = next((b for b in species_blocks if b["specie"] == "Fe"), None)
    if fe_block is None or fe_block["count"] < 1:
        return path.read_text(encoding="utf-8"), False
    pick = rng.randrange(fe_block["count"])
    picked_line = fe_block["atom_lines"][pick]
    new_cr_line = ensure_sc_suffix(flip_magnetic_moment(picked_line))
    new_fe_lines = list(fe_block["atom_lines"])
    del new_fe_lines[pick]
    rebuilt = lines[: pos_idx + 1]
    rebuilt.append(lines[coord_type_idx])
    rebuilt.append("")
    for block in species_blocks:
        rebuilt.append(block["specie"])
        if block["specie"] == "Fe":
            rebuilt.append(lines[block["param_idx"]])
            rebuilt.append(str(block["count"] - 1))
            rebuilt.extend(new_fe_lines)
        elif block["specie"] == "Cr":
            rebuilt.append("0")
            rebuilt.append("1")
            rebuilt.append(new_cr_line)
        else:
            rebuilt.append(lines[block["param_idx"]])
            rebuilt.append(str(block["count"]))
            rebuilt.extend(block["atom_lines"])
        rebuilt.append("")
    if not any(b["specie"] == "Cr" for b in species_blocks):
        rebuilt.append("Cr")
        rebuilt.append("0")
        rebuilt.append("1")
        rebuilt.append(new_cr_line)
        rebuilt.append("")
    if cursor < len(lines):
        rebuilt.extend(lines[cursor:])
    return "\n".join(rebuilt).rstrip() + "\n", True


def insert_random_cr(input_dir: Path, output_dir: Path, seed: int | None = None) -> Path:
    rng = random.Random(seed)
    files = sorted(input_dir.glob("STRU-*"))
    if not files:
        raise ValueError("no STRU-* files found")
    output_dir.mkdir(parents=True, exist_ok=True)
    pick_order = files[:]
    rng.shuffle(pick_order)
    for f in pick_order:
        new_text, did_change = insert_cr_in_one_file(f, rng)
        if did_change:
            out = output_dir / f.name
            out.write_text(new_text, encoding="utf-8")
            return out
    raise ValueError("no structure with Fe atoms found")


def load_perturb_config(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    cfg = data["prepare"]["pert_stru"]
    required = ["pert_number", "cell_pert_frac", "atom_pert_dist", "mag_rotate_angle", "mag_tilt_angle", "mag_norm_dist"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing keys in perturb config: {missing}")
    return cfg


def parse_atom_line(line: str, coord_type: str) -> dict[str, Any]:
    parts = line.split()
    if len(parts) < 3:
        raise ValueError(f"Atom line too short: {line.rstrip()}")
    coord = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
    mag_index = next((i for i, t in enumerate(parts) if t.lower() == "mag"), None)
    if mag_index is None or mag_index + 3 >= len(parts):
        raise ValueError(f"Invalid mag in atom line: {line.rstrip()}")
    return {
        "coord": coord,
        "middle_tokens": parts[3:mag_index],
        "mag_vec": np.array([float(parts[mag_index + 1]), float(parts[mag_index + 2]), float(parts[mag_index + 3])], dtype=float),
        "tail_tokens": parts[mag_index + 4:],
        "coord_type": coord_type,
    }


def parse_stru_for_perturb(stru_path: Path) -> dict[str, Any]:
    lines = stru_path.read_text(encoding="utf-8").splitlines(keepends=True)
    lc_line = lv_line = ap_line = None
    for i, line in enumerate(lines):
        tag = line.strip().upper()
        if tag == "LATTICE_CONSTANT":
            lc_line = i
        elif tag == "LATTICE_VECTORS":
            lv_line = i
        elif tag == "ATOMIC_POSITIONS":
            ap_line = i
    if lc_line is None or lv_line is None or ap_line is None:
        raise ValueError("STRU missing required sections")
    lattice = np.array([[float(x) for x in lines[lv_line + 1 + r].split()[:3]] for r in range(3)], dtype=float)
    coord_type_line = ap_line + 1
    coord_type = lines[coord_type_line].split("#")[0].strip().lower()
    if coord_type not in ("cartesian", "direct"):
        raise ValueError(f"Unsupported coordinate type: {lines[coord_type_line].rstrip()}")
    species_blocks: list[dict[str, Any]] = []
    i = coord_type_line + 2
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        name = lines[i].strip()
        if i + 2 >= len(lines):
            raise ValueError("Incomplete species block")
        natom = int(lines[i + 2].split()[0])
        atom_start = i + 3
        atom_end = atom_start + natom
        atoms = [parse_atom_line(lines[j], coord_type) for j in range(atom_start, atom_end)]
        species_blocks.append({"atom_start": atom_start, "atoms": atoms, "name": name})
        i = atom_end
    return {"lines": lines, "lattice": lattice, "lattice_vectors_line": lv_line, "coord_type": coord_type, "species_blocks": species_blocks}


def unit_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def perturb_magnetic_moment(mag_vec: np.ndarray, rotate_angle_deg: float, tilt_angle_deg: float, norm_dist: float, rng: np.random.Generator) -> np.ndarray:
    mag_norm = np.linalg.norm(mag_vec)
    if mag_norm < 1e-15:
        return mag_vec.copy()
    axis = unit_vector(mag_vec)
    ref = np.array([1.0, 0.0, 0.0], dtype=float) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = unit_vector(np.cross(axis, ref))
    e2 = unit_vector(np.cross(axis, e1))
    phi = rng.uniform(0.0, math.radians(max(0.0, rotate_angle_deg)))
    theta = rng.uniform(0.0, math.radians(max(0.0, tilt_angle_deg)))
    transverse_dir = math.cos(phi) * e1 + math.sin(phi) * e2
    new_dir = unit_vector(math.cos(theta) * axis + math.sin(theta) * transverse_dir)
    new_norm = max(0.0, mag_norm + rng.uniform(-norm_dist, norm_dist))
    return new_norm * new_dir


def perturb_position(coord: np.ndarray, coord_type: str, lattice: np.ndarray, atom_pert_dist: float, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    disp = np.zeros(3) if n < 1e-15 else (v / n) * rng.uniform(0.0, atom_pert_dist)
    if coord_type == "cartesian":
        return coord + disp
    cart = coord.reshape(1, 3) @ lattice
    cart_new = cart.reshape(3) + disp
    frac = cart_new.reshape(1, 3) @ np.linalg.inv(lattice)
    return np.mod(frac.reshape(3), 1.0)


def format_atom_line(atom: dict[str, Any]) -> str:
    xyz = atom["coord"]
    mag = atom["mag_vec"]
    tokens = [f"{xyz[0]: .12f}", f"{xyz[1]: .12f}", f"{xyz[2]: .12f}", *atom["middle_tokens"], "mag", f"{mag[0]: .12f}", f"{mag[1]: .12f}", f"{mag[2]: .12f}", *atom["tail_tokens"]]
    return " ".join(tokens) + "\n"


def generate_perturbed_structures(stru_path: Path, json_path: Path, out_dir: Path, seed: int | None = None, prefix: str = "STRU_") -> list[Path]:
    cfg = load_perturb_config(json_path)
    parsed = parse_stru_for_perturb(stru_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    new_lines_base = parsed["lines"][:]
    new_lattice = (np.eye(3) + rng.uniform(-float(cfg["cell_pert_frac"]), float(cfg["cell_pert_frac"]), size=(3, 3))) @ parsed["lattice"]
    lv_line = parsed["lattice_vectors_line"]
    for i in range(3):
        new_lines_base[lv_line + 1 + i] = f"{new_lattice[i, 0]: .15f} {new_lattice[i, 1]: .15f} {new_lattice[i, 2]: .15f}\n"
    outputs: list[Path] = []
    for idx in range(int(cfg["pert_number"])):
        lines = new_lines_base[:]
        for block in parsed["species_blocks"]:
            for offset, atom in enumerate(block["atoms"]):
                atom_new = {
                    "coord": perturb_position(atom["coord"], parsed["coord_type"], new_lattice, float(cfg["atom_pert_dist"]), rng),
                    "middle_tokens": atom["middle_tokens"][:],
                    "mag_vec": perturb_magnetic_moment(atom["mag_vec"], float(cfg["mag_rotate_angle"]), float(cfg["mag_tilt_angle"]), float(cfg["mag_norm_dist"]), rng),
                    "tail_tokens": atom["tail_tokens"][:],
                }
                lines[block["atom_start"] + offset] = format_atom_line(atom_new)
        out_path = out_dir / f"{prefix}{idx:04d}"
        out_path.write_text("".join(lines), encoding="utf-8")
        outputs.append(out_path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_insert = sub.add_parser("insert")
    p_insert.add_argument("--input-dir", type=Path, default=Path.cwd())
    p_insert.add_argument("--output-dir", type=Path, default=Path.cwd() / "random_cr_inserted")
    p_insert.add_argument("--seed", type=int, default=None)

    p_pert = sub.add_parser("perturb")
    p_pert.add_argument("--stru", type=Path, required=True)
    p_pert.add_argument("--config", type=Path, required=True)
    p_pert.add_argument("--out-dir", type=Path, required=True)
    p_pert.add_argument("--seed", type=int, default=None)
    p_pert.add_argument("--prefix", default="STRU_")

    p_both = sub.add_parser("both")
    p_both.add_argument("--input-dir", type=Path, default=Path.cwd())
    p_both.add_argument("--insert-out", type=Path, default=Path.cwd() / "random_cr_inserted")
    p_both.add_argument("--config", type=Path, required=True)
    p_both.add_argument("--pert-out", type=Path, required=True)
    p_both.add_argument("--seed", type=int, default=None)
    p_both.add_argument("--prefix", default="STRU_")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "insert":
        selected = insert_random_cr(args.input_dir.resolve(), args.output_dir.resolve(), args.seed)
        print(f"insert_done selected={selected.name} output={selected.parent}")
        return
    if args.cmd == "perturb":
        outputs = generate_perturbed_structures(args.stru.resolve(), args.config.resolve(), args.out_dir.resolve(), args.seed, args.prefix)
        print(f"perturb_done count={len(outputs)} output={args.out_dir.resolve()}")
        return
    selected = insert_random_cr(args.input_dir.resolve(), args.insert_out.resolve(), args.seed)
    outputs = generate_perturbed_structures(selected, args.config.resolve(), args.pert_out.resolve(), args.seed, args.prefix)
    print(f"both_done selected={selected.name} perturbed_count={len(outputs)} insert_output={selected.parent} perturb_output={args.pert_out.resolve()}")


if __name__ == "__main__":
    main()
