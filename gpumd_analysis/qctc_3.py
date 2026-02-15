# -*- coding: utf-8 -*-
"""
用途
- 批量读取多个 fold（默认 fold1/fold2/fold3）中的 HNEMD 输出文件，分别计算并汇总平均热导率：
  - k：来自 kappa.out 的时间累计平均最终值（final_k）
  - k_spec：来自 shc.out 的频谱热导率积分
  - k_spec_quantum：对频谱热导率做量子修正后的积分
- 最终输出：每个 fold 的结果 + 多个 fold 的平均值（mean）与样本标准差（std, ddof=1）。

输入文件（每个 fold 目录必须包含）
- kappa.out：用于计算 k（累计平均）
- shc.out：用于计算 k_spec 与 k_spec_quantum（按本脚本固定格式读取）
- run.in：用于读取外场驱动力 Fe（从包含 compute_hnemd 的那一行取“最后三个数字”Fx,Fy,Fz）
- dump.xyz：用于读取晶胞参数并计算体积 V（从第 2 行的 Lattice="..." 解析 3x3 晶胞矩阵）

用法
- 在本脚本所在目录（COF/compute/COF）下运行：
  - 计算 x 方向（which=0），温度 300 K（默认）：
    python qctc_3.py --which 0 --T 300
  - 指定要统计的 fold 目录（可写多个）：
    python qctc_3.py --folds fold1 fold2 fold3 --which 0 --T 300

参数说明
- --which: 0/1/2 分别对应 x/y/z 方向
  - 影响 kappa.out 的列选择（与原 qctc.py 一致）
  - 同时决定从 run.in 读取 Fx/Fy/Fz 中的哪一个作为 Fe
- --T: 温度（K），默认 300
- --folds: fold 目录列表，默认 fold1 fold2 fold3（相对于脚本所在目录）

注意事项
- 体积 V 的读取以 dump.xyz 为准：本脚本从 dump.xyz 第 2 行的 Lattice="a b c d e f g h i" 构造 3x3 晶胞矩阵，
  用 det(cell) 计算体积。若晶胞存在剪切（非正交），Lx/Ly/Lz 打印值为三条晶格矢量的范数，仅用于展示。
- Fe 由 run.in 的 compute_hnemd 行末 3 个数字读取；如果你选择的方向对应 Fe=0，会导致频谱热导率计算除以 0 而报错。
  请确保对应方向的驱动力分量非零，或改用正确的 --which。
- shc.out 的读取假设与原脚本一致：前 (2*num_corr_points-1) 行是相关函数段，后面是频域段；默认 num_corr_points=250。
  如果你的 shc.out 格式不同，需要相应修改 compute_spectral_integrals 的读取逻辑。
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_NUMBER_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


def read_Fe_from_runin(direction_index: int, runin_path: Path) -> float:
    if direction_index not in (0, 1, 2):
        raise ValueError("direction_index 必须是 0(x), 1(y), 或 2(z)")

    if not runin_path.exists():
        raise FileNotFoundError(f"Error: '{runin_path}' not found in {runin_path.parent}")

    matches: List[Tuple[int, List[float], str]] = []
    with open(runin_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            line_wo_comment = re.split(r"[#;!]", line, maxsplit=1)[0].strip()
            if not line_wo_comment:
                continue
            if re.search(r"\bcompute_hnemd\b", line_wo_comment):
                nums = re.findall(_NUMBER_RE, line_wo_comment)
                if len(nums) < 3:
                    raise ValueError(
                        f"Found 'compute_hnemd' at line {ln} but fewer than 3 numbers:\n{line.strip()}"
                    )
                last_three = list(map(float, nums[-3:]))
                matches.append((ln, last_three, line.strip()))

    if not matches:
        raise ValueError("Error: No line containing 'compute_hnemd' was found in run.in")

    ln, last_three, _raw = matches[0]
    Fe = last_three[direction_index]
    print(f"[{runin_path.parent.name}] Fe from run.in line {ln}: last_three={last_three}, Fe={Fe}")
    return Fe


def calculate_volume_from_dumpxyz(dump_path: Path) -> float:
    with open(dump_path, "r", encoding="utf-8", errors="ignore") as f:
        _line1 = f.readline()
        line2 = f.readline()

    if not line2:
        raise ValueError("dump.xyz missing the second line with Lattice.")

    m = re.search(r'Lattice="([^"]+)"', line2)
    if not m:
        raise ValueError(f"dump.xyz second line has no Lattice field: {line2.strip()}")

    lattice_str = m.group(1).strip()
    parts = lattice_str.split()
    if len(parts) != 9:
        raise ValueError(f"Lattice expects 9 numbers, got {len(parts)}: {lattice_str}")

    cell = np.array(list(map(float, parts)), dtype=float).reshape(3, 3)
    a, b, c = cell[0], cell[1], cell[2]
    Lx = float(np.linalg.norm(a))
    Ly = float(np.linalg.norm(b))
    Lz = float(np.linalg.norm(c))
    V = float(abs(np.linalg.det(cell)))
    print(f"[{dump_path.parent.name}] V from dump.xyz: Lx={Lx}, Ly={Ly}, Lz={Lz}, V={V}")
    return V


def compute_k_from_kappa(kappa_path: Path, which_position: int) -> float:
    kappa = np.loadtxt(kappa_path)
    M = kappa.shape[0]
    if which_position == 0:
        ki_ave = np.cumsum(kappa[:, 0]) / np.arange(1, M + 1)
        ko_ave = np.cumsum(kappa[:, 1]) / np.arange(1, M + 1)
        k = ki_ave + ko_ave
    elif which_position == 1:
        ki_ave = np.cumsum(kappa[:, 2]) / np.arange(1, M + 1)
        ko_ave = np.cumsum(kappa[:, 3]) / np.arange(1, M + 1)
        k = ki_ave + ko_ave
    elif which_position == 2:
        k = np.cumsum(kappa[:, 4]) / np.arange(1, M + 1)
    else:
        raise ValueError("which_position 必须是 0/1/2")
    return float(k[-1])


def calc_spectral_kappa(shc: Dict[str, np.ndarray], force_parameter: float, temperature: float, volume: float) -> None:
    convert = 1602.17662
    shc["kwi"] = shc["jwi"] * convert / (force_parameter * temperature * volume)
    shc["kwo"] = shc["jwo"] * convert / (force_parameter * temperature * volume)


def compute_spectral_integrals(
    shc_path: Path, Fe: float, T: float, V: float, num_corr_points: int = 250
) -> Tuple[float, float]:
    num_corr_points_in_run = num_corr_points * 2 - 1

    coor_array = np.loadtxt(shc_path, max_rows=num_corr_points_in_run)
    omega_array = np.loadtxt(shc_path, skiprows=num_corr_points_in_run)

    labels_corr = ["t", "Ki", "Ko"]
    labels_omega = ["omega", "jwi", "jwo"]

    shc: Dict[str, np.ndarray] = {}
    for label_num, key in enumerate(labels_corr):
        shc[key] = coor_array[:, label_num]
    for label_num, key in enumerate(labels_omega):
        shc[key] = omega_array[:, label_num]

    shc["nu"] = shc["omega"] / (2 * np.pi)
    calc_spectral_kappa(shc, force_parameter=Fe, temperature=T, volume=V)
    shc["kw"] = shc["kwi"] + shc["kwo"]

    k_spec = float(np.trapz(shc["kw"], shc["nu"]))

    h = 1.054e-34
    kb = 1.38e-23
    x__ = h * shc["omega"] / (kb * T) * 1e12
    quantum_factor = x__**2 * np.exp(x__) / ((np.exp(x__) - 1) ** 2)
    quantum_spectral_kappa = shc["kw"] * quantum_factor
    k_spec_quantum = float(np.trapz(quantum_spectral_kappa, shc["nu"]))
    return k_spec, k_spec_quantum


def compute_for_fold(fold_dir: Path, which_position: int, T: float) -> Dict[str, float]:
    kappa_path = fold_dir / "kappa.out"
    dump_path = fold_dir / "dump.xyz"
    runin_path = fold_dir / "run.in"
    shc_path = fold_dir / "shc.out"

    if not kappa_path.exists():
        raise FileNotFoundError(f"Missing {kappa_path}")
    if not dump_path.exists():
        raise FileNotFoundError(f"Missing {dump_path}")
    if not runin_path.exists():
        raise FileNotFoundError(f"Missing {runin_path}")
    if not shc_path.exists():
        raise FileNotFoundError(f"Missing {shc_path}")

    k = compute_k_from_kappa(kappa_path, which_position=which_position)
    V = calculate_volume_from_dumpxyz(dump_path)
    Fe = read_Fe_from_runin(which_position, runin_path=runin_path)
    k_spec, k_spec_quantum = compute_spectral_integrals(shc_path, Fe=Fe, T=T, V=V)

    return {"k": k, "k_spec": k_spec, "k_spec_quantum": k_spec_quantum}


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return mean, std


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--T", type=float, default=300.0)
    parser.add_argument(
        "--folds",
        nargs="*",
        default=["fold1", "fold2", "fold3"],
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    fold_dirs = [base_dir / name for name in args.folds]

    results: List[Tuple[str, Dict[str, float]]] = []
    for d in fold_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Fold directory not found: {d}")
        res = compute_for_fold(d, which_position=args.which, T=args.T)
        results.append((d.name, res))
        print(f"[{d.name}] k={res['k']}, k_spec={res['k_spec']}, k_spec_quantum={res['k_spec_quantum']}")

    ks = [r["k"] for _name, r in results]
    k_specs = [r["k_spec"] for _name, r in results]
    k_specs_q = [r["k_spec_quantum"] for _name, r in results]

    k_mean, k_std = mean_std(ks)
    ks_mean, ks_std = mean_std(k_specs)
    kq_mean, kq_std = mean_std(k_specs_q)

    print("")
    print(f"Average over {len(results)} folds (which={args.which}, T={args.T} K)")
    print(f"k_mean={k_mean}, k_std={k_std}")
    print(f"k_spec_mean={ks_mean}, k_spec_std={ks_std}")
    print(f"k_spec_quantum_mean={kq_mean}, k_spec_quantum_std={kq_std}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

