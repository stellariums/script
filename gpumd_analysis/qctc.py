# -*- coding: utf-8 -*-
"""
用途 / Use:
- 读取 HNEMD 输出（kappa.out / shc.out / thermo.out），计算：
  1) 时间累计平均热导率 k（来自 kappa.out）
  2) 频谱热导率积分 k_spec（来自 shc.out）
  3) 量子修正后的频谱热导率积分 k_spec_quantum（来自 shc.out）

用法 / How to use:
1) 在脚本顶部修改：
   - model: 0=全部绘图数据都读；1=HNEMD(kappa)；2=MSD；3=RDF；4=DOS
   - which_position: 0=x方向, 1=y方向, 2=z方向
2) 确保当前目录包含：
   - run.in（用于读取 Fe；需要包含一行 compute_hnemd ... Fx Fy Fz）
   - thermo.out（用于读取盒子尺寸计算体积 V）
   - shc.out（用于频谱热导率）
   - kappa.out（若 model=0 或 1）
3) 直接运行脚本

重要改动说明:
- Fe 不再手动指定常数，而是从当前文件夹 run.in 中读取：
  查找包含 "compute_hnemd" 的那一行，取其“最后三个数字”作为 (Fx, Fy, Fz)。
  which_position=0/1/2 时，分别取 Fx/Fy/Fz 作为 Fe。
- 除 Fe 获取方式外，其余计算流程、变量、输出逻辑保持不变。
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ===========================
# NEW: 从 run.in 读取 Fe
# ===========================

# 支持整数、小数、科学计数法（例如 1e-4、-2.3E+01 等）
_NUMBER_RE = r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'


def read_Fe_from_runin(direction_index: int, runin_path: str = "run.in") -> float:
    """
    从 run.in 中读取 compute_hnemd 行末尾的三个驱动力参数，并按方向返回 Fe。

    参数:
    - direction_index: 0 表示 x，1 表示 y，2 表示 z（对应 which_position）
    - runin_path: run.in 文件路径（默认当前目录的 run.in）

    规则:
    - 找到包含关键字 "compute_hnemd" 的那一行（去掉行尾注释后判断）
    - 提取该行所有数字，取最后三个数作为 (Fx, Fy, Fz)
    - direction_index=0/1/2 分别返回 Fx/Fy/Fz

    返回:
    - Fe (float)

    异常:
    - 找不到 run.in / 找不到 compute_hnemd / 数字不足 3 个时抛出异常
    """
    if direction_index not in (0, 1, 2):
        raise ValueError("direction_index 必须是 0(x), 1(y), 或 2(z)")

    if not os.path.exists(runin_path):
        raise FileNotFoundError(f"Error: '{runin_path}' not found in {os.getcwd()}")

    matches = []
    with open(runin_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            # 去掉常见行尾注释（# ; !）避免把注释里的数字也读进去
            line_wo_comment = re.split(r"[#;!]", line, maxsplit=1)[0].strip()
            if not line_wo_comment:
                continue

            if re.search(r"\bcompute_hnemd\b", line_wo_comment):
                nums = re.findall(_NUMBER_RE, line_wo_comment)
                if len(nums) < 3:
                    raise ValueError(
                        f"Found 'compute_hnemd' at line {ln} but fewer than 3 numbers:\n{line.strip()}"
                    )
                last_three = list(map(float, nums[-3:]))  # (Fx, Fy, Fz) at end
                matches.append((ln, last_three, line.strip()))

    if not matches:
        raise ValueError("Error: No line containing 'compute_hnemd' was found in run.in")

    # 若存在多行 compute_hnemd，默认使用第一行（保持稳定且可预期）
    if len(matches) > 1:
        print(f"Warning: Found {len(matches)} 'compute_hnemd' lines in run.in; using the first one.")

    ln, last_three, raw = matches[0]
    Fe = last_three[direction_index]
    print(
        f"Read Fe from run.in (line {ln}): last_three={last_three}, "
        f"using Fe={Fe} for direction_index={direction_index}"
    )
    return Fe


# Function to get the parent directory name
def get_parent_folder_name():
    current_path = os.getcwd()
    parent_folder = os.path.dirname(current_path)
    return os.path.basename(parent_folder)


# Main program
model = 1  # 0 for all plots, 1 for HNEMD, 2 for MSD, 3 for RDF, 4 for DOS
which_position = 0  # 0 for kx, 1 for ky, 2 for kz

print("Please modify the parameters in the script as needed before running.")

# Load data
if model == 0:
    kappa = np.loadtxt("kappa.out")
    msd = np.loadtxt("msd.out")
    rdf = np.loadtxt("rdf.out")
    dos = np.loadtxt("dos.out")
else:
    if model == 1:
        kappa = np.loadtxt("kappa.out")
    elif model == 2:
        msd = np.loadtxt("msd.out")
    elif model == 3:
        rdf = np.loadtxt("rdf.out")
    elif model == 4:
        dos = np.loadtxt("dos.out")

parent_folder = get_parent_folder_name()
if parent_folder:
    print(f"Parent directory: {parent_folder}")

# Plot thermal conductivity
if model == 1 or model == 0:
    M = kappa.shape[0]
    t = np.arange(1, M + 1) * 0.001  # Time in ns

    if which_position == 0:
        ki_ave = np.cumsum(kappa[:, 0]) / np.arange(1, M + 1)
        ko_ave = np.cumsum(kappa[:, 1]) / np.arange(1, M + 1)
        k = ki_ave + ko_ave
    elif which_position == 1:
        ki_ave = np.cumsum(kappa[:, 2]) / np.arange(1, M + 1)
        ko_ave = np.cumsum(kappa[:, 3]) / np.arange(1, M + 1)
        k = ki_ave + ko_ave
    else:
        k = np.cumsum(kappa[:, 4]) / np.arange(1, M + 1)

    final_k = k[-1]
    print(f"Thermal conductivity (k) is: {final_k}")


# Calculate spectral thermal conductivity
# with open('thermo.out', 'r') as file:
#     lines = file.readlines()
# last_line = lines[-1]
# params = last_line.split()
# Lx, Ly, Lz = params[-3:]
# Lx, Ly, Lz = map(float, (Lx, Ly, Lz))
# V = Lx * Ly * Lz

def calculate_volume():
    """
    从 thermo.out 的最后一行读取盒子尺寸并计算体积 V。

    新规则（按你的要求）：
    - 直接取最后一行的 (倒数第1、倒数第5、倒数第9) 作为 (Lx, Ly, Lz)
      即：Lx=params[-1], Ly=params[-5], Lz=params[-9]
    """
    try:
        with open("thermo.out", "r") as file:
            lines = file.readlines()

        if not lines:
            print("Error: thermo.out is empty.")
            return None

        last_line = lines[-1].strip()
        params = last_line.split()

        try:
            Lx = float(params[-1])
            Ly = float(params[-5])
            Lz = float(params[-9])
        except (ValueError, IndexError):
            print("Error: Not enough parameters or invalid values in the last line to calculate volume.")
            print("Last line:", last_line)
            return None

        V = Lx * Ly * Lz

        print(f"Read box from thermo.out last line (idx -1,-5,-9): Lx={Lx}, Ly={Ly}, Lz={Lz}")
        print(f"Volume: {V}")
        return V

    except FileNotFoundError:
        print("Error: File 'thermo.out' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# 调用函数
V = calculate_volume()
##################
T = 300  # Temperature (K)

# Fe 改为从 run.in 读取（按 which_position 对应 x/y/z）
Fe = read_Fe_from_runin(which_position, runin_path="run.in")

num_corr_points, num_omega = 250, 1000
###################

labels_corr = ["t", "Ki", "Ko"]
labels_omega = ["omega", "jwi", "jwo"]

num_corr_points_in_run = num_corr_points * 2 - 1
coor_array = np.loadtxt("shc.out", max_rows=num_corr_points_in_run)
omega_array = np.loadtxt("shc.out", skiprows=num_corr_points_in_run)

shc = dict()
for label_num, key in enumerate(labels_corr):
    shc[key] = coor_array[:, label_num]

for label_num, key in enumerate(labels_omega):
    shc[key] = omega_array[:, label_num]
shc["nu"] = shc["omega"] / (2 * np.pi)


def calc_spectral_kappa(shc, force_parameter, temperature, volume):
    # ev*A/ps/THz * 1/A^3 *1/K * A ==> W/m/K/THz
    convert = 1602.17662
    shc["kwi"] = shc["jwi"] * convert / (force_parameter * temperature * volume)
    shc["kwo"] = shc["jwo"] * convert / (force_parameter * temperature * volume)


calc_spectral_kappa(shc, force_parameter=Fe, temperature=T, volume=V)
shc["kw"] = shc["kwi"] + shc["kwo"]
spectral_kappa_integral = np.trapz(shc["kw"], shc["nu"])
print(f"Spectral thermal conductivity (k_spec) is {spectral_kappa_integral}")

# Quantum correlation
hbar = 1.054e-34
h = 1.054e-34
boltzmann_constant = 1.38e-23
kb = 1.38e-23
x__ = h * shc["omega"] / (kb * T) * 1e12

den = np.expm1(x__)                 # den = exp(x__) - 1，更稳
quantum_factor = x__ ** 2 * np.exp(x__) / ((np.exp(x__) - 1) ** 2)
quantum_spectral_kappa = shc["kw"] * quantum_factor
quantum_kappa_integral = np.trapz(quantum_spectral_kappa, shc["nu"])
print(f"Quantum correlated spectral thermal conductivity (k_spec) is {quantum_kappa_integral}")





current_dir = os.getcwd()
with open(f"../qctc.data", "a") as f:
    f.write(f"{current_dir},{spectral_kappa_integral},{spectral_kappa_integral},{quantum_kappa_integral}\n")
