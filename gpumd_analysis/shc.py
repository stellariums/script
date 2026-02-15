import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator


num_corr_points, num_omega = 250, 1000

labels_corr = ['t', 'Ki', 'Ko']
labels_omega = ['omega', 'jwi', 'jwo']

num_corr_points_in_run = num_corr_points * 2 - 1
coor_array = np.loadtxt("shc.out", max_rows=num_corr_points_in_run)
omega_array = np.loadtxt("shc.out", skiprows=num_corr_points_in_run)

shc = dict()
for label_num, key in enumerate(labels_corr):
    shc[key] = coor_array[:, label_num]

for label_num, key in enumerate(labels_omega):
    shc[key] = omega_array[:, label_num]
shc["nu"] = shc["omega"] / (2*np.pi)



def calc_spectral_kappa(shc, driving_force, temperature, volume):
    # ev*A/ps/THz * 1/A^3 *1/K * A ==> W/m/K/THz
    convert = 1602.17662
    shc['kwi'] = shc['jwi'] * convert / (driving_force * temperature * volume)
    shc['kwo'] = shc['jwo'] * convert / (driving_force * temperature * volume)

def read_Fe_from_runin(direction_index: int = 0, runin_path: str = "run.in") -> float:
    with open(runin_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "compute_hnemd" in s:
            parts = s.split()
            if len(parts) < 4:
                continue
            last_three = parts[-3:]
            try:
                values = [float(x) for x in last_three]
            except ValueError:
                continue
            if not (0 <= direction_index < 3):
                raise ValueError(f"direction_index must be 0/1/2, got {direction_index}")
            return float(values[direction_index])

    raise ValueError("run.in 中未找到包含 compute_hnemd 的行，无法读取 Fe")


def calculate_volume(path: str = "dump.xyz") -> float:
    if path.lower().endswith(".xyz"):
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            _ = file.readline()
            header = file.readline().strip()

        key = 'Lattice="'
        start = header.find(key)
        if start == -1:
            raise ValueError(f"{path} 第二行未找到 Lattice 字段")
        start += len(key)
        end = header.find('"', start)
        if end == -1:
            raise ValueError(f"{path} 第二行 Lattice 字段缺少结束引号")

        lattice_numbers = header[start:end].split()
        if len(lattice_numbers) != 9:
            raise ValueError(f"{path} 第二行 Lattice 字段不是 9 个数: {header[start:end]}")

        mat = np.array([float(x) for x in lattice_numbers], dtype=float).reshape(3, 3)
        return float(abs(np.linalg.det(mat)))

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()
    if not lines:
        raise ValueError(f"{path} 为空，无法读取体积")

    last_line = lines[-1].strip()
    params = last_line.split()
    try:
        Lx = float(params[-1])
        Ly = float(params[-5])
        Lz = float(params[-9])
    except (IndexError, ValueError) as e:
        raise ValueError(f"{path} 最后一行盒子尺寸字段不足或格式错误: {e}")
    return Lx * Ly * Lz


which_position = 0
T = 300.0
Fe = read_Fe_from_runin(which_position, runin_path="run.in")
V = calculate_volume("dump.xyz")



calc_spectral_kappa(shc, driving_force=Fe, temperature=T, volume=V)
shc['kw'] = shc['kwi'] + shc['kwo']
shc['K'] = shc['Ki'] + shc['Ko']
length = np.logspace(1,6,100)
k_L = np.zeros_like(length)

# Quantum correlation
hbar = 1.054e-34
h=1.054e-34
boltzmann_constant = 1.38e-23
kb=1.38e-23
x__=h*shc["omega"]/kb/T*1e12;
quantum_factor = x__**2*np.exp(x__)/((np.exp(x__)-1)**2)
quantum_spectral_kappa = shc["kw"] * quantum_factor
#######################
nu = shc["nu"]
kw_classical = shc["kw"]
kw_quantum = quantum_spectral_kappa

x_max_data = float(np.nanmax(nu))
x_max = min(50.0, x_max_data) if np.isfinite(x_max_data) else 50.0
x_max = float(np.ceil(x_max / 5.0) * 5.0) if x_max > 0 else 50.0

in_view = (nu >= 0) & (nu <= x_max)
y_max_data = float(np.nanmax(np.r_[kw_classical[in_view], kw_quantum[in_view]]))
y_max = y_max_data * 1.08 if np.isfinite(y_max_data) and y_max_data > 0 else 1.0

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.2), sharex=True, sharey=True, constrained_layout=True)

axes[0].plot(nu, kw_classical, color='tab:blue', linewidth=2.5)
axes[0].set_title("Classical")

axes[1].plot(nu, kw_quantum, color='tab:red', linewidth=2.5)
axes[1].set_title("Quantum-corrected")

for ax in axes:
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(r'$\nu$ (THz)')
    ax.set_ylabel(r'$\kappa(\omega)$ (W/m/K/THz)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

fig_c, ax_c = plt.subplots(1, 1, figsize=(5.2, 4.2), constrained_layout=True)
ax_c.plot(nu, kw_classical, color='tab:blue', linewidth=2.5)
ax_c.set_title("Classical")
ax_c.set_xlim(0, x_max)
ax_c.set_ylim(0, y_max)
ax_c.set_xlabel(r'$\nu$ (THz)')
ax_c.set_ylabel(r'$\kappa(\omega)$ (W/m/K/THz)')
ax_c.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax_c.yaxis.set_major_locator(MaxNLocator(nbins=6))

fig_q, ax_q = plt.subplots(1, 1, figsize=(5.2, 4.2), constrained_layout=True)
ax_q.plot(nu, kw_quantum, color='tab:red', linewidth=2.5)
ax_q.set_title("Quantum-corrected")
ax_q.set_xlim(0, x_max)
ax_q.set_ylim(0, y_max)
ax_q.set_xlabel(r'$\nu$ (THz)')
ax_q.set_ylabel(r'$\kappa(\omega)$ (W/m/K/THz)')
ax_q.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax_q.yaxis.set_major_locator(MaxNLocator(nbins=6))

k_spec = float(np.trapz(kw_classical, nu))
k_spec_q = float(np.trapz(kw_quantum, nu))
print(f"Fe={Fe}  T={T}  V={V}")
print(f"Spectral thermal conductivity (classical) ∫k(ν)dν = {k_spec}")
print(f"Spectral thermal conductivity (quantum)   ∫k_q(ν)dν = {k_spec_q}")

out_png = os.path.join(os.getcwd(), "shc_spectral_kappa.png")
out_pdf = os.path.join(os.getcwd(), "shc_spectral_kappa.pdf")
fig.savefig(out_png, dpi=300)
fig.savefig(out_pdf)

fig_c.savefig(os.path.join(os.getcwd(), "shc_spectral_kappa_classical.png"), dpi=300)
fig_q.savefig(os.path.join(os.getcwd(), "shc_spectral_kappa_quantum.png"), dpi=300)

np.savetxt('kv.out', np.c_[shc['nu'], quantum_spectral_kappa], header='Frequency (THz), Power (W/mK/THz)', comments='', fmt='%.6e')



if os.environ.get("SHC_SHOW", "1") == "1":
    plt.show()
else:
    plt.close(fig)
    plt.close(fig_c)
    plt.close(fig_q)
