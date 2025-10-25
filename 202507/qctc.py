import numpy as np
import matplotlib.pyplot as plt
import re
import os

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
    kappa = np.loadtxt('kappa.out')
    msd = np.loadtxt('msd.out')
    rdf = np.loadtxt('rdf.out')
    dos = np.loadtxt('dos.out')
else:
    if model == 1:
        kappa = np.loadtxt('kappa.out')
    elif model == 2:
        msd = np.loadtxt('msd.out')
    elif model == 3:
        rdf = np.loadtxt('rdf.out')
    elif model == 4:
        dos = np.loadtxt('dos.out')

parent_folder = get_parent_folder_name()
if parent_folder:
    print(f'Parent directory: {parent_folder}')

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
#with open('thermo.out', 'r') as file:
#    lines = file.readlines()
#last_line = lines[-1]
#params = last_line.split()
#Lx, Ly, Lz = params[-3:]  
#Lx, Ly, Lz = map(float, (Lx, Ly, Lz))
#V = Lx * Ly * Lz
def calculate_volume():
    try:
        with open('thermo.out', 'r') as file:
            lines = file.readlines()
        last_line = lines[-1]
        params = last_line.split()
        
        # ³¢ÊÔ´Óµ¹ÊýµÚÈý¸ö¡¢µ¹ÊýµÚÎå¸öºÍµ¹ÊýµÚÆß¸ö²ÎÊýÖÐ»ñÈ¡³¤¶È²ÎÊý
        try:
            Lx, Ly, Lz = map(float, params[-3:])
        except ValueError:
            print("Error: Not enough parameters at the end of the last line.")
            return None
        
        V = Lx * Ly * Lz
        
        # Èç¹û V Îª 0£¬ÔòÊ¹ÓÃµ¹ÊýµÚÒ»¸ö¡¢µ¹ÊýµÚÎå¸öºÍµ¹ÊýµÚ¾Å¸öÊýÖØÐÂ¼ÆËã V
        if V == 0:
            try:
                Lx, Ly, Lz = map(float, (params[-1], params[-5], params[-9]))
                V = Lx * Ly * Lz
            except (ValueError, IndexError):
                print("Error: Not enough parameters or invalid values to calculate volume.")
                return None
        
        print(f"Volume: {V}")
        return V
    except FileNotFoundError:
        print("Error: File 'thermo.out' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# µ÷ÓÃº¯Êý
V = calculate_volume()
##################
T = 300  # Temperature (K)
Fe = 1e-3  # Driving force parameter (1/A)
num_corr_points, num_omega = 250, 1000
###################


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


def calc_spectral_kappa(shc, force_parameter, temperature, volume):
    # ev*A/ps/THz * 1/A^3 *1/K * A ==> W/m/K/THz
    convert = 1602.17662
    shc['kwi'] = shc['jwi'] * convert / (force_parameter * temperature * volume)
    shc['kwo'] = shc['jwo'] * convert / (force_parameter * temperature * volume)

calc_spectral_kappa(shc, force_parameter=Fe, temperature=T, volume=V)
shc['kw'] = shc['kwi'] + shc['kwo']
spectral_kappa_integral = np.trapz(shc['kw'], shc["nu"])
print(f"Spectral thermal conductivity (k_spec) is {spectral_kappa_integral}")

# Quantum correlation
hbar = 1.054e-34
h=1.054e-34
boltzmann_constant = 1.38e-23
kb=1.38e-23
x__=h*shc["omega"]/kb/T*1e12;
quantum_factor = x__**2*np.exp(x__)/((np.exp(x__)-1)**2)
quantum_spectral_kappa = shc["kw"] * quantum_factor
quantum_kappa_integral = np.trapz( quantum_spectral_kappa, shc["nu"])
print(f"Quantum correlated spectral thermal conductivity (k_spec) is {quantum_kappa_integral}")
current_dir = os.getcwd()
with open(f"../qctc.data", "a") as f:
    f.write(f"{current_dir},{spectral_kappa_integral},{spectral_kappa_integral},{quantum_kappa_integral}\n")

