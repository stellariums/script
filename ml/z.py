import os
from ase.io import read, write
import numpy as np

base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, "filtered")
os.makedirs(output_dir, exist_ok=True)

for name in os.listdir(base_dir):
    if not name.lower().endswith(".xyz"):
        continue
    if "_filtered" in name:
        continue
    input_path = os.path.join(base_dir, name)
    atoms = read(input_path)
    z = atoms.positions[:, 2]
    mask = np.abs(z) <= 100.0
    filtered_atoms = atoms[mask]
    output_path = os.path.join(output_dir, name)
    write(output_path, filtered_atoms)
