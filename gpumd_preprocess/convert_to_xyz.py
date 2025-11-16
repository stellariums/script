import argparse
from pathlib import Path
from ase.io import read, write

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="GWHR-CONTCAR")
    parser.add_argument("--out", dest="outfile", default="model.xyz")
    args = parser.parse_args()
    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    atoms = read(in_path, format="vasp")
    atoms.arrays.pop("momenta", None)
    write(out_path, atoms, format="extxyz")

if __name__ == "__main__":
    main()