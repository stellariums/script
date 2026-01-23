#!/bin/bash
#SBATCH -J 800
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
#SBATCH --no-requeue

cd $SLURM_SUBMIT_DIR

module purge
module load compiler/gcc-12.2.0
module load gpumd-4.7

echo "Python sanity check:"
which python3
python3 -V
python3 - << EOF
import ase
print("ASE:", ase.__version__)
EOF

bash sw.sh
