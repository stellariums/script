#!/bin/bash
#SBATCH -J UN
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
#SBATCH --array=1-100%10
#SBATCH --exclude=gpu01
###############

SUB_DIR="${SLURM_ARRAY_TASK_ID}"
cd $SUB_DIR

source /etc/profile

module purge
module load compiler/gcc-12.2.0
module load compiler/cmake-3.31.2
module load cuda/12.4
module load mpi/oneapi

source /public/home/like/abacus-gpu/abacus-develop-LTS/toolchain/abacus_env.sh
source "$HOME/venvs/abacus/bin/activate"

export LD_LIBRARY_PATH=/public/software/oneapi/compiler/2022.2.0/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's/::/:/g')

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_PATH=/usr/local/cuda-12.4
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc

abacus >> log.file 