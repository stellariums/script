#!/bin/bash
#SBATCH -J lzy
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
###############
cd $SLURM_SUBMIT_DIR
module load compiler/gcc-12.2.0
module load gpumd-4.4
gpumd >> log.file 
