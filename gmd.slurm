#!/bin/bash
#SBATCH -J lzy_HDCG
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH --exclude=gpu01
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
###############
cd $SLURM_SUBMIT_DIR
module load compiler/gcc-12.2.0
module load gpumd-4.4
gpumd >> log.file 
