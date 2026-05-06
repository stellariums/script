#!/bin/bash
#SBATCH -J nep
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
###############
cd $SLURM_SUBMIT_DIR
module load compiler/gcc-12.2.0 
module load cuda/12.4 
/public/home/like/GPUMD-4.9.1/src/nep >> log.file 


