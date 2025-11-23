#!/bin/bash
#SBATCH -J lzy_1270
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH --nodelist=gpu02,gpu03,gpu04,gpu01
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
###############
cd $SLURM_SUBMIT_DIR
<<<<<<< HEAD
module load gpumd-4.0
gpumd >> log.file
=======
module load compiler/gcc-12.2.0
module load gpumd-4.4
gpumd >> log.file 
>>>>>>> c46d075b9061fdd5bfa81bb78b86963f7b1210e2
