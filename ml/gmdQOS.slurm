#!/bin/bash
#SBATCH -J lzy_1000k
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12


cd $SLURM_SUBMIT_DIR
module load compiler/gcc-12.2.0
module load gpumd-4.4

# 赋予脚本可执行权限
chmod +x run_simulations.sh

# 执行批量模拟脚本并记录日志
echo "Job started at $(date)" >> batch_run.log
./run_simulations.sh >> batch_run.log 2>&1
echo "Job finished at $(date)" >> batch_run.log

