#!/bin/bash
#SBATCH -J lzy_500
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12

# 1. 加载环境 (请根据超算实际 module 名修改)
module purge                    # 清除已加载的环境，防止冲突
module load anaconda3           # 加载 Python
module load compiler/gcc-12.2.0
module load gpumd-4.4
gpumd >> log.file 



# 3. 打印调试信息
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Date: $(date)"
echo "CUDA Info:"
nvidia-smi

# 4. 进入提交目录
cd $SLURM_SUBMIT_DIR

# 5. 运行主脚本
bash sw.sh
