#!/bin/bash
#SBATCH -J lzybatch
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12

# 加载模块（只需加载一次）
module load gpumd-4.0

# 依次处理当前目录下的所有子目录
for DIR in */; do
    # 确认是目录
    [ -d "$DIR" ] || continue

    echo "===== 进入目录 $DIR ====="

    # 将 run.in 复制到子目录
    cp run.in "$DIR"

    # 进入子目录并运行 gpumd
    cd "$DIR"
    gpumd >> log.file 2>&1
    cd ..
done
