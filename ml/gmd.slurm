#!/bin/bash
#SBATCH -J lzy_1500k
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
#SBATCH --array=131-230%6


cd $SLURM_SUBMIT_DIR

SUB_DIR="${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SUB_DIR"

cp run.in nep.txt "$SUB_DIR"/
cp "${SLURM_ARRAY_TASK_ID}.xyz" "$SUB_DIR/model.xyz"

# 在 model.xyz 第二行末尾追加 pbc="T T F"
sed -i '2s/$/ pbc="T T F"/' "$SUB_DIR/model.xyz"

cd "$SUB_DIR"

module load compiler/gcc-12.2.0
module load gpumd-4.7

gpumd >> log.file
