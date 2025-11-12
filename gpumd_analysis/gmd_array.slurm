#!/bin/bash
#SBATCH -J lzy_array
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
#SBATCH --array=1-3%1

###############
# Define the sub-directory for this task
SUB_DIR="fold${SLURM_ARRAY_TASK_ID}"

# Create the sub-directory if it doesn't exist
mkdir -p $SUB_DIR

# Copy necessary input files to the sub-directory
# Assuming run.in, model.xyz, and the potential file are the inputs
cp run.in model.xyz Graphene_Lindsay_2010_modified.txt $SUB_DIR/

# Navigate to the sub-directory
cd $SUB_DIR

# Load the required module
module load gpumd-4.0

# Run the simulation and redirect output
gpumd \u003e\u003e log.file
