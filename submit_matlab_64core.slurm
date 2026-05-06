#!/bin/bash
#SBATCH -J cof4f_matlab
#SBATCH -p intel
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

module purge

# Use your own MATLAB installation
export MATLABROOT=$HOME/apps/MATLAB/R2022b
export PATH=$MATLABROOT/bin:$PATH
export MLM_LICENSE_FILE=$HOME/licenses/network.lic

# Avoid CPU over-subscription inside MATLAB and math libraries
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

"$MATLABROOT/bin/matlab" -nodisplay -nosplash -nodesktop -batch "maxNumCompThreads(${SLURM_CPUS_PER_TASK:-1}); run_ilp_fit_hpc"

echo "Job finished at $(date)"