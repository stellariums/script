#!/bin/bash
#SBATCH -J aims_H2O
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

module purge
module load compiler/gcc-12.2.0
module load mpi/mpich-4.0-gcc12

export AIMS_ROOT=/public/home/LK/FHI_aims/fhi-aims.250822
export AIMS_SPECIES_DIR=$AIMS_ROOT/species_defaults

ulimit -s unlimited || ulimit -s 500000
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=false

export LD_LIBRARY_PATH=/public/software/gcc-12.2/lib64:/public/software/mpich-4.0-gcc12.2.0/lib:$HOME/local/gcc12/lib:${LD_LIBRARY_PATH:-}

cd "${SLURM_SUBMIT_DIR}"

echo "Running on: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "AIMS_SPECIES_DIR=$AIMS_SPECIES_DIR"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
ulimit -s

srun --cpu-bind=cores -n ${SLURM_NTASKS} $AIMS_ROOT/build/aims.250822.scalapack.mpi.x </dev/null \
  | tee H2O_test.slurm.${SLURM_JOB_ID}.own

grep -n "Total energy of the DFT / Hartree-Fock s.c.f. calculation" H2O_test.slurm.${SLURM_JOB_ID}.own | tail -n 1 || true
