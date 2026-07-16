#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source /path/to/ipi/env.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ulimit -s unlimited

i-pi input.xml > ipi.log 2>&1 &

sleep 5

mpirun -np 36 /path/to/abacus > abacus.log 2>&1

wait
