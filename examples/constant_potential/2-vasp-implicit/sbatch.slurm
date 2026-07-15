#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source /path/to/ipi/env.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

i-pi input.xml > ipi.log 2>&1 &

sleep 5

mpirun -np 64 /path/to/vasp_gam > vasp.log 2>&1

wait
