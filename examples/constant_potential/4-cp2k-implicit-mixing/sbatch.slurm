#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source /path/to/ipi/env.sh

i-pi input.xml > ipi.log 2>&1 &

sleep 5

mpirun -np 64 /path/to/cp2k.psmp -i cp2k_farming.inp -o cp2k_farming.out

wait
