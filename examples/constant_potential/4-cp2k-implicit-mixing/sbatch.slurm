#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -euo pipefail
# Activate i-PI/CP2K and set CP2K_EXECUTABLE before submitting this script.
cd "${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]}")}"
exec bash run.sh --ranks "${SLURM_NTASKS:-64}" --output "run-${SLURM_JOB_ID:-local}" "$@"
