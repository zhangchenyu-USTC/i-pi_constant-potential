# ABACUS PW + standard i-PI example

This example uses the non-constant-potential `abacus-3.10.0-ipi.patch`. The
ABACUS process connects to the inet socket on port 19003 and supplies ordinary
energy, forces, and virial data to a one-bead NVT simulation.

Before running, set `pseudo_dir` in `INPUT`, the i-PI environment setup path,
and the ABACUS executable path in `sbatch.slurm`. The supplied Al structure has
24 fixed atoms, matching the `<fixatoms>` list in `input.xml`.
