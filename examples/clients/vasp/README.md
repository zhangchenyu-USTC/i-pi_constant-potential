# VASP clients

The combined VASP 5.4.4 patch supports ordinary i-PI calculations with
`IBRION=23` and constant-potential calculations with `IBRION=24`. The tested
constant-potential executables are `vasp_gam` and `vasp_std`.

## Build the constant-potential client

Start with a clean VASP 5.4.4 source tree. Apply the following preparations in
order:

1. The official `patch.5.4.4.16052018` update.
2. `vaspsol++-vasp_5.4.4.patch` from VASPsol++.
3. The matching VASPsol++ `solvation_intel.F`, copied to `src/solvation.F`.
4. This directory's `vasp-5.4.4-ipi-constant-potential.patch`.

```bash
cd /path/to/vasp.5.4.4
gzip -dc /path/to/patch.5.4.4.16052018.gz | patch --dry-run -p0
gzip -dc /path/to/patch.5.4.4.16052018.gz | patch -p0
patch --dry-run -p1 < /path/to/vaspsol++-vasp_5.4.4.patch
patch -p1 < /path/to/vaspsol++-vasp_5.4.4.patch
cp /path/to/solvation_intel.F src/solvation.F
patch --dry-run -p1 < /path/to/vasp-5.4.4-ipi-constant-potential.patch
patch -p1 < /path/to/vasp-5.4.4-ipi-constant-potential.patch
```

The combined patch already includes ordinary i-PI socket support. Apply it
directly after the VASPsol++ preparation; do not also apply the older
`vasp.5.4.4-ipi.patch` or run `patch_vasp.py` on this tree.

Configure `makefile.include` for the compiler and MPI installation, then build:

```bash
make std gam
```

Use `vasp_gam` for Gamma-only calculations and `vasp_std` for general k-point
calculations. VASP sources and POTCAR files come from your VASP installation.

## Run directly on a node

Copy either the [implicit-solvent example](../../constant_potential/2-vasp-implicit/)
or the [Ne example](../../constant_potential/3-vasp-Ne/) to a run directory.
Prepare a local `POTCAR` in the species order of `POSCAR`, and keep the atom order
consistent with `init.xyz`. The first example uses VASPsol++; the second uses Ne
core-charge adjustment with implicit solvation disabled.

Start i-PI in one terminal:

```bash
cd /path/to/run
python /path/to/i-pi-constant-potential/bin/i-pi -V medium input.xml > ipi.log 2>&1
```

When `ipi.log` reports `Created inet socket`, start VASP in a second terminal:

```bash
cd /path/to/run
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -s unlimited
mpiexec -n 96 /path/to/vasp_gam > vasp.log 2>&1
```

Choose the MPI process count together with `NCORE` in `INCAR`. Set `IHOST`,
`PORT`, and `INET` to match the socket in `input.xml`. Electron dynamics and the
potential-reference window are configured in the XML input. For a remote
client, use a host address and port reachable from the VASP node. Each VASP
process group needs its own run directory.

## Solvent update interval

`SOL_UPDATE` is an INCAR integer with a default of `1`. A value of `1` updates
the solvent at every configuration evaluation. For example:

```text
IBRION = 24
LSOL = .TRUE.
LSOL_SCF = .TRUE.
SOL_UPDATE = 5
```

This refreshes the solvent at local evaluation numbers `0, 5, 10, ...`. Between
refreshes, VASP keeps the solvent correction potential and solvent charge and
performs the electronic SCF for the current geometry and electron number.
The energy and ionic forces include the interaction with this frozen solvent
field. The Fermi level and work function are evaluated for the current state.

The counter advances after a successful configuration evaluation. `STATUS` and
repeated retrieval of a result do not advance it. A new connection or restart
starts with a solvent refresh and a new counter.

Values greater than `1` require `IBRION=24`, `LSOL`, and `LSOL_SCF`. Use them
with one bead and a separate VASP process group for each endpoint; auxiliary
evaluations need separate process groups. Keep `SOL_UPDATE=1` for ordinary
`IBRION=23` and native VASP calculations.

## Ordinary i-PI examples

For ordinary socket support on other VASP versions, the existing helper can
patch VASP 5.4 and later:

```bash
python patch_vasp.py /path/to/vasp.X.Y.Z/src
```

Use `IBRION=23`. The water and diamond examples in this directory use the
ordinary socket interface. After preparing local POTCAR files, pass the
executable paths to make:

```bash
make h2o VASP=/path/to/vasp_std IPI=/path/to/i-pi-constant-potential/bin/i-pi
make dia VASP=/path/to/vasp_std IPI=/path/to/i-pi-constant-potential/bin/i-pi
```

The targets run water and diamond, respectively. `make clean` removes the
generated output. These are small working examples; select bead counts and
simulation parameters for the system you study.
