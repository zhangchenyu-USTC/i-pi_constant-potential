# VASP clients

The separate combined patches for VASP 5.4.4 and 6.3.2 support ordinary i-PI calculations with
`IBRION=23` and constant-potential calculations with `IBRION=24`. The tested
constant-potential executables are `vasp_gam` and `vasp_std`.

## Build the VASP 5.4.4 client

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

## Build the VASP 6.3.2 client

Start with the original VASP 6.3.2 archive and prepare it in this order:

1. Apply `vaspsol++-vasp_6.3.2.patch`.
2. Copy the matching `solvation_intel.F` to `src/solvation.F`, using LF line endings.
3. Apply [vasp-6.3.2-ipi-constant-potential.patch](vasp-6.3.2-ipi-constant-potential.patch).

```bash
cd /path/to/vasp.6.3.2
patch --dry-run --fuzz=0 -p1 < /path/to/vaspsol++-vasp_6.3.2.patch
patch --fuzz=0 -p1 < /path/to/vaspsol++-vasp_6.3.2.patch
python - /path/to/solvation_intel.F <<'PY'
from pathlib import Path
import sys
Path('src/solvation.F').write_bytes(Path(sys.argv[1]).read_bytes().replace(b'\r\n', b'\n'))
PY
patch --dry-run --fuzz=0 -p1 < /path/to/vasp-6.3.2-ipi-constant-potential.patch
patch --fuzz=0 -p1 < /path/to/vasp-6.3.2-ipi-constant-potential.patch
cp arch/makefile.include.intel makefile.include
# Set the compiler, MPI and MKL paths for your installation.
make -j 12 DEPS=1 gam
make -j 12 DEPS=1 std
```

Keep the VASP 6 compiler options, including `-Dvasp6`. Intel Fortran 2021.3
uses `-mkl=sequential` in place of the template's `-qmkl=sequential`.
The combined patch includes the ordinary socket interface; apply it directly
to this prepared baseline. The VASP 5.4.4 update, separate ordinary i-PI patch,
and `cp-vaspsol++.patch` are not part of this installation sequence.

The 6.3.2 port uses CPU `gam` and `std` for constant-potential calculations;
`ncl` is included in the build checks. The XML input and socket protocol are
the same as for 5.4.4. Use the two examples below with the new executable.

## Run directly on a node

Copy either the [implicit-solvent example](../../constant_potential/2-vasp-implicit/)
or the [Ne example](../../constant_potential/3-vasp-Ne/) to a run directory.
Prepare a local `POTCAR` in the species order of `POSCAR`, and keep the atom order
consistent with `init.xyz`. The first example uses VASPsol++; the second uses Ne
core-charge adjustment with implicit solvation disabled.

Combining Ne core-charge adjustment with implicit solvation is currently
unsupported in both versions. Use `LSOL=.FALSE.` for the Ne example.

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

To run either repository example for three steps and check its four socket
evaluations, use the same Python environment as i-PI:

```bash
ulimit -s unlimited
python examples/clients/vasp/check_example.py \
  --example 2-vasp-implicit --vasp /path/to/vasp_gam \
  --potcar /path/to/implicit/POTCAR --ranks 48 --work /path/to/new-test-directory
```

For the Ne example, select `--example 3-vasp-Ne` and its matching POTCAR.
The script checks electron numbers, finite responses, the work-function
relation, SCF convergence, evaluation counts, and both exit codes. It writes
the inputs and results into the specified new directory.

In both patched VASP versions, `IBRION=23/24` writes and flushes one
`F= ... E0= ... d E = ...` summary to OSZICAR for each configuration
evaluation. The local counter starts at 1, including the initial configuration;
`STATUS` and repeated `GETFORCE` requests do not add summaries. `F` is the
electronic free energy and `E0` is the zero-smearing extrapolation, in eV.
The `d E` field follows VASP's static/socket output convention (the electronic
entropy contribution); it is not the difference between successive frames.

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
