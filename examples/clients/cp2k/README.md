CP2K - i-PI example
===================

Build and run the constant-potential examples
--------------------------------------------

Apply the complete patch to a clean CP2K `v2026.1` tree, then build and install
with the CP2K toolchain or CMake:

```bash
git apply --check /path/to/cp2k-2026.1-ipi-constant-potential.patch
git apply /path/to/cp2k-2026.1-ipi-constant-potential.patch
```

Use the installed executable and its matching `libcp2k` when using a shared
build. Load the compiler/MPI environment used to build CP2K. Its data directory
must provide `BASIS_MOLOPT`, `GTH_POTENTIALS`, and `dftd3.dat`; set `CP2K_DATA_DIR`
to the installed data directory if needed.

From this directory, run either example directly on a node:

```bash
export CP2K_EXECUTABLE=/path/to/cp2k.psmp
python run_example.py --example mixing --output run-mixing --ranks 96 --steps 3
python run_example.py --example ne --output run-ne --ranks 96 --steps 3
```

The launcher copies the input files into a new output directory, starts this
i-PI checkout, waits for its sockets, and launches CP2K. `--steps` overrides the
100-step example setting. For FARMING, `--ranks` is the total MPI process count;
the launcher divides it between the two endpoints and updates `GROUP_PARTITION`.
Use `--mpi-command 'mpiexec ...'` or `MPIEXEC` to select the MPI launcher.
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` default to 1;
explicitly exported values are retained. `--timeout` sets an optional total
runtime limit. A failed process stops its partner and produces a nonzero exit.
Logs are saved as `ipi.log`, `cp2k.log`, and the CP2K output files.

Each example directory also has `bash run.sh --output NEW_DIR --ranks 96`
and a Slurm wrapper. Run the wrapper from the example directory after loading
the software environment. For custom i-PI instrumentation, `--ipi-command`
accepts a command to which the launcher appends `input.xml`.

The Al36 mixing example uses STANDARD diagonalization with Fermi-Dirac smearing.
Its initial 107/108-electron endpoints use `CHARGE 1/0`, implicit-Poisson
`TOL 1e-8`, and `EXTRAPOLATION USE_PREV_P` for density continuation. Both this
example and the Ne example completed three MD steps with four evaluations per
endpoint and normal i-PI/CP2K exits in the release checks.

Constant-potential Ne electrodes (CP2K 2026.1)
---------------------------------------------

The accompanying constant-potential patch includes fractional RKS occupations,
Fermi-Dirac electron-count conservation, and the Ne CHGDATA driver. Apply the
complete patch and rebuild CP2K; changing the driver alone is insufficient.

Ne mode requires Gamma-point RKS, self-consistent `DIAGONALIZATION ALGORITHM
STANDARD`, `CORE_CORR_DIP TRUE`, `SURFACE_DIPOLE_CORRECTION TRUE`, and `CHARGE 0`.
The i-PI `neutral_electrons` value must equal the uncorrected neutral valence
electron count. Non-Ne kinds must have zero `CORE_CORRECTION`. Existing Ne
corrections are replaced by the current absolute request, not accumulated.
Ne mode supports no smearing or positive-temperature `FERMI_DIRAC` smearing.
Provide sufficient `ADDED_MOS` for the largest requested electron number and
the thermal occupation tail. OT, UKS, explicit K points, MOM, XAS, and custom
occupation methods are outside this Ne implementation's supported scope.

For trajectories, retain a usable wavefunction/density guess between requests.
CP2K itself recommends `SCF_GUESS RESTART` when a nonzero core correction is
combined with the surface dipole correction. `EXTRAPOLATION USE_PREV_P` avoids
rebuilding an atomic guess at every geometry. Place the dipole discontinuity
in a low-density vacuum plane.

The `constant_potential/5-cp2k-Ne` input uses these settings and Broyden
`ALPHA 0.1`, `NBROYDEN 16`. Its `SURF_DIP_POS 35.0` applies to the supplied
45-A cell, outside the 27--30 A workfunction reference window. The release
three-step test starts from the neutral geometry without a bundled wavefunction.
GPW falls back to an atomic guess when the RESTART wavefunction is absent.
`USE_PREV_P` retains the density for subsequent fractional-electron requests.

The driver changes each Ne effective nuclear charge by
`(requested_electrons - neutral_electrons) / number_of_Ne_atoms`. CP2K's native
core-correction occupation mechanism then adjusts the electron density. The
response reports the occupations actually used to construct the density;
both this count and the full MO occupation sum must agree with the request
within `1e-8` electrons. A discarded small-occupation tail requires a tighter
`EPS_FERMI_DIRAC`; a significantly occupied highest allocated MO requires more
`ADDED_MOS`. These are separate convergence checks.

Each completed result caches energy, forces, virial, Fermi level, workfunction,
and electron count together. A new INIT invalidates that result and requires
a fresh CHGDATA/POSDATA evaluation. A client process cannot switch between Ne
and ordinary CHGDATA modes. The current CHGDATA flags are 0 for ordinary mode
and 2 for Ne mode; INIT carries no solvent-update interval.

Ordinary integer-electron OT updates
----------------------------------

When an ordinary CHGDATA request changes the electron count, the driver releases
and reinitializes the native QS electronic environment with a private copy of
the FORCE_EVAL input and the new CHARGE. It retains the nuclear subsystem,
connection, and process. An explicit CORE or RANDOM guess is preserved. ATOMIC,
RESTART, and HISTORY_RESTART use a fresh CORE guess for the changed count: the
atomic density can lack rank for newly occupied orbitals, and saved wavefunctions
belong to the previous orbital dimensions. The original input remains unchanged.
MO occupations, matrices, OT history, preconditioners, and solvent state are
rebuilt together. Identical counts preserve the existing state and repeated
results keep their normal cache behavior.

After each actual zero-temperature ordinary OT SCF, the driver recomputes the
occupied-subspace Ritz HOMO from the current orbitals and KS matrix. A nonzero
cached native mu can refer to the previous calculation and is not a valid reason
to skip this update. Repeated GETFORCE reads use the complete cached response;
Ne and finite-temperature native chemical potentials retain their own paths.

Ordinary mode supports native RKS and fixed-MULTIPLICITY UKS integer populations.
OT uses its native zero-temperature occupied-orbital setup; Ne uses STANDARD
diagonalization. Changing an OT count costs a new electronic initialization
and a cold SCF.

This CORE initialization lets the He2 RKS driver populate the third occupied
orbital when the requested count changes from four to six electrons. Count,
occupations, and density are checked against independent fixed-charge CORE
calculations. The electron-count tolerance remains `1e-8`.

The patch also releases the previous dielectric real-space grid before
replacement and recreates implicit-Poisson solver caches when the PW grid is
rebuilt. The normal offload memory-pool exit checks remain enabled.

## Ordinary CP2K examples

The `npt_*` and `nvt_*` subdirectories contain the classical and PIGLET water
inputs from Ceriotti, More, and Manolopoulos, *Computer Physics
Communications* (2013). Each case contains its own `input.xml` and `in.cp2k`.
For example, the NPT PIGLET inputs are in `npt_piglet/`.

Run i-PI from the selected case directory and launch each CP2K client in a
separate working directory. Set the XML `ffsocket` address and port to match
`HOST` and `PORT` in the CP2K `DRIVER` section. These examples use the ordinary
socket interface; the constant-potential examples and node launcher are
described above.
