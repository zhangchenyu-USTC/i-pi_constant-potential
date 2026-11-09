# ABACUS constant-potential examples and tests

This directory contains the Si2 PW/LCAO examples, the Al36 electrode
examples, and the real-client regression suite. ABACUS keeps the focused
C++ unit tests alongside its source.

The current PR source uses upstream commit
`e58ffe5e40f8e70713aa879f40da07fe591d5294`. It passes each electronic request
through a temporary, non-owning `UnitCell` pointer instead of adding ESolver
interfaces. The included `abacus-develop-constant-potential.patch` remains on
`d20090cc137fdc3b3baddd9217f792bb7af922d7` and does not contain this refactor.
The beta10 patch in main uses its separate beta10 baseline. Build the original
reference from the same upstream revision as the candidate being tested.
Earlier validation commits remain available at their fixed links.

## Run an example

From this directory, set the executable and data paths:

```bash
export ABACUS_CP_EXECUTABLE=/path/to/abacus
export ABACUS_PP_ORB=/path/to/abacus-source/tests/PP_ORB
python run_example.py si2-pw --output /path/to/new-run --ranks 4 --steps 3
```

The case names are `si2-pw`, `si2-lcao`, `al36-pw`, and `al36-lcao`.
The launcher copies the selected case into a new output directory and writes
the configured data paths into its `INPUT`. Case inputs use `./PP_ORB` as a
placeholder. The launcher also creates a unique socket and starts i-PI and
ABACUS with one OpenMP thread per MPI rank.

`ABACUS_SOURCE_ROOT` can replace `ABACUS_PP_ORB`; data are then read from
`$ABACUS_SOURCE_ROOT/tests/PP_ORB`. i-PI is taken from this repository, or
from `IPI_ROOT` when set. `MPIEXEC` selects the launcher (default `mpirun`),
and `ABACUS_TEST_MPI_OPTIONS` supplies additional options. For example,
OpenMPI users can set `ABACUS_TEST_MPI_OPTIONS="--bind-to none"`.

Use `--temperature`, `--dt`, and `--timeout` to set kelvin, femtoseconds, and
the total runtime limit in seconds. Without `--steps`, the case's step count
is used. Results are in `ipi.log`, `abacus.log`, `OUT.ABACUS/`, and `cp.out`.

The larger production-style inputs remain in
[`examples/constant_potential`](../../../constant_potential/).

## Run the regression suite

Install the Python dependencies for this i-PI checkout, plus `pytest`, `ase`,
and ABACUS's `abacuslite` ASE interface from the ABACUS checkout used to
build the executables. Configure both ABACUS executables:

```bash
export ABACUS_CP_EXECUTABLE=/path/to/modified/abacus
export ABACUS_REFERENCE_EXECUTABLE=/path/to/original/abacus
export ABACUS_PP_ORB=/path/to/abacus-source/tests/PP_ORB
bash tests/run_develop_core.sh /path/to/new-test-output
```

Both executables should use the same compiler, MPI library, and optional
features. For the current PR, use the e58 baseline; for the included develop
patch, use d200 and apply the patch only to the candidate.
Load their runtime libraries in the shell before running the tests. Use this
i-PI checkout for the server as well as the examples.

`run_develop_core.sh` currently selects **52 checks**: 46 core checks and six
PW/LCAO output-mode checks. These include Makov–Payne electronic-control
rejection, ordinary SCF/socket comparisons, and `out_app_flag=0/1/2` with
`out_freq_ion=2` on the current PR baseline.
They cover dynamic electron counts and positions,
static references, cube potentials, ordinary SCF/MD/relax and socket sessions,
native implicit solvent, k-point pools, ASE, i-PI trajectories, and restart.
The default is four MPI ranks, needed for the parallel cases. The launcher
records executable hashes and treats skipped tests as failures. Logs and
JUnit results are saved in the new output directory.

To check dependencies and list the selected tests without starting a
calculation, append `--collect-only` and use a separate new output directory.
Collection requires the executable and data paths above. The full regression
suite, including additional occupation, electronic-method, and dynamics
checks, remains available through:

```bash
bash tests/run_acceptance.sh /path/to/full-test-output
```

For a focused run, invoke `python -m pytest tests/test_socket_driver.py` with
the same environment. Set `ABACUS_TEST_RANKS=1` for serial checks, or use
`ABACUS_TEST_RANKS=4` for the full suite's parallel cases. The reference
executable is a separate, unmodified ABACUS build.

## Compare the electronic-step refactor

Keep the executable and data paths above, with the current PR as the candidate
and unmodified e58 as the reference. Set the additional executable to a build
of PR commit `977f6b8770ec5f1c9e9721728a828ebf6fa7d717`, before this refactor:

```bash
export ABACUS_PREVIOUS_EXECUTABLE=/path/to/pre-refactor/abacus
export ABACUS_TEST_IPI_ROOT=/path/to/i-pi-constant-potential
export ABACUS_REQUIRE_REAL_TESTS=1 ABACUS_TEST_RANKS=4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -p no:cacheprovider -v \
  tests/test_socket_driver.py::test_electronic_step_matches_previous_binary \
  --basetemp=/path/to/new-before-after-work \
  --junitxml=/path/to/before-after.xml
```

The four PW/LCAO, solvent-on/off cases compare five frames each, including a
coordinate change at unchanged electron count. The local validation passed
all four cases: the 20 frame pairs had zero differences in energy, forces,
occupations, Fermi level, reference potential, and work function. Reference
potential sampling stays after force evaluation, preserving the electric-field
state used for forces. These four checks are separate from the 52 core checks.

## Focused C++ checks

After configuring ABACUS with MPI and `BUILD_TESTING=ON`, build and run the
electronic-control tests using their current module names:

```bash
cmake --build build --target MODULE_ESTATE_electronic_occupations_test \
  MODULE_ESTATE_electrostatic_potential_test MODULE_RELAX_socket_driver_test \
  MODULE_RELAX_socket_constant_potential_test
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure \
  -R 'electronic_occupations|electrostatic_potential|MODULE_RELAX_socket_(driver|constant_potential)'
```

This selects six CTest targets, including MPI 1/4 occupation and potential
averaging tests. The socket tests cover request validation before SCF and
electron updates across consecutive frames.
