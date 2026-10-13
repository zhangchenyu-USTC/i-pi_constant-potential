# ABACUS constant-potential examples and tests

This directory contains the Si2 PW/LCAO examples, the Al36 electrode
examples, and the real-client regression suite. ABACUS keeps the focused
C++ unit tests alongside its source.

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
features. For the develop patch, build both from the baseline recorded in
the [client README](../README.md), applying the patch only to the candidate.
Load their runtime libraries in the shell before running the tests. Use this
i-PI checkout for the server as well as the examples.

`run_develop_core.sh` selects the **38 checks** used to validate the develop
patch after upstream #8119. They cover dynamic electron counts and positions,
static references, cube potentials, ordinary SCF/MD/relax and socket sessions,
native implicit solvent, k-point pools, ASE, i-PI trajectories, and restart.
The default is four MPI ranks, needed for the parallel cases. The launcher
records executable hashes and treats skipped tests as failures. Logs and
JUnit results are saved in the new output directory.

To check dependencies and list the same 38 tests without starting a
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
