# A Flexible and Generalized Constant-Potential Framework in i-PI

This project extends [i-PI 3.3.0](https://github.com/i-pi/i-pi) with a shared
electron-number coordinate and a potentiostat. An electronic-structure client
computes energy, atomic forces, the Fermi level, and optionally the work
function; i-PI propagates the nuclei and electron number toward the target
potential.

The `abacus-develop-validation` branch publishes the **ABACUS PW/LCAO
constant-potential integration**, its matching i-PI server, and runnable
examples. Start with the [ABACUS client instructions](examples/clients/abacus/README.md)
and [examples and verification suite](examples/clients/abacus/validation/README.md).
The VASP and CP2K release remains on
[`main`](https://github.com/zhangchenyu-USTC/i-pi_constant-potential/tree/main).
Its client patches use that release's protocol and are not paired with this
branch's server.

Related publication:

> Chenyu Zhang, Ruoting Zhao, Zhengda He, Marcella Iannuzzi, Yanxia Chen, and Jinggang Lan,
> *A Flexible and Generalized Constant-Potential Framework in i-PI*,
> Journal of Chemical Theory and Computation (2026),
> [doi:10.1021/acs.jctc.6c00504](https://doi.org/10.1021/acs.jctc.6c00504).

## Install and run

Install this checkout in a Python environment:

```bash
python -m pip install -e .
```

Build ABACUS develop using the
[constant-potential patch](examples/clients/abacus/abacus-develop-constant-potential.patch).
Apply it to the clean commit listed in the
[client README](examples/clients/abacus/README.md), then follow ABACUS's normal
build instructions. The baseline includes the upstream AO interpolation
fix #8119. Its PCC contribution is included in both electrostatic cube output
and the potential used for work-function feedback.

Run a small example directly on a node:

```bash
export ABACUS_CP_EXECUTABLE=/path/to/modified/abacus
export ABACUS_PP_ORB=/path/to/abacus-source/tests/PP_ORB
cd examples/clients/abacus/validation
python run_example.py si2-pw --output /path/to/new-run --ranks 4 --steps 3
```

The launcher starts i-PI and ABACUS, waits for both, and stores their outputs
in the new directory. Use `si2-lcao`, `al36-pw`, or `al36-lcao` for the other
cases. Slurm is not required. Larger gate-compensated electrode inputs are in
[`6-abacus-pw-compensating-charge`](examples/constant_potential/6-abacus-pw-compensating-charge/)
and [`7-abacus-lcao-compensating-charge`](examples/constant_potential/7-abacus-lcao-compensating-charge/).

## Electron control and communication

ABACUS uses a single socket and accepts fractional total electron numbers.
i-PI sends the current number through `CHGDATA` before each `POSDATA`; ABACUS
updates occupations and charge-dependent potentials, then performs SCF at that
number. It returns the converged energy and forces together with extras such as:

```json
{
  "fermi_level_eV": -1.2,
  "nelect": 107.5,
  "workfunction_eV": 5.0
}
```

`nelect` is the actual occupation sum. Both programs check it against the
request with an absolute tolerance of `1e-8`. The work function is the selected
electrostatic-potential average minus the current Fermi level. The reference
potential is evaluated in memory using the same construction as cube output.

The ABACUS constant-potential path uses fixed-cell CPU double-precision
norm-conserving PW/LCAO KSDFT, `calculation=scf`, `socket_driver=1`,
`cal_force=1`, `cal_stress=0`, and `nspin=1` or `2`. Socket calculations default
to `symmetry=0`; an explicit `symmetry=-1` is also supported. Connect with
`ABACUS_SOCKET_ADDRESS=/tmp/ipi_<address>:UNIX` or `host:port`.

`CHGDATA` contains a float64 total electron number and int32 flags. ABACUS
uses flags `0`. i-PI no longer sends a solvent-update schedule; the removed
`solvation_update_stride` XML/INIT field and old flags `1`/`3` are rejected.
Native ABACUS implicit solvation updates through its usual SCF path. The
supplied electrode examples use the native planar gate, dipole correction,
and blocking region for compensation, with `imp_sol=0`.

## Configure constant-potential dynamics

Place the electronic settings under `<motion><dynamics>`:

```xml
<electrons enabled="true" charge_mixing="false">
  <target_workfunction units="electronvolt">4.0</target_workfunction>
  <potential_average_axis>z</potential_average_axis>
  <potential_average_region units="angstrom">[2, 4]</potential_average_region>
  <neutral_electrons>8</neutral_electrons>
  <q_init>8.05</q_init>
  <mass units="dalton">50</mass>
  <potentiostat mode="langevin">
    <tau units="femtosecond">300</tau>
  </potentiostat>
</electrons>
```

These values follow the small Si2 example. Set the neutral count, starting
count, target, and reference interval for the chosen structure. Use
`target_fermi_level` instead of the work-function target and reference window
for constant-Fermi-level dynamics. The electronic mass is a fictitious
dynamical parameter.

Langevin and SVR electronic thermostats are available. Omitting
`potentiostat/temp` makes the electronic temperature follow the physical
ensemble temperature continuously; an explicit value remains independent.

All beads share one electron number and electronic momentum. The electronic
feedback is the arithmetic mean over bead Fermi levels or work functions.
Each bead is checked against the same requested electron count before its
feedback is accepted. The common server also retains integer-endpoint mixing;
the ABACUS examples use direct fractional-electron SCF.

Each system has one electronic-feedback force component, covering every bead
without contraction and with unit effective weight. Other charge-independent
force components retain their own weights. Each system uses its own
constant-potential backend and connection pool. Returned forces are restored
to full atomic order, including for reordered or partial `active` selections.

## Requests, trajectories, and restart

Electronic metadata, geometry, and cell form one immutable request snapshot
before the request enters a worker queue. Energy, full nuclear force, virial,
and electronic feedback are checked together before use. Failed evaluations
wake waiting consumers and propagate their original error.

The integrator prepares initial feedback before propagation. Outputs and
repeated reads reuse that result: a continuous single-endpoint run of `N`
steps requires `N+1` evaluations per bead. Window samples count accepted
physical frames, starting with frame 0.

Constant-potential checkpoints use `state_format="checkpoint_v1"`. They save
feedback initialization, the accepted frame, history, random state, electronic
momentum, and thermostat/protocol work, with at least 17 significant digits
for physical floating-point state. Restart recomputes the saved boundary and
replaces that frame's feedback sample before advancing. Checkpoints without
the required lifecycle fields are rejected. Normal new inputs use the default
`state_format="auto"`; explicit inline initial states can use `initial`.

## Verification

ABACUS keeps its focused C++ protocol, state, occupation, and INPUT tests. This
repository contains the real-client cases and Python verification suite.
With matched modified and unmodified ABACUS builds, run:

```bash
export ABACUS_CP_EXECUTABLE=/path/to/modified/abacus
export ABACUS_REFERENCE_EXECUTABLE=/path/to/original/abacus
export ABACUS_PP_ORB=/path/to/abacus-source/tests/PP_ORB
cd examples/clients/abacus/validation
bash tests/run_develop_core.sh /path/to/new-test-output
```

The develop-core selection contains the 38 checks used for electron-count
changes, static references, cube potentials, ordinary calculations and socket
sessions, ASE, trajectories, and restart. The full suite remains available
through `tests/run_acceptance.sh`. Dependencies and collection-only commands
are documented in the [validation README](examples/clients/abacus/validation/README.md).

## Citation

If you use this project in your research, please cite both the
constant-potential framework paper and the i-PI 3.0 paper:

```bibtex
@article{zhang2026flexible,
  title   = {A Flexible and Generalized Constant-Potential Framework in i-PI},
  author  = {Zhang, Chenyu and Zhao, Ruoting and He, Zhengda and Iannuzzi, Marcella and Chen, Yanxia and Lan, Jinggang},
  journal = {Journal of Chemical Theory and Computation},
  year    = {2026},
  doi     = {10.1021/acs.jctc.6c00504}
}

@article{litman2024ipi,
  title   = {i-PI 3.0: a flexible and efficient framework for advanced atomistic simulations},
  author  = {Litman, Yair and Kapil, Venkat and Feldman, Yotam M. Y. and Tisi, Davide and Begušić, Tomislav and others},
  journal = {The Journal of Chemical Physics},
  volume  = {161},
  pages   = {062505},
  year    = {2024},
  doi     = {10.1063/5.0215869}
}
```
