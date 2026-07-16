# A Flexible and Generalized Constant-Potential Framework in i-PI

This project is a constant-potential molecular dynamics extension developed on
top of [i-PI 3.3.0](https://github.com/i-pi/i-pi). It retains i-PI's
server-client architecture while introducing a system-level electronic degree
of freedom in addition to the nuclear degrees of freedom. This allows the
electron number to evolve dynamically during molecular dynamics toward a
target Fermi level or target work function.

The current version supports constant-potential simulations through i-PI with
the following first-principles clients:

- VASP 5.4.4 + vaspsol++;
- CP2K 2026.1;
- ABACUS 3.10.0 (PW and LCAO).

The constant-potential patches for these clients are located at:

- `examples/clients/vasp/vasp-5.4.4-ipi-constant-potential.patch`
- `examples/clients/cp2k/cp2k-2026.1-ipi-constant-potential.patch`
- `examples/clients/abacus/abacus-3.10.0-ipi-constant-potential.patch`

Examples are provided in `examples/constant_potential/`.

Related publication:

> Chenyu Zhang, Ruoting Zhao, Zhengda He, Marcella Iannuzzi, Yanxia Chen, and Jinggang Lan,
> *A Flexible and Generalized Constant-Potential Framework in i-PI*,
> Journal of Chemical Theory and Computation (2026),
> [https://doi.org/10.1021/acs.jctc.6c00504](https://doi.org/10.1021/acs.jctc.6c00504).

## Core Concept

i-PI normally propagates the nuclear coordinates, while an external electronic
structure program computes energies and atomic forces. This project extends
that division of responsibilities by introducing a total-electron-number
coordinate `q` and its conjugate momentum `p`. A newly added potentiostat
propagates this extended degree of freedom. For each configuration, the
external client returns the actual Fermi level, electron number, and, when
required in work-function mode, the work function. i-PI then evaluates the
generalized electronic force from the deviation from the target value and
updates the total electron number for the next step.

The constant-potential implementation does not dispatch by the names VASP,
CP2K, or ABACUS. i-PI sends the total electron number and mode flags through a
unified `CHGDATA` message and receives the following force extras:

```json
{
  "fermi_level_eV": -1.2,
  "nelect": 107.5,
  "workfunction_eV": 5.0
}
```

Two paths are available, depending on how the client supports electron-number
changes:

1. **Single-socket fractional-electron path**: for DFT clients that support
   fractional electron numbers. At every step, i-PI sends the current total
   electron number directly to one client.
2. **Dual-socket integer-endpoint mixing path**: for DFT clients that require
   integer electron numbers. Two clients evaluate adjacent integer-electron
   states, and i-PI interpolates the energy, forces, virial, Fermi level, and
   work function with the same linear weight.

For the Ne counter-electrode method, i-PI still sends the total electron number
of the whole system. The client performs the internal mapping using the neutral
electron count, current total electron count, and number of Ne atoms. The wire
protocol does not transmit an "electron-number change per Ne atom."

For implicit-solvent calculations, `solvation_update_stride` can treat the
solvent reaction field as a slow variable. The electron number is updated for
every configuration, whereas the complete implicit-solvent response is updated
only at the specified steps. i-PI coordinates the update/freeze schedule.

### Charge-Compensation Strategies

Changing the total electron number in a constant-potential simulation generally
leaves the electrode electron-ion subsystem with a net charge. Dielectric
polarization alone can only redistribute bound charge, whose integral is zero,
and therefore cannot by itself ensure charge neutrality of the full periodic
cell. The current examples use the following compensation strategies:

- **VASP + vaspsol++**: `2-vasp-implicit` uses `ISOL=2`, `C_MOLAR=1.0`, and
  `R_ION=4.0`. Mobile ions in the Poisson-Boltzmann continuum generate the
  compensating charge self-consistently. `C_MOLAR` specifies the bulk
  electrolyte concentration.
- **ABACUS**: the PW and LCAO examples do not use implicit solvation. They use
  `gate_flag=1`, dipole correction, and a blocking region, so that a planar gate
  charge compensates the charged electrode. The gate and blocking region must
  be placed in a physically appropriate atom-free region.
- **CP2K**: the implicit Poisson model used by the current CP2K implementation
  provides a dielectric response but does not include a mobile continuum-ion
  compensation model corresponding to the vaspsol++ `C_MOLAR`/`R_ION` model.

The Ne counter-electrode method provides another compensation path. The current
implementation updates the system-wide total electron number and the associated
Ne counter-electrode mapping at every MD step. Its numerical stability depends
on the fictitious electronic mass, potentiostat time constant, SCF convergence,
work-function averaging window, and the number and positions of the Ne atoms.
Convergence must therefore be tested for each system. If necessary, a future
implementation may update the electron number only once every several steps.

## Main Features

- Langevin and stochastic velocity rescaling (SVR) electronic thermostats;
- single-socket fractional-electron and dual-socket integer-endpoint mixing;
- Ne counter-electrode support with VASP and CP2K;
- planar-averaged work functions computed by the DFT client directly from the
  in-memory electrostatic potential along the x, y, or z lattice direction,
  including systems whose vacuum region is not along z;
- implicit-solvent slow-variable updates with
  `solvation_update_stride>1` in VASP and ABACUS;
- NVE, NVT, NPT, and NST ensembles;
- one shared system-level electron number for multiple beads, with the
  generalized electronic force obtained by averaging the chemical-potential or
  work-function deviations over the beads;
- `inet` and `unix` sockets, including the consolidated and non-consolidated
  message paths introduced in i-PI 3.3.0;
- unchanged upstream i-PI behavior when `<electrons>` is not enabled.

## Installation and Usage

### 1. Install the Constant-Potential i-PI Version

i-PI is a Python server and does not need to be compiled. A recent Python and
NumPy installation is recommended. From the project root, run:

```bash
source env.sh
i-pi --help
```

Alternatively, install the project in editable mode in an isolated Python
environment:

```bash
python -m pip install -e .
```

This project is based on i-PI 3.3.0. See the
[i-PI documentation](https://docs.ipi-code.org/) for the standard i-PI
installation, testing, and usage instructions. Constant-potential simulations
require this project together with matching client patches; do not combine this
modified i-PI server with unmodified DFT clients.

### 2. Supported Client Versions and Patches

| Client | Supported source version | Constant-potential patch | Main capabilities |
|---|---|---|---|
| VASP | 5.4.4 + vaspsol++ | `examples/clients/vasp/vasp-5.4.4-ipi-constant-potential.patch` | Fractional electrons, Ne, implicit electrolyte and continuum-ion compensation, x/y/z work functions |
| CP2K | 2026.1 | `examples/clients/cp2k/cp2k-2026.1-ipi-constant-potential.patch` | Integer-endpoint mixing, OT, Ne, implicit Poisson (without continuum-ion compensation) |
| ABACUS | LTSv3.10.0 | `examples/clients/abacus/abacus-3.10.0-ipi-constant-potential.patch` | PW/LCAO, fractional electrons, planar gate compensation, implicit solvent, x/y/z work functions |

The patches were compiled and tested against the source versions listed above.
Other versions have not been tested and require a new interface audit,
compilation, and numerical validation.

#### VASP 5.4.4 + vaspsol++

Before applying the VASP constant-potential patch, prepare vaspsol++ following
the instructions in the
[VASPsol project](https://github.com/VASPsol/VASPsol/) and use the official
`solvation_intel.F` compatible with VASP 5.4.4 and your compiler. This project
does not distribute `vaspsol++-vasp_5.4.4.patch` or `solvation_intel.F`.

The recommended installation order is:

```bash
cd /path/to/vasp.5.4.4

# 1. Check and apply the vaspsol++ patch
patch --dry-run -p1 < /path/to/vaspsol++-vasp_5.4.4.patch
patch -p1 < /path/to/vaspsol++-vasp_5.4.4.patch

# 2. Replace solvation_intel.F
cp /path/to/solvation_intel.F src/solvation.F

# 3. Check and apply this project's combined patch
patch --dry-run -p1 < /path/to/vasp-5.4.4-ipi-constant-potential.patch
patch -p1 < /path/to/vasp-5.4.4-ipi-constant-potential.patch
```

Then rebuild `vasp_std`, `vasp_gam`, or `vasp_ncl` following the normal VASP
5.4.4 build procedure. This project's VASP patch already incorporates the
original `examples/clients/vasp/vasp.5.4.4-ipi.patch` and adds the `IBRION=24`
constant-potential state machine, `CHGDATA`, fractional electron numbers, Ne
mapping, in-memory work functions, and vaspsol++ slow-variable caching.
**Do not apply `vasp.5.4.4-ipi.patch` separately.**

Standard VASP-i-PI simulations continue to use `IBRION=23`; the
constant-potential examples in this project use `IBRION=24`. Implicit-solvent
settings must also be enabled explicitly as required by vaspsol++; the client
patch does not turn on the solvent model automatically.

The release example `examples/constant_potential/2-vasp-implicit/` uses the
nonlinear, nonlocal vaspsol++ implicit-electrolyte model and enables mobile-ion
compensation with `C_MOLAR=1.0` and `R_ION=4.0`. Its
`solvation_update_stride=5` setting demonstrates slow solvent updates. Set the
stride to `1` if continuum ions must strictly compensate every electron-number
change at every step.

#### CP2K 2026.1

Apply the CP2K patch directly to a clean CP2K 2026.1 source tree:

```bash
cd /path/to/cp2k-2026.1
patch --dry-run -p1 < /path/to/cp2k-2026.1-ipi-constant-potential.patch
patch -p1 < /path/to/cp2k-2026.1-ipi-constant-potential.patch
```

Then rebuild CP2K with its standard toolchain or CMake workflow. The current
CP2K implicit Poisson path does not provide mobile continuum-ion compensation.

#### ABACUS 3.10.0

Apply the constant-potential patch directly to a clean ABACUS `LTSv3.10.0`
source tree:

```bash
cd /path/to/abacus-LTSv3.10.0
patch --dry-run -p1 < /path/to/abacus-3.10.0-ipi-constant-potential.patch
patch -p1 < /path/to/abacus-3.10.0-ipi-constant-potential.patch
```

Then rebuild ABACUS using its normal build procedure. The constant-potential
patch supports PW and LCAO, work-function extras, and implicit-solvent
update/freeze operations.

The released PW/LCAO constant-potential examples use ABACUS's built-in planar
gate charge rather than implicit-solvent compensation. They set `imp_sol=0` and
enable `efield_flag`, `dip_cor_flag`, `gate_flag`, and `block`. These parameters
depend on the cell orientation and the location of the vacuum region; when
adapting the examples to another structure, their fractional positions must be
set again rather than copied unchanged.

In addition,
`examples/clients/abacus/abacus-3.10.0-ipi.patch` provides a standard
i-PI-ABACUS interface **without constant-potential functionality**. It
implements only a conventional i-PI force driver and neither recognizes nor
communicates `CHGDATA`. The corresponding standard PW example is located in
`examples/clients/abacus/pw-example/`.

The standard and constant-potential patches are mutually exclusive alternatives.
Apply only one of them to a clean ABACUS 3.10.0 source tree; do not stack the
patches. When using the standard patch, the i-PI input must not contain an
`<electrons>` section.

### 3. Prepare a Constant-Potential Input

Constant-potential settings are placed under
`<motion><dynamics><electrons>`. A representative input is:

```xml
<electrons enabled="true">
  <target_workfunction units="electronvolt">4.6</target_workfunction>
  <potential_average_axis>z</potential_average_axis>
  <potential_average_region units="angstrom">[12.5, 15.0]</potential_average_region>
  <neutral_electrons>108</neutral_electrons>
  <q_init>107.5</q_init>
  <mass units="atomic_unit">400.0</mass>
  <solvation_update_stride>1</solvation_update_stride>
  <potentiostat mode="svr">
    <tau units="femtosecond">200.0</tau>
    <temp units="kelvin">300.0</temp>
  </potentiostat>
</electrons>
```

Key parameters:

| Parameter | Purpose |
|---|---|
| `target_workfunction` | Target in constant-work-function mode; requires a potential-averaging direction and interval |
| `neutral_electrons` | Total electron number of the neutral system; required when constant-potential dynamics is enabled |
| `q_init`, `p_init` | Initial total electron number and its conjugate momentum |
| `mass` | Fictitious generalized mass of the electron-number coordinate, not the physical electron mass |
| `potential_average_axis` | Plane normal for the work-function reference potential: `x`, `y`, or `z` |
| `potential_average_region` | Electrostatic-potential averaging interval along the selected direction |
| `solvation_update_stride` | Update interval for the complete implicit-solvent slow variable; default: 1 |
| `charge_mixing` | Enables dual-socket integer-endpoint mixing |
| `Ne_doping` | Enables the Ne counter-electrode mode; mutually exclusive with `charge_mixing` |
| `workfunction_average_steps` | Moving-average window for the work function in Ne mode |
| `potentiostat` | Electronic thermostat: `langevin` or `svr` |

Dual-socket mode requires two `ffsocket` definitions in the simulation. Under
`<forces>`, reference either one of the sockets as the logical force. An
internal i-PI wrapper requests both endpoints. Do not add both the low and high
forces to `<forces>`, because doing so would apply the forces twice.

### 4. Run the Examples

`examples/constant_potential/` contains:

| Directory | Description |
|---|---|
| `1-pes-fermi` | Constant-Fermi-level test using an analytic asymmetric double-well potential |
| `2-vasp-implicit` | VASP + vaspsol++ constant-work-function simulation with implicit-electrolyte mobile-ion compensation |
| `3-vasp-Ne` | VASP single-socket Ne counter-electrode constant-work-function simulation |
| `4-cp2k-implicit-mixing` | CP2K FARMING dual-integer-endpoint implicit-solvent mixing without mobile-ion compensation |
| `5-cp2k-Ne` | CP2K single-socket Ne counter-electrode constant-work-function simulation |
| `6-abacus-pw-compensating-charge` | ABACUS PW constant-work-function simulation with planar gate compensation |
| `7-abacus-lcao-compensating-charge` | ABACUS LCAO constant-work-function simulation with planar gate compensation |

These examples demonstrate typical constant-potential settings and can be used
as starting points for production calculations. Users must adapt executable
paths, MPI process counts, pseudopotentials, orbital files, and batch-system
settings to their environment. A typical launch sequence is:

```bash
source /path/to/i-pi-3.3.0/env.sh
cd /path/to/copied-example
i-pi input.xml > ipi.log 2>&1 &

# Wait for i-PI to create the socket, then start the matching VASP, CP2K,
# or ABACUS client. The sbatch.slurm files are templates for cluster jobs.
```

## Current Limitations and Important Notes

1. **CP2K implicit Poisson requires `solvation_update_stride=1`.** CP2K does
   not expose the complete implicit-solvent reaction potential separately from
   the normal Hartree/electrostatic state as an independently cacheable and
   replayable object. It is therefore unsafe to freeze the full solvent
   potential, solvent energy, and solvent forces while the geometry and
   electron number change. The client terminates explicitly if the stride is
   greater than 1.
2. **CP2K implicit Poisson does not provide mobile continuum-ion
   compensation.** It describes dielectric polarization but cannot generate an
   integrated ionic distribution that cancels the solute net charge in the way
   provided by vaspsol++ through `C_MOLAR`/`R_ION`.
3. **The Ne counter-electrode path requires system-specific convergence
   testing.** The current implementation updates the electron number and the Ne
   counter-electrode mapping at every step. Convergence with respect to the
   fictitious electronic mass, potentiostat time constant, time step, and other
   relevant parameters must be checked systematically.
4. **ABACUS does not currently support the Ne counter-electrode mode.** The
   ABACUS constant-potential client hard-fails when it receives the Ne mode bit;
   it never silently ignores the flag.
5. **Constant-potential sockets do not currently support SHM, FFMPI, or
   `batch_size>1`.** The supported combinations are `inet`/`unix` with
   `batch_size=1`; both consolidated and non-consolidated socket message paths
   are supported.
6. **The supported ensembles are NVE, NVT, NPT, and NST.** NVT-CC, SC, SCNPT,
   and related modes reject `<electrons>` during input validation.
7. **CP2K requires integer electron numbers.** Continuous electron-number
   dynamics must use dual-socket mixing and must not rely on silent client-side
   rounding. VASP and ABACUS support single-socket fractional electron numbers.
8. **`solvation_update_stride>1` is a multiple-time-step approximation for the
   slow solvent field.** Nuclear coordinates and the electron number still
   change during frozen steps, while VASP/ABACUS reuse the complete solvent
   response from the preceding update step. Check convergence of energies,
   forces, and sampling with respect to the stride. Use
   `solvation_update_stride=1` when VASP continuum ions must compensate every
   electron-number change.
9. **The work-function averaging interval must lie in a physically meaningful
   plateau.** Inspect the planar-averaged electrostatic potential to validate
   the selected direction and `[min,max]` interval; successful input parsing
   alone does not make the work-function reference physically valid.
10. **Clients must return real, converged data.** An unconverged SCF calculation,
    missing extras, NaN/Inf values, an inconsistent electron number, or a
    missing cache terminates a constant-potential simulation. The code never
    enables dummy behavior automatically.
11. **Client patches are version-specific.** The constant-potential patches
    depend on the listed source versions. Other versions have not been tested
    and require a new interface audit, compilation, and numerical validation.
12. **CP2K FARMING can encounter an exit-cleanup assertion.** In some build
    environments, an offload mempool cleanup assertion can occur after all
    endpoints have already terminated normally.

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
