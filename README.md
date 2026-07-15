# A Flexible and Generalized Constant-Potential Framework in i-PI

This project is a constant-potential molecular dynamics extension developed on top of
[i-PI 3.3.0](https://github.com/i-pi/i-pi). It preserves the i-PI server-client
architecture and introduces a system-level electronic degree of freedom in addition to
the nuclear degrees of freedom, allowing the electron number to evolve dynamically in
response to a target Fermi level or work function.

The current version supports constant-potential simulations through i-PI with the
following first-principles clients:

- VASP 5.4.4 + vaspsol++;
- CP2K 2026.1;
- ABACUS 3.10.0 (PW and LCAO).

The constant-potential patches for these clients are provided at:

- `examples/clients/vasp/vasp-5.4.4-ipi-constant-potential.patch`
- `examples/clients/cp2k/cp2k-2026.1-ipi-constant-potential.patch`
- `examples/clients/abacus/abacus-3.10.0-ipi-constant-potential.patch`

Example inputs are available in `examples/constant_potential/`.

Related publication:

> Chenyu Zhang, Ruoting Zhao, Zhengda He, Marcella Iannuzzi, Yanxia Chen, and Jinggang Lan,
> *A Flexible and Generalized Constant-Potential Framework in i-PI*,
> Journal of Chemical Theory and Computation (2026),
> [https://doi.org/10.1021/acs.jctc.6c00504](https://doi.org/10.1021/acs.jctc.6c00504).

## Main idea

i-PI normally propagates the nuclear coordinates, while an external electronic-structure
program computes the energy and atomic forces. This project extends that division of
labor by introducing the total electron-number coordinate `q` and its conjugate momentum
`p`. A new potentiostat propagates this extended degree of freedom. At every
configuration, the external client returns the actual Fermi level and electron number,
together with the work function when work-function control is used. i-PI evaluates the
generalized electronic force from the deviation from the target and updates the total
electron number for the next step.

The constant-potential implementation does not dispatch on client names such as VASP,
CP2K, or ABACUS. i-PI sends the total electron number and mode flags through the unified
`CHGDATA` message and receives the following force extras:

```json
{
  "fermi_level_eV": -1.2,
  "nelect": 107.5,
  "workfunction_eV": 5.0
}
```

Two paths are available, depending on the electron-number capabilities of the client:

1. **Single-socket fractional-electron path**: for DFT clients that support fractional
   electron numbers. i-PI sends the current total electron number directly to one client
   at every step.
2. **Two-socket integer-endpoint mixing path**: for DFT clients that require integer
   electron numbers. Two clients evaluate neighboring integer-electron states, and i-PI
   applies the same linear-interpolation weight to the energy, forces, virial, Fermi
   level, and work function.

In the Ne counter-electrode scheme, i-PI still sends the total electron number of the
system. The client performs its internal mapping using the neutral electron number, the
current total electron number, and the number of Ne atoms. The wire protocol does not
send an "electron-number change per Ne atom."

Implicit-solvent calculations can use `solvation_update_stride` to treat the solvent
reaction field as a slow variable. The electron number is updated at every configuration,
whereas the complete implicit-solvent response is updated only at the specified steps.
i-PI schedules the update/freeze state consistently.

## Main features

- Langevin and stochastic velocity rescaling (SVR) potentiostats;
- single-socket fractional-electron and two-socket integer-endpoint interpolation paths;
- Ne counter-electrode support in VASP and CP2K;
- in-memory planar-averaged work functions along the x, y, or z lattice direction,
  including cells whose vacuum direction is not z;
- implicit-solvent slow-variable updates with `solvation_update_stride>1` in VASP and
  ABACUS;
- NVE, NVT, NPT, and NST ensembles;
- one system-level shared electron number for multiple beads, with the generalized
  electronic force obtained by averaging the chemical-potential or work-function
  deviations over the beads;
- `inet` and `unix` sockets, including the consolidated and non-consolidated message paths
  introduced in i-PI 3.3.0;
- unchanged upstream behavior when `<electrons>` is not enabled.

## Installation and usage

### 1. Install the constant-potential i-PI version

i-PI is a Python server and does not itself require compilation. A recent version of
Python and NumPy is recommended. From the source root, run:

```bash
source env.sh
i-pi --help
```

Alternatively, install it in editable mode in an isolated Python environment:

```bash
python -m pip install -e .
```

This project is based on i-PI 3.3.0. See the
[i-PI documentation](https://docs.ipi-code.org/) for standard installation, testing,
and usage instructions. Constant-potential calculations require this version of i-PI
and a matching patched client; do not use this constant-potential i-PI implementation
with an unmodified DFT client.

### 2. Supported client versions and patches

| Client | Supported source version | Constant-potential patch | Main capabilities |
|---|---|---|---|
| VASP | 5.4.4 + vaspsol++ | `examples/clients/vasp/vasp-5.4.4-ipi-constant-potential.patch` | Fractional electrons, Ne, implicit solvent, x/y/z work function |
| CP2K | 2026.1 | `examples/clients/cp2k/cp2k-2026.1-ipi-constant-potential.patch` | Integer-endpoint mixing, OT, Ne, implicit Poisson |
| ABACUS | LTSv3.10.0 | `examples/clients/abacus/abacus-3.10.0-ipi-constant-potential.patch` | PW/LCAO, fractional electrons, implicit solvent, x/y/z work function |

The patches have been compiled and tested only against the source versions listed above.
Other versions have not been validated and require a fresh interface audit, compilation,
and numerical verification.

#### VASP 5.4.4 + vaspsol++

Before applying the VASP constant-potential patch, prepare vaspsol++ according to the
[VASPsol project](https://github.com/VASPsol/VASPsol/) instructions and use the official
`solvation_intel.F` appropriate for VASP 5.4.4 and the selected compiler. This project
does not distribute `vaspsol++-vasp_5.4.4.patch` or `solvation_intel.F`.

The recommended order is:

```bash
cd /path/to/vasp.5.4.4

# 1. Check and apply the vaspsol++ patch
patch --dry-run -p1 < /path/to/vaspsol++-vasp_5.4.4.patch
patch -p1 < /path/to/vaspsol++-vasp_5.4.4.patch

# 2. Replace src/solvation.F with the official solvation_intel.F
cp /path/to/solvation_intel.F src/solvation.F

# 3. Check and apply the combined i-PI/constant-potential patch
patch --dry-run -p1 < /path/to/vasp-5.4.4-ipi-constant-potential.patch
patch -p1 < /path/to/vasp-5.4.4-ipi-constant-potential.patch
```

Then rebuild `vasp_std`, `vasp_gam`, or `vasp_ncl` following the normal VASP 5.4.4
procedure. The VASP patch provided by this project already incorporates the original
`examples/clients/vasp/vasp.5.4.4-ipi.patch` and additionally implements the `IBRION=24`
constant-potential state machine, `CHGDATA`, fractional electrons, Ne mapping, in-memory
work functions, and the vaspsol++ slow-variable cache. **Do not apply
`vasp.5.4.4-ipi.patch` separately.**

The regular VASP i-PI mode continues to use `IBRION=23`, whereas the constant-potential
examples use `IBRION=24`. The implicit-solvent options must still be enabled explicitly
according to the vaspsol++ input requirements; the client patch does not enable the
solvent model automatically.

#### CP2K 2026.1

Apply the CP2K patch directly to a clean CP2K 2026.1 source tree:

```bash
cd /path/to/cp2k-2026.1
patch --dry-run -p1 < /path/to/cp2k-2026.1-ipi-constant-potential.patch
patch -p1 < /path/to/cp2k-2026.1-ipi-constant-potential.patch
```

Then rebuild CP2K using its standard toolchain or CMake procedure.

#### ABACUS 3.10.0

Apply the constant-potential patch directly to a clean ABACUS `LTSv3.10.0` source tree:

```bash
cd /path/to/abacus-LTSv3.10.0
patch --dry-run -p1 < /path/to/abacus-3.10.0-ipi-constant-potential.patch
patch -p1 < /path/to/abacus-3.10.0-ipi-constant-potential.patch
```

Then rebuild ABACUS following its normal build procedure. The constant-potential patch
supports PW and LCAO, work-function extras, and implicit-solvent update/freeze behavior.

In addition, `examples/clients/abacus/abacus-3.10.0-ipi.patch` provides a **standard
i-PI-ABACUS interface without constant-potential functionality**. It implements only the
regular i-PI force driver and neither recognizes nor communicates `CHGDATA`. A standard
PW example is provided in `examples/clients/abacus/pw-example/`.

The standard and constant-potential patches are mutually exclusive alternatives. Apply
exactly one of them to a clean ABACUS 3.10.0 source tree; do not apply them on top of one
another. Inputs used with the standard patch must not contain an `<electrons>` section.

### 3. Prepare a constant-potential input

Constant-potential settings are placed in `<motion><dynamics><electrons>`. The input
structure is:

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
| `target_workfunction` | Target work function; requires a potential-averaging direction and region |
| `neutral_electrons` | Total electron number of the neutral system; required whenever constant-potential dynamics is enabled |
| `q_init`, `p_init` | Initial total electron number and its conjugate momentum |
| `mass` | Generalized mass of the electron-number coordinate, not the physical electron mass |
| `potential_average_axis` | Lattice direction normal to the potential-averaging plane: `x`, `y`, or `z` |
| `potential_average_region` | Electrostatic-potential averaging interval along the selected direction |
| `solvation_update_stride` | Update interval for the complete implicit-solvent slow state; default: 1 |
| `charge_mixing` | Enable two-socket integer-endpoint mixing |
| `Ne_doping` | Enable the Ne counter-electrode mode; mutually exclusive with `charge_mixing` |
| `workfunction_average_steps` | Moving-average window for the work function in Ne mode |
| `potentiostat` | Electronic thermostat: `langevin` or `svr` |

Two-socket mode requires two `ffsocket` definitions in the simulation. Reference either
one of them in `<forces>` as the logical force; the internal i-PI wrapper requests both
endpoints. Do not add both the low and high forces to `<forces>`, because doing so defines
an erroneous double-counted force setup.

### 4. Run the examples

`examples/constant_potential/` contains:

| Directory | Description |
|---|---|
| `1-pes-fermi` | Constant-Fermi-level test with an analytical asymmetric double well |
| `2-vasp-implicit` | VASP + vaspsol++ implicit-solvent constant-work-function simulation |
| `3-vasp-Ne` | VASP single-socket Ne counter-electrode constant-work-function simulation |
| `4-cp2k-implicit-mixing` | CP2K FARMING two-integer-endpoint implicit-solvent mixing |
| `5-cp2k-Ne` | CP2K single-socket Ne counter-electrode constant-work-function simulation |
| `6-abacus-pw-implicit` | ABACUS PW implicit-solvent constant-work-function simulation |
| `7-abacus-lcao-implicit` | ABACUS LCAO implicit-solvent constant-work-function simulation |

These examples provide representative constant-potential setups and can be used as
starting points. Adapt executable paths, MPI process counts, pseudopotentials, orbital
files, and scheduler options to the local environment. A typical launch sequence is:

```bash
source /path/to/i-pi-3.3.0/env.sh
cd /path/to/copied-example
i-pi input.xml > ipi.log 2>&1 &

# Wait for i-PI to create the socket, then launch the corresponding VASP,
# CP2K, or ABACUS client. The supplied sbatch.slurm files are cluster-job templates.
```

## Important notes and current limitations

1. **CP2K implicit Poisson requires `solvation_update_stride=1`.** CP2K does not currently
   separate the complete implicit-solvent reaction potential from the ordinary
   Hartree/electrostatic state into independently cacheable and replayable objects.
   Consequently, it cannot safely freeze the complete solvent potential, solvent energy,
   and solvent forces while the geometry and electron number change. The client aborts
   explicitly when a value greater than 1 is requested.
2. **ABACUS does not yet support the Ne counter-electrode mode.** The ABACUS
   constant-potential client hard fails when it receives the Ne mode bit; it never ignores
   the flag or silently falls back to another mode.
3. **Constant-potential sockets do not yet support SHM, FFMPI, or `batch_size>1`.** The
   supported transports are `inet` and `unix` with `batch_size=1`. Both consolidated and
   non-consolidated socket message paths are supported.
4. **The supported ensembles are NVE, NVT, NPT, and NST.** NVT-CC, SC, SCNPT, and other
   unsupported modes are rejected before running when `<electrons>` is enabled.
5. **CP2K requires integer electron numbers.** Continuous-electron-number simulations
   should use two-socket mixing and must not rely on silent rounding by the client. VASP
   and ABACUS support single-socket fractional electrons.
6. **`solvation_update_stride>1` is an MTS approximation for the slow solvent field.** The
   atomic coordinates and electron number continue to change on frozen steps. VASP and
   ABACUS reuse the complete solvent response from the most recent update. Check the
   convergence of energies, forces, and sampled observables with respect to the stride
   before using a large value.
7. **The work-function averaging region must lie in a physically meaningful plateau.**
   Inspect the planar-averaged electrostatic potential and verify the selected direction
   and `[min,max]` interval; successful input parsing alone does not validate the physical
   work-function reference.
8. **Clients must return real, converged data.** Unconverged SCF calculations, missing
   extras, NaN/Inf values, inconsistent electron numbers, or missing caches terminate a
   constant-potential simulation. The implementation never enables dummy behavior as a
   fallback.
9. **Client patches are version-specific.** The current constant-potential patches depend
   on the listed client versions. Other versions have not been tested and require a fresh
   interface audit, compilation, and numerical verification.
10. **CP2K FARMING has a known exit-cleanup issue.** After the endpoints have terminated
    normally, some build environments may still trigger an offload-mempool cleanup
    assertion during process shutdown.

## Citation

If you use this project in your research, please cite both the constant-potential
framework paper and the i-PI 3.0 paper:

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
