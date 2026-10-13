# A Flexible and Generalized Constant-Potential Framework in i-PI

[简体中文](README.zh-CN.md)

This project extends [i-PI 3.3.0](https://github.com/i-pi/i-pi) with constant-potential molecular dynamics.
It retains i-PI's server–client architecture and adds a system-wide electronic degree of freedom alongside the nuclear degrees of freedom, allowing the electron count to evolve with the target work function during molecular dynamics.

The implementation supports fractional electron counts, interpolation between two integer endpoints, the Ne counter-electrode method, and constant-potential path-integral molecular dynamics: **all beads share the same electron count, and their mean work function or mean Fermi level drives the electronic degree of freedom.**

The following first-principles clients support constant-potential simulations through i-PI:

- VASP 5.4.4 + vaspsol++;
- CP2K 2026.1;
- ABACUS 3.11.0-beta10 (PW and LCAO).

The constant-potential patches for these clients are located at:

- `examples/clients/vasp/vasp-5.4.4-ipi-constant-potential.patch`
- `examples/clients/cp2k/cp2k-2026.1-ipi-constant-potential.patch`
- `examples/clients/abacus/abacus-3.11.0-beta10-pr7939-ipi-constant-potential.patch`

Examples are provided in `examples/constant_potential/`.

Related papers: Chenyu Zhang et al., *A Flexible and Generalized Constant-Potential Framework in i-PI*, Journal of Chemical Theory and Computation (2026), [DOI: 10.1021/acs.jctc.6c00504](https://doi.org/10.1021/acs.jctc.6c00504).

Chenyu Zhang et al., *Electrode Potentials in Computational Electrochemistry: Referencing, Control and Uncertainty*. Current Opinion in Electrochemistry (2026), [DOI: 10.1016/j.coelec.2026.101947](https://doi.org/10.1016/j.coelec.2026.101947).

## Methods and features

The total electron count $q$ serves as a generalized coordinate, with conjugate momentum $p_q$. For each configuration evaluation, the client receives the current $q$, performs an SCF calculation, and returns the actual electron count, Fermi level, and, in work-function mode, the work function.

The electronic driving forces in the two control modes are:

$$
F_q=\mu_\star-\overline{\mu},
\qquad
F_q=\overline{\Phi}-\Phi_\star.
$$

The star denotes the target value, and the overbar denotes an average over beads. For a single bead, its feedback is used directly. The client computes the work function as $\Phi=V_{\mathrm{ref}}-E_F$, where $V_{\mathrm{ref}}$ is the planar-averaged electrostatic potential over the specified direction and interval. Ne mode also supports a moving average of the work function over accepted physical frames.

Constant-Fermi-level control requires a consistent potential reference; a fixed Fermi level corresponds to a fixed work function when the reference potential remains unchanged.

Electron counts are handled through two approaches:

- **Fractional electron counts**: one client directly evaluates the current real-valued electron count.
- **Two integer endpoints**: two clients evaluate adjacent integer electron counts. i-PI combines their energies, nuclear forces, virials, Fermi levels, and work functions using the same linear weights. The electronic driving force follows the mixed-feedback definition above.

The electronic thermostat can be Langevin or SVR. When its `temp` is omitted, it continuously follows the physical system temperature $T$; this remains $T$ for multiple beads. Setting `temp` explicitly gives the electronic thermostat an independent temperature.

Constant-potential integration in i-PI supports NVE, NVT, NPT, and NST, with cell and electronic-structure settings supported by the client. Checkpoints save the electronic state, thermostat energy, random-number state, work-function window, and target-scan progress. Inputs without `<electrons>` use the original i-PI simulation workflow.

### Charge compensation

Changing the total electron count in a constant-potential simulation generally gives the electrode's electron–ion subsystem a net charge. Dielectric polarization redistributes bound charge, whose integral over the entire periodic cell is zero; it cannot by itself ensure cell neutrality. The current examples use the following compensation strategies:

- **VASP + vaspsol++**: `2-vasp-implicit` uses `ISOL=2`, `C_MOLAR=1.0`, and
  `R_ION=4.0`. Mobile ions in the Poisson–Boltzmann continuum produce the compensating charge self-consistently.
  `C_MOLAR` specifies the bulk electrolyte concentration.
- **ABACUS**: the PW and LCAO examples use `gate_flag=1`, dipole correction, and a blocking region, without implicit solvent.
  A planar gate charge compensates the charged electrode; the gate and blocking region must be placed in a suitable region without atoms.
- **CP2K**: the implicit Poisson model used here provides a dielectric response but does not include a mobile continuum-ion compensation model corresponding to vaspsol++'s
  `C_MOLAR`/`R_ION`.

The Ne counter-electrode provides another compensation approach. Each configuration evaluation uses the current electron count and the corresponding effective Ne nuclear charge. Numerical stability depends on the electronic generalized mass, potentiostat time constant, SCF convergence, work-function averaging window, and the number and positions of Ne atoms. These settings require convergence tests for the system of interest. Future changes may allow updating the electron count once every few steps.

## Installation

### i-PI

Install with Python 3.10 or later in a dedicated environment:

```bash
git clone https://github.com/zhangchenyu-USTC/i-pi_constant-potential.git
cd i-pi_constant-potential
python -m pip install -e .
i-pi --help
```

Alternatively, run `source env.sh` in the project directory to use the source tree directly. Compile the clients with the matching patches below, configuring build and link dependencies according to each program's build instructions.

### VASP 5.4.4 + VASPsol++

Apply the following to your VASP 5.4.4 source directory, in order:

1. The official `patch.5.4.4.16052018` fix.
2. The VASPsol++ patch for VASP 5.4.4.
3. The `solvation_intel.F` file matching this version and compiler, copied to `src/solvation.F`.
4. This project's [combined VASP patch](examples/clients/vasp/vasp-5.4.4-ipi-constant-potential.patch).

```bash
cd /path/to/vasp.5.4.4
patch -p0 < /path/to/patch.5.4.4.16052018
patch -p1 < /path/to/vaspsol++-vasp_5.4.4.patch
cp /path/to/solvation_intel.F src/solvation.F
patch --dry-run -p1 < /path/to/vasp-5.4.4-ipi-constant-potential.patch
patch -p1 < /path/to/vasp-5.4.4-ipi-constant-potential.patch
```

The combined patch includes the ordinary i-PI interface. Compile it using the VASP build procedure. Use `IBRION=23` for ordinary socket calculations and `IBRION=24` for constant-potential calculations.

The new INCAR parameter `SOL_UPDATE` controls the VASPsol update interval. It takes a positive integer and defaults to `1`. For example, `SOL_UPDATE=5` updates the solvent response at local configuration-evaluation indices `0, 5, 10…`. Other evaluations reuse the frozen solvent field while performing electronic SCF at the current electron count and computing the energy, nuclear forces, and work function. A restart or new INIT forces an initial refresh and resets the counter; STATUS and repeated result reads do not advance it. The current setup for `SOL_UPDATE>1` uses a single bead and an independent VASP process for each endpoint. Other VASP modes use the default value `1`.

### CP2K 2026.1

Apply the [complete patch](examples/clients/cp2k/cp2k-2026.1-ipi-constant-potential.patch) to a clean CP2K `v2026.1` source tree, then build and install it using CP2K's toolchain or CMake procedure:

```bash
cd /path/to/cp2k-2026.1
git apply --check /path/to/cp2k-2026.1-ipi-constant-potential.patch
git apply /path/to/cp2k-2026.1-ipi-constant-potential.patch
```

Run the newly installed executable with its matching shared libraries, and set `CP2K_DATA_DIR` to the CP2K data directory. Ne mode requires `CORE_CORR_DIP` and `SURFACE_DIPOLE_CORRECTION`; see [example 5](examples/constant_potential/5-cp2k-Ne/cp2k.inp) for the complete settings.

### ABACUS 3.11.0-beta10

The patch is based on the original `3.11.0-beta10`, at upstream commit `40e7ce8c575abb2ae7fc0df1bd6ec4b48a695d2d`:

```bash
git clone https://github.com/deepmodeling/abacus-develop.git
cd abacus-develop
git checkout 40e7ce8c575abb2ae7fc0df1bd6ec4b48a695d2d
git apply --check /path/to/abacus-3.11.0-beta10-pr7939-ipi-constant-potential.patch
git apply /path/to/abacus-3.11.0-beta10-pr7939-ipi-constant-potential.patch
```

The [patch](examples/clients/abacus/abacus-3.11.0-beta10-pr7939-ipi-constant-potential.patch) extends the native socket driver with electron-count control, actual-occupation checks, and Fermi-level and work-function feedback. The reference potential shares its calculation with the native electrostatic-potential cube output, and charge compensation uses the native gate. Follow the ABACUS documentation to compile; the constant-potential examples use a fixed cell.

## Input and execution

### Start with an analytical potential

[Example 1](examples/constant_potential/1-pes-fermi/) runs without a DFT client and provides a direct check of the installation and electronic dynamics:

```bash
cp -r examples/constant_potential/1-pes-fermi /path/to/pes-run
cd /path/to/pes-run
i-pi input.xml > ipi.log 2>&1
```

The following is a complete three-step input. Place example 1's `init.xyz` in the same directory to run it:

```xml
<simulation verbosity="low">
  <output prefix="simulation">
    <properties stride="1" filename="out">[step, time{femtosecond}, potential{electronvolt}, electronic_charge, fermi_level{electronvolt}]</properties>
    <checkpoint filename="restart" stride="1" overwrite="True"/>
  </output>
  <total_steps>3</total_steps>
  <prng><seed>31245</seed></prng>
  <ffdirect name="toy_pes">
    <pes>asym_dw_constant_potential</pes>
    <parameters>{a:1.0, b:3.0, d:1.0, x0:0.0, c:-10.0, pzc:5.0, cap:10.0, q_default:1.0}</parameters>
  </ffdirect>
  <system>
    <initialize nbeads="1">
      <cell mode="abc" units="angstrom">[10.0, 10.0, 10.0]</cell>
      <file mode="xyz" units="angstrom">init.xyz</file>
      <velocities mode="thermal" units="kelvin">300.0</velocities>
    </initialize>
    <forces><force forcefield="toy_pes"/></forces>
    <ensemble><temperature units="kelvin">300.0</temperature></ensemble>
    <motion mode="dynamics">
      <fixatoms_dof>[1, 2]</fixatoms_dof>
      <dynamics mode="nvt" splitting="baoab">
        <timestep units="femtosecond">0.5</timestep>
        <thermostat mode="svr"><tau units="femtosecond">200.0</tau></thermostat>
        <electrons enabled="true">
          <target_fermi_level units="electronvolt">-4.5</target_fermi_level>
          <neutral_electrons>1</neutral_electrons>
          <q_init>1.0</q_init>
          <mass units="atomic_unit">200.0</mass>
          <potentiostat mode="svr"><tau units="femtosecond">200.0</tau></potentiostat>
        </electrons>
      </dynamics>
    </motion>
  </system>
</simulation>
```

### Constant work function and electronic parameters

Constant-potential settings belong under `<motion><dynamics><electrons>`. For an external DFT client, configure `ffsocket` and the client connection address following the corresponding example. In work-function mode, replace the Fermi-level target with the following target and reference window:

```xml
<target_workfunction units="electronvolt">4.6</target_workfunction>
<potential_average_axis>z</potential_average_axis>
<potential_average_region units="angstrom">[12.5, 15.0]</potential_average_region>
```

Choose a reference window on the electrostatic-potential plateau in the intended vacuum or solvent reference region. When changing the structure, set the direction and interval from its planar-averaged potential profile.

| Parameter | Meaning |
|---|---|
| `target_fermi_level` / `target_workfunction` | Select one control target |
| `neutral_electrons` | Total electron count of the neutral system |
| `q_init`, `p_init` | Initial total electron count and its generalized momentum |
| `mass` | Fictitious mass of the electron-count degree of freedom |
| `potential_average_axis`, `potential_average_region` | Direction and averaging interval for the work-function reference potential |
| `charge_mixing="true"` | Enable interpolation between two integer endpoints on `<electrons>` |
| `Ne_doping="true"` | Enable Ne counter-electrode mode on `<electrons>` |
| `workfunction_average_steps` | Moving-average window for the work function in Ne mode |
| `potentiostat` | Electronic thermostat type, time constant, and optional temperature |

Two-endpoint mode defines two sockets; i-PI calls both and combines their results. See [example 4](examples/constant_potential/4-cp2k-implicit-mixing/input.xml) for the endpoint configuration.

### Output and restart

`electronic_charge` reports the total electron count $q$, while `fermi_level` and `workfunction` report the current feedback. The Ne moving average is available as `workfunction_steps_average`. Targets, electronic momentum, and the grand potential can also be added to the properties list.

To restart, read the generated checkpoint. `total_steps` specifies the final step count: set it to a value greater than the current `step` in the checkpoint, then run:

```bash
i-pi simulation.restart > ipi-restart.log 2>&1
```

## Examples and validation

| Directory | Contents |
|---|---|
| [1-pes-fermi](examples/constant_potential/1-pes-fermi/) | Constant-Fermi-level dynamics with an analytical potential |
| [2-vasp-implicit](examples/constant_potential/2-vasp-implicit/) | Constant-work-function dynamics with VASPsol++ implicit electrolyte and `SOL_UPDATE` |
| [3-vasp-Ne](examples/constant_potential/3-vasp-Ne/) | Constant-work-function dynamics with a VASP Ne counter-electrode |
| [4-cp2k-implicit-mixing](examples/constant_potential/4-cp2k-implicit-mixing/) | CP2K implicit dielectric environment with two integer endpoints through FARMING |
| [5-cp2k-Ne](examples/constant_potential/5-cp2k-Ne/) | Constant-work-function dynamics with fractional electron counts and a CP2K Ne counter-electrode |
| [6-abacus-pw-compensating-charge](examples/constant_potential/6-abacus-pw-compensating-charge/) | ABACUS PW with planar compensating charge |
| [7-abacus-lcao-compensating-charge](examples/constant_potential/7-abacus-lcao-compensating-charge/) | ABACUS LCAO with planar compensating charge |

## Citation

When using this project, please cite the constant-potential method paper and the i-PI 3.0 paper:

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
