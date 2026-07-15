# ABACUS 3.10.0 i-PI client

This directory contains two patches for the ABACUS 3.10.0 source tree:

- `abacus-3.10.0-ipi.patch`: standard i-PI force driver. It implements the
  regular i-PI protocol and does not recognize or send `CHGDATA`.
- `abacus-3.10.0-ipi-constant-potential.patch`: constant-potential extension.
  It adds `CHGDATA`, electronic-state updates, Fermi-level/work-function
  extras, and the ABACUS implicit-solvent update/freeze path.

Apply exactly one patch to a clean ABACUS 3.10.0 source tree. The patches are
alternatives; do not apply them on top of one another.

`pw-example/` is a normal NVT i-PI example for the standard patch. It contains
neither implicit-solvent input nor an i-PI `<electrons>` section. Update the
pseudopotential and executable paths before submitting the job.
