# ABACUS constant-potential client

The release patch targets ABACUS develop commit
`cd2c7ce5b2c8b24893d3426922a20fb7d4b7fd3c`, including AO interpolation fix #8119.
Apply `abacus-develop-constant-potential.patch` with `git apply <patch>` to that
clean baseline, then follow ABACUS's standard build instructions.

The patch extends the native socket driver with total-electron-count control
and electronic observables. Implicit solvent uses the native implementation.

The develop patch preserves the upstream PCC contribution to electrostatic
cube output and uses the same potential for the work-function reference.
Use the source commit listed above so its AO implementation and integration
references match the patch baseline.

Before publishing a constant-potential result, the client checks the actual
SCF occupation sum against the requested electron count with an absolute
tolerance of `1e-8`. Occupations already include k-point weights and spin
degeneracy; MPI pool replicas are counted once. Non-finite occupations or a
mismatch reject the result. `extras.nelect` reports the checked occupation
sum, without adding INPUT parameters or changing the native occupations.

Use `calculation=scf`, `socket_driver=1`, `cal_force=1`, `cal_stress=0` and
`ABACUS_SOCKET_ADDRESS=/tmp/ipi_<address>:UNIX` (or `host:port`). The supplied
PW/LCAO electrode examples use native gate compensation. CP requires fixed
cells; ordinary socket runs retain physical stress support.

`historical/` contains the unmaintained 3.10.0 patches. `pw-example/` accompanies
that historical ordinary interface.

[Examples and real-client tests](validation/README.md) are maintained here in
the i-PI repository. They include Si2 and Al36 PW/LCAO inputs, a local MPI
launcher, and the electron-count, static-reference, ASE, and restart tests.
