# ABACUS constant-potential client

The release patch is `abacus-3.11.0-beta10-pr7939-ipi-constant-potential.patch`.
Apply it to ABACUS **3.11.0-beta10**, upstream commit
`40e7ce8c575abb2ae7fc0df1bd6ec4b48a695d2d`, then use the normal ABACUS build.
Installation and constant-potential settings are described in the
[project README](../../../README.zh-CN.md#安装).

The PW and LCAO examples are in `../../constant_potential/`. They use native
plane-charge compensation and the socket interface. `historical/` contains
the 3.10.0 patches; `pw-example/` accompanies that older ordinary interface.
