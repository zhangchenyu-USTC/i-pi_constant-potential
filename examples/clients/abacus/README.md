# ABACUS constant-potential client

The release patch is `abacus-3.11.0-beta10-pr7939-ipi-constant-potential.patch`.
Apply it to ABACUS **3.11.0-beta10**, upstream commit
`40e7ce8c575abb2ae7fc0df1bd6ec4b48a695d2d`, then use the normal ABACUS build.
Installation and constant-potential settings are described in the
[project README](../../../README.zh-CN.md#安装).

The PW and LCAO examples are in `../../constant_potential/`. They use native
plane-charge compensation and the socket interface. `historical/` contains
the 3.10.0 patches; `pw-example/` accompanies that older ordinary interface.

Constant-potential electron control rejects `assume_isolated=makov-payne`
(including `mp` and `m-p`), because the native correction changes the energy
without a matching electronic potential. Ordinary Makov–Payne calculations
retain their native behavior.

[PW/LCAO examples and real-client validation](https://github.com/zhangchenyu-USTC/i-pi_constant-potential/tree/630a6254112eb885f0c428bdf16540bb436b701d/examples/clients/abacus/validation)
are available on the validation branch. Its runner selects 46 checks, including
the Makov–Payne rejection and ordinary SCF/socket comparisons.
