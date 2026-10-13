# Al36 LCAO constant-potential example

This fixed-cell example uses the Al36 PW slab, PD04 PBE pseudopotential (three valence electrons per Al), 80 bands, a 30 Ry grid cutoff, and an SCF threshold of `1e-7`. The `Al_gga_10au_100Ry_3s3p2d.orb` basis is paired with this pseudopotential in the upstream `tests/03_NAO_multik/scf_in_restart` example.

The numerical example uses 10 K, a 0.1 fs nuclear timestep, one bead, a 50 dalton electronic mass, and a 5 eV target workfunction averaged over z = 10–12.5 Å. It exercises the interface over short physical times. The checkpoint float format retains 17 significant decimal digits for electronic restart state.

From the `validation` directory, run:

```bash
export ABACUS_CP_EXECUTABLE=/path/to/abacus
export ABACUS_PP_ORB=/path/to/abacus-source/tests/PP_ORB
python run_example.py al36-lcao --output /path/to/new-run --ranks 4 --steps 3 --temperature 10 --dt 0.1 --timeout 1800
```

The launcher sets one OpenMP thread per rank. The output directory must be new. Inspect `ipi.log`, `abacus.log`, `OUT.ABACUS/running_scf.log`, and `cp.out` after completion. See the [validation guide](../../README.md) for MPI options and regression tests.
