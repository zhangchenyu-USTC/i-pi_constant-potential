"""Real ASE socket interface smoke tests for fixed-cell energy and forces."""

import os
import time

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import PropertyNotImplementedError
from abacuslite import AbacusProfile, AbacusSocketIO

from test_socket_driver import BINARY, CELL, POSITIONS, PP_ORB

pytestmark = pytest.mark.skipif(
    not BINARY or not PP_ORB, reason="real ABACUS not configured"
)


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_fixed_cell_ase_energy_forces_without_stress(tmp_path, basis):
    atoms = Atoms(
        "Si2", positions=POSITIONS * units.Bohr, cell=CELL * units.Bohr, pbc=True
    )
    profile = AbacusProfile(
        command=BINARY, pseudo_dir=PP_ORB, orbital_dir=PP_ORB, omp_num_threads=1
    )
    calculator = AbacusSocketIO(
        profile=profile,
        directory=tmp_path,
        unixsocket=f"abacus_ase_cp_{os.getpid()}_{time.time_ns()}",
        pseudopotentials={"Si": "Si_ONCV_PBE-1.0.upf"},
        basissets={"Si": "Si_gga_8au_100Ry_2s2p1d.orb"},
        inp={
            "basis_type": basis,
            "nbands": 12,
            "ecutwfc": 40,
            "scf_thr": 1e-9,
            "cal_force": 1,
            "cal_stress": 0,
            "symmetry": 0,
            "smearing_method": "fd",
            "smearing_sigma": 0.02,
            "mixing_beta": 0.3,
            "ks_solver": "scalapack_gvx" if basis == "lcao" else "cg",
        },
        kpts={
            "mode": "mp-sampling",
            "nk": [1, 1, 1],
            "gamma-centered": True,
            "kshift": [0, 0, 0],
        },
    )
    with calculator:
        atoms.calc = calculator
        with pytest.raises(PropertyNotImplementedError):
            atoms.get_stress()
        values = []
        for step in range(2):
            atoms.positions[1, 0] += 0.01 * step
            values.append(atoms.get_potential_energy())
            assert np.isfinite(atoms.get_forces()).all()
            assert "stress" not in calculator.results
            assert calculator.last_scf_converged is True
        assert abs(values[1] - values[0]) > 1e-8
    text = (tmp_path / "OUT.ABACUS/running_scf.log").read_text()
    assert text.count("Sending extras JSON") == 2


def test_ase_socket_input_preserves_independent_properties():
    assert AbacusSocketIO._socket_inp({"cal_stress": "0"})["cal_stress"] == 0
    for value in (True, "1"):
        assert AbacusSocketIO._socket_inp({"cal_stress": value})["cal_stress"] == 1
    for settings in (
        {"calculation": "md"},
        {"cal_force": 2},
        {"cal_stress": "invalid"},
    ):
        with pytest.raises(ValueError):
            AbacusSocketIO._socket_inp(settings)
