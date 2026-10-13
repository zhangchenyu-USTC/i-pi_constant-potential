"""Combined CP controls and ordinary native-dynamics regression tests."""

import json
import os
import re
import subprocess

import numpy as np
import pytest

from test_socket_driver import (
    BINARY,
    REFERENCE,
    RY_EV,
    BOHR_A,
    Client,
    command,
    cube_average,
    inputs,
    response_calls,
)

pytestmark = pytest.mark.skipif(
    not BINARY or not REFERENCE, reason="real ABACUS executables not configured"
)


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_gate_solvent_and_kpoint_pools_together(tmp_path, basis):
    """Exercise charged gate+solvent updates and freezes across two pool layouts."""
    ranks = int(os.environ["ABACUS_TEST_RANKS"])
    assert ranks >= 4 and ranks % 2 == 0
    samples = []
    for pools in (1, 2):
        directory = tmp_path / f"pools_{pools}"
        extra = {
            "kpar": pools,
            "imp_sol": 1,
            "eb_k": 80,
            "gate_flag": 1,
            "zgate": 0.75,
            "efield_flag": 1,
            "dip_cor_flag": 1,
            "efield_pos_max": 0.9,
            "efield_pos_dec": 0.1,
        }
        inputs(directory, basis, extra=extra)
        (directory / "KPT").write_text("K_POINTS\n0\nGamma\n2 1 1 0 0 0\n")
        values = []
        with Client(directory) as client:
            client.initialize()
            for q, flags in [(8.0, 0), (8.2, 0), (7.8, 0)]:
                result = client.calculate(q, flags=flags)
                assert result[3]["nelect"] == pytest.approx(q, abs=1e-8, rel=0)
                assert result[3]["electrostatic_reference_eV"] == pytest.approx(
                    cube_average(directory, 2), abs=1e-4
                )
                values.append(result)
            client.finish()
        text = (directory / "OUT.ABACUS/running_scf.log").read_text()
        assert text.count("Sending extras JSON") == 3
        assert response_calls(directory) > 0
        samples.append(values)
    errors = []
    for one, two in zip(*samples):
        error = [
            abs(one[0] - two[0]),
            np.max(np.abs(one[1] - two[1])),
            abs(one[3]["fermi_level_eV"] - two[3]["fermi_level_eV"]),
            abs(
                one[3]["electrostatic_reference_eV"]
                - two[3]["electrostatic_reference_eV"]
            ),
        ]
        assert error[0] < 1e-6
        assert error[1] < 1e-5
        assert error[2] < 1e-4
        assert error[3] < 1e-4
        errors.append([float(value) for value in error])
    (tmp_path / "combined-pool-comparison.json").write_text(
        json.dumps(errors, indent=2) + "\n"
    )


def run_native(directory, binary, basis, mode):
    """Run the native driver with deterministic initialization and no socket mode."""
    extra = {"calculation": mode, "nelec": 8.0, "out_pot": 0}
    if mode == "md":
        # Native ABACUS counts the initial frame among md_nstep output frames.
        extra.update(
            md_type="nve",
            md_nstep=3,
            md_dt=0.1,
            md_tfirst=300,
            md_seed=17,
            md_dumpfreq=1,
            dump_force=1,
            dump_vel=1,
        )
    else:
        extra.update(relax_nmax=3, force_thr=1e-12)
    inputs(directory, basis, socket_mode=False, extra=extra)
    with (directory / "abacus.log").open("w") as log:
        subprocess.run(
            command(binary),
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=180,
        )
    text = (directory / f"OUT.ABACUS/running_{mode}.log").read_text()
    assert "Sending extras JSON" not in text
    assert "CHGDATA" not in text
    energy = float(re.findall(r"!FINAL_ETOT_IS\s+([-+\d.eE]+)", text)[-1]) / (2 * RY_EV)
    forces = []
    for block in text.split("TOTAL-FORCE (eV/Angstrom)")[1:]:
        rows = re.findall(r"Si\d+\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)", block)[
            :2
        ]
        assert len(rows) == 2
        forces.append(np.array(rows, dtype=float) * BOHR_A / (2 * RY_EV))
    assert len(forces) >= 2, "native run must evaluate more than its initial geometry"
    trajectory = None
    if mode == "md":
        dump = (directory / "OUT.ABACUS/MD_dump").read_text()
        assert re.findall(r"MDSTEP:\s+(\d+)", dump) == ["0", "1", "2"]
        rows = re.findall(r"^\s*\d+\s+Si\s+(.+)$", dump, flags=re.MULTILINE)
        trajectory = np.array([[float(value) for value in row.split()] for row in rows])
        assert trajectory.shape == (6, 9)
    return energy, np.array(forces), trajectory


@pytest.mark.parametrize("basis", ["pw", "lcao"])
@pytest.mark.parametrize("mode", ["md", "relax"])
def test_native_dynamics_matches_ordinary_build(tmp_path, basis, mode):
    reference = run_native(tmp_path / "reference", REFERENCE, basis, mode)
    modified = run_native(tmp_path / "modified", BINARY, basis, mode)
    assert abs(reference[0] - modified[0]) < 1e-6
    assert reference[1].shape == modified[1].shape
    force_error = np.max(np.abs(reference[1] - modified[1]))
    assert force_error < 1e-5
    evidence = {
        "energy_error_Ha": abs(reference[0] - modified[0]),
        "max_force_error_Ha_per_Bohr": float(force_error),
        "force_frames": len(reference[1]),
    }
    if mode == "md":
        delta = np.max(np.abs(reference[2] - modified[2]), axis=0)
        assert np.max(delta[:3]) < 1e-7
        assert np.max(delta[3:6]) * BOHR_A / (2 * RY_EV) < 1e-5
        assert np.max(delta[6:]) < 1e-7
        evidence["md_dump_max_error"] = delta.tolist()
    (tmp_path / "native-comparison.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
