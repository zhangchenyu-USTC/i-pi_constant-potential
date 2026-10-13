"""Real method-level regressions for changing electron number through the socket.

Each frame is compared with a separate ordinary ABACUS process using the same
functional and numerical settings. These tests require the LibXC/LibRI builds.
"""

import json
import re
import struct

import numpy as np
import pytest

from test_socket_driver import (
    Client,
    POSITIONS,
    cube_average,
    header,
    inputs,
    standalone,
)

METHODS = [
    pytest.param("pw", {"dft_functional": "SCAN", "mixing_tau": 1}, id="scan-pw"),
    pytest.param("lcao", {"dft_functional": "SCAN", "mixing_tau": 1}, id="scan-lcao"),
    pytest.param(
        "pw",
        {
            "dft_plus_u": 1,
            "orbital_corr": 1,
            "hubbard_u": 2.0,
            "onsite_radius": 3.0,
            "nspin": 2,
        },
        id="dftu1-pw",
    ),
    pytest.param(
        "lcao",
        {
            "dft_plus_u": 1,
            "orbital_corr": 1,
            "hubbard_u": 2.0,
            "onsite_radius": 3.0,
            "nspin": 2,
        },
        id="dftu1-lcao",
    ),
]
for _basis, _functional in (("pw", "PBE0"), ("lcao", "HSE")):
    for _separate in (0, 1):
        METHODS.append(
            pytest.param(
                _basis,
                {
                    "dft_functional": _functional,
                    "exx_separate_loop": _separate,
                    "exx_hybrid_step": 80,
                    "exx_thr_type": "density",
                    "exx_pca_threshold": 1e-4,
                    "exx_ccp_rmesh_times": 1,
                    **(
                        {"smearing_sigma": 0.05, "scf_thr": 1e-12}
                        if _basis == "pw"
                        else {}
                    ),
                    **(
                        {
                            "exx_mixing_beta": 0.5,
                            "exx_hybrid_step": 160,
                            "smearing_sigma": 0.05,
                        }
                        if _basis == "lcao" and _separate == 1
                        else {}
                    ),
                },
                id=f"{_functional.lower()}-{_basis}-loop{_separate}",
            )
        )
METHODS += [
    pytest.param(
        "lcao",
        {
            "dft_functional": "HSE",
            "nspin": 2,
            "nupdown": 0.4,
            "exx_separate_loop": 1,
            "exx_hybrid_step": 160,
            "exx_pca_threshold": 1e-4,
            "exx_ccp_rmesh_times": 1,
            "exx_mixing_beta": 0.5,
            "smearing_sigma": 0.10,
        },
        id="hse-lcao-fixed-spin",
    ),
    pytest.param(
        "lcao",
        {
            "dft_functional": "SCAN0",
            "mixing_tau": 1,
            "exx_separate_loop": 1,
            "exx_hybrid_step": 160,
            "exx_pca_threshold": 1e-4,
            "exx_ccp_rmesh_times": 1,
            "exx_mixing_beta": 0.5,
            "smearing_sigma": 0.05,
        },
        id="scan0-lcao",
    ),
]


@pytest.mark.parametrize("basis,method", METHODS)
def test_electronic_method_charge_and_geometry_sequence(tmp_path, basis, method):
    options = {
        "scf_thr": 1e-10,
        "scf_nmax": 200,
        "mixing_beta": 0.2,
        "chg_extrap": "none",
        **method,
    }
    socket_dir = tmp_path / "socket"
    inputs(socket_dir, basis, extra=options)
    evidence = []
    spin_evidence = []
    with Client(socket_dir) as client:
        client.connection.settimeout(600)
        client.initialize()
        for index, nelect in enumerate((8.0, 8.15, 7.85, 8.0)):
            positions = POSITIONS.copy()
            if index == 3:
                positions[1] += np.array([0.04, -0.03, 0.02])
            frame = client.calculate(nelect, positions=positions)
            reference_dir = tmp_path / f"reference_{index}"
            reference = standalone(
                reference_dir, basis, nelect, positions, extra=options
            )
            if "nupdown" in method:
                socket_log = (socket_dir / "OUT.ABACUS/running_scf.log").read_text()
                spin_values = re.findall(
                    r"E_Fermi_(?:up|dw)\s+([-+\d.eE]+)\s+([-+\d.eE]+)", socket_log
                )
                assert len(spin_values) >= 2
                spin_eV = [float(value[1]) for value in spin_values[-2:]]
                assert abs(spin_eV[0] - spin_eV[1]) > 1e-4
                assert frame[3]["fermi_level_eV"] == pytest.approx(
                    np.mean(spin_eV), abs=2e-9
                )
                spin_evidence.append(
                    {
                        "nelect": nelect,
                        "fermi_up_eV": spin_eV[0],
                        "fermi_down_eV": spin_eV[1],
                        "reported_fermi_eV": frame[3]["fermi_level_eV"],
                    }
                )
                (tmp_path / "spin-evidence.json").write_text(
                    json.dumps(spin_evidence, indent=2) + "\n"
                )
            errors = {
                "energy_hartree": float(abs(frame[0] - reference[0])),
                "force_hartree_per_bohr": float(
                    np.max(np.abs(frame[1] - reference[1]))
                ),
                "fermi_eV": float(abs(frame[3]["fermi_level_eV"] - reference[2])),
                "reference_eV": float(
                    abs(
                        frame[3]["electrostatic_reference_eV"]
                        - cube_average(reference_dir, 2)
                    )
                ),
            }
            evidence.append({"nelect": nelect, **errors})
            (tmp_path / "method-comparison.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )
            assert frame[3]["nelect"] == pytest.approx(nelect, abs=1e-8, rel=0)
            assert errors["energy_hartree"] < 1e-6, errors
            assert errors["force_hartree_per_bohr"] < 1e-5, errors
            assert errors["fermi_eV"] < 2e-4, errors
            assert errors["reference_eV"] < 2e-4, errors
            assert frame[3]["workfunction_eV"] == pytest.approx(
                frame[3]["electrostatic_reference_eV"] - frame[3]["fermi_level_eV"],
                abs=1e-12,
            )
            reference_log = (reference_dir / "abacus.log").read_text()
            assert "!!EXX IS NOT CONVERGED!!" not in reference_log
            assert "!!SCF IS NOT CONVERGED!!" not in reference_log
            energy_log = (reference_dir / "OUT.ABACUS/running_scf.log").read_text()
            if method.get("dft_plus_u"):
                correction = re.findall(r"E_plusU\s+([-+\d.eE]+)", energy_log)
                assert correction and abs(float(correction[-1])) > 1e-6
            if "exx_hybrid_step" in method:
                correction = re.findall(r"E_exx\s+([-+\d.eE]+)", energy_log)
                assert correction and abs(float(correction[-1])) > 1e-6
                if method.get("exx_separate_loop"):
                    updates = (reference_log + energy_log).count("Updating EXX")
                    assert 0 < updates < method["exx_hybrid_step"], updates
            if index == 3:
                assert np.max(np.abs(reference[1])) > 1e-4
        client.finish()
    (tmp_path / "method-comparison.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


@pytest.mark.parametrize("basis,functional", [("pw", "PBE0"), ("lcao", "HSE")])
def test_exhausted_hybrid_outer_loop_never_publishes_cp_frame(
    tmp_path, basis, functional
):
    inputs(
        tmp_path,
        basis,
        extra={
            "dft_functional": functional,
            "exx_separate_loop": 1,
            "exx_hybrid_step": 1,
            "exx_thr_type": "density",
            "scf_thr": 1e-11,
            "scf_nmax": 200,
        },
    )
    with Client(tmp_path) as client:
        client.connection.settimeout(600)
        client.initialize()
        client.connection.sendall(
            header("CHGDATA")
            + struct.pack("=di", 8.15, 0)
            + client.posdata(POSITIONS, np.eye(3) * 10.26)
            + header("GETFORCE")
        )
        with pytest.raises((EOFError, ConnectionResetError)):
            client.read(12)
        assert client.process.wait(timeout=30) != 0
    text = (tmp_path / "abacus.log").read_text()
    assert "EXX outer loop did not converge" in text


def test_legacy_dftu_forces_rejected_before_scf(tmp_path):
    inputs(
        tmp_path,
        "lcao",
        extra={"dft_plus_u": 2, "orbital_corr": 1, "hubbard_u": 2.0, "nspin": 2},
    )
    with Client(tmp_path) as client:
        client.initialize()
        client.connection.sendall(
            header("CHGDATA")
            + struct.pack("=di", 8.15, 0)
            + client.posdata(POSITIONS, np.eye(3) * 10.26)
            + header("GETFORCE")
        )
        with pytest.raises((EOFError, ConnectionResetError)):
            client.read(12)
        assert client.process.wait(timeout=30) != 0
    assert (
        "legacy dft_plus_u=2 force implementation is unavailable"
        in (tmp_path / "abacus.log").read_text()
    )


@pytest.mark.parametrize("separate_loop", [0, 1])
def test_pw_hybrid_meta_requires_upstream_tau_support(tmp_path, separate_loop):
    inputs(
        tmp_path,
        "pw",
        extra={
            "dft_functional": "SCAN0",
            "mixing_tau": 1,
            "exx_separate_loop": separate_loop,
        },
    )
    with Client(tmp_path) as client:
        client.initialize()
        client.connection.sendall(
            header("CHGDATA")
            + struct.pack("=di", 8.15, 0)
            + client.posdata(POSITIONS, np.eye(3) * 10.26)
            + header("GETFORCE")
        )
        with pytest.raises((EOFError, ConnectionResetError)):
            client.read(12)
        assert client.process.wait(timeout=30) != 0
    text = (tmp_path / "abacus.log").read_text()
    assert (
        "upstream kinetic-energy-density updates do not handle hybrid functionals"
        in text
    )
    assert "@IPI DRIVER: Sending extras JSON" not in text
