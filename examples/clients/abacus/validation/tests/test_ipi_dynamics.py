"""Real fixed-cell i-PI/ABACUS dynamics and restart tests."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from test_socket_driver import BINARY, BOHR_A, POSITIONS, command, inputs

IPI_ROOT = os.environ.get("ABACUS_TEST_IPI_ROOT")
pytestmark = pytest.mark.skipif(
    not BINARY or not IPI_ROOT, reason="real ABACUS/i-PI not configured"
)


def configuration(
    directory,
    basis,
    mode="nvt",
    *,
    steps=3,
    target="workfunction",
    consolidate=True,
    angstrom=False,
):
    inputs(directory, basis, extra={"out_pot": 0, "imp_sol": 0})
    scale = BOHR_A if angstrom else 1.0
    length_units = "angstrom" if angstrom else "atomic_unit"
    xyz = (
        f"2\nPositions in {length_units}\n"
        + "\n".join(
            "Si " + " ".join(str(x * scale) for x in position) for position in POSITIONS
        )
        + "\n"
    )
    (directory / "init.xyz").write_text(xyz)
    address = f"abacus_cp_md_{os.getpid()}_{time.time_ns()}"
    thermostat = (
        '<thermostat mode="langevin"><tau units="femtosecond">50</tau></thermostat>'
        if mode != "nve"
        else ""
    )
    observable = "fermi_level" if target == "fermi" else "workfunction"
    target_xml = (
        '<target_fermi_level units="electronvolt">-3.5</target_fermi_level>'
        if target == "fermi"
        else '<target_workfunction units="electronvolt">4.0</target_workfunction>'
        "<potential_average_axis>z</potential_average_axis>"
        '<potential_average_region units="angstrom">[2, 4]</potential_average_region>'
    )
    text = f"""<simulation verbosity="medium" floatformat="%24.16e">
  <prng><seed>51723</seed></prng>
  <output prefix="cp">
    <properties stride="1" filename="out">[ step, potential{{electronvolt}}, electronic_charge,
      fermi_level{{electronvolt}}, {observable}{{electronvolt}}, Grand_canonical_potential{{electronvolt}} ]</properties>
    <trajectory filename="pos" stride="1" format="xyz">positions{{atomic_unit}}</trajectory>
    <checkpoint stride="1" filename="checkpoint" overwrite="true" />
  </output>
  <total_steps>{steps}</total_steps>
  <ffsocket name="abacus" mode="unix">
    <address>{address}</address><timeout>120</timeout><latency>0.01</latency>
    <batch_size>1</batch_size><consolidate_messages>{str(consolidate).lower()}</consolidate_messages>
  </ffsocket>
  <system>
    <initialize nbeads="1">
      <file mode="xyz" units="{length_units}">init.xyz</file>
      <cell mode="abc" units="{length_units}">{[10.26 * scale] * 3}</cell>
      <velocities mode="thermal" units="kelvin">10</velocities>
    </initialize>
    <forces><force forcefield="abacus" /></forces>
    <ensemble><temperature units="kelvin">10</temperature></ensemble>
    <motion mode="dynamics"><fixcom>false</fixcom>
      <dynamics mode="{mode}" splitting="baoab">
        <timestep units="femtosecond">0.1</timestep>{thermostat}
        <electrons enabled="true" charge_mixing="false">
          {target_xml}
          <neutral_electrons>8</neutral_electrons><q_init>8.05</q_init>
          <mass units="dalton">50</mass>
          <potentiostat mode="langevin"><tau units="femtosecond">300</tau></potentiostat>
        </electrons>
      </dynamics>
    </motion>
  </system>
</simulation>
"""
    (directory / "input.xml").write_text(text)
    return address


def run_pair(directory, address, *, xml="input.xml", suffix="", timeout=600):
    path = Path("/tmp") / ("ipi_" + address)
    environment = dict(
        os.environ,
        PYTHONPATH=str(IPI_ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        ABACUS_SOCKET_ADDRESS=str(path) + ":UNIX",
    )
    processes = []
    with (directory / f"ipi{suffix}.log").open("w") as ipilog, (
        directory / f"abacus{suffix}.log"
    ).open("w") as abalog:
        try:
            ipi = subprocess.Popen(
                [sys.executable, str(Path(IPI_ROOT) / "bin/i-pi"), xml],
                cwd=directory,
                env=environment,
                stdout=ipilog,
                stderr=subprocess.STDOUT,
            )
            processes.append(ipi)
            deadline = time.monotonic() + 30
            while (
                not path.exists() and ipi.poll() is None and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert path.exists(), (directory / f"ipi{suffix}.log").read_text()
            abacus = subprocess.Popen(
                command(BINARY),
                cwd=directory,
                env=environment,
                stdout=abalog,
                stderr=subprocess.STDOUT,
            )
            processes.append(abacus)
            ipi_status = ipi.wait(timeout=timeout)
            abacus_status = abacus.wait(timeout=45)
            return ipi_status, abacus_status
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            path.unlink(missing_ok=True)


def read_charge_requests(directory):
    text = (directory / "OUT.ABACUS/running_scf.log").read_text()
    return [
        (float(q), int(flag))
        for q, flag in re.findall(r"CHGDATA nelect=([-+\d.eE]+) flag=(\d+)", text)
    ]


def audit(directory, count):
    text = (directory / "OUT.ABACUS/running_scf.log").read_text()
    extras = [
        json.loads(line.split("Sending extras JSON = ")[1])
        for line in text.splitlines()
        if "Sending extras JSON = " in line
    ]
    assert len(extras) == count
    charges = read_charge_requests(directory)
    assert len(charges) == count
    assert all(flag == 0 for _, flag in charges)
    for received, (sent, _) in zip(extras, charges):
        assert received["nelect"] == pytest.approx(sent, abs=1e-8, rel=0)
    assert all(np.isfinite([x["fermi_level_eV"], x["nelect"]]).all() for x in extras)
    assert "Exiting cleanly" in (directory / "ipi.log").read_text()
    return extras, [flag for _, flag in charges]


@pytest.mark.parametrize("basis", ["pw", "lcao"])
@pytest.mark.parametrize("mode", ["nve", "nvt"])
def test_real_dynamics_and_force_count(tmp_path, basis, mode):
    address = configuration(
        tmp_path,
        basis,
        mode,
        target="fermi" if mode == "nve" else "workfunction",
        consolidate=mode == "nvt",
    )
    assert run_pair(tmp_path, address) == (0, 0)
    extras, _ = audit(tmp_path, 4)
    assert abs(extras[-1]["nelect"] - extras[0]["nelect"]) > 1e-8
    assert np.isfinite(np.loadtxt(tmp_path / "cp.out")).all()


def test_workfunction_restart(tmp_path):
    address = configuration(tmp_path, "pw")
    assert run_pair(tmp_path, address) == (0, 0)
    initial, flags = audit(tmp_path, 4)
    assert 0 in flags
    first_log = (tmp_path / "OUT.ABACUS/running_scf.log").read_text()
    (tmp_path / "running_scf.first.log").write_text(first_log)
    tree = ET.parse(tmp_path / "RESTART")
    saved_q = float(tree.getroot().find("system/motion/dynamics/electrons/q_init").text)
    assert saved_q == read_charge_requests(tmp_path)[-1][0]
    assert initial[-1]["nelect"] == pytest.approx(saved_q, abs=1e-8, rel=0)
    tree.getroot().find("total_steps").text = "6"
    tree.write(tmp_path / "restart.xml")
    assert run_pair(tmp_path, address, xml="restart.xml", suffix="-restart") == (0, 0)
    restarted, flags = audit(tmp_path, 4)
    assert read_charge_requests(tmp_path)[0][0] == saved_q
    assert restarted[0]["nelect"] == pytest.approx(saved_q, abs=1e-8, rel=0)
    assert 0 in flags
    assert "Exiting cleanly" in (tmp_path / "ipi-restart.log").read_text()


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_real_dynamics_with_angstrom_initialization(tmp_path, basis):
    address = configuration(tmp_path, basis, angstrom=True)
    assert run_pair(tmp_path, address) == (0, 0)
    extras, _ = audit(tmp_path, 4)
    assert abs(extras[-1]["nelect"] - extras[0]["nelect"]) > 1e-8
