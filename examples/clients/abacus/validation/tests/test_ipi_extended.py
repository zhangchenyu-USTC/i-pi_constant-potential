"""Real i-PI target scans, two-endpoint interpolation and restart comparisons."""

from contextlib import ExitStack
import copy
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

from test_ipi_dynamics import (
    IPI_ROOT,
    audit,
    configuration,
    read_charge_requests,
    run_pair,
)
from test_socket_driver import BINARY, BOHR_A, RY_EV, command, inputs
from ipi.utils.units import unit_to_internal

pytestmark = pytest.mark.skipif(
    not BINARY or not IPI_ROOT, reason="real ABACUS/i-PI not configured"
)


def read_electronic_frames(directory):
    text = (directory / "OUT.ABACUS/running_scf.log").read_text()
    extras = [
        json.loads(line.split("Sending extras JSON = ", 1)[1])
        for line in text.splitlines()
        if "Sending extras JSON = " in line
    ]
    energies = np.array(
        [
            float(value) / (2 * RY_EV)
            for value in re.findall(r"#TOTAL ENERGY#\s+([-+\d.eE]+)\s+eV", text)
        ]
    )
    forces = []
    for block in text.split("TOTAL-FORCE (eV/Angstrom)")[1:]:
        values = re.findall(
            r"Si\d+\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)", block
        )[:2]
        assert len(values) == 2
        forces.append(np.asarray(values, float) * BOHR_A / (2 * RY_EV))
    assert len(extras) == len(energies) == len(forces)
    return extras, energies, np.asarray(forces)


def read_xyz_values(path):
    lines = path.read_text().splitlines()
    frames = []
    index = 0
    while index < len(lines):
        nat = int(lines[index])
        frames.append(
            [
                [float(x) for x in row.split()[1:4]]
                for row in lines[index + 2 : index + 2 + nat]
            ]
        )
        index += nat + 2
    return np.asarray(frames)


def run_endpoints(directory, endpoints, *, timeout=900):
    processes = []
    paths = [Path("/tmp") / ("ipi_" + address) for _, address in endpoints]
    environment = dict(
        os.environ,
        PYTHONPATH=str(IPI_ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    with ExitStack() as stack:
        try:
            ipilog = stack.enter_context((directory / "ipi.log").open("w"))
            ipi = subprocess.Popen(
                [sys.executable, str(Path(IPI_ROOT) / "bin/i-pi"), "input.xml"],
                cwd=directory,
                env=environment,
                stdout=ipilog,
                stderr=subprocess.STDOUT,
            )
            processes.append(ipi)
            deadline = time.monotonic() + 30
            while (
                not all(path.exists() for path in paths)
                and ipi.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert all(path.exists() for path in paths), (
                directory / "ipi.log"
            ).read_text()
            for (folder, _), path in zip(endpoints, paths):
                log = stack.enter_context((folder / "abacus.log").open("w"))
                env = dict(environment, ABACUS_SOCKET_ADDRESS=str(path) + ":UNIX")
                processes.append(
                    subprocess.Popen(
                        command(BINARY),
                        cwd=folder,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            assert ipi.wait(timeout=timeout) == 0, (directory / "ipi.log").read_text()
            for process in processes[1:]:
                assert process.wait(timeout=45) == 0
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            for path in paths:
                path.unlink(missing_ok=True)


@pytest.mark.parametrize("basis", ["pw", "lcao"])
@pytest.mark.parametrize("target", ["fermi", "workfunction"])
def test_linear_target_scan_and_restart(tmp_path, basis, target):
    address = configuration(tmp_path, basis, steps=3, target=target)
    tree = ET.parse(tmp_path / "input.xml")
    root = tree.getroot()
    electrons = root.find("system/motion/dynamics/electrons")
    name = "fermi_level" if target == "fermi" else "workfunction"
    initial, final = (-3.5, -3.0) if target == "fermi" else (4.0, 4.4)
    electrons.remove(electrons.find("target_" + name))
    for prefix, value in (("initial", initial), ("final", final)):
        ET.SubElement(
            electrons, prefix + "_target_" + name, units="electronvolt"
        ).text = str(value)
    ET.SubElement(electrons, "transition_steps").text = "6"
    properties = root.find("output/properties")
    properties.text = properties.text.replace(
        "]", ", " + name + "_target{electronvolt} ]"
    )
    tree.write(tmp_path / "input.xml")
    assert run_pair(tmp_path, address) == (0, 0)
    first, _ = audit(tmp_path, 4)
    values = np.atleast_2d(np.loadtxt(tmp_path / "cp.out"))
    # i-PI records the target used during completed step n at output step n+1.
    expected = initial + np.clip(np.maximum(values[:, 0] - 1, 0) / 6, 0, 1) * (
        final - initial
    )
    np.testing.assert_allclose(values[:, -1], expected, atol=1e-8, rtol=0)
    checkpoint = ET.parse(tmp_path / "RESTART")
    saved_electrons = checkpoint.getroot().find("system/motion/dynamics/electrons")
    saved_q_text = saved_electrons.find("q_init").text.strip()
    saved_q = float(saved_q_text)
    # The checkpoint stores the requested count; SCF returns actual occupations.
    assert saved_q == read_charge_requests(tmp_path)[-1][0]
    assert first[-1]["nelect"] == pytest.approx(saved_q, abs=1e-8, rel=0)
    saved_targets = [
        float(saved_electrons.find(prefix + "_target_" + name).text)
        / unit_to_internal("energy", "electronvolt", 1.0)
        for prefix in ("initial", "final")
    ]
    checkpoint.getroot().find("total_steps").text = "7"
    checkpoint.write(tmp_path / "restart.xml")
    assert run_pair(tmp_path, address, xml="restart.xml", suffix="-restart") == (0, 0)
    restarted, _ = audit(tmp_path, 5)
    assert read_charge_requests(tmp_path)[0][0] == saved_q
    assert restarted[0]["nelect"] == pytest.approx(saved_q, abs=1e-8, rel=0)
    final_values = np.atleast_2d(np.loadtxt(tmp_path / "cp.out"))
    progress = np.clip(np.maximum(final_values[:, 0] - 1, 0) / 6, 0, 1)
    expected = initial + progress * (final - initial)
    resumed = final_values[:, 0] > 3
    expected[resumed] = saved_targets[0] + progress[resumed] * (
        saved_targets[1] - saved_targets[0]
    )
    np.testing.assert_allclose(final_values[:, -1], expected, atol=1e-8, rtol=0)
    assert final_values[-1, -1] == pytest.approx(saved_targets[1], abs=1e-8, rel=0)


@pytest.mark.parametrize("basis", ["pw", "lcao"])
@pytest.mark.parametrize("span", [1, 2])
@pytest.mark.parametrize("direction", [-1, 1])
def test_charge_mixing_crosses_integer_endpoints(tmp_path, basis, span, direction):
    steps = 5
    address = configuration(tmp_path, basis, steps=steps, consolidate=direction > 0)
    tree = ET.parse(tmp_path / "input.xml")
    root = tree.getroot()
    low_socket = root.find("ffsocket")
    low_socket.set("name", "abacus_low")
    low_socket.find("address").text = address + "_low"
    high_socket = copy.deepcopy(low_socket)
    high_socket.set("name", "abacus_high")
    high_socket.find("address").text = address + "_high"
    root.insert(list(root).index(low_socket) + 1, high_socket)
    root.find("system/forces/force").set("forcefield", "abacus_low")
    electrons = root.find("system/motion/dynamics/electrons")
    electrons.set("charge_mixing", "true")
    ET.SubElement(electrons, "charge_mixing_span").text = str(span)
    ET.SubElement(electrons, "charge_mixing_low").text = "abacus_low"
    ET.SubElement(electrons, "charge_mixing_high").text = "abacus_high"
    electrons.find("q_init").text = str(8.0 - direction * 0.005)
    ET.SubElement(electrons, "p_init").text = str(direction * 100.0)
    potentiostat = electrons.find("potentiostat")
    potentiostat.find("tau").text = "1e8"
    ET.SubElement(potentiostat, "temp", units="kelvin").text = "0"
    keys = (
        "nelect,charge_mixing_lambda,charge_mixing_nelect_low,charge_mixing_nelect_high,"
        "fermi_level_eV,workfunction_eV,endpoint_low_energy,endpoint_high_energy"
    )
    ET.SubElement(
        root.find("output"), "trajectory", filename="mixed", stride="1", extra_type=keys
    ).text = "extras"
    ET.SubElement(
        root.find("output"), "trajectory", filename="force", stride="1", format="xyz"
    ).text = "forces{atomic_unit}"
    tree.write(tmp_path / "input.xml")
    endpoints = []
    for label in ("low", "high"):
        folder = tmp_path / label
        inputs(folder, basis, extra={"out_pot": 0})
        endpoints.append((folder, address + "_" + label))
    run_endpoints(tmp_path, endpoints)
    low, energy_low, force_low = read_electronic_frames(endpoints[0][0])
    high, energy_high, force_high = read_electronic_frames(endpoints[1][0])
    assert len(low) == len(high) == steps + 1
    mixed_paths = list(tmp_path.glob("cp.mixed*"))
    assert len(mixed_paths) == 1
    mixed = np.atleast_2d(np.loadtxt(mixed_paths[0]))
    assert mixed.shape == (steps + 1, 8)
    q, weight, qlow, qhigh = mixed[:, :4].T
    assert direction * (q[-1] - q[0]) > 0.01
    assert np.min(q) < 8.0 < np.max(q)
    np.testing.assert_allclose(qlow, span * np.floor(q / span), atol=1e-10, rtol=0)
    np.testing.assert_allclose(qhigh, qlow + span, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        [frame["nelect"] for frame in low], qlow, atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        [frame["nelect"] for frame in high], qhigh, atol=1e-8, rtol=0
    )
    for column, key in ((4, "fermi_level_eV"), (5, "workfunction_eV")):
        expected = (1 - weight) * np.array([frame[key] for frame in low])
        expected += weight * np.array([frame[key] for frame in high])
        np.testing.assert_allclose(mixed[:, column], expected, atol=1e-7, rtol=0)
    properties = np.atleast_2d(np.loadtxt(tmp_path / "cp.out"))
    # ABACUS log eV and i-PI output eV use different historical conversion constants.
    expected_energy = (
        (1 - weight) * energy_low + weight * energy_high
    ) / unit_to_internal("energy", "electronvolt", 1.0)
    np.testing.assert_allclose(mixed[:, 6], energy_low, atol=1e-8, rtol=0)
    np.testing.assert_allclose(mixed[:, 7], energy_high, atol=1e-8, rtol=0)
    np.testing.assert_allclose(properties[:, 1], expected_energy, atol=1e-5, rtol=0)
    force_paths = list(tmp_path.glob("cp.force*.xyz"))
    assert len(force_paths) == 1
    force = read_xyz_values(force_paths[0])
    expected_force = (1 - weight[:, None, None]) * force_low + weight[
        :, None, None
    ] * force_high
    np.testing.assert_allclose(force, expected_force, atol=1e-7, rtol=0)
    (tmp_path / "mixing-audit.json").write_text(
        json.dumps(
            {
                "span": span,
                "direction": direction,
                "q": q.tolist(),
                "low": qlow.tolist(),
                "high": qhigh.tolist(),
                "max_force_error": float(np.max(np.abs(force - expected_force))),
                "max_energy_error_eV": float(
                    np.max(np.abs(properties[:, 1] - expected_energy))
                ),
            },
            indent=2,
        )
        + "\n"
    )
