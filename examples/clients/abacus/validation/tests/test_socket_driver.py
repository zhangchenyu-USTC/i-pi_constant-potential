"""Opt-in integration tests against real ABACUS binaries (no model force substitutes).

Set ABACUS_CP_EXECUTABLE, ABACUS_REFERENCE_EXECUTABLE, ABACUS_PP_ORB and optionally
ABACUS_TEST_RANKS. Run pytest with --basetemp outside the source/backup trees.
"""

import json
import os
from pathlib import Path
import re
import shlex
import socket
import struct
import subprocess
import time

import numpy as np
import pytest

BINARY = os.environ.get("ABACUS_CP_EXECUTABLE")
REFERENCE = os.environ.get("ABACUS_REFERENCE_EXECUTABLE")
PP_ORB = os.environ.get("ABACUS_PP_ORB")
pytestmark = pytest.mark.skipif(
    not BINARY or not PP_ORB, reason="real ABACUS executable/data not configured"
)
RY_EV = 13.605698  # ABACUS beta9 constants.h
BOHR_A = 0.5291770
RY_BOHR3_TO_KBAR = (4.35974394e-18 / 2.0) / (0.529177e-10**3) * 1e-8
CELL = np.eye(3) * 10.26
POSITIONS = np.array([[0.0, 0.0, 0.0], [2.4, 2.4, 2.4]])


def header(name):
    return name.encode().ljust(12)


def command(binary):
    ranks = int(os.environ.get("ABACUS_TEST_RANKS", "1"))
    mpi_options = shlex.split(os.environ.get("ABACUS_TEST_MPI_OPTIONS", ""))
    launcher = shlex.split(os.environ.get("MPIEXEC", "mpirun"))
    return (
        [*launcher, *mpi_options, "-np", str(ranks)] if ranks > 1 else []
    ) + shlex.split(str(binary))


def inputs(
    directory,
    basis="pw",
    *,
    positions=POSITIONS,
    cell=CELL,
    socket_mode=True,
    extra=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    parameters = dict(
        calculation="scf",
        basis_type=basis,
        pseudo_dir=PP_ORB,
        orbital_dir=PP_ORB,
        nbands=12,
        ecutwfc=40,
        scf_thr=1e-10,
        scf_nmax=150,
        symmetry=0,
        smearing_method="fd",
        smearing_sigma=0.02,
        mixing_beta=0.3,
        cal_force=1,
        cal_stress=0,
        chg_extrap="first-order",
        out_pot="2 12",
        out_ndigits=12,
    )
    if socket_mode:
        parameters["socket_driver"] = 1
    if basis == "lcao":
        parameters["ks_solver"] = "scalapack_gvx"
    if extra:
        parameters.update(extra)
    parameters = {key: value for key, value in parameters.items() if value is not None}
    (directory / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        + "\n".join(f"{k} {v}" for k, v in parameters.items())
        + "\n"
    )
    needs_orbitals = basis == "lcao" or bool(parameters.get("dft_plus_u", 0))
    orbital = (
        "\nNUMERICAL_ORBITAL\nSi_gga_8au_100Ry_2s2p1d.orb\n" if needs_orbitals else ""
    )
    stru = "ATOMIC_SPECIES\nSi 28.0855 Si_ONCV_PBE-1.0.upf\n" + orbital
    stru += "\nLATTICE_CONSTANT\n1.0\n\nLATTICE_VECTORS\n"
    stru += "\n".join(" ".join(f"{v:.16g}" for v in row) for row in cell)
    stru += "\n\nATOMIC_POSITIONS\nCartesian\nSi\n0.0\n2\n"
    stru += (
        "\n".join(" ".join(f"{v:.16g}" for v in row) + " 1 1 1" for row in positions)
        + "\n"
    )
    (directory / "STRU").write_text(stru)
    (directory / "KPT").write_text("K_POINTS\n0\nGamma\n1 1 1 0 0 0\n")


class Client:
    def __init__(self, directory, *, transport="unix", binary=None):
        self.directory = directory
        if transport == "unix":
            self.path = Path("/tmp") / f"abacus_cp_test_{os.getpid()}_{time.time_ns()}"
            self.server = socket.socket(socket.AF_UNIX)
            self.server.bind(str(self.path))
            address = str(self.path) + ":UNIX"
        else:
            self.path = None
            self.server = socket.socket(socket.AF_INET)
            self.server.bind(("127.0.0.1", 0))
            address = "127.0.0.1:" + str(self.server.getsockname()[1])
        self.server.listen(1)
        self.server.settimeout(180)
        env = dict(
            os.environ,
            ABACUS_SOCKET_ADDRESS=address,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        self.log = (directory / "abacus.log").open("w")
        self.process = subprocess.Popen(
            command(binary or BINARY),
            cwd=directory,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self.connection = None
        try:
            self.connection, _ = self.server.accept()
            self.connection.settimeout(180)
        except BaseException:
            self.close()
            raise

    def read(self, size):
        result = b""
        while len(result) < size:
            data = self.connection.recv(size - len(result))
            if not data:
                raise EOFError(
                    f"ABACUS disconnected; inspect {self.directory / 'abacus.log'}"
                )
            result += data
        return result

    def status(self):
        self.connection.sendall(header("STATUS"))
        return self.read(12).decode().strip()

    def initialize(self, *, cp=True, axis="z", raw=None):
        assert self.status() == "NEEDINIT"
        text = (
            raw
            if raw is not None
            else (
                f"NELECT : 8, neutral_electrons : 8, "
                f"potential_average_axis : {axis}, potential_average_min_A : 2, potential_average_max_A : 4"
                if cp
                else ""
            )
        )
        payload = text.encode()
        self.connection.sendall(
            header("INIT") + struct.pack("=ii", 0, len(payload)) + payload
        )
        assert self.status() == "READY"

    @staticmethod
    def posdata(positions, cell):
        return (
            header("POSDATA")
            + np.asarray(cell.T, dtype="=f8").tobytes()
            + np.asarray(np.linalg.inv(cell.T), dtype="=f8").tobytes()
            + struct.pack("=i", len(positions))
            + np.asarray(positions, dtype="=f8").tobytes()
        )

    def calculate(
        self, nelect=None, *, flags=0, positions=POSITIONS, cell=CELL, consolidated=True
    ):
        assert self.status() == "READY"
        charge = (
            b""
            if nelect is None
            else header("CHGDATA") + struct.pack("=di", nelect, flags)
        )
        payload = charge + self.posdata(positions, cell)
        if consolidated:
            self.connection.sendall(payload + header("GETFORCE"))
        else:
            # Exercise stream fragmentation, including splits inside binary payloads.
            for start in range(0, len(payload), 7):
                self.connection.sendall(payload[start : start + 7])
            assert self.status() == "HAVEDATA"
            self.connection.sendall(header("GETFORCE"))
        assert self.read(12).decode().strip() == "FORCEREADY"
        energy = struct.unpack("=d", self.read(8))[0]
        nat = struct.unpack("=i", self.read(4))[0]
        force = np.frombuffer(self.read(24 * nat), dtype="=f8").copy().reshape(nat, 3)
        virial = np.frombuffer(self.read(72), dtype="=f8").copy().reshape(3, 3)
        size = struct.unpack("=i", self.read(4))[0]
        extras = json.loads(self.read(size))
        metadata = {"schema", "present", "scf_converged"}
        assert extras["schema"] == "abacus.socket.properties.v1"
        assert isinstance(extras["scf_converged"], bool)
        assert "energy" in extras["present"]
        assert set(extras["present"]) <= {"energy", "forces", "stress"}
        electronic = (
            {
                "fermi_level_eV",
                "nelect",
                "workfunction_eV",
                "electrostatic_reference_eV",
            }
            if nelect is not None
            else set()
        )
        assert set(extras) == metadata | electronic
        if nelect is not None:
            assert extras["scf_converged"]
            assert extras["present"] == ["energy", "forces"]
        assert self.status() == "READY"
        return energy, force, virial, extras

    def finish(self):
        self.connection.sendall(header("EXIT"))
        self.connection.close()
        self.connection = None
        status = self.process.wait(timeout=30)
        assert status == 0, (status, self.directory)

    def close(self):
        if self.connection:
            self.connection.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        self.server.close()
        self.log.close()
        if self.path:
            self.path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def standalone(directory, basis, nelect, positions, *, extra=None):
    if not REFERENCE:
        raise RuntimeError(
            "ABACUS_REFERENCE_EXECUTABLE must select an independent ordinary build"
        )
    options = {"nelec": nelect, **(extra or {})}
    inputs(directory, basis, positions=positions, socket_mode=False, extra=options)
    with (directory / "abacus.log").open("w") as log:
        subprocess.run(
            command(REFERENCE),
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=180,
        )
    text = (directory / "OUT.ABACUS/running_scf.log").read_text()
    energy = float(re.findall(r"!FINAL_ETOT_IS\s+([-+\d.eE]+)", text)[-1]) / (2 * RY_EV)
    fermi_values = re.findall(r"E_Fermi\s+([-+\d.eE]+)\s+([-+\d.eE]+)", text)
    if fermi_values:
        fermi = float(fermi_values[-1][1])
    else:
        spin_values = re.findall(
            r"E_Fermi_(?:up|dw)\s+([-+\d.eE]+)\s+([-+\d.eE]+)", text
        )
        assert len(spin_values) >= 2
        fermi = np.mean([float(x[1]) for x in spin_values[-2:]])
    section = text.rsplit("TOTAL-FORCE (eV/Angstrom)", 1)[-1]
    forces = np.array(
        [
            list(map(float, match))
            for match in re.findall(
                r"Si\d+\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)", section
            )[:2]
        ]
    )
    assert forces.shape == (2, 3)
    return energy, forces * BOHR_A / (2 * RY_EV), fermi


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_charge_and_position_sequence(tmp_path, basis):
    inputs(tmp_path / "socket", basis)
    errors = []
    with Client(
        tmp_path / "socket", transport="inet" if basis == "lcao" else "unix"
    ) as client:
        client.initialize()
        for index, q in enumerate([8.0, 8.2, 7.8, 8.0]):
            pos = POSITIONS.copy()
            if index == 3:
                pos[1] += [0.08, -0.04, 0.02]
            result = client.calculate(q, positions=pos, consolidated=index % 2 == 0)
            reference = standalone(tmp_path / f"reference_{index}", basis, q, pos)
            error = [
                abs(result[0] - reference[0]),
                np.max(np.abs(result[1] - reference[1])),
                abs(result[3]["fermi_level_eV"] - reference[2]),
            ]
            errors.append([float(v) for v in error])
            assert error[0] < 1e-6
            assert error[1] < 1e-5
            assert error[2] < 1e-4
            assert result[3]["nelect"] == pytest.approx(q, abs=1e-8, rel=0)
        client.finish()
    (tmp_path / "comparison.json").write_text(json.dumps(errors, indent=2))


def cube_average(directory, axis, minimum=2.0, maximum=4.0):
    lines = (directory / "OUT.ABACUS/potes.cube").read_text().splitlines()
    nat = abs(int(lines[2].split()[0]))
    grid = [int(line.split()[0]) for line in lines[3:6]]
    steps = np.array([[float(v) for v in line.split()[1:4]] for line in lines[3:6]])
    lattice = steps * np.array(grid)[:, None]
    other = [i for i in range(3) if i != axis]
    height = (
        abs(np.linalg.det(lattice))
        / np.linalg.norm(np.cross(lattice[other[0]], lattice[other[1]]))
        * BOHR_A
    )
    values = np.fromstring(" ".join(lines[6 + nat :]), sep=" ").reshape(grid)
    indices = np.arange(grid[axis]) / grid[axis] * height
    return (
        float(
            np.take(
                values,
                np.flatnonzero((indices >= minimum) & (indices < maximum)),
                axis=axis,
            ).mean()
        )
        * RY_EV
    )


@pytest.mark.parametrize(
    "basis,axis", [(b, a) for b in ("pw", "lcao") for a in ("x", "y", "z")]
)
def test_workfunction_matches_cube_on_triclinic_cell(tmp_path, basis, axis):
    cell = CELL.copy()
    cell[1, 0] = 0.9
    cell[2, :2] = [0.4, 0.7]
    inputs(tmp_path, basis, cell=cell)
    with Client(tmp_path) as client:
        client.initialize(axis=axis)
        result = client.calculate(8.1, cell=cell)
        average = cube_average(tmp_path, "xyz".index(axis))
        assert abs(average - result[3]["electrostatic_reference_eV"]) < 1e-4
        assert result[3]["workfunction_eV"] == pytest.approx(
            average - result[3]["fermi_level_eV"], abs=1e-4
        )
        client.finish()


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_ordinary_socket_default_symmetry_and_multiple_positions(tmp_path, basis):
    inputs(tmp_path / "socket", basis, extra={"symmetry": None})
    with Client(tmp_path / "socket") as client:
        client.initialize(cp=False)
        for index in range(2):
            positions = POSITIONS.copy()
            positions[1] += np.array([0.05, -0.03, 0.02]) * index
            result = client.calculate(positions=positions)
            reference = standalone(
                tmp_path / f"reference_{index}", basis, 8.0, positions
            )
            assert abs(result[0] - reference[0]) < 1e-6
            assert np.max(np.abs(result[1] - reference[1])) < 1e-5
            assert "fermi_level_eV" not in result[3]
        client.finish()


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_zero_request_exit(tmp_path, basis):
    inputs(tmp_path, basis)
    with Client(tmp_path) as client:
        assert client.status() == "NEEDINIT"
        client.finish()


@pytest.mark.parametrize(
    "failure",
    [
        "force_before_pos",
        "pos_before_init",
        "missing_charge",
        "duplicate_charge",
        "unknown_header",
        "negative_init",
        "truncated_header",
        "bad_count",
        "bad_cell",
        "bad_inverse",
        "nan_position",
        "ne_bit",
        "nan_charge",
        "band_capacity",
        "legacy_flag",
        "legacy_flag_three",
        "unconverged",
        "bad_window",
        "fixed_fractional",
    ],
)
def test_invalid_requests_fail_without_results(tmp_path, failure):
    extra = {"scf_nmax": 1} if failure == "unconverged" else {}
    if failure == "fixed_fractional":
        extra["smearing_method"] = "fixed"
    inputs(tmp_path, extra=extra)
    with Client(tmp_path) as client:
        if failure not in ("pos_before_init", "negative_init", "truncated_header"):
            if failure == "bad_window":
                client.initialize(
                    raw="NELECT:8,neutral_electrons:8,potential_average_axis:z,potential_average_min_A:20,potential_average_max_A:30"
                )
            else:
                client.initialize()
                if failure == "legacy_flag_three":
                    client.calculate(8.0)
        if failure == "force_before_pos":
            payload = header("GETFORCE")
        elif failure == "pos_before_init" or failure == "missing_charge":
            payload = client.posdata(POSITIONS, CELL)
        elif failure == "negative_init":
            payload = header("INIT") + struct.pack("=ii", 0, -1)
        elif failure == "truncated_header":
            payload = b"STAT"
        elif failure == "unknown_header":
            payload = header("UNKNOWN")
        else:
            q = (
                float("nan")
                if failure == "nan_charge"
                else (99.0 if failure == "band_capacity" else 8.1)
            )
            flag = {"ne_bit": 2, "legacy_flag": 1, "legacy_flag_three": 3}.get(
                failure, 0
            )
            payload = header("CHGDATA") + struct.pack("=di", q, flag)
            if failure == "duplicate_charge":
                payload += payload
            else:
                positions, cell = POSITIONS.copy(), CELL.copy()
                if failure == "bad_cell":
                    cell[0, 0] += 0.01
                if failure == "nan_position":
                    positions[0, 0] = float("nan")
                posdata = client.posdata(positions, cell)
                if failure == "bad_count":
                    posdata = posdata[:156] + struct.pack("=i", 2147483647)
                if failure == "bad_inverse":
                    posdata = (
                        posdata[:84]
                        + np.zeros(9, dtype="=f8").tobytes()
                        + posdata[156:]
                    )
                payload += posdata + header("GETFORCE")
        client.connection.sendall(payload)
        client.connection.shutdown(socket.SHUT_WR)
        with pytest.raises((EOFError, ConnectionResetError)):
            client.read(12)
        status = client.process.wait(timeout=30)
        assert status != 0
    text = (tmp_path / "abacus.log").read_text()
    if failure == "unconverged":
        assert "SCF did not converge" in text
    if failure == "fixed_fractional":
        assert "fractional electron populations" in text


def response_calls(directory):
    text = (directory / "OUT.ABACUS/running_scf.log").read_text()
    matches = re.findall(r"surchem\s+v_cor\s+[\d.]+\s+(\d+)\s", text)
    assert matches, "missing actual surchem timer accounting"
    return int(matches[-1])


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_native_solvent_updates_on_every_changed_request(tmp_path, basis):
    for name, count in [("update_only", 1), ("with_freeze", 2), ("with_refresh", 3)]:
        directory = tmp_path / name
        inputs(directory, basis, extra={"imp_sol": 1, "eb_k": 80})
        with Client(directory) as client:
            client.initialize()
            for index in range(count):
                result = client.calculate(8.0 - index * 0.1, flags=0)
                assert result[3]["nelect"] == pytest.approx(
                    8.0 - index * 0.1, abs=1e-8, rel=0
                )
                assert result[3]["electrostatic_reference_eV"] == pytest.approx(
                    cube_average(directory, 2), abs=1e-4
                )
            client.finish()
        text = (directory / "OUT.ABACUS/running_scf.log").read_text()
        assert text.count("Sending extras JSON") == count
    assert response_calls(tmp_path / "update_only") < response_calls(
        tmp_path / "with_freeze"
    )
    assert response_calls(tmp_path / "with_refresh") > response_calls(
        tmp_path / "with_freeze"
    )


@pytest.mark.parametrize("basis", ["pw", "lcao"])
@pytest.mark.parametrize("setting", ["spin", "fixed_magnetization", "gate"])
def test_electronic_control_with_spin_or_gate(tmp_path, basis, setting):
    extra = (
        {"nspin": 2, "nupdown": 0.4 if setting == "fixed_magnetization" else 0}
        if setting != "gate"
        else {
            "gate_flag": 1,
            "zgate": 0.75,
            "efield_flag": 1,
            "dip_cor_flag": 1,
            "efield_pos_max": 0.9,
            "efield_pos_dec": 0.1,
        }
    )
    if setting == "gate":
        extra["scf_thr"] = 1e-12
    inputs(tmp_path / "socket", basis, extra=extra)
    with Client(tmp_path / "socket") as client:
        client.initialize()
        for i, q in enumerate([8.0, 8.2]):
            result = client.calculate(q)
            reference = standalone(
                tmp_path / f"reference_{i}", basis, q, POSITIONS, extra=extra
            )
            assert abs(result[0] - reference[0]) < 1e-6
            assert np.max(np.abs(result[1] - reference[1])) < 1e-5
            assert abs(result[3]["fermi_level_eV"] - reference[2]) < 1e-4
        client.finish()


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_ordinary_scf_matches_independent_reference(tmp_path, basis, monkeypatch):
    reference = standalone(
        tmp_path / "reference", basis, 7.8, POSITIONS, extra={"imp_sol": 1}
    )
    monkeypatch.setitem(globals(), "REFERENCE", BINARY)
    modified = standalone(
        tmp_path / "modified", basis, 7.8, POSITIONS, extra={"imp_sol": 1}
    )
    assert abs(modified[0] - reference[0]) < 1e-6
    assert np.max(np.abs(modified[1] - reference[1])) < 1e-5
    assert abs(modified[2] - reference[2]) < 1e-4


def log_virial(directory, cell=CELL):
    """Read independently calculated ABACUS stress and convert to wire virial."""
    text = (directory / "OUT.ABACUS/running_scf.log").read_text()
    section = text.rsplit("#TOTAL-STRESS (kbar)#", 1)
    assert len(section) == 2, "missing calculated stress in standalone reference"
    rows = re.findall(
        r"^\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*$",
        section[1],
        flags=re.MULTILINE,
    )[:3]
    stress = np.array(rows, dtype=float)
    assert stress.shape == (3, 3)
    return (stress / RY_BOHR3_TO_KBAR * np.linalg.det(cell) / 2.0).T


@pytest.mark.parametrize("basis", ["pw", "lcao"])
@pytest.mark.parametrize(
    "forces,stress", [(False, False), (True, False), (False, True), (True, True)]
)
def test_ordinary_properties_and_physical_stress(tmp_path, basis, forces, stress):
    directory = tmp_path / "socket"
    inputs(
        directory, basis, extra={"cal_force": int(forces), "cal_stress": int(stress)}
    )
    with Client(directory) as client:
        client.initialize(cp=False)
        result = client.calculate()
        assert result[3]["scf_converged"]
        assert result[3]["present"] == (
            ["energy"] + (["forces"] if forces else []) + (["stress"] if stress else [])
        )
        client.finish()
    reference_dir = tmp_path / "reference"
    reference = standalone(
        reference_dir,
        basis,
        8.0,
        POSITIONS,
        extra={"cal_force": 1, "cal_stress": int(stress)},
    )
    assert abs(result[0] - reference[0]) < 1e-6
    if forces:
        assert np.max(np.abs(result[1] - reference[1])) < 1e-5
    else:
        assert np.array_equal(result[1], np.zeros_like(result[1]))
    if stress:
        expected = log_virial(reference_dir)
        assert np.max(np.abs(expected)) > 1e-5
        assert np.max(np.abs(result[2] - expected)) < 1e-5
    else:
        assert np.array_equal(result[2], np.zeros((3, 3)))


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_ordinary_unconverged_reports_metadata(tmp_path, basis):
    inputs(tmp_path, basis, extra={"scf_nmax": 1})
    with Client(tmp_path) as client:
        client.initialize(cp=False)
        result = client.calculate()
        assert result[3]["scf_converged"] is False
        assert np.isfinite(result[0])
        assert np.all(np.isfinite(result[1]))
        client.finish()


@pytest.mark.parametrize(
    "extra,message",
    [({"cal_force": 0}, "cal_force=1"), ({"cal_stress": 1}, "cal_stress=0")],
)
def test_constant_potential_requires_supported_properties(tmp_path, extra, message):
    inputs(tmp_path, extra=extra)
    with Client(tmp_path) as client:
        with pytest.raises((EOFError, ConnectionResetError)):
            client.initialize()
        assert client.process.wait(timeout=30) != 0
    text = (tmp_path / "abacus.log").read_text()
    assert message in text


@pytest.mark.skipif(
    int(os.environ.get("ABACUS_TEST_RANKS", "1")) < 2,
    reason="comparison requires at least two MPI ranks",
)
@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_kpoint_pools_preserve_charge_and_potential(tmp_path, basis):
    """Check distributed potential reduction against the same k mesh in one pool."""
    ranks = int(os.environ["ABACUS_TEST_RANKS"])
    assert ranks % 2 == 0, "multi-pool integration test needs an even MPI rank count"
    samples = []
    for pools in (1, 2):
        directory = tmp_path / f"pools_{pools}"
        inputs(directory, basis, extra={"kpar": pools, "imp_sol": 1, "eb_k": 80})
        (directory / "KPT").write_text("K_POINTS\n0\nGamma\n2 1 1 0 0 0\n")
        with Client(directory) as client:
            client.initialize()
            sequence = []
            for index, q in enumerate([8.0, 8.2]):
                result = client.calculate(q, flags=0)
                assert result[3]["nelect"] == pytest.approx(q, abs=1e-8, rel=0)
                assert result[3]["electrostatic_reference_eV"] == pytest.approx(
                    cube_average(directory, 2), abs=1e-4
                )
                sequence.append(result)
            samples.append(sequence)
            client.finish()
        log = (directory / "OUT.ABACUS/running_scf.log").read_text()
        assert log.count("Sending extras JSON") == 2
    comparisons = []
    for first, second in zip(*samples):
        error = [
            abs(first[0] - second[0]),
            np.max(np.abs(first[1] - second[1])),
            abs(first[3]["fermi_level_eV"] - second[3]["fermi_level_eV"]),
            abs(
                first[3]["electrostatic_reference_eV"]
                - second[3]["electrostatic_reference_eV"]
            ),
        ]
        assert error[0] < 1e-6
        assert error[1] < 1e-5
        assert error[2] < 1e-4
        assert error[3] < 1e-4
        comparisons.append([float(value) for value in error])
    (tmp_path / "pool-comparison.json").write_text(json.dumps(comparisons, indent=2))


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_first_cell_unit_roundoff_does_not_allow_later_cell_changes(tmp_path, basis):
    inputs(tmp_path, basis)
    # The two applications historically use separately rounded length constants.
    wire_cell = CELL * (BOHR_A * 1.8897261)
    relative_roundoff = np.max(np.abs(wire_cell - CELL)) / np.max(np.abs(CELL))
    assert 0 < relative_roundoff < 1e-6
    with Client(tmp_path) as client:
        client.initialize()
        client.calculate(8.0, cell=wire_cell)
        client.calculate(8.1, cell=wire_cell)
        changed = wire_cell.copy()
        changed[0, 0] += 1e-7
        payload = (
            header("CHGDATA")
            + struct.pack("=di", 8.0, 0)
            + client.posdata(POSITIONS, changed)
            + header("GETFORCE")
        )
        client.connection.sendall(payload)
        with pytest.raises((EOFError, ConnectionResetError)):
            client.read(12)
        assert client.process.wait(timeout=30) != 0
    assert "variable-cell socket updates" in (tmp_path / "abacus.log").read_text()


@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_second_order_extrapolation_across_periodic_boundary(tmp_path, basis):
    initial = POSITIONS.copy()
    initial[0, 0] = CELL[0, 0] - 0.03
    inputs(
        tmp_path / "socket",
        basis,
        positions=initial,
        extra={"chg_extrap": "second-order"},
    )
    errors = []
    with Client(tmp_path / "socket") as client:
        client.initialize()
        for index, x in enumerate([CELL[0, 0] - 0.03, 0.01, 0.06, 0.1]):
            positions = POSITIONS.copy()
            positions[0, 0] = x
            result = client.calculate(8.1, positions=positions)
            reference = standalone(
                tmp_path / f"reference_{index}", basis, 8.1, positions
            )
            error = [
                abs(result[0] - reference[0]),
                np.max(np.abs(result[1] - reference[1])),
                abs(result[3]["fermi_level_eV"] - reference[2]),
            ]
            assert error[0] < 1e-6
            assert error[1] < 1e-5
            assert error[2] < 1e-4
            errors.append([float(value) for value in error])
        client.finish()
    (tmp_path / "pbc-comparison.json").write_text(json.dumps(errors, indent=2))


@pytest.mark.parametrize(
    "raw",
    [
        "solvation_update_stride:1",
        "NELECT:8,neutral_electrons:8,solvation_update_stride:5",
    ],
)
def test_removed_init_is_rejected(tmp_path, raw):
    inputs(tmp_path)
    with Client(tmp_path) as client:
        with pytest.raises((EOFError, ConnectionResetError)):
            client.initialize(raw=raw)
        assert client.process.wait(timeout=30) != 0
    assert (
        "solvation_update_stride has been removed"
        in (tmp_path / "abacus.log").read_text()
    )
