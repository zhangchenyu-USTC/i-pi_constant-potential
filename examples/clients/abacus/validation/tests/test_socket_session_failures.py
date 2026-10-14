"""Socket lifecycle failures and Makov-Payne electronic-control compatibility."""

import re
import socket
import struct

import numpy as np
import pytest
import test_socket_driver as driver

from test_socket_driver import (
    BINARY,
    REFERENCE,
    PP_ORB,
    CELL,
    POSITIONS,
    Client,
    header,
    inputs,
)

pytestmark = pytest.mark.skipif(
    not BINARY or not PP_ORB, reason="real ABACUS executable/data not configured"
)


@pytest.mark.skipif(not REFERENCE, reason="ordinary ABACUS reference not configured")
@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_ordinary_exit_can_discard_unread_result(tmp_path, basis):
    for name, binary in (("modified", BINARY), ("reference", REFERENCE)):
        directory = tmp_path / name
        inputs(directory, basis)
        with Client(directory, binary=binary) as client:
            client.initialize(cp=False)
            client.connection.sendall(client.posdata(POSITIONS, CELL))
            assert client.status() == "HAVEDATA"
            client.finish()


@pytest.mark.parametrize("pending", ["charge", "force"])
@pytest.mark.parametrize("termination", ["exit", "eof"])
def test_unfinished_cp_session_is_rejected_on_every_rank(
    tmp_path, pending, termination
):
    inputs(tmp_path)
    with Client(tmp_path) as client:
        client.initialize()
        client.connection.sendall(header("CHGDATA") + struct.pack("=di", 8.0, 0))
        if pending == "force":
            client.connection.sendall(client.posdata(POSITIONS, CELL))
            assert client.status() == "HAVEDATA"
        else:
            assert client.status() == "READY"
        if termination == "exit":
            client.connection.sendall(header("EXIT"))
        else:
            client.connection.shutdown(socket.SHUT_WR)
        with pytest.raises((EOFError, ConnectionResetError)):
            client.read(12)
        assert client.process.wait(timeout=30) != 0
    text = (tmp_path / "abacus.log").read_text()
    expected = "unfinished request/result" if termination == "exit" else "pending"
    assert expected in text


def _cubic_makov_inputs(directory, basis, **kwargs):
    """Use the native simple-cubic STRU syntax with the existing Si2 fixture."""
    positions = np.asarray(kwargs.pop("positions", POSITIONS)) / CELL[0, 0]
    extra = {
        "latname": "sc",
        "assume_isolated": "makov-payne",
        **kwargs.pop("extra", {}),
    }
    inputs(directory, basis, positions=positions, extra=extra, **kwargs)
    stru = directory / "STRU"
    text = stru.read_text().replace(
        "LATTICE_CONSTANT\n1.0", f"LATTICE_CONSTANT\n{CELL[0, 0]}"
    )
    text = re.sub(r"\nLATTICE_VECTORS\n(?:[^\n]*\n){3}", "\n", text)
    stru.write_text(
        text.replace("ATOMIC_POSITIONS\nCartesian", "ATOMIC_POSITIONS\nDirect")
    )


@pytest.mark.parametrize("basis", ["pw", "lcao"])
@pytest.mark.parametrize("correction", ["makov-payne", "mp", "m-p"])
def test_makov_payne_electron_control_is_rejected(tmp_path, basis, correction):
    _cubic_makov_inputs(tmp_path, basis, extra={"assume_isolated": correction})
    with Client(tmp_path) as client:
        client.initialize()
        client.connection.sendall(
            header("CHGDATA")
            + struct.pack("=di", 7.8, 0)
            + client.posdata(POSITIONS, CELL)
            + header("GETFORCE")
        )
        with pytest.raises((EOFError, ConnectionResetError)):
            client.read(12)
        assert client.process.wait(timeout=30) != 0
    text = (tmp_path / "abacus.log").read_text()
    assert (
        "constant-potential electron control is incompatible with the energy-only Makov-Payne correction"
        in text
    )
    running = (tmp_path / "OUT.ABACUS/running_scf.log").read_text()
    assert "ABACUS socket return energy" not in running
    assert "!FINAL_ETOT_IS" not in running


@pytest.mark.skipif(not REFERENCE, reason="ordinary ABACUS reference not configured")
@pytest.mark.parametrize("basis", ["pw", "lcao"])
def test_ordinary_makov_payne_scf_and_socket_match_reference(
    tmp_path, basis, monkeypatch
):
    monkeypatch.setattr(driver, "inputs", _cubic_makov_inputs)
    reference = driver.standalone(tmp_path / "reference", basis, 7.8, POSITIONS)
    monkeypatch.setattr(driver, "REFERENCE", BINARY)
    candidate = driver.standalone(tmp_path / "candidate", basis, 7.8, POSITIONS)
    directory = tmp_path / "socket"
    _cubic_makov_inputs(directory, basis, extra={"nelec": 7.8})
    with Client(directory) as client:
        client.initialize(cp=False)
        frame = client.calculate()
        assert frame[3]["scf_converged"]
        client.finish()
    for result in (candidate, frame):
        assert abs(result[0] - reference[0]) < 1e-6
        np.testing.assert_allclose(result[1], reference[1], atol=1e-5, rtol=0)
    assert abs(candidate[2] - reference[2]) < 1e-4
    for name in ("reference", "candidate", "socket"):
        log = (tmp_path / name / "OUT.ABACUS/running_scf.log").read_text()
        assert "Makov-Payne correction" in log
        assert "#SCF IS CONVERGED#" in log
