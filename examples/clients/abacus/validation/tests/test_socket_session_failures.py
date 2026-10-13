"""Ordinary EXIT and constant-potential unfinished-request handling."""

import socket
import struct

import pytest

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
