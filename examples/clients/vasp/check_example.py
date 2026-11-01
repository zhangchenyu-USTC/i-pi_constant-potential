#!/usr/bin/env python3
"""Run a three-step VASP example on a node and check the socket responses."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def run_ipi():
    sys.path.insert(0, str(ROOT))
    from ipi.interfaces.sockets import Driver, DriverSocket

    send, receive = DriverSocket.sendall, Driver._recv_force_data
    receive_bulk = Driver._recv_forces_bulk
    lock = threading.Lock()

    def record(data):
        with lock, open("wire-trace.jsonl", "a") as stream:
            stream.write(json.dumps(data) + "\n")

    def observed_send(self, data, *args, **kwargs):
        if bytes(data[:12]).strip() == b"CHGDATA":
            q, flags = struct.unpack_from("=di", data, 12)
            record(dict(event="request", nelect=q, flags=flags))
        return send(self, data, *args, **kwargs)

    def observe(original):
        def wrapped(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            record(
                dict(
                    event="result",
                    energy=float(result[0]),
                    extras=result[3],
                    finite=bool(
                        np.isfinite(result[0])
                        and np.isfinite(result[1]).all()
                        and np.isfinite(result[2]).all()
                    ),
                )
            )
            return result

        return wrapped

    DriverSocket.sendall = observed_send
    Driver._recv_force_data = observe(receive)
    Driver._recv_forces_bulk = observe(receive_bulk)
    sys.argv = [str(ROOT / "bin/i-pi"), "-V", "medium", "input.xml"]
    runpy.run_path(str(ROOT / "bin/i-pi"), run_name="__main__")


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example", choices=["2-vasp-implicit", "3-vasp-Ne"], required=True
    )
    parser.add_argument("--vasp", type=Path, required=True)
    parser.add_argument("--potcar", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="New test directory")
    parser.add_argument("--ranks", type=int, default=48)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--timeout", type=float, default=20000)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    source = ROOT / "examples/constant_potential" / args.example
    for name in ["input.xml", "init.xyz", "POSCAR", "INCAR", "KPOINTS"]:
        shutil.copy2(source / name, work / name)
    shutil.copy2(args.potcar, work / "POTCAR")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    tree = ET.parse(work / "input.xml")
    tree.getroot().find("total_steps").text = "3"
    ff = tree.getroot().find("ffsocket")
    ff.find("address").text = "127.0.0.1"
    ff.find("port").text = str(port)
    tree.write(work / "input.xml")
    incar = (work / "INCAR").read_text()
    incar = re.sub(r"(?im)^\s*PORT\s*=.*$", f"PORT = {port}", incar)
    incar = re.sub(r"(?im)^\s*IHOST\s*=.*$", "IHOST = 127.0.0.1", incar)
    (work / "INCAR").write_text(incar)
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    ipi = client = None
    started = time.monotonic()
    status = dict(
        example=args.example,
        status="RUNNING",
        ranks=args.ranks,
        binary=str(args.vasp.resolve()),
        binary_sha256=hashlib.sha256(args.vasp.read_bytes()).hexdigest(),
        input_sha256={
            name: hashlib.sha256((work / name).read_bytes()).hexdigest()
            for name in [
                "input.xml",
                "init.xyz",
                "INCAR",
                "POSCAR",
                "KPOINTS",
                "POTCAR",
            ]
        },
    )
    try:
        with (work / "ipi.log").open("w") as ilog, (work / "vasp.log").open(
            "w"
        ) as vlog:
            ipi = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--ipi-child"],
                cwd=work,
                env=env,
                stdout=ilog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while "Created inet socket" not in (work / "ipi.log").read_text():
                if ipi.poll() is not None or time.monotonic() - started > 60:
                    raise RuntimeError("i-PI did not start; see ipi.log")
                time.sleep(0.2)
            client = subprocess.Popen(
                [args.mpiexec, "-n", str(args.ranks), str(args.vasp.resolve())],
                cwd=work,
                env=env,
                stdout=vlog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while ipi.poll() is None and client.poll() is None:
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError("Example exceeded the time limit")
                time.sleep(0.5)
            status.update(
                ipi_exit=ipi.wait(timeout=60), vasp_exit=client.wait(timeout=60)
            )
        assert status["ipi_exit"] == status["vasp_exit"] == 0, status
        events = [
            json.loads(line)
            for line in (work / "wire-trace.jsonl").read_text().splitlines()
        ]
        requests = [e for e in events if e["event"] == "request"]
        results = [e for e in events if e["event"] == "result"]
        assert (
            len(requests) == len(results) == 4
        ), "Three steps must use four evaluations"
        errors = []
        for request, result in zip(requests, results):
            extras = result["extras"]
            if isinstance(extras, str):
                extras = json.loads(extras)
            error = abs(request["nelect"] - extras["nelect"])
            assert result["finite"] and error <= 1e-8
            assert all(
                np.isfinite(extras[k])
                for k in [
                    "nelect",
                    "fermi_level_eV",
                    "vacuum_level_eV",
                    "workfunction_eV",
                ]
            )
            assert (
                abs(
                    extras["workfunction_eV"]
                    - extras["vacuum_level_eV"]
                    + extras["fermi_level_eV"]
                )
                < 1e-8
            )
            errors.append(error)
        assert (work / "OUTCAR").read_text().count(
            "aborting loop because EDIFF is reached"
        ) == 4
        status.update(
            status="PASS", requests=4, responses=4, max_electron_error=max(errors)
        )
    except BaseException as error:
        status.update(status="FAIL", error=repr(error))
        raise
    finally:
        stop(client)
        stop(ipi)
        status["seconds"] = time.monotonic() - started
        (work / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status))


if __name__ == "__main__":
    if sys.argv[1:] == ["--ipi-child"]:
        run_ipi()
    else:
        main()
