#!/usr/bin/env python3
"""Run a fixed-cell constant-potential example in a new output directory."""

import argparse
import math
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "example", choices=("si2-pw", "si2-lcao", "al36-pw", "al36-lcao")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--temperature",
        type=float,
        help="Initial and thermostat temperature in kelvin.",
    )
    parser.add_argument(
        "--dt", type=float, help="Nuclear integration timestep in femtoseconds."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Maximum total runtime in seconds, including startup.",
    )
    args = parser.parse_args()
    if args.ranks < 1 or any(
        value is not None and value < 1 for value in (args.steps,)
    ):
        parser.error("ranks and steps must be positive")
    for name in ("temperature", "dt", "timeout"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    root = Path(__file__).resolve().parent
    ipi_root = (
        Path(os.environ.get("IPI_ROOT", str(root.parents[3]))).expanduser().resolve()
    )
    binary = os.environ.get("ABACUS_CP_EXECUTABLE")
    data = os.environ.get("ABACUS_PP_ORB")
    if not data and os.environ.get("ABACUS_SOURCE_ROOT"):
        data = str(Path(os.environ["ABACUS_SOURCE_ROOT"]).expanduser() / "tests/PP_ORB")
    if not binary or not data:
        parser.error(
            "Set ABACUS_CP_EXECUTABLE and ABACUS_PP_ORB (or ABACUS_SOURCE_ROOT)"
        )
    executable = Path(binary).expanduser().resolve()
    pp_orb = Path(data).expanduser().resolve()
    if not executable.is_file() or not (ipi_root / "bin/i-pi").is_file():
        parser.error("ABACUS or i-PI is missing; set ABACUS_CP_EXECUTABLE and IPI_ROOT")
    if not pp_orb.is_dir():
        parser.error(
            "ABACUS_PP_ORB must point to the pseudopotential/orbital directory"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.output.resolve()
    source = root / "cases" / args.example
    for name in ("INPUT", "STRU", "KPT", "init.xyz", "input.xml"):
        shutil.copy2(source / name, directory / name)
    input_file = directory / "INPUT"
    input_file.write_text(
        "\n".join(
            (
                f"{line.split()[0]} {pp_orb}"
                if line.split() and line.split()[0] in ("pseudo_dir", "orbital_dir")
                else line
            )
            for line in input_file.read_text().splitlines()
        )
        + "\n"
    )
    address = f"cpipi_{os.getpid()}_{time.time_ns()}"
    socket_path = Path("/tmp") / ("ipi_" + address)
    tree = ET.parse(directory / "input.xml")
    tree.getroot().find("ffsocket/address").text = address
    if args.steps is not None:
        tree.getroot().find("total_steps").text = str(args.steps)
    if args.temperature is not None:
        for element in (
            tree.getroot().find(".//initialize/velocities"),
            tree.getroot().find(".//ensemble/temperature"),
        ):
            element.set("units", "kelvin")
            element.text = str(args.temperature)
    if args.dt is not None:
        element = tree.getroot().find(".//dynamics/timestep")
        element.set("units", "femtosecond")
        element.text = str(args.dt)
    tree.write(directory / "input.xml")
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(ipi_root),
        PYTHONDONTWRITEBYTECODE="1",
        ABACUS_SOCKET_ADDRESS=str(socket_path) + ":UNIX",
    )
    processes = []
    started = time.monotonic()
    run_deadline = started + args.timeout if args.timeout is not None else float("inf")
    with (directory / "ipi.log").open("w") as ipilog, (directory / "abacus.log").open(
        "w"
    ) as abalog:
        try:
            ipi = subprocess.Popen(
                [sys.executable, str(ipi_root / "bin/i-pi"), "input.xml"],
                cwd=directory,
                env=environment,
                stdout=ipilog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(ipi)
            deadline = min(time.monotonic() + 30, run_deadline)
            while (
                not socket_path.exists()
                and ipi.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.1)
            if not socket_path.exists():
                raise RuntimeError(
                    f"i-PI did not open its socket; inspect {directory / 'ipi.log'}"
                )
            command = (
                shlex.split(os.environ.get("MPIEXEC", "mpirun"))
                + shlex.split(os.environ.get("ABACUS_TEST_MPI_OPTIONS", ""))
                + ["-np", str(args.ranks)]
                if args.ranks > 1
                else []
            ) + [str(executable)]
            abacus = subprocess.Popen(
                command,
                cwd=directory,
                env=environment,
                stdout=abalog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(abacus)
            early_exit_deadline = None
            while ipi.poll() is None:
                if time.monotonic() >= run_deadline:
                    raise TimeoutError(
                        f"Run exceeded {args.timeout} seconds; inspect {directory}"
                    )
                status = abacus.poll()
                if status is not None:
                    if status != 0:
                        raise RuntimeError(
                            f"ABACUS failed (status={status}); inspect {directory}"
                        )
                    # Allow i-PI to finish writing its outputs after a normal driver shutdown.
                    if early_exit_deadline is None:
                        early_exit_deadline = time.monotonic() + 5
                    elif time.monotonic() >= early_exit_deadline:
                        raise RuntimeError(
                            f"ABACUS exited before i-PI completed; inspect {directory}"
                        )
                time.sleep(0.2)
            status = abacus.wait(
                timeout=max(0.01, min(45, run_deadline - time.monotonic()))
            )
            if ipi.returncode != 0 or status != 0:
                raise RuntimeError(
                    f"Run failed (i-PI={ipi.returncode}, ABACUS={status}); inspect {directory}"
                )
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            socket_path.unlink(missing_ok=True)
    text = (directory / "OUT.ABACUS/running_scf.log").read_text()
    print(
        f"Completed: {directory}; force evaluations: {text.count('Sending extras JSON')}; "
        f"elapsed: {time.monotonic()-started:.3f} s"
    )


if __name__ == "__main__":
    main()
