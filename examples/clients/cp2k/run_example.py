#!/usr/bin/env python3
"""Run a CP2K constant-potential example in a new output directory."""

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("mixing", "ne"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--cp2k", default=os.environ.get("CP2K_EXECUTABLE", "cp2k.psmp")
    )
    parser.add_argument("--mpi-command", default=os.environ.get("MPIEXEC", "mpiexec"))
    parser.add_argument("--ipi-command", help="i-PI command; input.xml is appended")
    parser.add_argument("--timeout", type=float, help="Total runtime limit in seconds")
    args = parser.parse_args()
    if args.ranks < (2 if args.example == "mixing" else 1):
        parser.error("mixing needs at least two ranks; ne needs at least one")
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("timeout must be positive")
    executable = shutil.which(os.path.expanduser(args.cp2k))
    if executable is None:
        parser.error("CP2K is missing; set CP2K_EXECUTABLE or --cp2k")
    executable = str(Path(executable).resolve())

    root = Path(__file__).resolve().parents[3]
    name = "4-cp2k-implicit-mixing" if args.example == "mixing" else "5-cp2k-Ne"
    source = root / "examples/constant_potential" / name
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.output.resolve()
    files = (
        (
            "input.xml",
            "36Al.xyz",
            "cp2k_farming.inp",
            "cp2k_low/cp2k_low.inp",
            "cp2k_high/cp2k_high.inp",
        )
        if args.example == "mixing"
        else ("input.xml", "init.xyz", "cp2k.inp")
    )
    for name in files:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    tree = ET.parse(directory / "input.xml")
    tree.getroot().set("verbosity", "medium")
    tree.getroot().set("floatformat", "%24.16e")
    if args.steps is not None:
        tree.find("total_steps").text = str(args.steps)
    tree.write(directory / "input.xml")
    ports = [node.findtext("port").strip() for node in tree.findall("ffsocket")]
    cp_input = "cp2k.inp"
    if args.example == "mixing":
        cp_input = "cp2k_farming.inp"
        farming = directory / cp_input
        half = args.ranks // 2
        farming.write_text(
            re.sub(
                r"GROUP_PARTITION[^\n]*",
                f"GROUP_PARTITION {half} {args.ranks - half}",
                farming.read_text(),
            )
        )

    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(name, "1")
    environment["PYTHONUNBUFFERED"] = "1"
    ipi_command = (
        shlex.split(args.ipi_command)
        if args.ipi_command
        else [sys.executable, str(root / "bin/i-pi")]
    )
    cp_command = shlex.split(args.mpi_command) + [
        "-np",
        str(args.ranks),
        executable,
        "-i",
        cp_input,
        "-o",
        Path(cp_input).stem + ".out",
    ]
    processes = []
    start = time.monotonic()
    deadline = start + args.timeout if args.timeout is not None else float("inf")
    with (directory / "ipi.log").open("w") as ipilog, (directory / "cp2k.log").open(
        "w"
    ) as cplog:
        try:
            ipi = subprocess.Popen(
                ipi_command + ["input.xml"],
                cwd=directory,
                env=environment,
                stdout=ipilog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(ipi)
            startup_deadline = min(deadline, time.monotonic() + 60)
            while True:
                if ipi.poll() is not None:
                    raise RuntimeError(
                        f"i-PI exited during startup ({ipi.returncode}); see {directory / 'ipi.log'}"
                    )
                log = (directory / "ipi.log").read_text()
                if all(f"and port number {port}\n" in log for port in ports):
                    break
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError(
                        f"i-PI did not open its sockets; see {directory / 'ipi.log'}"
                    )
                time.sleep(0.1)
            cp2k = subprocess.Popen(
                cp_command,
                cwd=directory,
                env=environment,
                stdout=cplog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(cp2k)
            exit_deadline = float("inf")
            while True:
                statuses = [process.poll() for process in processes]
                if any(code not in (None, 0) for code in statuses):
                    raise RuntimeError(
                        f"Run failed: i-PI={statuses[0]}, CP2K={statuses[1]}; see {directory}"
                    )
                if all(code == 0 for code in statuses):
                    break
                if 0 in statuses and exit_deadline == float("inf"):
                    exit_deadline = time.monotonic() + 60
                if time.monotonic() >= min(deadline, exit_deadline):
                    raise TimeoutError(f"Run or shutdown timed out; see {directory}")
                time.sleep(0.2)
        finally:
            for process in reversed(processes):
                stop(process)
    print(
        f"Completed {args.example}: {directory}; i-PI=0, CP2K=0; elapsed={time.monotonic() - start:.2f}s"
    )


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except (OSError, RuntimeError, TimeoutError) as error:
        sys.exit(str(error))
