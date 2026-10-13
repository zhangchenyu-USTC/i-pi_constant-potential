"""Real CP occupation counts, independent of the requested/socket electron count.

Run this module separately with ABACUS_TEST_RANKS=1 and 4. The serial group
collects six basis/spin cases; the four-rank group also collects k-point/band
decompositions. Inapplicable MPI layouts are not collected as skipped tests.
The existing opt-in executable/data environment and Client helper are reused.
"""

import json
import math
import os
import re

import pytest

from test_socket_driver import BINARY, PP_ORB, Client, inputs

pytestmark = pytest.mark.skipif(
    not BINARY or not PP_ORB, reason="real ABACUS executable/data not configured"
)

ELECTRON_TOLERANCE = 1.0e-8
REQUESTS = (8.0, 8.2, 7.8, 8.0)
RANKS = int(os.environ.get("ABACUS_TEST_RANKS", "1"))

SPINS = [
    pytest.param(1, 0.0, id="rks"),
    pytest.param(2, 0.0, id="spin-unconstrained"),
    pytest.param(2, 0.4, id="spin-fixed-0.4"),
]
LAYOUTS = [
    pytest.param("pw", 1, 1, id="pw-pool1-band1"),
    pytest.param("lcao", 1, 1, id="lcao-pool1-band1"),
]
if RANKS >= 4 and RANKS % 4 == 0:
    LAYOUTS += [
        pytest.param("pw", 2, 1, id="pw-pool2-band1"),
        pytest.param("pw", 1, 2, id="pw-bpcg-pool1-band2"),
        pytest.param("pw", 2, 2, id="pw-bpcg-pool2-band2"),
        pytest.param("lcao", 2, 1, id="lcao-pool2-band1"),
    ]


def read_occupation_frames(path, *, nspin, nkpoints, nbands):
    """Read global weighted occupations once; reject missing/duplicate records.

    ABACUS write_eig_file gathers the distributed band columns and writes each
    (spin, global k-point) block once. Its final column is wg, already including
    k-point weights and the applicable spin degeneracy. No extra weight,
    degeneracy, MPI-rank factor, or pool factor belongs in this sum.
    """
    text = path.read_text()
    frame_headers = list(re.finditer(r"^\s*(\d+)\s+# ionic step\s*$", text, re.M))
    assert frame_headers, f"No completed occupation frames in {path}"
    frames = []
    for index, frame_header in enumerate(frame_headers):
        frame_id = int(frame_header.group(1))
        assert frame_id == index + 1, f"Missing/duplicate ionic frame in {path}"
        end = (
            frame_headers[index + 1].start()
            if index + 1 < len(frame_headers)
            else len(text)
        )
        body = text[frame_header.end() : end]
        spin_header = re.search(r"^\s*Spin number\s+(\d+)\s*$", body, re.M)
        assert spin_header and int(spin_header.group(1)) == nspin
        k_headers = list(
            re.finditer(
                r"^\s*spin=(\d+)\s+k-point=(\d+)/(\d+)\s+Cartesian=.*$", body, re.M
            )
        )
        blocks = {}
        for kindex, k_header in enumerate(k_headers):
            spin, kpoint, total_k = map(int, k_header.groups())
            assert 1 <= spin <= nspin and total_k == nkpoints
            assert 1 <= kpoint <= nkpoints
            key = (spin, kpoint)
            assert (
                key not in blocks
            ), f"Duplicated spin/k-point {key} in frame {frame_id}"
            kend = (
                k_headers[kindex + 1].start()
                if kindex + 1 < len(k_headers)
                else len(body)
            )
            rows = [
                line.split()
                for line in body[k_header.end() : kend].splitlines()
                if line.strip()
            ]
            assert len(rows) == nbands, (frame_id, key, len(rows), nbands)
            assert all(len(row) == 3 for row in rows), (frame_id, key, rows)
            assert [int(row[0]) for row in rows] == list(range(1, nbands + 1))
            eigenvalues = [float(row[1]) for row in rows]
            occupations = [float(row[2]) for row in rows]
            assert all(math.isfinite(value) for value in eigenvalues + occupations)
            blocks[key] = occupations
        expected = {
            (spin, kpoint)
            for spin in range(1, nspin + 1)
            for kpoint in range(1, nkpoints + 1)
        }
        assert set(blocks) == expected, (frame_id, set(blocks), expected)
        spin_counts = [
            math.fsum(
                value
                for (s, _), values in blocks.items()
                if s == spin
                for value in values
            )
            for spin in range(1, nspin + 1)
        ]
        frames.append(
            {
                "frame": frame_id,
                "occupation_sum": math.fsum(spin_counts),
                "spin_counts": spin_counts,
                "unique_spin_kpoint_blocks": len(blocks),
                "bands_per_block": nbands,
            }
        )
    return frames


@pytest.mark.parametrize("basis,kpar,bndpar", LAYOUTS)
@pytest.mark.parametrize("nspin,nupdown", SPINS)
def test_actual_occupations_follow_charge_sequence(
    tmp_path, basis, kpar, bndpar, nspin, nupdown
):
    # Odd band count checks uneven 7/6 partitions, not only equal-size groups.
    nbands = 13 if bndpar > 1 else 12
    nkpoints = 2 if kpar > 1 else 1
    extra = {
        "nspin": nspin,
        "nupdown": nupdown,
        "nbands": nbands,
        "kpar": kpar,
        "bndpar": bndpar,
        "scf_thr": 1.0e-11,
        "scf_nmax": 250,
        "chg_extrap": "none",
        "out_pot": 0,
    }
    if bndpar > 1:
        extra["ks_solver"] = "bpcg"
    directory = tmp_path / "socket"
    inputs(directory, basis, extra=extra)
    if nkpoints > 1:
        (directory / "KPT").write_text("K_POINTS\n0\nGamma\n2 1 1 0 0 0\n")
    evidence = {
        "basis": basis,
        "mpi_ranks": RANKS,
        "kpar": kpar,
        "bndpar": bndpar,
        "nspin": nspin,
        "nupdown": nupdown,
        "nbands": nbands,
        "absolute_electron_tolerance": ELECTRON_TOLERANCE,
        "frames": [],
    }
    evidence_file = tmp_path / "occupation-count.json"
    with Client(directory) as client:
        client.connection.settimeout(600)
        client.initialize()
        for frame_id, requested in enumerate(REQUESTS, start=1):
            extras = client.calculate(requested)[3]
            frames = read_occupation_frames(
                directory / "OUT.ABACUS/eig_occ.txt",
                nspin=nspin,
                nkpoints=nkpoints,
                nbands=nbands,
            )
            assert len(frames) == frame_id
            actual = frames[-1]
            reported = extras["nelect"]
            errors = {
                "occupation_minus_request": actual["occupation_sum"] - requested,
                "reported_minus_request": reported - requested,
                "reported_minus_occupation": reported - actual["occupation_sum"],
            }
            if nupdown:
                for spin, target in enumerate(
                    ((requested + nupdown) / 2, (requested - nupdown) / 2)
                ):
                    errors[f"spin_{spin + 1}_minus_target"] = (
                        actual["spin_counts"][spin] - target
                    )
            evidence["frames"].append(
                {
                    **actual,
                    "requested_nelect": requested,
                    "reported_nelect": reported,
                    "errors": errors,
                }
            )
            evidence_file.write_text(json.dumps(evidence, indent=2) + "\n")
            assert all(
                abs(error) <= ELECTRON_TOLERANCE for error in errors.values()
            ), evidence
        client.finish()
    evidence["exit_code"] = 0
    evidence_file.write_text(json.dumps(evidence, indent=2) + "\n")
