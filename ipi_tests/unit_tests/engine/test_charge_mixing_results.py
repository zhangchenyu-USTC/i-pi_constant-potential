"""Socket force remapping and complete electronic replies at mixed endpoints."""

import json

import numpy as np
import pytest

from ipi.engine.atoms import Atoms
from ipi.engine.beads import Beads
from ipi.engine.cell import Cell
from ipi.engine.constant_potential import (
    bind_constant_potential_provider,
    read_constant_potential_feedback,
)
from ipi.engine.forcefields import FFDirect
from ipi.engine.forcefields_cp import FFChargeMixTwoSockets
from ipi.engine.forces import ForceComponent
from ipi.interfaces.sockets import Driver, Status
from ipi_tests.unit_tests.engine.test_electronic_requests import owner, socket_provider
from types import SimpleNamespace as NS


def provider(natoms, active, nbeads=1):
    dynamics = owner(nbeads=nbeads, mode="workfunction")
    dynamics.beads = Beads(natoms, nbeads)
    low, _ = socket_provider(dynamics)
    high, _ = socket_provider(dynamics)
    low.active = high.active = np.array(active)
    wrapper = FFChargeMixTwoSockets(low, high, active=np.array(active), name="mixed")
    wrapper.configure_electrons(
        dict(
            enabled=True,
            neutral_electrons=8,
            charge_mixing=True,
            charge_mixing_span=1,
            potential_average_axis="z",
            potential_average_region=[1.0, 2.0],
        ),
        dynamics,
    )
    wrapper.set_electronic_state(8.2)
    return wrapper, dynamics


def reply(request, force, phi=4.0):
    q = request["electronic_context"].nelect
    return [
        q,
        force,
        np.eye(3) * q,
        dict(nelect=q, fermi_level_eV=q * 0.1, workfunction_eV=phi),
    ]


def receive(child, force, phi=4.0):
    """Use the production socket receive/scatter path with an in-memory payload."""
    driver = Driver.__new__(Driver)
    driver._recv_forces_bulk = lambda natoms: reply(child, force, phi)
    driver.get_status = lambda: Status.Up | Status.Ready
    child["offset"] = 0.0
    assert driver.dispatch_recv(child)


@pytest.mark.parametrize(
    "natoms,active", [(2, [-1]), (2, [1]), (3, [2, 0]), (2, [1, 0])]
)
def test_mixing_uses_full_force_coordinates_after_socket_receive(natoms, active):
    wrapper, _ = provider(natoms, active)
    request = wrapper.queue(Atoms(natoms), Cell(np.eye(3)))
    indices = request["active"]
    wire = np.arange(indices.size, dtype=float) + 1
    receive(request["child_low"], wire)
    receive(request["child_high"], 3 * wire)
    wrapper.poll()
    assert request["status"] == "Done"
    expected = np.zeros(3 * natoms)
    expected[indices] = 1.4 * wire
    np.testing.assert_allclose(request["result"][1], expected, atol=1e-14, rtol=0)
    assert request["result"][3]["nelect"] == pytest.approx(8.2, abs=1e-14)
    wrapper.release(request)
    assert (
        wrapper.requests
        == wrapper.endpoint_low.requests
        == wrapper.endpoint_high.requests
        == []
    )


def test_short_internal_endpoint_force_is_still_rejected():
    wrapper, _ = provider(2, [1])
    request = wrapper.queue(Atoms(2), Cell(np.eye(3)))
    with pytest.raises(RuntimeError, match="shape"):
        wrapper._mix_results(
            reply(request["child_low"], np.zeros(3)),
            reply(request["child_high"], np.zeros(3)),
            request,
        )
    wrapper.release(request)


def test_each_endpoint_electron_count_is_checked_before_mixing():
    wrapper, _ = provider(1, [-1])
    wrapper.set_electronic_state(8.5)
    request = wrapper.queue(Atoms(1), Cell(np.eye(3)))
    low = reply(request["child_low"], np.zeros(3))
    high = reply(request["child_high"], np.zeros(3))
    low[3]["nelect"] += 0.1
    high[3]["nelect"] -= 0.1  # Their weighted error cancels; both are still wrong.
    with pytest.raises(RuntimeError, match="snapshot"):
        wrapper._mix_results(low, high, request)
    wrapper.release(request)


@pytest.mark.parametrize("backend", ["direct", "mix"])
@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_all_cp_backends_share_nonfinite_result_checks(backend, component, bad):
    if backend == "mix":
        wrapper, _ = provider(1, [-1])
        request = wrapper.queue(Atoms(1), Cell(np.eye(3)))
        first = reply(request["child_low"], np.zeros(3))
        second = reply(request["child_high"], np.zeros(3))
    else:
        dynamics = owner()
        wrapper = FFDirect(pes="dummy", threaded=False)
        wrapper.driver.set_electronic_state = lambda q: None
        wrapper.configure_electrons(dict(enabled=True, neutral_electrons=8), dynamics)
        wrapper.set_electronic_state(8.2)
        request = wrapper._build_request(Atoms(1), Cell(np.eye(3)))
        first = reply(request, np.zeros(3))
        first[3] = json.dumps(first[3])
    if component == 0:
        first[0] = bad
    else:
        first[component].flat[0] = bad
    with pytest.raises(RuntimeError, match="non-finite"):
        if backend == "mix":
            wrapper._mix_results(first, second, request)
        else:
            wrapper._process_results(first, request)
    wrapper.release(request)


def test_two_beads_share_charge_and_average_only_complete_mixed_feedback():
    wrapper, dynamics = provider(2, [1], nbeads=2)
    dynamics.beads.q[1] = np.arange(6) * 0.01
    component = ForceComponent(ffield="mixed", nbeads=2, mts_weights=[1.0])
    component.bind(
        dynamics.beads, Cell(np.eye(3)), {"mixed": wrapper}, output_maker=None
    )
    dynamics.forces = NS(mforces=[component])
    bind_constant_potential_provider(dynamics, component, wrapper)
    for bead in component._forces:
        bead.queue()
    requests = [bead.request for bead in component._forces]
    assert not np.array_equal(requests[0]["pos"], requests[1]["pos"])
    assert len({r["electronic_context"].bead_id for r in requests}) == 2
    wrapper.set_electronic_state(9.7)  # Pending requests retain their original q.
    for index, request in enumerate(requests):
        assert request["electronic_context"].nelect == 8.2
        assert request["child_low"]["nelect"] == 8.0
        assert request["child_high"]["nelect"] == 9.0
        receive(request["child_low"], np.ones(3), phi=4.0 + index * 2)
        receive(request["child_high"], np.ones(3) * 3, phi=5.0 + index * 2)
    wrapper.poll()
    assert all(request["status"] == "Done" for request in requests)
    fermi, phi = read_constant_potential_feedback(dynamics)
    assert fermi == pytest.approx(0.82)
    assert phi == pytest.approx(5.2)
    assert dynamics.electronic_state._workfunction_average_total_samples == 0
    assert (
        wrapper.requests
        == wrapper.endpoint_low.requests
        == wrapper.endpoint_high.requests
        == []
    )
