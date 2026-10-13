"""Request publication, cancellation and electronic-provider ownership regressions."""

from dataclasses import FrozenInstanceError
from concurrent.futures import Future
from types import SimpleNamespace as NS
import threading
import time

import numpy as np
import pytest

from ipi.engine.atoms import Atoms
from ipi.engine.beads import Beads
from ipi.engine.cell import Cell
from ipi.engine.constant_potential import (
    bind_constant_potential_provider,
    electronic_request_context,
    read_constant_potential_feedback,
    validate_constant_potential_ownership,
)
from ipi.engine.forcefields import FFDirect, FFSocket
from ipi.engine.forcefields_cp import FFChargeMixTwoSockets
from ipi.engine.forces import ForceComponent, Forces, ScaledForceComponent
from ipi.engine.potentiostat import ElectronicState
from ipi.interfaces.sockets import Driver, InterfaceSocket, Status


def owner(q=8.2, nbeads=1, mode="fermi"):
    targets = {"target_ef": 0.3} if mode == "fermi" else {"target_workfunction": 0.3}
    state = ElectronicState(q_init=q, mass=1.0, mode=mode, **targets)
    state.neutral_electrons = 8
    return NS(
        electronic_state=state, beads=Beads(1, nbeads), ensemble=NS(hweights=np.ones(1))
    )


def socket_provider(dynamics=None, *, threaded=True, pars=None):
    dynamics = owner() if dynamics is None else dynamics
    interface = NS(
        mode="inet", batch_size=1, requests=None, offset=0.0, poll=lambda: None
    )
    ff = FFSocket(name="cp", interface=interface, threaded=threaded, pars=pars)
    config = dict(
        enabled=True,
        neutral_electrons=8,
        potential_average_axis="z",
        potential_average_region=[1.0, 2.0],
    )
    ff.configure_electrons(config, dynamics=dynamics)
    ff.set_electronic_state(dynamics.electronic_state.q)
    return ff, dynamics


@pytest.mark.parametrize("threaded", [False, True])
def test_complete_request_is_published_before_any_consumer(threaded):
    ff, dynamics = socket_provider(threaded=threaded)
    observed = []

    class ObservedQueue(list):
        def append(self, request):
            # This is the earliest possible observation of the published object.
            observed.append(
                (
                    request["nelect"],
                    request["mode_flags"],
                    request["pars"],
                    request["electronic_context"],
                )
            )
            super().append(request)

    ff.requests = ff.socket.requests = ObservedQueue()
    ff.socket.poll = lambda: observed.append("synchronous consumer")
    request = ff.queue(Atoms(1), Cell(np.eye(3)))
    assert observed[0][:2] == (8.2, 0)
    assert (
        "NELECT : 8.199999999999999" in observed[0][2]
        or "NELECT : 8.2" in observed[0][2]
    )
    assert "neutral_electrons : 8" in observed[0][2]
    assert ("synchronous consumer" in observed) is (not threaded)
    ff.set_electronic_state(8.7)
    dynamics.electronic_state.q = 8.7
    assert request["nelect"] == 8.2
    with pytest.raises(FrozenInstanceError):
        request["electronic_context"].nelect = 8.7


def test_invalid_request_never_enters_queue():
    ff, _ = socket_provider(pars={"NELECT": 99})
    with pytest.raises(ValueError, match="reserved"):
        ff.queue(Atoms(1), Cell(np.eye(3)))
    assert ff.requests == []


def test_workfunction_init_is_a_snapshot():
    ff, dynamics = socket_provider(owner(mode="workfunction"))
    snapshot = electronic_request_context(ff)
    ff.electrons_config["potential_average_axis"] = "x"
    ff.electrons_config["potential_average_region"][:] = [3.0, 4.0]
    request = ff.queue(Atoms(1), Cell(np.eye(3)), request_context=snapshot)
    assert "potential_average_axis : z" in request["pars"]
    assert request["electronic_context"].potential_region == snapshot.potential_region


def test_direct_evaluation_and_validation_use_the_queued_charge():
    ff = FFDirect(pes="asym_dw_constant_potential", threaded=True)
    dynamics = owner()
    ff.configure_electrons(dict(enabled=True, neutral_electrons=8), dynamics=dynamics)
    ff.set_electronic_state(8.2)
    first = ff.queue(Atoms(1), Cell(np.eye(3)))
    ff.set_electronic_state(8.7)
    second = ff.queue(Atoms(1), Cell(np.eye(3)))
    ff.poll()
    assert first["result"][3]["nelect"] == 8.2
    assert second["result"][3]["nelect"] == 8.7


def mixed_provider():
    low, dynamics = socket_provider()
    high, _ = socket_provider(dynamics)
    wrapper = FFChargeMixTwoSockets(low, high)
    wrapper.configure_electrons(
        dict(
            enabled=True, neutral_electrons=8, charge_mixing=True, charge_mixing_span=1
        ),
        dynamics,
    )
    wrapper.set_electronic_state(8.2)
    return wrapper, low, high, dynamics


def test_both_endpoint_requests_share_one_immutable_snapshot(monkeypatch):
    wrapper, low, high, _ = mixed_provider()
    original = low._publish_request

    def publish(request):
        wrapper.set_electronic_state(9.7)
        return original(request)

    monkeypatch.setattr(low, "_publish_request", publish)
    request = wrapper.queue(Atoms(1), Cell(np.eye(3)))
    assert request["q_current"] == 8.2
    assert request["lambda"] == pytest.approx(0.2)
    assert request["child_low"]["nelect"] == 8.0
    assert request["child_high"]["nelect"] == 9.0


def test_endpoint_geometry_does_not_change_during_construction(monkeypatch):
    wrapper, low, high, _ = mixed_provider()
    atoms, cell = Atoms(1), Cell(np.eye(3) * 10)
    atoms.q[:] = [0.1, 0.2, 0.3]
    original = low._build_request

    def build(*args, **kwargs):
        request = original(*args, **kwargs)
        atoms.q[:] = [1, 2, 3]
        cell.h[:] = np.eye(3) * 20
        return request

    monkeypatch.setattr(low, "_build_request", build)
    request = wrapper.queue(atoms, cell)
    for child in (request["child_low"], request["child_high"]):
        assert np.allclose(child["pos"], [0.1, 0.2, 0.3])
        assert np.array_equal(child["cell"][0], np.eye(3) * 10)


def test_second_endpoint_validation_failure_publishes_neither():
    wrapper, low, high, _ = mixed_provider()
    high.pars = {"NELECT": 99}
    with pytest.raises(ValueError, match="reserved"):
        wrapper.queue(Atoms(1), Cell(np.eye(3)))
    assert wrapper.requests == low.requests == high.requests == []


def test_endpoint_publication_failure_cancels_queued_sibling(monkeypatch):
    wrapper, low, high, _ = mixed_provider()
    monkeypatch.setattr(
        high,
        "_publish_request",
        lambda request: (_ for _ in ()).throw(RuntimeError("broken endpoint")),
    )
    with pytest.raises(RuntimeError, match="broken endpoint"):
        wrapper.queue(Atoms(1), Cell(np.eye(3)))
    assert wrapper.requests == low.requests == high.requests == []


def test_endpoint_publication_failure_retains_running_sibling_until_drained(
    monkeypatch,
):
    wrapper, low, high, _ = mixed_provider()
    original = low._publish_request

    def publish(request):
        original(request)
        request["status"] = "Running"
        return request

    monkeypatch.setattr(low, "_publish_request", publish)
    monkeypatch.setattr(
        high,
        "_publish_request",
        lambda request: (_ for _ in ()).throw(RuntimeError("broken endpoint")),
    )
    with pytest.raises(RuntimeError, match="broken endpoint"):
        wrapper.queue(Atoms(1), Cell(np.eye(3)))
    assert wrapper.requests == high.requests == []
    running = low.requests[0]
    assert running["cancelled"]
    low.socket.poll = lambda: running.update(status="Done")
    low.poll()
    assert low.requests == []


def test_endpoint_failure_marks_parent_failed_and_cancels_sibling(monkeypatch):
    wrapper, low, high, _ = mixed_provider()
    request = wrapper.queue(Atoms(1), Cell(np.eye(3)))
    request["child_low"]["status"] = "Running"
    request["child_high"]["status"] = "Exit"
    failures = []
    monkeypatch.setattr(
        wrapper,
        "notify_failure",
        lambda: failures.append(wrapper._failure.message),
    )
    wrapper.poll()
    assert request["status"] == "Exit"
    assert request._event_done.is_set()
    assert "endpoint failed" in wrapper._failure.message
    assert not failures  # A synchronous poll returns the failed request to its caller.
    assert request["child_low"] in low.requests
    assert request["child_low"]["cancelled"]
    assert high.requests == []
    # A failed wrapper never publishes another pair of endpoint evaluations.
    with pytest.raises(RuntimeError, match="endpoint failed"):
        wrapper.queue(Atoms(1), Cell(np.eye(3)))
    assert low.requests == [request["child_low"]]
    assert high.requests == []


def test_inflight_cancel_is_drained_and_disconnect_never_requeues():
    ff, _ = socket_provider()
    request = ff.queue(Atoms(1), Cell(np.eye(3)))
    request["status"] = "Running"
    ff.cancel_request(request)
    assert request in ff.requests  # The transport must still drain this response.
    ff.socket.poll = lambda: request.update(status="Done", result=[0, [], [], {}])
    ff.poll()
    assert request not in ff.requests

    request = ff.queue(Atoms(1), Cell(np.eye(3)))
    request["status"] = "Running"
    ff.cancel_request(request)
    interface = InterfaceSocket()
    client = NS(status=Status.Up)
    interface._requeue_disconnected(request, client)
    assert request["status"] == "Exit"
    assert request._event_done.is_set()
    interface.prlist = [request]
    ready = NS(status=Status.Up | Status.Ready)
    assert not interface.dispatch_free_client(ready)
    assert interface.prlist == []


def test_cancel_after_pending_list_snapshot_prevents_dispatch():
    ff, _ = socket_provider()
    request = ff.queue(Atoms(1), Cell(np.eye(3)))
    interface = InterfaceSocket()
    interface.prlist = [request]
    # Dispatch pauses while checking the client, after entering its pending list.
    entered, proceed = threading.Event(), threading.Event()

    class PausedClient:
        status = Status.Up | Status.Ready
        locked = False

        @property
        def lastreq(self):
            entered.set()
            assert proceed.wait(2)
            return None

    outcomes = []
    thread = threading.Thread(
        target=lambda: outcomes.append(interface.dispatch_free_client(PausedClient()))
    )
    thread.start()
    assert entered.wait(2)
    ff.cancel_request(request)
    proceed.set()
    thread.join(2)
    assert not thread.is_alive()
    assert outcomes == [False]
    assert ff.requests == interface.prlist == []


def test_cancelled_running_result_is_received_before_removal(monkeypatch):
    ff, _ = socket_provider()
    request = ff.queue(Atoms(1), Cell(np.eye(3)))
    request.update(status="Running", offset=0)
    ff.cancel_request(request)
    interface = InterfaceSocket()
    client = NS(status=Status.Up | Status.HasData)
    received = []

    def receive(natoms):
        assert request in ff.requests
        received.append(natoms)
        return [1.0, np.zeros(3), np.zeros((3, 3)), {"raw": ""}]

    client._recv_forces_bulk = receive
    client.get_status = lambda: setattr(client, "status", Status.Up | Status.Ready)
    client.dispatch_recv = lambda req: Driver.dispatch_recv(client, req)
    interface.clients = [client]
    interface.requests = ff.requests
    interface.jobs = [[request, client, None]]
    interface.prlist = []
    monkeypatch.setattr(
        "ipi.interfaces.sockets.select.select", lambda *args: ([client], [], [])
    )
    ff.socket.poll = interface._pool_distribute_select
    ff.poll()
    assert received == [1]
    assert request._event_done.is_set()
    assert ff.requests == interface.jobs == []


@pytest.mark.parametrize("cancelled", [False, True])
def test_legacy_disconnect_requeues_only_live_requests(monkeypatch, cancelled):
    ff, _ = socket_provider()
    request = ff.queue(Atoms(1), Cell(np.eye(3)))
    request["status"] = "Running"
    if cancelled:
        ff.cancel_request(request)
    interface = InterfaceSocket(consolidate_messages=False)
    client = NS(
        status=Status.Disconnected,
        peername="test",
        shutdown=lambda *_: None,
        close=lambda: None,
    )
    future = Future()
    future.set_result(None)
    interface.clients = [client]
    interface.jobs = [[request, client, future]]
    interface.server = object()
    monkeypatch.setattr(
        "ipi.interfaces.sockets.select.select", lambda *args: ([], [], [])
    )
    interface.pool_update()
    assert request["status"] == ("Exit" if cancelled else "Queued")
    assert interface.clients == interface.jobs == []


def test_cancelled_legacy_timeout_does_not_retry():
    ff, _ = socket_provider()
    request = ff.queue(Atoms(1), Cell(np.eye(3)))
    request.update(status="Running", start=time.time() - 10)
    ff.cancel_request(request)
    interface = InterfaceSocket(timeout=1, consolidate_messages=False)
    client = NS(
        status=Status.Up, peername="test", shutdown=lambda *_: None, close=lambda: None
    )
    assert interface.check_job_finished(request, client, None) == 0
    assert request["status"] == "Exit"
    assert request._event_done.is_set()


def bound_provider(nbeads=1, mode="fermi"):
    ff, dynamics = socket_provider(owner(nbeads=nbeads, mode=mode))
    component = ForceComponent(ffield="cp", nbeads=nbeads, mts_weights=[1.0])
    component.bind(dynamics.beads, Cell(np.eye(3)), {"cp": ff}, output_maker=None)
    dynamics.forces = NS(
        mforces=[component], extras={"nelect": [8.2], "fermi_level_eV": [999]}
    )
    bind_constant_potential_provider(dynamics, component, ff)
    return dynamics, component, ff


def test_feedback_uses_provider_and_checks_each_bead():
    dynamics, component, ff = bound_provider(4)
    component._extras.set(
        dict(nelect=[8.2] * 4, fermi_level_eV=[1.0, 2.0, 3.0, 4.0]), manual=False
    )
    assert read_constant_potential_feedback(dynamics) == (2.5, None)
    component._extras.set(
        dict(nelect=[8.1, 8.3, 8.2, 8.2], fermi_level_eV=[1.0] * 4), manual=False
    )
    with pytest.raises(RuntimeError, match="disagree"):
        read_constant_potential_feedback(dynamics)


def test_workfunction_feedback_averages_beads_without_updating_history():
    dynamics, component, ff = bound_provider(4, mode="workfunction")
    extras = dict(
        nelect=[8.2] * 4,
        fermi_level_eV=[1.0, 2.0, 3.0, 4.0],
        workfunction_eV=[4.0, 6.0, 8.0, 10.0],
    )
    component._extras.set(extras, manual=False)
    history = list(dynamics.electronic_state._workfunction_average_history)
    assert read_constant_potential_feedback(dynamics) == (2.5, 7.0)
    ff.update()
    assert dynamics.electronic_state._workfunction_average_history == history


@pytest.mark.parametrize("field", ["weight", "mts", "hweight"])
def test_changed_nonunit_weight_is_rejected_before_feedback(field):
    dynamics, component, _ = bound_provider()
    if field == "weight":
        component.weight = 2
    elif field == "mts":
        component.mts_weights = np.array([2.0])
    else:
        dynamics.ensemble.hweights[0] = 2
    with pytest.raises(ValueError, match="weight"):
        read_constant_potential_feedback(dynamics)


@pytest.mark.parametrize("reverse", [False, True])
def test_plain_system_cannot_borrow_cp_backend_in_either_bind_order(reverse):
    dynamics, cp_component, ff = bound_provider()
    cp = NS(forces=dynamics.forces, ensemble=NS(bias=NS(mforces=[])))
    plain_component = NS(ff=ff, _forces=[NS(ff=ff)])
    plain = NS(forces=NS(mforces=[plain_component]), ensemble=NS(bias=NS(mforces=[])))
    systems = [cp, plain] if not reverse else [plain, cp]
    with pytest.raises(ValueError, match="shared"):
        validate_constant_potential_ownership(NS(syslist=systems))
    validate_constant_potential_ownership(NS(syslist=[cp]))


def test_owned_bias_consumer_is_rejected():
    dynamics, component, ff = bound_provider()
    bias = NS(ff=ff, _forces=[NS(ff=ff)])
    system = NS(forces=dynamics.forces, ensemble=NS(bias=NS(mforces=[bias])))
    with pytest.raises(ValueError, match="shared"):
        validate_constant_potential_ownership(NS(syslist=[system]))


def test_scaled_hamiltonian_proxy_is_not_a_second_backend_consumer():
    dynamics, component, _ = bound_provider()
    proxy = ScaledForceComponent(component, scaling=0.0)
    system = NS(forces=dynamics.forces, ensemble=NS(bias=NS(mforces=[proxy])))
    validate_constant_potential_ownership(NS(syslist=[system]))


def test_plain_consumer_of_mixing_endpoint_is_rejected():
    wrapper, low, _, dynamics = mixed_provider()
    component = NS(ff=wrapper, _forces=[], _electronic_owner=dynamics)
    main = NS(forces=NS(mforces=[component]), ensemble=NS(bias=NS(mforces=[])))
    plain_component = NS(ff=low, _forces=[])
    plain = NS(forces=NS(mforces=[plain_component]), ensemble=NS(bias=NS(mforces=[])))
    with pytest.raises(ValueError, match="shared"):
        validate_constant_potential_ownership(NS(syslist=[main, plain]))


def test_auxiliary_clone_inherits_owner_and_q_dependency():
    ff, dynamics = socket_provider()
    cell = Cell(np.eye(3))
    forces = Forces()
    forces.bind(
        dynamics.beads,
        cell,
        [ForceComponent(ffield="cp", mts_weights=[1.0])],
        {"cp": ff},
        [],
        None,
    )
    dynamics.forces = forces
    bind_constant_potential_provider(dynamics, forces.mforces[0], ff)
    clone = forces.clone(dynamics.beads.clone(), cell.clone())
    bead = clone.mforces[0]._forces[0]
    bead._ufvx.set([0.0, np.zeros(3), np.zeros((3, 3)), {}], manual=False)
    assert not bead._ufvx.tainted()
    dynamics.electronic_state.q = 8.7
    assert bead._ufvx.tainted()
    bead.queue()
    assert bead.request["nelect"] == 8.7
    assert bead.request["electronic_context"].owner_id == id(dynamics)
    assert bead.request["electronic_context"].purpose == "auxiliary"


def test_wrong_bead_reply_is_rejected_after_releasing_request(monkeypatch):
    dynamics, component, ff = bound_provider()
    # Process-wide cleanup is covered by isolated failure subprocess tests.
    monkeypatch.setattr(ff, "notify_failure", lambda: None)
    bead = component._forces[0]
    bead.queue()
    bead.request.update(
        status="Done",
        result=[
            0.0,
            np.zeros(3),
            np.zeros((3, 3)),
            dict(nelect=8.3, fermi_level_eV=0.2),
        ],
    )
    with pytest.raises(RuntimeError, match="snapshot"):
        bead.get_all()
    assert bead.request is None
    assert ff.requests == []
