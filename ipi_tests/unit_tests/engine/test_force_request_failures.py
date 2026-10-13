"""Completed failure requests, finite CP responses, and real threaded shutdown.

Numerical contract tests isolate the global exit notifier. Process tests retain
the real notifier, polling threads, consumer futures, and Simulation.run boundary.
No socket server or external electronic-structure executable is needed.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ipi.engine.atoms import Atoms
from ipi.engine.beads import Beads
from ipi.engine.cell import Cell
from ipi.engine.forcefields import FFDirect, FFSocket, ForceField
from ipi.engine.forces import ForceBead
from ipi.engine.potentiostat import ElectronicState
from ipi.engine.simulation import Simulation
from ipi.utils.softexit import softexit

ROOT = Path(__file__).resolve().parents[3]
SENTINEL = "deliberate-force-request-failure"


def _owner():
    state = ElectronicState(q_init=8.2, mass=1.0, mode="fermi", target_ef=0.0)
    state.neutral_electrons = 8
    return NS(electronic_state=state, beads=Beads(1, 1))


def _socket_bead(cp=True, name="contract"):
    interface = NS(
        mode="inet",
        batch_size=1,
        requests=None,
        offset=0.0,
        poll=lambda: None,
        close=lambda: None,
    )
    ff = FFSocket(name=name, interface=interface, threaded=True)
    bead = ForceBead()
    bead.bind(Atoms(1), Cell(np.eye(3)), ff, None)
    if cp:
        owner = _owner()
        ff.configure_electrons(dict(enabled=True, neutral_electrons=8), owner)
        ff.set_electronic_state(owner.electronic_state.q)
        bead.bind_electronic_context(owner)
    return ff, bead


def _result():
    return [0.0, np.zeros(3), np.zeros((3, 3)), dict(nelect=8.2, fermi_level_eV=0.0)]


def _publish(bead, result):
    bead.queue()
    request = bead.request
    request.update(status="Done", result=result, t_finished=time.time())
    request._event_done.set()
    return request


def _causes(error):
    values = []
    while error is not None:
        values.append(error)
        error = error.__cause__
    return values


@pytest.fixture
def isolated_notifications(monkeypatch):
    """Keep unit cases local; process regressions below use actual cleanup."""
    notifications = []
    monkeypatch.setattr(
        ForceField, "notify_failure", lambda self: notifications.append(self)
    )
    monkeypatch.setattr(softexit, "register_thread", lambda *args: None)
    monkeypatch.setattr(softexit, "register_function", lambda *args: None)
    return notifications


@pytest.mark.parametrize("component", [0, 1, 2], ids=["energy", "force", "virial"])
@pytest.mark.parametrize(
    "bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "minus-inf"]
)
def test_cp_socket_nonfinite_result_is_rejected_and_released(
    component, bad, isolated_notifications
):
    ff, bead = _socket_bead()
    result = _result()
    if component == 0:
        result[0] = bad
    else:
        result[component].flat[0] = bad
    _publish(bead, result)
    with pytest.raises(RuntimeError):
        bead.get_all()
    assert bead.request is None and bead._getallcount == 0
    assert ff.requests == []
    assert isolated_notifications


@pytest.mark.parametrize(
    "component,value",
    [(0, np.zeros(2)), (1, np.zeros(2)), (1, np.zeros((1, 3))), (2, np.zeros(9))],
)
def test_cp_socket_result_shape_contract(component, value, isolated_notifications):
    ff, bead = _socket_bead()
    result = _result()
    result[component] = value
    _publish(bead, result)
    with pytest.raises(RuntimeError):
        bead.get_all()
    assert bead.request is None and bead._getallcount == 0
    assert ff.requests == []


def test_cp_zero_result_is_valid(isolated_notifications):
    ff, bead = _socket_bead()
    _publish(bead, _result())
    result = bead.get_all()
    assert result[0] == 0.0
    np.testing.assert_array_equal(result[1], np.zeros(3))
    assert not isolated_notifications
    assert ff.requests == []


def test_new_finiteness_gate_does_not_change_ordinary_socket(isolated_notifications):
    ff, bead = _socket_bead(cp=False)
    result = _result()
    result[0] = np.nan
    result[1][:] = np.inf
    _publish(bead, result)
    accepted = bead.get_all()
    assert np.isnan(accepted[0]) and np.isinf(accepted[1]).all()
    assert not isolated_notifications
    assert ff.requests == []


class _FailingPES:
    def __init__(self, error=None, barrier=None):
        self.error = error if error is not None else ValueError(SENTINEL)
        self.barrier = barrier

    def __call__(self, cell, positions):
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        raise self.error


def test_direct_background_failure_finishes_all_pending_requests(
    isolated_notifications,
):
    ff = FFDirect(name="failing-direct", pes="dummy", threaded=True)
    error = ValueError(SENTINEL)
    ff.driver = _FailingPES(error)
    requests = [ff.queue(Atoms(1), Cell(np.eye(3)), reqid=i) for i in range(3)]
    ff.start()
    try:
        assert all(request._event_done.wait(3) for request in requests)
        assert all(request["status"] == "Exit" for request in requests)
        assert all(request["result"] is None for request in requests)
        assert any(request.get("exception") is error for request in requests)
        assert isolated_notifications
        for request in requests:
            ff.release(request)
        assert ff.requests == []
    finally:
        ff.stop()
        ff._thread.join(timeout=3)
    assert not ff._thread.is_alive()


def test_direct_synchronous_failure_preserves_first_cause_and_releases(
    isolated_notifications,
):
    ff = FFDirect(name="sync-direct", pes="dummy", threaded=False)
    original = ValueError(SENTINEL)
    ff.driver = _FailingPES(original)
    bead = ForceBead()
    bead.bind(Atoms(1), Cell(np.eye(3)), ff, None)
    with pytest.raises(RuntimeError) as failure:
        bead.get_all()
    assert original in _causes(failure.value)
    assert bead.request is None and bead._getallcount == 0 and ff.requests == []
    ff.record_failure(ValueError("later failure must not replace the first"))
    later = ff.queue(Atoms(1), Cell(np.eye(3)), reqid=10)
    assert later._event_done.is_set() and later["status"] == "Exit"
    assert later["exception"] is original and later["result"] is None
    assert ff.requests == []


def test_direct_background_electronic_contract_failure_finishes_request(
    isolated_notifications,
):
    class WrongElectrons:
        def set_electronic_state(self, q):
            self.q = q

        def __call__(self, cell, positions):
            return [
                0.0,
                np.zeros_like(positions),
                np.zeros((3, 3)),
                json.dumps(dict(nelect=self.q - 1.0, fermi_level_eV=0.0)),
            ]

    ff = FFDirect(name="wrong-electrons", pes="dummy", threaded=True)
    ff.driver = WrongElectrons()
    owner = _owner()
    ff.configure_electrons(dict(enabled=True, neutral_electrons=8), owner)
    ff.set_electronic_state(owner.electronic_state.q)
    request = ff.queue(Atoms(1), Cell(np.eye(3)))
    ff.start()
    try:
        assert request._event_done.wait(3)
        assert request["status"] == "Exit" and request["result"] is None
        assert isinstance(request["exception"], RuntimeError)
        assert "electronic" in str(request["exception"]).lower()
        ff.release(request)
    finally:
        ff.stop()
        ff._thread.join(timeout=3)
    assert not ff._thread.is_alive() and ff.requests == []


@pytest.mark.parametrize(
    "short_batch", [False, True], ids=["driver-raises", "short-result-list"]
)
def test_direct_batch_failure_does_not_leave_running_requests(
    short_batch, isolated_notifications
):
    ff = FFDirect(name="batch-direct", pes="dummy", batch_size=2, threaded=True)
    if short_batch:
        ff.driver = lambda cells, positions: [_result()]
    else:
        ff.driver = _FailingPES()
    requests = [ff.queue(Atoms(1), Cell(np.eye(3)), reqid=i) for i in range(3)]
    ff.start()
    try:
        assert all(request._event_done.wait(3) for request in requests)
        assert all(request["status"] == "Exit" for request in requests)
        assert all(request["result"] is None for request in requests)
        assert ff.request_batch == []
        for request in requests:
            ff.release(request)
        assert ff.requests == []
    finally:
        ff.stop()
        ff._thread.join(timeout=3)
    assert not ff._thread.is_alive()


@pytest.mark.parametrize(
    "bad_last", [False, True], ids=["valid-batches", "invalid-last-result"]
)
def test_direct_batch_prepares_all_results_before_publication(
    bad_last, isolated_notifications
):
    ff = FFDirect(
        name="ordinary-batch", pes="dummy", batch_size=2, threaded=True, offset=1.5
    )
    calls = []

    def driver(cells, positions):
        calls.append(len(positions))
        results = [
            [
                float(pos[0, 0]),
                np.full_like(pos, pos[0, 0]),
                np.eye(3) * pos[0, 0],
                json.dumps(dict(label=float(pos[0, 0]))),
            ]
            for pos in positions
        ]
        if bad_last:
            results[-1][2] = np.zeros(8)
        return results

    ff.driver = driver
    requests = []
    for index in range(3):
        atoms = Atoms(1)
        atoms.q = np.array([index + 2.0, 0.0, 0.0])
        requests.append(ff.queue(atoms, Cell(np.eye(3)), reqid=index))
    ff.start()
    try:
        assert all(request._event_done.wait(3) for request in requests)
        if bad_last:
            assert calls == [2]
            assert all(
                request["status"] == "Exit" and request["result"] is None
                for request in requests
            )
        else:
            assert calls == [2, 1]
            for index, request in enumerate(requests):
                assert request["status"] == "Done"
                energy, forces, virial, extra = request["result"]
                assert energy == index + 0.5
                np.testing.assert_array_equal(forces, np.full(3, index + 2.0))
                np.testing.assert_array_equal(virial, np.eye(3) * (index + 2.0))
                assert extra["label"] == index + 2.0
            assert not isolated_notifications
        assert ff.request_batch == []
        for request in requests:
            ff.release(request)
    finally:
        ff.stop()
        ff._thread.join(timeout=3)
    assert not ff._thread.is_alive() and ff.requests == []


class _TwoWaiters:
    """Control a real producer/consumer interleaving without sleeps as a barrier."""

    def __init__(self):
        self.event = threading.Event()
        self.both_waiting = threading.Event()
        self.lock = threading.Lock()
        self.waiters = set()

    def wait(self, timeout=None):
        with self.lock:
            self.waiters.add(threading.get_ident())
            if len(self.waiters) == 2:
                self.both_waiting.set()
        return self.event.wait(timeout)

    def set(self):
        self.event.set()

    def is_set(self):
        return self.event.is_set()


@pytest.mark.parametrize("action", ["cancel", "stop"])
def test_direct_failure_during_cancel_or_stop_releases_consumers(
    action, isolated_notifications
):
    """An actor waits for the polling lock while the driver owns it and fails."""
    ff = FFDirect(name="interleaved-direct", pes="dummy", threaded=True)
    original = ValueError(SENTINEL)
    model_entered = threading.Event()
    model_can_fail = threading.Event()
    actor_waiting = threading.Event()
    actor_finished = threading.Event()
    consumer_errors, actor_errors = [], []

    def driver(cell, positions):
        model_entered.set()
        assert model_can_fail.wait(5)
        raise original

    ff.driver = driver
    bead = ForceBead()
    bead.bind(Atoms(1), Cell(np.eye(3)), ff, None)
    bead.queue()
    request = bead.request
    request._event_done = completion = _TwoWaiters()
    polling_lock = ff._threadlock

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread() is actor:
                actor_waiting.set()
            polling_lock.acquire()
            return self

        def __exit__(self, *args):
            polling_lock.release()

    def act():
        try:
            if action == "cancel":
                ff.cancel_request(request)
            else:
                ff.stop()
        except BaseException as error:
            actor_errors.append(error)
        finally:
            actor_finished.set()

    def consume():
        try:
            bead.get_all()
        except BaseException as error:
            consumer_errors.append(error)

    actor = threading.Thread(target=act, daemon=True)
    consumers = [threading.Thread(target=consume, daemon=True) for _ in range(2)]
    ff._threadlock = ObservedLock()
    ff.start()
    try:
        assert model_entered.wait(3)
        for consumer in consumers:
            consumer.start()
        assert completion.both_waiting.wait(3)
        actor.start()
        assert actor_waiting.wait(3)
        assert polling_lock.locked() and not actor_finished.is_set()
        if action == "cancel":
            # cancel_request owns dispatch before waiting on the FF lock.
            assert request._dispatch_lock.locked()
        model_can_fail.set()
    finally:
        model_can_fail.set()
        ff._doloop[0] = False
        for thread in [actor, *consumers, ff._thread]:
            if thread.ident is not None:
                thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in [actor, *consumers, ff._thread])
    assert actor_finished.is_set() and not actor_errors
    assert completion.is_set() and request["status"] == "Exit"
    assert request["exception"] is original and request["result"] is None
    assert len(consumer_errors) == 2
    assert all(
        isinstance(error, RuntimeError) and original in _causes(error)
        for error in consumer_errors
    )
    assert bead._getallcount == 0 and bead.request is None and ff.requests == []
    assert not request._dispatch_lock.locked() and not polling_lock.locked()
    assert isolated_notifications


@pytest.mark.parametrize("invalid", [False, True], ids=["success", "failure"])
def test_two_waiters_release_shared_request_once(
    invalid, isolated_notifications, monkeypatch
):
    ff, bead = _socket_bead()
    bead.queue()
    request = bead.request
    request._event_done = event = _TwoWaiters()
    released, answers, errors = [], [], []
    original_release = ff.release

    def release(item, *args, **kwargs):
        released.append(item)
        return original_release(item, *args, **kwargs)

    monkeypatch.setattr(ff, "release", release)

    def consume():
        try:
            answers.append(bead.get_all())
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=consume, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert event.both_waiting.wait(3)
    result = _result()
    if invalid:
        result[2][0, 0] = np.inf
    request.update(status="Done", result=result, t_finished=time.time())
    event.set()
    for thread in threads:
        thread.join(timeout=3)
    assert all(not thread.is_alive() for thread in threads)
    assert bead._getallcount == 0 and bead.request is None and ff.requests == []
    assert released == [request]
    if invalid:
        assert len(errors) == 2 and all(
            isinstance(error, RuntimeError) for error in errors
        )
        assert not answers
    else:
        assert len(answers) == 2 and not errors


def _run_subprocess(command, directory, timeout=15):
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    process = subprocess.Popen(
        command,
        cwd=directory,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate(timeout=5)
        pytest.fail(
            "force failure did not terminate the process before timeout:\n" + output
        )
    return process.returncode, output


def test_real_cli_force_failure_exits_nonzero(tmp_path):
    from ipi_tests.unit_tests.engine.test_electronic_lifecycle import input_xml

    driver = tmp_path / "failing_pes.py"
    driver.write_text(
        "__DRIVER_CLASS__ = 'Failing'\n"
        "class Failing:\n"
        "    def __init__(self, **kwargs): pass\n"
        "    def set_electronic_state(self, q): pass\n"
        "    def __call__(self, cell, positions):\n"
        f"        raise ValueError({SENTINEL!r})\n"
    )
    xml = input_xml(steps=1).replace(
        "<ffdirect name='cp'><pes>asym_dw_constant_potential</pes></ffdirect>",
        f"<ffdirect name='cp' threaded='true'><pes>custom</pes><pes_path>{driver}</pes_path></ffdirect>",
    )
    (tmp_path / "input.xml").write_text(xml)
    status, output = _run_subprocess(
        [sys.executable, str(ROOT / "bin/i-pi"), "input.xml"], tmp_path
    )
    assert status != 0, output
    assert SENTINEL in output and "RuntimeError" in output, output


def _skeleton_simulation(forcefields, threaded=False):
    """Keep production run/run_step and futures; provide no-op physical fixtures."""
    sim = Simulation.__new__(Simulation)
    from ipi.utils.depend import depend_value

    sim._step = depend_value(name="step", value=0)
    sim.tsteps = 1
    sim.safe_stride = 1
    sim.threading = threaded
    sim.fflist = {ff.name: ff for ff in forcefields}
    sim.syslist = []
    sim.outputs = []
    sim.finished = False
    sim.rollback = True
    sim.ttime = 0
    sim.smotion = None
    sim.chk = NS(store=lambda: None, write=lambda **kwargs: None)
    sim._initialize_constant_potential_observables = lambda: None
    sim._update_constant_potential_forcefields = lambda: None
    if threaded:
        sim.executor = ThreadPoolExecutor(max_workers=2)
    return sim


def _process_scenario(scenario):
    """Invoked only in a child interpreter; real global softexit is intentional."""
    softexit.reset()
    if scenario == "simultaneous-polls":
        barrier = threading.Barrier(2)
        forcefields = [
            FFDirect(name=f"direct-{i}", pes="dummy", threaded=True) for i in range(2)
        ]
        beads = []
        for ff in forcefields:
            ff.driver = _FailingPES(barrier=barrier)
            bead = ForceBead()
            bead.bind(Atoms(1), Cell(np.eye(3)), ff, None)
            bead.queue()
            beads.append(bead)
        sim = _skeleton_simulation(forcefields, threaded=True)
        sim.syslist = [
            NS(motion=NS(step=lambda step, bead=bead: bead.get_all(), finished=False))
            for bead in beads
        ]
        for ff in forcefields:
            ff.start()
    else:
        bad_ff, bad_bead = _socket_bead(name="bad-later")
        result = _result()
        result[0] = np.nan
        _publish(bad_bead, result)
        softexit.register_function(bad_ff.stop)
        sim = _skeleton_simulation([bad_ff], threaded=scenario == "later-system")
        if scenario == "initial":
            sim._initialize_constant_potential_observables = bad_bead.get_all
        elif scenario == "output":
            sim.outputs = [NS(write=bad_bead.get_all, active=lambda: True)]
        elif scenario == "later-system":
            blocked_ff, blocked_bead = _socket_bead(cp=False, name="blocked-first")
            blocked_bead.queue()
            softexit.register_function(blocked_ff.stop)
            sim.fflist[blocked_ff.name] = blocked_ff
            entered = threading.Event()

            def first_step(step):
                entered.set()
                return blocked_bead.get_all()

            def second_step(step):
                assert entered.wait(3)
                return bad_bead.get_all()

            sim.syslist = [
                NS(motion=NS(step=first_step, finished=False)),
                NS(motion=NS(step=second_step, finished=False)),
            ]
        else:
            raise AssertionError(scenario)
    try:
        sim.run(write_outputs=True)
    except RuntimeError as error:
        assert any(
            isinstance(cause, (RuntimeError, ValueError)) for cause in _causes(error)
        )
        if scenario == "simultaneous-polls":
            assert SENTINEL in " ".join(str(cause) for cause in _causes(error))
            assert all(not ff._thread.is_alive() for ff in forcefields)
        else:
            assert bad_bead._getallcount == 0 and bad_bead.request is None
            assert bad_ff.requests == []
        print("FAILURE_REACHED_RUN_BOUNDARY", scenario, flush=True)
        raise
    finally:
        if hasattr(sim, "executor"):
            sim.executor.shutdown(wait=False, cancel_futures=True)
    raise AssertionError("invalid force response was accepted")


@pytest.mark.parametrize(
    "scenario", ["initial", "output", "later-system", "simultaneous-polls"]
)
def test_real_simulation_failure_propagation_in_child_process(scenario, tmp_path):
    command = [
        sys.executable,
        "-c",
        "import runpy,sys; runpy.run_path(sys.argv[1])['_process_scenario'](sys.argv[2])",
        str(Path(__file__).resolve()),
        scenario,
    ]
    status, output = _run_subprocess(command, tmp_path)
    assert status != 0, output
    assert "FAILURE_REACHED_RUN_BOUNDARY " + scenario in output, output
    assert "RuntimeError" in output, output
