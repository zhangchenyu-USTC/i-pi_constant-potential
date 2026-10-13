"""Small, client-independent regressions for constant-potential dynamics."""

import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ipi.engine.atoms import Atoms
from ipi.engine.cell import Cell
from ipi.engine.constant_potential import validate_request_result
from ipi.engine.forcefields import FFSocket
from ipi.engine.forcefields_cp import FFChargeMixTwoSockets
from ipi.engine.potentiostat import ElectronicState, PotentiostatSVR
from ipi.engine.simulation import Simulation
from ipi.inputs.simulation import InputSimulation
from ipi.interfaces.sockets import Driver, HDRLEN, MESSAGE, Status
from ipi.utils.depend import dstrip
from ipi.utils.units import Constants


def socket_provider(owner=None, threaded=True, ne=False):
    owner = owner or NS(
        electronic_state=ElectronicState(
            q_init=8.2, mass=1.0, mode="workfunction", target_workfunction=0.3
        )
    )
    interface = NS(
        mode="inet", batch_size=1, requests=None, offset=0.0, poll=lambda: None
    )
    ff = FFSocket(name="cp", interface=interface, threaded=threaded)
    ff.configure_electrons(
        dict(
            enabled=True,
            neutral_electrons=8,
            Ne_doping=ne,
            potential_average_axis="z",
            potential_average_region=[0.0, 1.0],
        ),
        owner,
    )
    ff.set_electronic_state(owner.electronic_state.q)
    return ff, owner


def reply(q, force):
    return [
        0.0,
        np.asarray(force),
        np.zeros((3, 3)),
        dict(nelect=q, fermi_level_eV=0.0, workfunction_eV=0.0),
    ]


@pytest.mark.parametrize("threaded,ne", [(False, False), (True, True)])
def test_request_is_complete_when_published_and_retains_its_charge(threaded, ne):
    ff, owner = socket_provider(threaded=threaded, ne=ne)
    observed = []

    class Queue(list):
        def append(self, request):
            observed.append((request["nelect"], request["mode_flags"], request["pars"]))
            super().append(request)

    ff.requests = ff.socket.requests = Queue()
    request = ff.queue(Atoms(1), Cell(np.eye(3)))
    assert observed[0][:2] == (8.2, 2 if ne else 0)
    assert "neutral_electrons : 8" in observed[0][2]
    assert "potential_average_axis : z" in observed[0][2]
    owner.electronic_state.q = 9.0
    ff.set_electronic_state(9.0)
    assert request["electronic_context"].nelect == request["nelect"] == 8.2
    payloads = []
    Driver.sendchg(
        NS(status=Status.Up | Status.Ready, sendall=payloads.append),
        request["nelect"],
        request["mode_flags"],
    )
    assert payloads[0][:HDRLEN] == MESSAGE["chgdata"]
    assert len(payloads[0]) == HDRLEN + 12
    assert np.frombuffer(payloads[0][HDRLEN : HDRLEN + 8], np.float64)[0] == 8.2
    assert np.frombuffer(payloads[0][-4:], np.int32)[0] == (2 if ne else 0)
    ff.release(request)


@pytest.mark.parametrize("invalid", ["energy", "force", "virial", "nelect", "shape"])
def test_complete_reply_accepts_zero_values_and_rejects_invalid_results(invalid):
    ff, _ = socket_provider()
    request = ff.queue(Atoms(1), Cell(np.eye(3)))
    result = reply(8.2, np.zeros(3))
    validate_request_result(result, request)
    if invalid == "energy":
        result[0] = np.nan
    elif invalid in ("force", "virial"):
        result[1 if invalid == "force" else 2].flat[0] = np.inf
    elif invalid == "nelect":
        result[3]["nelect"] = 8.3
    else:
        result[1] = np.zeros(6)
    with pytest.raises(RuntimeError):
        validate_request_result(result, request)
    ff.release(request)


@pytest.mark.parametrize("active", [[1], [1, 0]])
def test_mixed_endpoints_use_full_forces_and_check_each_electron_count(active):
    low, owner = socket_provider()
    high, _ = socket_provider(owner)
    low.active = high.active = np.array(active)
    mixed = FFChargeMixTwoSockets(low, high, active=np.array(active))
    config = dict(low.electrons_config, charge_mixing=True, charge_mixing_span=1)
    mixed.configure_electrons(config, owner)
    mixed.set_electronic_state(8.2)
    request = mixed.queue(Atoms(2), Cell(np.eye(3)))
    indices = request["active"]
    wire = np.arange(indices.size, dtype=float) + 1
    for child, scale in [(request["child_low"], 1), (request["child_high"], 3)]:
        driver = Driver.__new__(Driver)
        driver._recv_forces_bulk = lambda n, c=child, s=scale: reply(
            c["nelect"], s * wire
        )
        driver.get_status = lambda: Status.Up | Status.Ready
        child["offset"] = 0.0
        assert driver.dispatch_recv(child)
    low_result, high_result = (
        request["child_low"]["result"],
        request["child_high"]["result"],
    )
    low_result[3]["nelect"] += 0.1
    with pytest.raises(RuntimeError, match="snapshot"):
        mixed._mix_results(low_result, high_result, request)
    low_result[3]["nelect"] -= 0.1
    mixed.poll()
    expected = np.zeros(6)
    expected[indices] = 1.4 * wire
    np.testing.assert_allclose(request["result"][1], expected, rtol=0, atol=1e-14)
    assert request["status"] == "Done"
    assert request["result"][3]["nelect"] == pytest.approx(8.2, abs=1e-14)
    mixed.release(request)
    assert mixed.requests == low.requests == high.requests == []


def test_svr_zero_time_and_finite_time_noise_from_zero_momentum():
    state = ElectronicState(q_init=8, mass=2, target_ef=0)
    draws = []
    thermostat = PotentiostatSVR(temp=0.01, tau=100)
    thermostat.bind(state, NS(gvec=lambda n: draws.append(n) or np.ones(n)))
    thermostat.O_step(0.0)
    assert state.p == thermostat.ethermo == 0.0
    assert draws == []
    thermostat.O_step(1.0)
    expected = 0.5 * Constants.kb * thermostat.temp * (-np.expm1(-0.01))
    assert state.kinetic_energy == pytest.approx(expected, rel=1e-14, abs=0)
    assert state.kinetic_energy + thermostat.ethermo == pytest.approx(0.0)
    assert draws == [1]


class CoupledPES:
    def __init__(self):
        self.q, self.calls = 8.0, []

    def set_electronic_state(self, q):
        self.q = q

    def __call__(self, cell, pos):
        x, dq = pos[0, 0], self.q - 8.0
        mu = x + dq + 0.01 * (cell[0, 0] - 100.0)
        self.calls.append((self.q, mu))
        force = np.zeros_like(pos)
        force[0, 0] = -(x + dq)
        return (
            0.5 * (x + dq) ** 2,
            force,
            np.zeros((3, 3)),
            json.dumps(
                dict(
                    nelect=self.q,
                    fermi_level_eV=mu * Constants.EV_PER_HARTREE,
                    workfunction_eV=-mu * Constants.EV_PER_HARTREE,
                )
            ),
        )


def simulation(xml=None, steps=3, beads=1, mode="langevin", temp=None, window=1):
    electronic_temp = "" if temp is None else f"<temp>{temp}</temp>"
    positions = [value for i in range(beads) for value in (0.02 * i, 0, 0)]
    if xml is None:
        xml = f"""<simulation verbosity='quiet' threading='false'>
          <total_steps>{steps}</total_steps><prng><seed>123</seed></prng><output/>
          <ffdirect name='cp'><pes>asym_dw_constant_potential</pes></ffdirect>
          <system><beads natoms='1' nbeads='{beads}'>
            <q shape='({beads},3)'>{positions}</q>
            <p shape='({beads},3)'>{[1,0,0] * beads}</p><m>[1]</m><names>[H]</names>
          </beads><cell shape='(3,3)'>[100,0,0,0,100,0,0,0,100]</cell>
          <forces><force forcefield='cp'/></forces>
          <ensemble><temperature>0.01</temperature></ensemble>
          <motion mode='dynamics'><fixcom>false</fixcom><dynamics mode='nve'>
            <timestep>0.1</timestep><electrons enabled='true' Ne_doping='true'>
              <target_workfunction>0</target_workfunction><potential_average_axis>z</potential_average_axis>
              <potential_average_region>[0,1]</potential_average_region><neutral_electrons>8</neutral_electrons>
              <q_init>8</q_init><mass>1</mass><workfunction_average_steps>{window}</workfunction_average_steps>
              <potentiostat mode='{mode}'><tau>100</tau>{electronic_temp}</potentiostat>
            </electrons></dynamics></motion></system></simulation>"""

    def configure(sim):
        sim.fflist["cp"].threaded = False
        sim.fflist["cp"].driver = CoupledPES()

    sim = Simulation.load_from_xml(xml, read_only=True, post_init_hook=configure)
    sim.outputs, sim.chk = [], NS(store=lambda: None)
    return sim


def checkpoint(sim, step):
    data = InputSimulation()
    data.store(sim)
    data.step.store(step)
    data.total_steps.store(3)
    return data.write("simulation")


def snapshot(sim):
    system = sim.syslist[0]
    es = system.motion.electronic_state
    return (
        dstrip(system.beads.q).copy(),
        dstrip(system.beads.p).copy(),
        [es.q, es.p, es.current_workfunction, system.motion.potentiostat.ethermo],
        list(es._workfunction_average_history),
        (es.accepted_frame_id, es._workfunction_average_total_samples),
        copy.deepcopy(sim.prng.state),
    )


@pytest.mark.parametrize("mode", ["langevin", "svr"])
@pytest.mark.parametrize("beads", [1, 4])
def test_output_restart_and_shared_bead_feedback(mode, beads):
    continuous, printed = simulation(mode=mode, beads=beads), simulation(
        mode=mode, beads=beads
    )
    printed.outputs = [
        NS(write=lambda: printed.syslist[0].motion._get_fermi_level_from_forces())
    ]
    continuous.run(write_outputs=False)
    printed.run(write_outputs=True)
    first = simulation(steps=1, mode=mode, beads=beads)
    first.run(write_outputs=False)
    resumed = simulation(checkpoint(first, 1))
    resumed.run(write_outputs=False)
    for candidate in (printed, resumed):
        for actual, expected in zip(snapshot(candidate)[:4], snapshot(continuous)[:4]):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        np.testing.assert_equal(snapshot(candidate)[4:], snapshot(continuous)[4:])
    for sim, evaluations in ((continuous, 4), (printed, 4), (first, 2), (resumed, 3)):
        calls = sim.fflist["cp"].driver.calls
        assert len(calls) == beads * evaluations
        for frame in range(evaluations):
            assert len({q for q, mu in calls[frame * beads : (frame + 1) * beads]}) == 1
        es = sim.syslist[0].motion.electronic_state
        assert es.current_workfunction == pytest.approx(
            -np.mean([mu for q, mu in calls[-beads:]])
        )
    assert snapshot(continuous)[4] == (3, 4)


@pytest.mark.parametrize("changed", ["positions", "cell"])
def test_inplace_boundary_refresh_replaces_one_window_sample(changed):
    sim = simulation(steps=1, window=3)
    sim.run(write_outputs=False)
    motion = sim.syslist[0].motion
    es, calls = motion.electronic_state, sim.fflist["cp"].driver.calls
    history, samples = (
        list(es._workfunction_average_history),
        es._workfunction_average_total_samples,
    )
    saved = motion._electronic_boundary_state
    assert not np.shares_memory(saved[0], dstrip(motion.beads.q))
    assert not np.shares_memory(saved[1], dstrip(motion.cell.h))
    if changed == "positions":
        motion.beads.q[0, 0] += 0.2
    else:
        motion.cell.h[0, 0] += 2.0
    motion._prepare_electronic_boundary(1)
    motion._prepare_electronic_boundary(1)
    assert len(calls) == 3
    assert (
        es.accepted_frame_id == 1 and es._workfunction_average_total_samples == samples
    )
    assert es._workfunction_average_history[:-1] == history[:-1]
    assert es._workfunction_average_history[-1] == pytest.approx(-calls[-1][1])
    assert es.current_workfunction_steps_average == pytest.approx(
        np.mean(es._workfunction_average_history)
    )


@pytest.mark.parametrize("mode", ["langevin", "svr"])
@pytest.mark.parametrize("temp", [None, 0.0])
def test_electronic_temperature_follows_physical_temperature_unless_explicit(
    mode, temp
):
    sim = simulation(beads=4, mode=mode, temp=temp)
    for state in (sim, simulation(checkpoint(sim, 0))):
        system = state.syslist[0]
        for temperature in (0.01, 0.02):
            system.ensemble.temp = temperature
            assert system.motion.potentiostat.temp == (
                temperature if temp is None else temp
            )
