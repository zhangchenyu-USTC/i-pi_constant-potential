"""Boundary feedback and checkpoint regression tests using a coupled analytic PES."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ipi import ipi_global_settings
from ipi.engine.simulation import Simulation
from ipi.engine.outputs import CheckpointOutput, PropertyOutput
from ipi.engine.potentiostat import PotentiostatSVR
from ipi.inputs.electrons import InputElectrons
from ipi.inputs.simulation import InputSimulation
from ipi.inputs.prng import NumpyEncoder
from ipi.utils.io.inputs.io_xml import write_float
from ipi.utils.io.inputs.serialization import checkpoint_serialization
from ipi.utils.units import Constants


class CoupledPES:
    def __init__(self, shift=0.0):
        self.q = 8.0
        self.shift = shift
        self.calls = []

    def set_electronic_state(self, q):
        self.q = float(q)

    def __call__(self, cell, pos):
        x = float(pos[0, 0])
        dq = self.q - 8.0
        mu = x + dq
        self.calls.append((x, self.q, mu))
        force = np.zeros_like(pos)
        force[0, 0] = -(x + dq)
        return (
            0.5 * x * x + x * dq + 0.5 * dq * dq,
            force,
            np.zeros((3, 3)),
            json.dumps(
                {
                    "nelect": self.q,
                    "fermi_level_eV": (mu + self.shift) * Constants.EV_PER_HARTREE,
                    "workfunction_eV": -(mu + self.shift) * Constants.EV_PER_HARTREE,
                }
            ),
        )


def input_xml(
    steps=3, ne=False, window=3, ensemble="nve", splitting="baoab", electronic_temp=0.0
):
    target = "<target_fermi_level>0</target_fermi_level>"
    if ne:
        target = (
            "<target_workfunction>0</target_workfunction>"
            "<potential_average_axis>z</potential_average_axis>"
            "<potential_average_region>[0,1]</potential_average_region>"
        )
    return f"""<simulation verbosity='quiet' threading='false'>
      <total_steps>{steps}</total_steps><prng><seed>123</seed></prng><output/>
      <ffdirect name='cp'><pes>asym_dw_constant_potential</pes></ffdirect>
      <system><beads natoms='1' nbeads='1'>
        <q shape='(1,3)'>[0,0,0]</q><p shape='(1,3)'>[1,0,0]</p>
        <m>[1]</m><names>[H]</names></beads>
        <cell shape='(3,3)'>[100,0,0,0,100,0,0,0,100]</cell>
        <forces><force forcefield='cp'/></forces><ensemble><temperature>0</temperature></ensemble>
        <motion mode='dynamics'><fixcom>false</fixcom><dynamics mode='{ensemble}' splitting='{splitting}'>
          <timestep>0.1</timestep>
          <electrons enabled='true' Ne_doping='{str(ne).lower()}'>
            {target}<neutral_electrons>8</neutral_electrons><q_init>8</q_init><mass>1</mass>
            <workfunction_average_steps>{window}</workfunction_average_steps>
            <potentiostat mode='langevin'><tau>100</tau><temp>{electronic_temp}</temp></potentiostat>
          </electrons></dynamics></motion></system></simulation>"""


def load(xml, shift=0.0):
    def hook(sim):
        forcefield = sim.fflist["cp"]
        forcefield.threaded = False
        forcefield.driver = CoupledPES(shift)

    sim = Simulation.load_from_xml(xml, read_only=True, post_init_hook=hook)
    sim.outputs = []
    sim.chk = SimpleNamespace(store=lambda: None)
    return sim


def state(sim):
    system = sim.syslist[0]
    es = system.motion.electronic_state
    return {
        "R": np.array(system.beads.q),
        "P": np.array(system.beads.p),
        "cell": np.array(system.cell.h),
        "barostat_p": np.array(getattr(system.motion.barostat, "p", [])),
        "nuclear_ethermo": system.motion.thermostat.ethermo,
        "barostat_ethermo": system.motion.barostat.thermostat.ethermo,
        "q": es.q,
        "p": es.p,
        "mu": es.current_ef,
        "phi": es.current_workfunction,
        "mean": es.current_workfunction_steps_average,
        "history": list(es._workfunction_average_history),
        "samples": es._workfunction_average_total_samples,
        "frame": es.accepted_frame_id,
        "initialized": es.feedback_initialized,
        "scan": es.current_step,
        "work": es.protocol_work,
        "ethermo": system.motion.potentiostat.ethermo,
        "rng": json.dumps(sim.prng.state, cls=NumpyEncoder),
    }


def assert_same(left, right, atol=0.0):
    left, right = state(left), state(right)
    for key in left:
        if isinstance(left[key], np.ndarray):
            np.testing.assert_allclose(
                left[key], right[key], rtol=0, atol=atol, err_msg=key
            )
        elif atol and isinstance(left[key], (float, list)):
            np.testing.assert_allclose(
                left[key], right[key], rtol=0, atol=atol, err_msg=key
            )
        else:
            assert left[key] == right[key], key


def checkpoint(sim, next_step, total_steps=6):
    data = InputSimulation()
    data.store(sim)
    data.step.store(next_step)
    data.total_steps.store(total_steps)
    return data.write("simulation")


@pytest.mark.parametrize("ne", [False, True])
@pytest.mark.parametrize(
    "ensemble,splitting", [("nve", "baoab"), ("nvt", "baoab"), ("nvt", "obabo")]
)
def test_output_switch_has_no_physical_or_request_effect(ne, ensemble, splitting):
    xml = input_xml(ne=ne, ensemble=ensemble, splitting=splitting, electronic_temp=0.01)
    enabled, disabled = load(xml), load(xml)
    # Exercise repeated feedback reads while producing output, not just an empty list.
    enabled.outputs = [
        SimpleNamespace(
            write=lambda: enabled.syslist[0].motion._get_fermi_level_from_forces()
        )
    ]
    enabled.run(write_outputs=True)
    disabled.run(write_outputs=False)
    assert_same(enabled, disabled)
    assert enabled.fflist["cp"].driver.calls == disabled.fflist["cp"].driver.calls
    assert enabled.syslist[0].motion.electronic_state.accepted_frame_id == 3


@pytest.mark.parametrize("ne", [False, True])
def test_real_property_output_stride_changes_only_written_rows(ne, tmp_path):
    xml = input_xml(steps=6, ne=ne, electronic_temp=0.01)
    simulations, rows = [], []
    properties = ["step", "potential", "conserved", "electronic_charge", "fermi_level"]
    if ne:
        properties += ["workfunction", "workfunction_steps_average"]
    for stride in (1, 3):
        sim = load(xml)
        path = tmp_path / f"properties-stride-{stride}.out"
        output = PropertyOutput(filename=str(path), stride=stride, outlist=properties)
        output.bind(sim.syslist[0])
        output.print_header()
        sim.outputs = [output]
        try:
            sim.run(write_outputs=True)
        finally:
            output.close_stream()
        simulations.append(sim)
        rows.append(np.loadtxt(path))
    assert_same(*simulations)
    assert (
        simulations[0].fflist["cp"].driver.calls
        == simulations[1].fflist["cp"].driver.calls
    )
    np.testing.assert_array_equal(rows[0][:, 0], np.arange(7))
    np.testing.assert_array_equal(rows[1], rows[0][::3])


@pytest.mark.parametrize("ne", [False, True])
def test_checkpoint_and_restore_after_every_single_step(ne, tmp_path):
    total = 6
    continuous = load(input_xml(steps=total, ne=ne, electronic_temp=0.01))
    continuous.run(write_outputs=False)
    repeated = load(input_xml(steps=1, ne=ne, electronic_temp=0.01))
    for frame in range(1, total + 1):
        repeated.tsteps = frame
        path = tmp_path / f"restart-frame-{frame}.xml"
        output = CheckpointOutput(filename=str(path), stride=1)
        output.bind(repeated)
        repeated.outputs = [output]
        repeated.run(write_outputs=True)
        es = repeated.syslist[0].motion.electronic_state
        assert es.accepted_frame_id == frame
        if ne:
            assert es._workfunction_average_total_samples == frame + 1
        # Extending total_steps is the only user edit needed to continue a
        # completed run; all physical/lifecycle/PRNG fields come from the file.
        tree = ET.parse(path).getroot()
        assert int(tree.find("step").text) == frame
        tree.find("total_steps").text = str(total + 1)
        previous = state(repeated)
        repeated = load(ET.tostring(tree, encoding="unicode"))
        for key, value in previous.items():
            np.testing.assert_equal(state(repeated)[key], value, err_msg=key)
    repeated._initialize_constant_potential_observables()
    assert_same(repeated, continuous)
    assert len(repeated.fflist["cp"].driver.calls) == 1


@pytest.mark.parametrize("ne", [False, True])
@pytest.mark.parametrize("splitting", ["baoab", "obabo"])
def test_svr_simulation_restart_preserves_rng_and_thermostat_energy(ne, splitting):
    def xml(steps):
        return input_xml(
            steps=steps,
            ne=ne,
            window=1,
            ensemble="nvt",
            splitting=splitting,
            electronic_temp=0.01,
        ).replace("<potentiostat mode='langevin'>", "<potentiostat mode='svr'>")

    first = load(xml(2))
    assert isinstance(first.syslist[0].motion.potentiostat, PotentiostatSVR)
    rng_before = state(first)["rng"]
    first.run(write_outputs=False)
    assert state(first)["rng"] != rng_before
    assert state(first)["ethermo"] != 0.0
    assert state(first)["q"] != 8.0
    resumed = load(checkpoint(first, 2))
    assert isinstance(resumed.syslist[0].motion.potentiostat, PotentiostatSVR)
    assert_same(first, resumed)
    resumed.run(write_outputs=False)
    continuous = load(xml(6))
    continuous.run(write_outputs=True)
    assert_same(resumed, continuous)


@pytest.mark.parametrize("ne", [False, True])
@pytest.mark.parametrize("split", [1, 2, 4])
def test_restart_matches_continuous_trajectory_and_random_state(ne, split):
    first = load(input_xml(steps=split, ne=ne, electronic_temp=0.01))
    first.run(write_outputs=False)
    resumed = load(checkpoint(first, split))
    assert not resumed.syslist[0].motion._fermi_cache_valid
    resumed.run(write_outputs=False)
    continuous = load(input_xml(steps=6, ne=ne, electronic_temp=0.01))
    continuous.run(write_outputs=True)
    assert_same(resumed, continuous)
    # Restart recomputes the saved full frame, never a nuclear half-step frame.
    assert resumed.fflist["cp"].driver.calls[0][0] == first.syslist[0].beads.q[0, 0]


def test_ne_feedback_read_and_same_frame_refresh_do_not_add_samples():
    sim = load(input_xml(steps=1, ne=True))
    sim.run(write_outputs=False)
    saved = state(sim)
    resumed = load(checkpoint(sim, 1), shift=0.01)
    motion = resumed.syslist[0].motion
    motion._prepare_electronic_boundary(1)
    refreshed = state(resumed)
    for key in (
        "R",
        "P",
        "q",
        "p",
        "samples",
        "frame",
        "scan",
        "work",
        "ethermo",
        "rng",
    ):
        np.testing.assert_equal(refreshed[key], saved[key], err_msg=key)
    assert refreshed["history"][:-1] == saved["history"][:-1]
    assert refreshed["history"][-1] == pytest.approx(saved["history"][-1] - 0.01)
    assert refreshed["mean"] == np.mean(refreshed["history"])
    motion._get_fermi_level_from_forces()
    motion._get_fermi_level_from_forces()
    motion._prepare_electronic_boundary(1)
    assert state(resumed)["samples"] == saved["samples"]
    assert len(resumed.fflist["cp"].driver.calls) == 1


@pytest.mark.parametrize("prepared", [False, True])
def test_frame_zero_checkpoint_initializes_or_replaces_once(prepared):
    sim = load(input_xml(ne=True))
    if prepared:
        sim._initialize_constant_potential_observables()
    resumed = load(checkpoint(sim, 0))
    resumed._initialize_constant_potential_observables()
    resumed._initialize_constant_potential_observables()
    es = resumed.syslist[0].motion.electronic_state
    assert es.feedback_initialized and es.accepted_frame_id == 0
    assert es._workfunction_average_total_samples == 1
    assert len(es._workfunction_average_history) == 1
    assert len(resumed.fflist["cp"].driver.calls) == 1


@pytest.mark.parametrize("field", InputElectrons.checkpoint_fields)
def test_checkpoint_rejects_missing_runtime_field(field):
    sim = load(input_xml())
    tree = ET.fromstring(checkpoint(sim, 0))
    electrons = tree.find("./system/motion/dynamics/electrons")
    electrons.remove(electrons.find(field))
    with pytest.raises(ValueError, match="checkpoint_v1 requires complete"):
        load(ET.tostring(tree, encoding="unicode"))


def test_legacy_frame_zero_and_ambiguous_inline_state_need_explicit_origin():
    sim = load(input_xml())
    tree = ET.fromstring(checkpoint(sim, 0))
    del tree.attrib["state_format"]
    electrons = tree.find("./system/motion/dynamics/electrons")
    for field in InputElectrons.runtime_fields:
        electrons.remove(electrons.find(field))
    pot = electrons.find("potentiostat")
    pot.remove(pot.find("ethermo"))
    xml = ET.tostring(tree, encoding="unicode")
    with pytest.raises(ValueError, match="ambiguous inline initial state"):
        load(xml)
    tree.set("state_format", "initial")
    initial = load(ET.tostring(tree, encoding="unicode"))
    assert not initial.syslist[0].motion.electronic_state.feedback_initialized


def test_checkpoint_precision_roundtrips_all_physical_doubles():
    sim = load(input_xml(ne=True, steps=2))
    sim.run(write_outputs=False)
    system = sim.syslist[0]
    es = system.motion.electronic_state
    history = [0.15012345123, 0.16123456876, 0.17234567321]
    es._workfunction_average_history = history
    es.current_workfunction = history[-1]
    es.current_workfunction_steps_average = float(np.mean(history))
    es.q = 8.123456789012345
    es.p = 0.012345678901234567
    system.beads.q[0, 0] = 0.23456789012345678
    system.beads.p[0, 0] = 0.34567890123456789
    restored = load(checkpoint(sim, 2))
    assert_same(sim, restored)


def test_checkpoint_precision_does_not_change_concurrent_output_format():
    ready, finish = Event(), Event()
    value = 0.12345678901234567
    before = dict(ipi_global_settings)
    expected = write_float(value)

    def serialize():
        with checkpoint_serialization():
            ready.set()
            assert finish.wait(10)
            return write_float(value)

    with ThreadPoolExecutor(1) as executor:
        pending = executor.submit(serialize)
        assert ready.wait(10)
        ordinary = write_float(value)
        finish.set()
        exact = pending.result()
    assert ordinary == expected
    assert float(exact) == value
    assert ipi_global_settings == before


def test_near_default_scalar_array_and_attribute_values_are_not_omitted():
    one = float(np.nextafter(1.0, np.inf))
    tau = float(np.nextafter(100.0, np.inf))
    tree = ET.fromstring(input_xml())
    tree.find("./system/motion/dynamics/timestep").text = repr(one)
    tree.find("./system/motion/dynamics/electrons/mass").text = repr(one)
    tree.find("./system/motion/dynamics/electrons/potentiostat/tau").text = repr(tau)
    component = tree.find("./system/forces/force")
    component.set("weight", repr(one))
    ET.SubElement(component, "mts_weights").text = f"[{one!r}]"
    original = load(ET.tostring(tree, encoding="unicode"))
    restored = load(checkpoint(original, 0))
    motion = restored.syslist[0].motion
    assert motion.dt == one
    assert motion.electronic_state.mass == one
    assert motion.potentiostat.tau == tau
    assert motion._electronic_provider_component.weight == one
    np.testing.assert_array_equal(
        motion._electronic_provider_component.mts_weights, [one]
    )


def test_runtime_weight_change_is_rejected_even_when_boundary_cache_is_valid():
    sim = load(input_xml())
    sim._initialize_constant_potential_observables()
    motion = sim.syslist[0].motion
    motion._electronic_provider_component.weight = 2
    with pytest.raises(ValueError, match="weight"):
        motion.step(0)


def scanned_xml(xml, ne=False):
    quantity = "workfunction" if ne else "fermi_level"
    return xml.replace(
        f"<target_{quantity}>0</target_{quantity}>",
        f"<initial_target_{quantity}>.003</initial_target_{quantity}>"
        f"<final_target_{quantity}>-.004</final_target_{quantity}>"
        "<transition_steps>4</transition_steps>",
    )


@pytest.mark.parametrize("ne", [False, True])
@pytest.mark.parametrize("split", [0, 1, 2, 4])
def test_scan_target_work_survives_restart_without_duplicate_update(ne, split):
    xml = scanned_xml(input_xml(steps=max(split, 1), ne=ne, electronic_temp=0.01), ne)
    first = load(xml)
    if split:
        first.run(write_outputs=False)
    resumed = load(checkpoint(first, split))
    resumed.run(write_outputs=False)
    continuous = load(scanned_xml(input_xml(steps=6, ne=ne, electronic_temp=0.01), ne))
    continuous.run(write_outputs=False)
    assert_same(resumed, continuous)
    quantity = "target_workfunction" if ne else "target_ef"
    assert getattr(resumed.syslist[0].motion.electronic_state, quantity) == getattr(
        continuous.syslist[0].motion.electronic_state, quantity
    )


def test_midstep_rollback_snapshot_keeps_entire_electronic_state():
    sim = load(scanned_xml(input_xml(ne=True, electronic_temp=0.01), True))
    sim.run_step(0)
    sim.step = 1
    before = state(sim)
    rollback = CheckpointOutput()
    rollback.bind(sim)
    rollback.store()
    motion = sim.syslist[0].motion
    motion._begin_electronic_step(1)
    motion.integrator.mtsprop_ba(0)
    motion.electronic_state.drift(0.05)
    motion.potentiostat.O_step(0.1)
    sim.prng.gvec(1)  # A midstep random draw must also be rolled back.
    restored = load(rollback.status.write("simulation"))
    after = state(restored)
    for key in before:
        np.testing.assert_equal(after[key], before[key], err_msg=key)


def variant_xml(xml, ensemble, mts=False):
    if ensemble in ("npt", "nst"):
        external = (
            "<pressure>0</pressure>"
            if ensemble == "npt"
            else "<stress shape='(3,3)'>[0,0,0,0,0,0,0,0,0]</stress>"
        )
        xml = xml.replace(
            "<temperature>0</temperature>", "<temperature>.5</temperature>" + external
        )
        mode = "isotropic" if ensemble == "npt" else "anisotropic"
        reference = (
            ""
            if ensemble == "npt"
            else "<h0 shape='(3,3)'>[100,0,0,0,100,0,0,0,100]</h0>"
        )
        xml = xml.replace(
            "<timestep>0.1</timestep>",
            "<timestep>0.1</timestep>"
            "<thermostat mode='langevin'><tau>100</tau></thermostat>"
            f"<barostat mode='{mode}'><tau>1000</tau>{reference}"
            "<thermostat mode='langevin'><tau>100</tau></thermostat></barostat>",
        )
    if mts:
        xml = xml.replace(
            "<system>",
            "<fflj name='fast'><parameters>"
            "{eps:1, sigma:1}</parameters></fflj><system>",
        )
        xml = xml.replace(
            "<force forcefield='cp'/>",
            "<force forcefield='fast'><mts_weights>[1,0]</mts_weights></force>"
            "<force forcefield='cp'><mts_weights>[0,1]</mts_weights></force>",
        )
        xml = xml.replace(
            "<timestep>0.1</timestep>", "<timestep>0.1</timestep><nmts>[1,2]</nmts>"
        )
    return xml


@pytest.mark.parametrize("ensemble", ["nve", "npt", "nst"])
@pytest.mark.parametrize("splitting", ["baoab", "obabo"])
@pytest.mark.parametrize("mts", [False, True])
def test_barostat_and_mts_output_and_restart_boundaries(ensemble, splitting, mts):
    def xml(steps):
        return variant_xml(
            input_xml(
                steps=steps,
                ensemble=ensemble,
                splitting=splitting,
                electronic_temp=0.01,
            ),
            ensemble,
            mts,
        )

    enabled, disabled = load(xml(3)), load(xml(3))
    enabled.run(write_outputs=True)
    disabled.run(write_outputs=False)
    assert_same(enabled, disabled)
    first = load(xml(1))
    first.run(write_outputs=False)
    resumed = load(checkpoint(first, 1, 3))
    resumed.run(write_outputs=False)
    assert_same(resumed, disabled)


@pytest.mark.parametrize("window", [1, 3])
def test_four_bead_trajectory_uses_shared_q_and_one_mean_phi_sample_per_frame(window):
    def xml(steps):
        tree = ET.fromstring(
            input_xml(steps=steps, ne=True, window=window, electronic_temp=0.01)
        )
        beads = tree.find("./system/beads")
        beads.set("nbeads", "4")
        for field, values in (
            ("q", "[-.03,0,0,.01,0,0,.02,0,0,.06,0,0]"),
            ("p", "[1,0,0,.6,0,0,.9,0,0,1.2,0,0]"),
        ):
            node = beads.find(field)
            node.set("shape", "(4,3)")
            node.text = values
        tree.find("./system/ensemble/temperature").text = ".05"
        return ET.tostring(tree, encoding="unicode")

    enabled, disabled = load(xml(3)), load(xml(3))
    enabled.run(write_outputs=True)
    disabled.run(write_outputs=False)
    assert_same(enabled, disabled)
    first = load(xml(1))
    first.run(write_outputs=False)
    resumed = load(checkpoint(first, 1, 3))
    resumed.run(write_outputs=False)
    # Rebuilding normal modes from checkpointed bead coordinates can introduce
    # transform round-off. Integrator phase and stochastic state must match.
    assert_same(resumed, disabled, atol=1.0e-14)
    for sim in (enabled, disabled, resumed):
        es = sim.syslist[0].motion.electronic_state
        calls = sim.fflist["cp"].driver.calls
        assert len(calls) % 4 == 0
        for start in range(0, len(calls), 4):
            assert len({call[1] for call in calls[start : start + 4]}) == 1
        expected_phi = -float(np.mean(sim.syslist[0].beads.q[:, 0] + es.q - 8.0))
        assert es.current_workfunction == pytest.approx(
            expected_phi, rel=0, abs=1.0e-14
        )
        assert es.accepted_frame_id == 3
        assert es._workfunction_average_total_samples == 4
