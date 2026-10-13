"""Physical-temperature inheritance and its checkpoint semantics."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ipi.utils.units import unit_to_internal
from ipi_tests.unit_tests.engine.test_electronic_lifecycle import (
    assert_same,
    checkpoint,
    input_xml,
    load,
    state,
)


def kelvin(value):
    return unit_to_internal("temperature", "kelvin", value)


def temperature_xml(mode, nbeads, electronic_temp, splitting="baoab"):
    root = ET.fromstring(input_xml(steps=4, ensemble="nvt", splitting=splitting))
    temperature = root.find("./system/ensemble/temperature")
    temperature.set("units", "kelvin")
    temperature.text = "300"
    dynamics = root.find("./system/motion/dynamics")
    thermostat = ET.SubElement(dynamics, "thermostat", mode="langevin")
    ET.SubElement(thermostat, "tau").text = "100"
    potentiostat = dynamics.find("electrons/potentiostat")
    potentiostat.set("mode", mode)
    if electronic_temp is None:
        potentiostat.remove(potentiostat.find("temp"))
    else:
        potentiostat.find("temp").set("units", "kelvin")
        potentiostat.find("temp").text = str(electronic_temp)
    beads = root.find("./system/beads")
    beads.set("nbeads", str(nbeads))
    for field, vector in (("q", [0, 0, 0]), ("p", [1, 0, 0])):
        node = beads.find(field)
        node.set("shape", str((nbeads, 3)))
        node.text = str(vector * nbeads)
    return ET.tostring(root, encoding="unicode")


@pytest.mark.parametrize("mode", ["langevin", "svr"])
@pytest.mark.parametrize("nbeads", [1, 4])
@pytest.mark.parametrize("electronic_temp", [None, 0.0, 150.0])
def test_temperature_dependency_and_existing_checkpoint_format(
    mode, nbeads, electronic_temp
):
    sim = load(temperature_xml(mode, nbeads, electronic_temp))
    system = sim.syslist[0]
    potentiostat = system.motion.potentiostat
    expected = kelvin(300.0 if electronic_temp is None else electronic_temp)
    assert potentiostat.temp == expected
    nuclear_noise = system.motion.thermostat.S

    system.ensemble.temp = kelvin(600.0)
    expected = kelvin(600.0 if electronic_temp is None else electronic_temp)
    assert potentiostat.temp == expected
    assert system.motion.thermostat.temp == nbeads * kelvin(600.0)
    assert system.motion.thermostat.S == pytest.approx(nuclear_noise * np.sqrt(2))
    assert sim.fflist["cp"].driver.calls == []

    # Existing checkpoint_v1 already distinguishes omission from explicit zero.
    saved = checkpoint(sim, 0, 4)
    root = ET.fromstring(saved)
    assert root.get("state_format") == "checkpoint_v1"
    temp = root.find("./system/motion/dynamics/electrons/potentiostat/temp")
    if electronic_temp is None:
        assert temp is None
    else:
        assert float(temp.text) == kelvin(electronic_temp)
    restored = load(saved)
    assert_same(restored, sim)
    system = restored.syslist[0]
    potentiostat = system.motion.potentiostat
    assert potentiostat.temp == expected
    system.ensemble.temp = kelvin(900.0)
    expected = kelvin(900.0 if electronic_temp is None else electronic_temp)
    assert potentiostat.temp == expected

    # Both electronic thermostats must use the new temperature in their noise,
    # even after their coefficients/properties have already been evaluated.
    potentiostat.prng = SimpleNamespace(gvec=lambda count: np.ones(count))
    momenta = []
    for temperature in (300.0, 600.0):
        system.ensemble.temp = kelvin(temperature)
        system.motion.electronic_state.p = 0.0
        potentiostat.O_step(0.1)
        momenta.append(system.motion.electronic_state.p)
    if electronic_temp is None:
        assert momenta[0] > 0.0
        assert momenta[1] == pytest.approx(momenta[0] * np.sqrt(2))
    elif electronic_temp == 0.0:
        assert momenta == [0.0, 0.0]
    else:
        assert momenta[0] > 0.0
        assert momenta[1] == momenta[0]


def advance_at_temperatures(sim, stop, temperatures):
    """Change physical temperature only at complete MD boundaries."""
    for step in range(int(sim.step), stop):
        sim.syslist[0].ensemble.temp = kelvin(temperatures[step])
        sim.run_step(step)
        sim.step = step + 1


@pytest.mark.parametrize("mode", ["langevin", "svr"])
@pytest.mark.parametrize("splitting", ["baoab", "obabo"])
@pytest.mark.parametrize("nbeads", [1, 4])
@pytest.mark.parametrize("electronic_temp", [None, 0.0, 150.0])
@pytest.mark.parametrize("temperatures", [(300.0,) * 4, (300.0, 600.0, 450.0, 900.0)])
def test_constant_and_changing_temperature_restart(
    mode, splitting, nbeads, electronic_temp, temperatures
):
    xml = temperature_xml(mode, nbeads, electronic_temp, splitting)
    continuous = load(xml)
    initial_rng = state(continuous)["rng"]
    advance_at_temperatures(continuous, 4, temperatures)

    first = load(xml)
    advance_at_temperatures(first, 1, temperatures)
    resumed = load(checkpoint(first, 1, 4))
    assert_same(resumed, first, atol=1e-14 if nbeads > 1 else 0.0)
    advance_at_temperatures(resumed, 4, temperatures)
    assert_same(resumed, continuous, atol=1e-14 if nbeads > 1 else 0.0)
    assert state(resumed)["rng"] != initial_rng
    assert state(resumed)["ethermo"] != 0.0
    expected = kelvin(temperatures[-1] if electronic_temp is None else electronic_temp)
    assert resumed.syslist[0].motion.potentiostat.temp == expected

    # Each uninterrupted segment needs its initial frame plus one per MD step.
    # A fresh process evaluates the saved boundary once without drawing noise.
    assert len(continuous.fflist["cp"].driver.calls) == nbeads * 5
    assert len(first.fflist["cp"].driver.calls) == nbeads * 2
    assert len(resumed.fflist["cp"].driver.calls) == nbeads * 4
