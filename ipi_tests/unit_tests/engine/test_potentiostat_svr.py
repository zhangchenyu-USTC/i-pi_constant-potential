"""Finite-time and restart contracts for the scalar electronic SVR bath."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from ipi.engine.potentiostat import ElectronicState, PotentiostatSVR
from ipi.utils.prng import Random
from ipi.utils.units import Constants


def bath(momentum=0.0, temperature=0.01, random=None):
    state = ElectronicState(q_init=8.0, mass=2.0, target_ef=0.0, p_init=momentum)
    state.neutral_electrons = 8
    thermostat = PotentiostatSVR(temp=temperature, tau=100.0, ethermo=0.125)
    thermostat.bind(state, Random(seed=61723) if random is None else random)
    return state, thermostat


@pytest.mark.parametrize("momentum", [0.0, -0.0, 0.3, -0.3, 1e-160, -1e-160])
def test_zero_time_preserves_state_and_random_stream(momentum):
    rng = Random(seed=912)
    saved = copy.deepcopy(rng.state)
    state, thermostat = bath(momentum, random=rng)
    before = state.p, thermostat.ethermo
    thermostat.O_step(0.0)
    assert (state.p, thermostat.ethermo) == before
    np.testing.assert_array_equal(rng.gvec(8), Random(state=saved).gvec(8))


@pytest.mark.parametrize("dt", [1e-18, 1e-8, 1.0, 100000.0])
@pytest.mark.parametrize("momentum", [0.0, 1e-160, -1e-160])
def test_zero_and_subnormal_kinetic_energy_have_finite_time_noise(dt, momentum):
    state, thermostat = bath(
        momentum, random=SimpleNamespace(gvec=lambda size: np.ones(size))
    )
    before = state.kinetic_energy + thermostat.ethermo
    thermostat.O_step(dt)
    # At vanishing initial momentum the kinetic energy has the exact finite
    # time value K_eq * (1-exp(-dt/tau)) for this unit Gaussian draw.
    expected = 0.5 * Constants.kb * thermostat.temp * (-np.expm1(-dt / 100.0))
    assert state.kinetic_energy == pytest.approx(expected, rel=2e-14, abs=0.0)
    assert state.kinetic_energy + thermostat.ethermo == pytest.approx(before)
    assert np.isfinite(state.p)
    assert (state.p < 0) == (momentum < 0)


@pytest.mark.parametrize("momentum", [0.0, 0.3, -0.3])
def test_zero_temperature_relaxes_kinetic_energy_and_consumes_one_gaussian(momentum):
    calls = []

    def draw(size):
        calls.append(size)
        return np.array([123.0])

    state, thermostat = bath(
        momentum, temperature=0.0, random=SimpleNamespace(gvec=draw)
    )
    initial = state.kinetic_energy
    thermostat.O_step(25.0)
    assert state.kinetic_energy == pytest.approx(initial * np.exp(-0.25))
    assert calls == [1]


def test_svr_keeps_sign_changes_without_an_artificial_minimum_scale():
    # Choose the Gaussian so that the original nonzero-momentum SVR scale
    # cancels exactly. The result may be zero, rather than a clipped scale.
    state, thermostat = bath(0.3)
    decay = np.exp(-0.5)
    sigma = np.sqrt(state.mass * Constants.kb * thermostat.temp * (-np.expm1(-1.0)))
    thermostat.prng = SimpleNamespace(
        gvec=lambda size: np.array([-decay * state.p / sigma])
    )
    thermostat.O_step(100.0)
    assert abs(state.p) < 1e-16

    state.p = -0.3
    thermostat.prng = SimpleNamespace(gvec=lambda size: np.array([-10.0]))
    thermostat.O_step(100.0)
    assert state.p > 0


@pytest.mark.parametrize("dt", [-1.0, float("nan"), float("inf")])
def test_invalid_time_rejected_without_drawing(dt):
    def fail_draw(size):
        raise AssertionError("Invalid time must not consume a Gaussian")

    state, thermostat = bath(random=SimpleNamespace(gvec=fail_draw))
    with pytest.raises(ValueError, match="timestep"):
        thermostat.O_step(dt)
    assert state.p == 0.0
    assert thermostat.ethermo == 0.125


def test_frozen_window_does_not_draw_or_exchange_heat():
    state = ElectronicState(
        q_init=8.0, mass=2.0, mode="workfunction", target_workfunction=0.2
    )
    state.Ne_doping = True
    state.workfunction_average_steps = 3
    thermostat = PotentiostatSVR()
    thermostat.bind(
        state, SimpleNamespace(gvec=lambda size: pytest.fail("Frozen bath drew noise"))
    )
    thermostat.O_step(1.0)
    assert state.p == 0.0
    assert thermostat.ethermo == 0.0


@pytest.mark.parametrize("initial", [0.0, -0.2, 0.2])
def test_conditional_mean_and_variance(initial):
    state, thermostat = bath(initial)
    values = np.empty(20000)
    for index in range(len(values)):
        state.p = initial
        thermostat.O_step(70.0)
        values[index] = state.p
    mean = np.exp(-0.35) * initial
    variance = state.mass * Constants.kb * thermostat.temp * (1.0 - np.exp(-0.7))
    assert abs(values.mean() - mean) < 5 * np.sqrt(variance / len(values))
    assert abs(values.var() / variance - 1.0) < 5 * np.sqrt(2.0 / (len(values) - 1))


def test_maxwell_distribution_is_invariant():
    state, thermostat = bath()
    variance = state.mass * Constants.kb * thermostat.temp
    initial = np.random.default_rng(888).normal(scale=np.sqrt(variance), size=20000)
    values = np.empty_like(initial)
    for index, momentum in enumerate(initial):
        state.p = momentum
        thermostat.O_step(37.0)
        values[index] = state.p
    assert abs(values.mean()) < 5 * np.sqrt(variance / len(values))
    assert abs(values.var() / variance - 1.0) < 5 * np.sqrt(2.0 / (len(values) - 1))
    assert abs(np.mean(values < 0) - 0.5) < 0.02


def test_restored_random_stream_and_heat_match_uninterrupted_bath():
    state, thermostat = bath()
    for _ in range(7):
        thermostat.O_step(0.1)
    restored, restart = bath(
        state.p, random=Random(state=copy.deepcopy(thermostat.prng.state))
    )
    restart.ethermo = thermostat.ethermo
    for dt in [0.0, 0.2, 1e-8, 13.0, 0.3]:
        thermostat.O_step(dt)
        restart.O_step(dt)
        assert restored.p == state.p
        assert restart.ethermo == thermostat.ethermo
