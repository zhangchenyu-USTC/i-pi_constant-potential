from types import SimpleNamespace

import numpy as np
import pytest

from ipi.engine.constant_potential import (
    ElectronicRequestContext,
    append_init_token,
    mean_extra,
)
from ipi.engine.atoms import Atoms
from ipi.engine.cell import Cell
from ipi.engine.ensembles import Ensemble
from ipi.engine.forcefields import FFSocket, ForceRequest
from ipi.engine.forcefields_cp import FFChargeMixTwoSockets
from ipi.engine.motion.dynamics import DummyIntegrator, Dynamics
from ipi.engine.potentiostat import (
    ElectronicState,
    PotentiostatLangevin,
    PotentiostatSVR,
)
from ipi.engine.properties import Properties
from ipi.engine.system import System
from ipi.inputs.electrons import InputElectrons
from ipi.interfaces.sockets import Driver, HDRLEN, MESSAGE, Status
from ipi.utils.units import Constants, unit_to_internal, unit_to_user
from ipi.utils.depend import depend_value


def state(mode="fermi", q=7.5):
    kwargs = {"target_ef": -0.2} if mode == "fermi" else {"target_workfunction": 0.15}
    result = ElectronicState(q_init=q, mass=2.0, mode=mode, **kwargs)
    result.neutral_electrons = 8
    return result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "fermi", "initial_target_ef": -0.2, "transition_steps": 10},
        {
            "mode": "workfunction",
            "final_target_workfunction": 0.2,
            "transition_steps": 10,
        },
    ],
)
def test_linear_targets_require_complete_scan(kwargs):
    with pytest.raises(ValueError):
        ElectronicState(q_init=8.0, mass=1.0, **kwargs)


def test_nonfinite_electronic_drift_hard_fails():
    with pytest.raises(ValueError):
        state().drift(float("nan"))


def test_electronvolt_conversion_matches_driver_protocol_constant():
    internal = unit_to_internal("energy", "electronvolt", 1.0)
    assert internal == pytest.approx(1.0 / Constants.EV_PER_HARTREE, rel=1.0e-15)
    assert unit_to_user("energy", "electronvolt", internal) == pytest.approx(
        1.0, rel=1.0e-15
    )


def test_framework_owned_init_token_rejects_manual_collision():
    assert append_init_token("label : test", "NELECT", "7.5").endswith("NELECT : 7.5")
    with pytest.raises(ValueError, match="reserved"):
        append_init_token("NELECT : 99", "NELECT", "7.5")


def test_scan_force_sign_and_workfunction_average():
    scan = ElectronicState(
        q_init=8.0,
        mass=1.0,
        mode="fermi",
        initial_target_ef=-0.4,
        final_target_ef=0.0,
        transition_steps=4,
    )
    scan.neutral_electrons = 8
    scan.update_target_fermi_level(2)
    scan.current_ef = -0.3
    assert scan.target_ef == pytest.approx(-0.2)
    assert scan.force == pytest.approx(0.1)

    average = state("workfunction")
    average.Ne_doping = True
    average.workfunction_average_steps = 2
    average.reset_workfunction_average()
    average.record_workfunction_sample(0.1)
    assert average.freeze_electronic_dynamics
    average.record_workfunction_sample(0.3)
    assert not average.freeze_electronic_dynamics
    assert average.current_workfunction_steps_average == pytest.approx(0.2)
    assert average.force == pytest.approx(0.05)


def test_complete_restart_state_roundtrip():
    inputs = InputElectrons()
    inputs.store(
        {
            "enabled": True,
            "Ne_doping": True,
            "target_workfunction": 0.15,
            "potential_average_axis": "y",
            "potential_average_region": np.array([10.0, 12.0]),
            "neutral_electrons": 8,
            "q_init": 7.5,
            "p_init": 0.25,
            "mass": 50.0,
            "workfunction_average_steps": 3,
            "runtime_step": 0,
            "feedback_initialized": True,
            "accepted_frame_id": 1,
            "current_fermi_level": -0.1,
            "current_workfunction": 0.2,
            "workfunction_average": 0.15,
            "workfunction_history": np.array([0.1, 0.2]),
            "workfunction_total_samples": 2,
            "protocol_work": 0.6,
            "potentiostat": PotentiostatSVR(tau=100.0, ethermo=0.4),
        }
    )
    result = inputs.create_electronic_state()
    assert result.q == pytest.approx(7.5)
    assert result.p == pytest.approx(0.25)
    assert result.current_step == 0
    assert result.feedback_initialized
    assert result.accepted_frame_id == 1
    assert result._workfunction_average_history == [0.1, 0.2]
    assert result.freeze_electronic_dynamics
    assert result.protocol_work == pytest.approx(0.6)
    restored = inputs.fetch()["potentiostat"]
    assert isinstance(restored, PotentiostatSVR)
    assert restored.ethermo == pytest.approx(0.4)


def test_electronic_legendre_terms_match_force_signs():
    fermi = state("fermi", q=7.5)
    fermi.p = 0.4
    assert fermi.target_potential_energy == pytest.approx(-0.1)
    assert fermi.conserved_energy == pytest.approx(-0.06)

    workfunction = state("workfunction", q=7.5)
    workfunction.p = 0.4
    assert workfunction.target_potential_energy == pytest.approx(-0.075)
    assert workfunction.conserved_energy == pytest.approx(-0.035)


def test_linear_target_protocol_work_keeps_switch_energy_continuous():
    scan = ElectronicState(
        q_init=7.5,
        mass=2.0,
        mode="fermi",
        initial_target_ef=-0.4,
        final_target_ef=0.0,
        transition_steps=4,
    )
    scan.neutral_electrons = 8
    before = scan.conserved_energy

    scan.update_target_fermi_level(2)

    assert scan.target_ef == pytest.approx(-0.2)
    assert scan.protocol_work == pytest.approx(-0.1)
    assert scan.conserved_energy == pytest.approx(before)


@pytest.mark.parametrize("potentiostat_class", [PotentiostatLangevin, PotentiostatSVR])
def test_electronic_thermostat_energy_tracks_kinetic_exchange(
    potentiostat_class,
):
    electronic_state = state()
    electronic_state.p = 0.3
    potentiostat = potentiostat_class(temp=0.01, tau=5.0, ethermo=0.2)
    potentiostat.bind(
        electronic_state,
        prng=SimpleNamespace(gvec=lambda size: np.array([0.7])),
    )
    before = electronic_state.kinetic_energy + potentiostat.ethermo

    potentiostat.O_step(0.5)

    assert electronic_state.kinetic_energy + potentiostat.ethermo == pytest.approx(
        before, abs=1.0e-15
    )


def test_electronic_econs_is_scaled_once_for_shared_multibead_coordinate():
    electronic_state = state(q=7.5)
    electronic_state.p = 0.4
    potentiostat = PotentiostatSVR(ethermo=0.2)
    dynamics = SimpleNamespace(
        beads=SimpleNamespace(nbeads=8),
        electronic_state=electronic_state,
        potentiostat=potentiostat,
    )

    raw = Dynamics.get_electronic_econs(dynamics)

    expected_per_bead = electronic_state.conserved_energy + potentiostat.ethermo
    assert raw == pytest.approx(8 * expected_per_bead)
    assert raw / dynamics.beads.nbeads == pytest.approx(expected_per_bead)


def test_electronic_input_has_no_client_discriminator():
    inputs = InputElectrons()
    assert "client" not in inputs.attribs
    inputs.store(
        {
            "enabled": True,
            "Ne_doping": True,
            "target_workfunction": 0.15,
            "potential_average_axis": "z",
            "potential_average_region": np.array([10.0, 12.0]),
            "neutral_electrons": 8,
            "q_init": 7.5,
            "mass": 50.0,
        }
    )
    assert "client" not in inputs.fetch()


@pytest.mark.parametrize("axis", ["x", "Y", "z"])
def test_workfunction_potential_average_axis_is_explicit_and_normalized(axis):
    inputs = InputElectrons()
    inputs.store(
        {
            "enabled": True,
            "target_workfunction": 0.15,
            "potential_average_axis": axis,
            "potential_average_region": np.array([10.0, 12.0]),
            "neutral_electrons": 8,
            "q_init": 7.5,
            "mass": 50.0,
        }
    )
    assert inputs.fetch()["potential_average_axis"] == axis.lower()


def test_workfunction_rejects_missing_axis_and_legacy_z_region():
    inputs = InputElectrons()
    assert "z_average_region" not in inputs.fields
    with pytest.raises(ValueError, match="potential_average_axis"):
        inputs.store(
            {
                "enabled": True,
                "target_workfunction": 0.15,
                "z_average_region": np.array([10.0, 12.0]),
                "neutral_electrons": 8,
                "q_init": 7.5,
                "mass": 50.0,
            }
        )
        inputs.fetch()


def test_workfunction_init_protocol_sends_general_axis_and_region_tokens():
    interface = SimpleNamespace(
        mode="inet",
        batch_size=1,
        consolidate_messages=True,
        requests=None,
        offset=0.0,
    )
    forcefield = FFSocket(name="driver", interface=interface)
    dynamics = SimpleNamespace(
        electronic_state=state("workfunction"),
    )
    forcefield.configure_electrons(
        {
            "enabled": True,
            "neutral_electrons": 8,
            "potential_average_axis": "y",
            "potential_average_region": np.array(
                [
                    unit_to_internal("length", "angstrom", 10.0),
                    unit_to_internal("length", "angstrom", 12.0),
                ]
            ),
        },
        dynamics=dynamics,
    )
    forcefield.set_electronic_state(7.5)
    request = forcefield.queue(Atoms(1), Cell(np.eye(3)))
    assert "potential_average_axis : y" in request["pars"]
    assert "potential_average_min_A : 10" in request["pars"]
    assert "potential_average_max_A : 12" in request["pars"]
    assert "z_avg_" not in request["pars"]


def test_removed_stride_rejected():
    inputs = InputElectrons()
    assert "solvation_update_stride" not in inputs.fields
    interface = SimpleNamespace(mode="inet", batch_size=1, requests=None, offset=0.0)
    forcefield = FFSocket(name="test", interface=interface)
    with pytest.raises(ValueError, match="removed"):
        forcefield.configure_electrons(
            {"enabled": True, "solvation_update_stride": 1},
            dynamics=SimpleNamespace(electronic_state=state()),
        )


@pytest.mark.parametrize("ne_mode,expected", [(False, 0), (True, 2)])
def test_flags_do_not_depend_on_step(ne_mode, expected):
    interface = SimpleNamespace(mode="inet", batch_size=1, requests=None, offset=0.0)
    electronic_state = state()
    forcefield = FFSocket(name="test", interface=interface)
    forcefield.configure_electrons(
        {"enabled": True, "neutral_electrons": 8, "Ne_doping": ne_mode},
        dynamics=SimpleNamespace(electronic_state=electronic_state),
    )
    for step in (0, 1, 5, 100):
        electronic_state.current_step = step
        assert forcefield._mode_flags() == expected


def test_electronic_drift_synchronizes_charge():
    calls = []
    electronic_state = state(q=7.5)
    integrator = SimpleNamespace(
        electronic_state=electronic_state,
        potentiostat=SimpleNamespace(A=lambda dt: setattr(electronic_state, "q", 7.6)),
        dynamics=SimpleNamespace(
            _sync_electronic_charge=lambda taint=False: calls.append(taint)
        ),
    )
    DummyIntegrator.electronic_A_step(integrator, 0.5)
    assert calls == [True]


def test_single_socket_electron_value_is_not_client_dispatched():
    interface = SimpleNamespace(
        mode="inet",
        batch_size=1,
        consolidate_messages=True,
        requests=None,
        offset=0.0,
    )
    forcefield = FFSocket(name="any_driver", interface=interface)
    forcefield.configure_electrons(
        {"enabled": True, "neutral_electrons": 8},
        dynamics=SimpleNamespace(electronic_state=state()),
    )
    forcefield.set_electronic_state(7.5)
    assert forcefield.current_nelect == pytest.approx(7.5)


def test_span_two_uses_global_even_lattice_and_mixes():
    low = SimpleNamespace(name="low")
    high = SimpleNamespace(name="high")
    electronic_state = state("workfunction", q=107.5)
    dynamics = SimpleNamespace(electronic_state=electronic_state)
    wrapper = FFChargeMixTwoSockets(low, high)
    # Endpoint configuration is tested separately from interpolation so simple
    # endpoint stubs can be used here.
    wrapper._dynamics_ref = dynamics
    wrapper.charge_mixing_span = 2
    wrapper.endpoint_low.set_electronic_state = lambda q: setattr(low, "q", q)
    wrapper.endpoint_high.set_electronic_state = lambda q: setattr(high, "q", q)
    wrapper.set_electronic_state(107.5)
    assert (wrapper.q_low, wrapper.q_high, wrapper.lambda_val) == pytest.approx(
        (106, 108, 0.75)
    )
    snapshot = ElectronicRequestContext(107.5, 0, "workfunction", id(dynamics))
    request = {
        "lambda": 0.75,
        "q_low": 106,
        "q_high": 108,
        "pos": np.zeros(3),
        "electronic_context": snapshot,
    }
    for label, charge in (("low", 106), ("high", 108)):
        request["child_" + label] = {
            "pos": request["pos"],
            "electronic_context": snapshot.at_charge(charge),
        }

    result = wrapper._mix_results(
        [
            1.0,
            np.array([1.0, 0.0, 0.0]),
            np.eye(3),
            {
                "fermi_level_eV": -2.0,
                "workfunction_eV": 4.0,
                "nelect": 106.0,
            },
        ],
        [
            3.0,
            np.array([5.0, 0.0, 0.0]),
            3.0 * np.eye(3),
            {
                "fermi_level_eV": 0.0,
                "workfunction_eV": 6.0,
                "nelect": 108.0,
            },
        ],
        request,
    )
    assert result[0] == pytest.approx(2.5)
    assert result[1] == pytest.approx([4.0, 0.0, 0.0])
    assert result[3]["nelect"] == pytest.approx(107.5)
    assert result[3]["workfunction_eV"] == pytest.approx(5.5)


@pytest.mark.parametrize(
    "span,q,expected",
    [(1, 107.5, (107, 108)), (2, 107.5, (106, 108)), (2, 108.0, (108, 110))],
)
def test_charge_mixing_chgdata_endpoints_are_integers(span, q, expected):
    low = SimpleNamespace(
        name="vasp_low",
        set_electronic_state=lambda value: setattr(low, "q", value),
    )
    high = SimpleNamespace(
        name="vasp_high",
        set_electronic_state=lambda value: setattr(high, "q", value),
    )
    wrapper = FFChargeMixTwoSockets(low, high)
    wrapper.charge_mixing_span = span
    wrapper.set_electronic_state(q)

    assert (wrapper.q_low, wrapper.q_high) == expected
    assert low.q == float(expected[0])
    assert high.q == float(expected[1])
    assert low.q.is_integer() and high.q.is_integer()


@pytest.mark.parametrize("explicit_endpoints", [False, True])
def test_one_logical_force_builds_client_independent_two_socket_wrapper(
    explicit_endpoints,
):
    def socket_forcefield(name):
        interface = SimpleNamespace(
            mode="inet",
            batch_size=1,
            consolidate_messages=True,
            requests=None,
            offset=0.0,
        )
        return FFSocket(name=name, interface=interface)

    low = socket_forcefield("vasp_low")
    high = socket_forcefield("vasp_high")
    config = {"enabled": True, "charge_mixing": True}
    if explicit_endpoints:
        config.update(charge_mixing_low="vasp_low", charge_mixing_high="vasp_high")
    component = SimpleNamespace(ffield="vasp_low", name="electronic_force", nbeads=0)
    system = System.__new__(System)
    system.motion = SimpleNamespace(electrons_config=config)
    system.simul = SimpleNamespace(fflist={"vasp_low": low, "vasp_high": high})
    system.fcomp = [component]
    system.beads = SimpleNamespace(nbeads=1)
    system.prefix = "test"

    system._prepare_charge_mixing_forcefield()

    wrapper = system.simul.fflist[component.ffield]
    assert isinstance(wrapper, FFChargeMixTwoSockets)
    assert wrapper.endpoint_low is low
    assert wrapper.endpoint_high is high
    assert component._charge_mixing_original_ffield == "vasp_low"
    assert component._charge_mixing_secondary_ffield == "vasp_high"


def test_multibead_observable_uses_bead_mean():
    extras = {"fermi_level_eV": [-2.0, 0.0, 1.0]}
    assert mean_extra(extras, "fermi_level_eV") == pytest.approx(-1.0 / 3.0)


@pytest.mark.parametrize("nbeads", [1, 2, 16])
def test_shared_electronic_coordinate_defaults_to_physical_temperature(
    monkeypatch, nbeads
):
    physical_temperature = 300.0
    potentiostat = PotentiostatLangevin(tau=100.0)
    config = {
        "enabled": True,
        "potentiostat": potentiostat,
        "potentiostat_temp": float("nan"),
    }
    forcefield = SimpleNamespace(
        constant_potential_capable=True,
        charge_enabled=True,
        configure_electrons=lambda config, dynamics=None: None,
        set_electronic_state=lambda q: None,
    )
    dynamics = Dynamics.__new__(Dynamics)
    dynamics.electrons_config = config
    dynamics.enstype = "nvt"
    dynamics.ensemble = Ensemble(temp=physical_temperature)
    dynamics.beads = SimpleNamespace(nbeads=nbeads)
    dynamics._ntemp = depend_value(
        name="test_ntemp", value=physical_temperature * nbeads
    )
    dynamics._dt = depend_value(name="test_dt", value=1.0)
    dynamics.forces = SimpleNamespace(
        mforces=[
            SimpleNamespace(
                nbeads=nbeads,
                weight=1.0,
                mts_weights=np.ones(1),
                _forces=[
                    SimpleNamespace(
                        ff=forcefield, bind_electronic_context=lambda dynamics: None
                    )
                ],
            )
        ]
    )
    dynamics._fermi_cache = {}
    dynamics._fermi_cache_valid = False
    monkeypatch.setattr(
        InputElectrons,
        "create_electronic_state",
        lambda self, config=None: state(),
    )

    dynamics._bind_electronic_dynamics(SimpleNamespace())

    assert dynamics.ntemp == pytest.approx(physical_temperature * nbeads)
    assert potentiostat.temp == pytest.approx(physical_temperature)


def test_charge_update_taints_only_charge_dependent_mts_component():
    class TaintCounter:
        def __init__(self):
            self.count = 0

        def taint(self):
            self.count += 1

    charge_ff = SimpleNamespace(
        charge_enabled=True,
        set_electronic_state=lambda q: setattr(charge_ff, "q", q),
    )
    independent_ff = SimpleNamespace(charge_enabled=False)
    charge_bead = SimpleNamespace(ff=charge_ff, _ufvx=TaintCounter())
    independent_bead = SimpleNamespace(ff=independent_ff, _ufvx=TaintCounter())
    dynamics = Dynamics.__new__(Dynamics)
    dynamics.electronic_state = state(q=7.25)
    dynamics.forces = SimpleNamespace(
        mforces=[
            SimpleNamespace(_forces=[charge_bead]),
            SimpleNamespace(_forces=[independent_bead]),
        ]
    )
    dynamics._clear_fermi_cache = lambda: None

    dynamics._sync_electronic_charge(taint=True)

    assert charge_ff.q == pytest.approx(7.25)
    assert charge_bead._ufvx.count == 1
    assert independent_bead._ufvx.count == 0


@pytest.mark.parametrize("ensemble", ["nvt-cc", "sc", "scnpt"])
def test_unimplemented_electronic_ensembles_hard_fail(ensemble):
    dynamics = Dynamics.__new__(Dynamics)
    dynamics.electrons_config = {"enabled": True}
    dynamics.enstype = ensemble
    with pytest.raises(ValueError, match="supports only NVE, NVT, NPT, and NST"):
        dynamics._bind_electronic_dynamics(SimpleNamespace())


@pytest.mark.parametrize("mode,batch", [("shm", 1), ("inet", 2)])
def test_unverified_socket_transports_are_rejected(mode, batch):
    interface = SimpleNamespace(
        mode=mode,
        batch_size=batch,
        consolidate_messages=True,
        requests=None,
        offset=0.0,
    )
    forcefield = FFSocket(name="test", interface=interface)
    dynamics = SimpleNamespace(electronic_state=state())
    with pytest.raises(ValueError):
        forcefield.configure_electrons(
            {"enabled": True, "neutral_electrons": 8}, dynamics=dynamics
        )


@pytest.mark.parametrize("consolidated", [False, True])
def test_both_socket_message_modes_are_accepted(consolidated):
    interface = SimpleNamespace(
        mode="unix",
        batch_size=1,
        consolidate_messages=consolidated,
        requests=None,
        offset=0.0,
    )
    forcefield = FFSocket(name="test", interface=interface)
    forcefield.configure_electrons(
        {"enabled": True, "neutral_electrons": 8},
        dynamics=SimpleNamespace(electronic_state=state()),
    )
    assert forcefield.charge_enabled


def test_constant_potential_socket_cannot_be_shared_by_systems():
    interface = SimpleNamespace(
        mode="inet",
        batch_size=1,
        consolidate_messages=True,
        requests=None,
        offset=0.0,
    )
    forcefield = FFSocket(name="shared", interface=interface)
    first = SimpleNamespace(electronic_state=state())
    forcefield.configure_electrons(
        {"enabled": True, "neutral_electrons": 8}, dynamics=first
    )
    with pytest.raises(RuntimeError, match="multiple systems"):
        forcefield.configure_electrons(
            {"enabled": True, "neutral_electrons": 8},
            dynamics=SimpleNamespace(electronic_state=state()),
        )


def test_socket_extras_require_one_scalar_per_bead():
    interface = SimpleNamespace(
        mode="inet",
        batch_size=1,
        consolidate_messages=True,
        requests=None,
        offset=0.0,
    )
    forcefield = FFSocket(name="strict", interface=interface)
    dynamics = SimpleNamespace(
        electronic_state=state(), beads=SimpleNamespace(nbeads=2)
    )
    forcefield.configure_electrons(
        {"enabled": True, "neutral_electrons": 8}, dynamics=dynamics
    )
    forcefield.set_electronic_state(7.5)
    with pytest.raises(RuntimeError, match="one scalar"):
        forcefield.validate_extras(
            {
                "fermi_level_eV": [[-0.2, -0.2], [-0.2, -0.2]],
                "nelect": [[7.5, 7.5], [7.5, 7.5]],
            }
        )


def test_single_socket_ring_polymer_contraction_is_rejected_at_bind(monkeypatch):
    forcefield = SimpleNamespace(
        constant_potential_capable=True,
        charge_enabled=False,
        configure_electrons=lambda config, dynamics=None: setattr(
            forcefield, "charge_enabled", True
        ),
        set_electronic_state=lambda q: None,
    )
    dynamics = Dynamics.__new__(Dynamics)
    dynamics.electrons_config = {
        "enabled": True,
        "potentiostat": PotentiostatLangevin(tau=100.0),
        "potentiostat_temp": 300.0,
    }
    dynamics.enstype = "nvt"
    dynamics.ensemble = SimpleNamespace(temp=300.0)
    dynamics.beads = SimpleNamespace(nbeads=4)
    dynamics._dt = depend_value(name="test_contraction_dt", value=1.0)
    dynamics.forces = SimpleNamespace(
        mforces=[SimpleNamespace(nbeads=1, _forces=[SimpleNamespace(ff=forcefield)])]
    )
    dynamics._fermi_cache = {}
    dynamics._fermi_cache_valid = False
    monkeypatch.setattr(
        InputElectrons,
        "create_electronic_state",
        lambda self, config=None: state(),
    )
    with pytest.raises(ValueError, match="cannot use ring-polymer contraction"):
        dynamics._bind_electronic_dynamics(SimpleNamespace())


def test_chgdata_wire_payload_is_12_plus_8_plus_4_bytes():
    capture = SimpleNamespace(status=Status.Up | Status.Ready, payload=None)
    capture.sendall = lambda payload: setattr(capture, "payload", payload)
    Driver.sendchg(capture, 7.5, 2)
    payload = capture.payload
    assert len(payload) == HDRLEN + 12
    assert payload[:HDRLEN] == MESSAGE["chgdata"]
    assert np.frombuffer(payload[HDRLEN : HDRLEN + 8], np.float64)[0] == 7.5
    assert np.frombuffer(payload[HDRLEN + 8 :], np.int32)[0] == 2


def test_consolidated_payload_orders_chgdata_before_posdata():
    capture = SimpleNamespace(
        status=Status.Up | Status.Ready,
        lastreq=None,
        peername="capture",
        payload=None,
    )
    capture.get_status = lambda: capture.status
    capture.sendall = lambda payload: setattr(capture, "payload", payload)
    request = ForceRequest(
        {
            "id": 0,
            "pars": " ",
            "pos": np.zeros(3),
            "active": np.arange(3),
            "cell": (np.eye(3), np.eye(3)),
            "nelect": 7.5,
            "mode_flags": 0,
            "t_dispatched": 0.0,
        }
    )
    assert Driver.dispatch_send(capture, request)
    expected = HDRLEN + 8 + 4 + HDRLEN + 9 * 8 * 2 + 4 + 3 * 8 + HDRLEN
    payload = capture.payload
    assert len(payload) == expected
    assert payload[:HDRLEN] == MESSAGE["chgdata"]
    assert payload[HDRLEN + 12 : HDRLEN + 12 + HDRLEN] == MESSAGE["posdata"]


def test_nonconsolidated_dispatch_orders_charge_before_positions():
    calls = []

    class ProtocolStub:
        status = Status.Up | Status.Ready
        lastreq = None

        def get_status(self):
            if self.status & Status.Busy:
                self.status = Status.Up | Status.HasData
            elif self.status & Status.HasData:
                self.status = Status.Up | Status.Ready
            return self.status

        def sendchg(self, nelect, flags):
            calls.append(("chgdata", nelect, flags))

        def sendpos(self, pos, cell):
            calls.append(("posdata",))
            self.status = Status.Up | Status.Busy

        def getforce(self):
            calls.append(("getforce",))
            return [0.0, np.zeros(3), np.zeros((3, 3)), {}]

    request = ForceRequest(
        {
            "id": 0,
            "pars": " ",
            "pos": np.zeros(3),
            "active": np.arange(3),
            "cell": (np.eye(3), np.eye(3)),
            "nelect": 7.5,
            "mode_flags": 0,
            "offset": 0.0,
            "status": "Running",
        }
    )
    Driver.dispatch(ProtocolStub(), request)
    assert [item[0] for item in calls] == ["chgdata", "posdata", "getforce"]
    assert request["status"] == "Done"


def test_nonconsolidated_reconnect_preserves_flags():
    calls = []

    class ProtocolStub:
        status = Status.Up | Status.NeedsInit
        lastreq = None

        def get_status(self):
            if self.status & Status.NeedsInit:
                return self.status
            if self.status & Status.Busy:
                self.status = Status.Up | Status.HasData
            elif self.status & Status.HasData:
                self.status = Status.Up | Status.Ready
            return self.status

        def initialize(self, rid, pars):
            calls.append(("init", rid, pars))
            self.status = Status.Up | Status.Ready

        def sendchg(self, nelect, flags):
            calls.append(("chgdata", nelect, flags))

        def sendpos(self, pos, cell):
            self.status = Status.Up | Status.Busy

        def getforce(self):
            return [0.0, np.zeros(3), np.zeros((3, 3)), {}]

    request = ForceRequest(
        {
            "id": 4,
            "pars": "NELECT : 7.5",
            "pos": np.zeros(3),
            "active": np.arange(3),
            "cell": (np.eye(3), np.eye(3)),
            "nelect": 7.5,
            "mode_flags": 0,
            "offset": 0.0,
            "status": "Running",
        }
    )
    Driver.dispatch(ProtocolStub(), request)
    assert calls[0][0] == "init"
    assert calls[1] == ("chgdata", 7.5, 0)


@pytest.mark.parametrize(
    "electronic_state,expected",
    [(state("fermi", 7.0), 9.8), (state("workfunction", 9.0), 10.15)],
)
def test_grand_canonical_potential_sign(electronic_state, expected):
    properties = Properties()
    properties.motion = SimpleNamespace(electronic_state=electronic_state)
    properties.get_pot = lambda: 10.0
    assert properties.get_grand_canonical_potential() == pytest.approx(expected)


def test_removed_xml_field_is_rejected():
    from ipi.utils.io.inputs.io_xml import xml_parse_string

    node = xml_parse_string(
        "<electrons><solvation_update_stride>1</solvation_update_stride></electrons>"
    )
    with pytest.raises(NameError, match="solvation_update_stride"):
        InputElectrons().parse(node.fields[0][1])


@pytest.mark.parametrize("flags", [-1, 1, 3, 4, 2147483647])
def test_sendchg_rejects_reserved_or_unknown_flags(flags):
    capture = SimpleNamespace(status=Status.Up | Status.Ready)
    capture.sendall = lambda payload: pytest.fail("Invalid flags reached the wire")
    with pytest.raises(ValueError, match="0 or 2"):
        Driver.sendchg(capture, 8.0, flags)


@pytest.mark.parametrize("flags", [0, 2])
def test_consolidated_reconnect_preserves_flags(flags):
    capture = SimpleNamespace(status=Status.Up | Status.NeedsInit, lastreq=None)
    capture.get_status = lambda: capture.status
    capture.initialize = lambda rid, pars: setattr(
        capture, "status", Status.Up | Status.Ready
    )
    capture.sendall = lambda payload: setattr(capture, "payload", payload)
    request = ForceRequest(
        dict(
            id=0,
            pars="NELECT:8",
            pos=np.zeros(3),
            active=np.arange(3),
            cell=(np.eye(3), np.eye(3)),
            nelect=8.0,
            mode_flags=flags,
        )
    )
    assert Driver.dispatch_send(capture, request)
    assert (
        np.frombuffer(capture.payload[HDRLEN + 8 : HDRLEN + 12], np.int32)[0] == flags
    )


@pytest.mark.parametrize("flags", [1, 3, -1, 4])
def test_consolidated_rejects_old_flags(flags):
    capture = SimpleNamespace(status=Status.Up | Status.Ready, lastreq=None)
    capture.sendall = lambda payload: pytest.fail("Invalid flags reached the wire")
    request = ForceRequest(
        dict(
            id=0,
            pars=" ",
            pos=np.zeros(3),
            active=np.arange(3),
            cell=(np.eye(3), np.eye(3)),
            nelect=8.0,
            mode_flags=flags,
        )
    )
    with pytest.raises(ValueError, match="0 or 2"):
        Driver.dispatch_send(capture, request)


def test_manual_removed_init_token_is_rejected():
    interface = SimpleNamespace(mode="inet", batch_size=1, requests=None, offset=0.0)
    ff = FFSocket(name="test", interface=interface, pars={"solvation_update_stride": 1})
    ff.configure_electrons(
        dict(enabled=True, neutral_electrons=8),
        dynamics=SimpleNamespace(electronic_state=state()),
    )
    ff.set_electronic_state(8.0)
    with pytest.raises(ValueError, match="removed"):
        ff.queue(Atoms(1), Cell(np.eye(3)))
