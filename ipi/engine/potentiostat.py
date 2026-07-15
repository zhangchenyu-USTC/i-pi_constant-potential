"""Electronic state and thermostats for constant-potential dynamics.

The electronic coordinate is the total electron number. Its conjugate force is
the difference between the requested and calculated electronic potential.

Reference: C. Zhang et al., A Flexible and Generalized Constant-Potential
Framework in i-PI, J. Chem. Theory Comput. (2026).
"""

import numpy as np

from ipi.utils.depend import depend_value, dproperties
from ipi.utils.prng import Random
from ipi.utils.units import Constants

__all__ = [
    "ElectronicChargeError",
    "ElectronicState",
    "Potentiostat",
    "PotentiostatLangevin",
    "PotentiostatSVR",
]


Q_MIN = 1.0e-6


class ElectronicChargeError(ValueError):
    """Raised when the total electron number is invalid."""


def _finite(value, label):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return value


class ElectronicState:
    """Electronic coordinate, momentum, targets, and restartable observables."""

    def __init__(
        self,
        q_init,
        mass,
        mode="fermi",
        target_ef=None,
        initial_target_ef=None,
        final_target_ef=None,
        target_workfunction=None,
        initial_target_workfunction=None,
        final_target_workfunction=None,
        transition_steps=None,
        p_init=0.0,
        protocol_work=0.0,
    ):
        self.mass = _finite(mass, "Electronic mass")
        if self.mass <= 0.0:
            raise ValueError("Electronic mass must be positive.")

        self._q = depend_value(name="electronic_q", value=1.0)
        self._p = depend_value(name="electronic_p", value=0.0)
        self._protocol_work = depend_value(
            name="electronic_protocol_work",
            value=_finite(protocol_work, "Electronic protocol work"),
        )
        self._neutral_electrons = None
        self.q = q_init
        self.p = p_init

        self.mode = str(mode)
        if self.mode not in ("fermi", "workfunction"):
            raise ValueError(f"Unsupported electronic control mode '{self.mode}'.")

        self.current_step = 0
        self.current_ef = 0.0
        self.current_workfunction = 0.0
        self.current_workfunction_steps_average = 0.0
        self.Ne_doping = False
        self.workfunction_average_steps = 0
        self._workfunction_average_history = []
        self._workfunction_average_total_samples = 0

        self._linear_mode = transition_steps is not None
        self._transition_steps = None
        self._initial_target_ef = None
        self._final_target_ef = None
        self._initial_target_workfunction = None
        self._final_target_workfunction = None

        supplied_fermi = [target_ef, initial_target_ef, final_target_ef]
        supplied_workfunction = [
            target_workfunction,
            initial_target_workfunction,
            final_target_workfunction,
        ]
        if self.mode == "fermi" and any(v is not None for v in supplied_workfunction):
            raise ValueError("Workfunction targets cannot be used in Fermi-level mode.")
        if self.mode == "workfunction" and any(v is not None for v in supplied_fermi):
            raise ValueError("Fermi-level targets cannot be used in workfunction mode.")

        if self._linear_mode:
            self._transition_steps = int(transition_steps)
            if self._transition_steps <= 0:
                raise ValueError("transition_steps must be positive.")
            if self.mode == "fermi":
                if (
                    initial_target_ef is None
                    or final_target_ef is None
                    or target_ef is not None
                ):
                    raise ValueError(
                        "A Fermi-level scan requires exactly two endpoints and transition_steps."
                    )
                self._initial_target_ef = _finite(
                    initial_target_ef, "Initial Fermi target"
                )
                self._final_target_ef = _finite(final_target_ef, "Final Fermi target")
                self.target_ef = self._initial_target_ef
            else:
                if (
                    initial_target_workfunction is None
                    or final_target_workfunction is None
                    or target_workfunction is not None
                ):
                    raise ValueError(
                        "A workfunction scan requires exactly two endpoints and transition_steps."
                    )
                self._initial_target_workfunction = _finite(
                    initial_target_workfunction, "Initial workfunction target"
                )
                self._final_target_workfunction = _finite(
                    final_target_workfunction, "Final workfunction target"
                )
                self.target_workfunction = self._initial_target_workfunction
        elif self.mode == "fermi":
            if (
                target_ef is None
                or initial_target_ef is not None
                or final_target_ef is not None
            ):
                raise ValueError("Fermi-level mode requires one constant target.")
            self.target_ef = _finite(target_ef, "Fermi target")
        else:
            if (
                target_workfunction is None
                or initial_target_workfunction is not None
                or final_target_workfunction is not None
            ):
                raise ValueError("Workfunction mode requires one constant target.")
            self.target_workfunction = _finite(
                target_workfunction, "Workfunction target"
            )

        self._conserved_energy = depend_value(
            name="electronic_conserved_energy",
            func=self.get_conserved_energy,
            dependencies=[self._q, self._p, self._protocol_work],
        )

    @property
    def q(self):
        value = self._q.get()
        if not np.isfinite(value) or value < Q_MIN:
            raise ElectronicChargeError(
                f"Electron number must be finite and at least {Q_MIN:.2e}, got {value!r}."
            )
        return value

    @q.setter
    def q(self, value):
        value = float(value)
        if not np.isfinite(value) or value < Q_MIN:
            raise ElectronicChargeError(
                f"Electron number must be finite and at least {Q_MIN:.2e}, got {value!r}."
            )
        self._q.set(value)

    @property
    def p(self):
        return self._p.get()

    @p.setter
    def p(self, value):
        self._p.set(_finite(value, "Electronic momentum"))

    @property
    def neutral_electrons(self):
        if self._neutral_electrons is None:
            raise ValueError("neutral_electrons has not been initialized.")
        return self._neutral_electrons

    @neutral_electrons.setter
    def neutral_electrons(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError("neutral_electrons must be a positive integer.")
        self._neutral_electrons = value
        if hasattr(self, "_conserved_energy"):
            self._conserved_energy.taint()

    @property
    def protocol_work(self):
        return self._protocol_work.get()

    @protocol_work.setter
    def protocol_work(self, value):
        self._protocol_work.set(_finite(value, "Electronic protocol work"))

    @property
    def kinetic_energy(self):
        return 0.5 * self.p**2 / self.mass

    @property
    def target_potential_energy(self):
        """Legendre term whose negative q derivative is the target force."""

        excess = self.q - self.neutral_electrons
        if self.mode == "workfunction":
            return self.target_workfunction * excess
        return -self.target_ef * excess

    @property
    def conserved_energy(self):
        """Electronic contribution excluding heat exchanged by the thermostat."""

        return self._conserved_energy.get()

    def get_conserved_energy(self):
        return self.kinetic_energy + self.target_potential_energy + self.protocol_work

    @property
    def is_linear_mode(self):
        return self._linear_mode

    @property
    def use_workfunction_average(self):
        return (
            self.mode == "workfunction"
            and self.Ne_doping
            and self.workfunction_average_steps > 0
        )

    @property
    def freeze_electronic_dynamics(self):
        return self.use_workfunction_average and (
            self._workfunction_average_total_samples < self.workfunction_average_steps
        )

    @property
    def force(self):
        if self.mode == "workfunction":
            current = (
                self.current_workfunction_steps_average
                if self.use_workfunction_average
                else self.current_workfunction
            )
            return current - self.target_workfunction
        return self.target_ef - self.current_ef

    def update_target_fermi_level(self, step, account_work=True):
        step = int(step)
        if step < 0:
            raise ValueError("Electronic scan step cannot be negative.")
        self.current_step = step
        if not self._linear_mode:
            return
        old_target_energy = self.target_potential_energy
        progress = min(float(step) / float(self._transition_steps), 1.0)
        if self.mode == "fermi":
            self.target_ef = self._initial_target_ef + progress * (
                self._final_target_ef - self._initial_target_ef
            )
        else:
            self.target_workfunction = self._initial_target_workfunction + progress * (
                self._final_target_workfunction - self._initial_target_workfunction
            )
        if account_work:
            # The target scan is an explicitly time-dependent Hamiltonian. Store
            # the opposite of its instantaneous Legendre-term change so that
            # the reported conserved quantity excludes externally applied work.
            self.protocol_work += old_target_energy - self.target_potential_energy
        else:
            self._conserved_energy.taint()

    def reset_workfunction_average(self):
        self._workfunction_average_history = []
        self._workfunction_average_total_samples = 0
        self.current_workfunction_steps_average = self.current_workfunction

    def record_workfunction_sample(self, workfunction):
        value = _finite(workfunction, "Workfunction sample")
        self.current_workfunction = value
        if not self.use_workfunction_average:
            self.current_workfunction_steps_average = value
            return
        self._workfunction_average_history.append(value)
        self._workfunction_average_history = self._workfunction_average_history[
            -self.workfunction_average_steps :
        ]
        self._workfunction_average_total_samples += 1
        self.current_workfunction_steps_average = float(
            np.mean(self._workfunction_average_history)
        )

    def drift(self, dt):
        self.q = self.q + self.p * _finite(dt, "Electronic drift timestep") / self.mass

    def kick(self, dt, force=None):
        if force is None:
            force = self.force
        self.p = self.p + _finite(force, "Electronic force") * _finite(
            dt, "Electronic kick timestep"
        )


class Potentiostat:
    """Base electronic thermostat."""

    def __init__(self, temp=1.0, dt=1.0, tau=100.0, ethermo=0.0):
        self._temp = depend_value(name="electronic_temp", value=temp)
        self._dt = depend_value(name="electronic_dt", value=dt)
        self._tau = depend_value(name="electronic_tau", value=tau)
        self._ethermo = depend_value(
            name="electronic_ethermo",
            value=_finite(ethermo, "Electronic thermostat energy"),
        )
        self.electronic_state = None
        self.prng = None

    def bind(self, electronic_state, prng=None):
        if float(self.tau) <= 0.0 or not np.isfinite(float(self.tau)):
            raise ValueError("Potentiostat tau must be finite and positive.")
        if float(self.temp) < 0.0 or not np.isfinite(float(self.temp)):
            raise ValueError(
                "Potentiostat temperature must be finite and non-negative."
            )
        self.electronic_state = electronic_state
        self.prng = Random() if prng is None else prng

    def half_B(self, dt, fermi_level_eV):
        self.electronic_state.current_ef = (
            _finite(fermi_level_eV, "Fermi level") / Constants.EV_PER_HARTREE
        )
        if not self.electronic_state.freeze_electronic_dynamics:
            self.electronic_state.kick(dt)

    def A(self, dt):
        if not self.electronic_state.freeze_electronic_dynamics:
            self.electronic_state.drift(dt)

    @property
    def temp(self):
        return self._temp.value

    @property
    def dt(self):
        return self._dt.value

    @property
    def tau(self):
        return self._tau.value

    @property
    def ethermo(self):
        return self._ethermo.get()

    @ethermo.setter
    def ethermo(self, value):
        self._ethermo.set(_finite(value, "Electronic thermostat energy"))


class PotentiostatLangevin(Potentiostat):
    """Langevin thermostat for the electronic momentum."""

    def O_step(self, dt):
        if self.electronic_state.freeze_electronic_dynamics:
            return
        dt = _finite(dt, "Potentiostat timestep")
        c = np.exp(-dt / float(self.tau))
        sigma = np.sqrt(
            self.electronic_state.mass * Constants.kb * float(self.temp) * (1.0 - c**2)
        )
        old_kinetic = self.electronic_state.kinetic_energy
        self.electronic_state.p = (
            c * self.electronic_state.p + sigma * self.prng.gvec(1)[0]
        )
        self.ethermo += old_kinetic - self.electronic_state.kinetic_energy

    def step(self, dt=None):
        self.O_step(self.dt if dt is None else dt)


class PotentiostatSVR(Potentiostat):
    """Stochastic velocity rescaling for one electronic degree of freedom."""

    def O_step(self, dt):
        if self.electronic_state.freeze_electronic_dynamics:
            return
        dt = _finite(dt, "Potentiostat timestep")
        c = np.exp(-dt / float(self.tau))
        kinetic = self.electronic_state.kinetic_energy
        target = 0.5 * Constants.kb * float(self.temp)
        random_normal = self.prng.gvec(1)[0]
        if kinetic <= 0.0:
            sigma = np.sqrt(self.electronic_state.mass * Constants.kb * self.temp)
            self.electronic_state.p = sigma * random_normal
            self.ethermo += kinetic - self.electronic_state.kinetic_energy
            return
        ratio = target / kinetic
        alpha2 = (
            c
            + (1.0 - c) * ratio * random_normal**2
            + 2.0 * random_normal * np.sqrt(c * (1.0 - c) * ratio)
        )
        alpha = np.sqrt(max(alpha2, 1.0e-14))
        if np.sqrt(c) + random_normal * np.sqrt((1.0 - c) * ratio) < 0.0:
            alpha = -alpha
        self.electronic_state.p *= alpha
        self.ethermo += kinetic - self.electronic_state.kinetic_energy

    def step(self, dt=None):
        self.O_step(self.dt if dt is None else dt)


dproperties(Potentiostat, ["temp", "dt", "tau", "ethermo"])
