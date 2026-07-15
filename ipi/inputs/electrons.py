"""Input and restart state for constant-potential electronic dynamics."""

import math

import numpy as np

from ipi.engine.potentiostat import (
    ElectronicState,
    PotentiostatLangevin,
    PotentiostatSVR,
)
from ipi.utils.inputvalue import (
    Input,
    InputArray,
    InputAttribute,
    InputValue,
    input_default,
)

__all__ = ["InputElectrons", "InputPotentiostat"]


class InputPotentiostat(Input):
    attribs = {
        "mode": (
            InputAttribute,
            {
                "dtype": str,
                "default": "langevin",
                "options": ["langevin", "svr"],
                "help": "Electronic thermostat: Langevin or stochastic velocity rescaling.",
            },
        )
    }
    fields = {
        "tau": (
            InputValue,
            {
                "dtype": float,
                "default": 100.0,
                "dimension": "time",
                "help": "Electronic thermostat relaxation time.",
            },
        ),
        "temp": (
            InputValue,
            {
                "dtype": float,
                "default": float("nan"),
                "dimension": "temperature",
                "help": "Electronic temperature; defaults to the physical system temperature.",
            },
        ),
        "ethermo": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "energy",
                "help": "Restartable energy exchanged with the electronic thermostat.",
            },
        ),
    }
    dynamic = {}
    default_help = "Thermostat for the electronic momentum."
    default_label = "POTENTIOSTAT"

    def store(self, value):
        if value is None:
            return
        if isinstance(value, InputPotentiostat):
            return
        super().store()
        if isinstance(value, dict):
            self.mode.store(value.get("mode", "langevin"))
            self.tau.store(value.get("tau", 100.0))
            self.temp.store(value.get("temp", float("nan")))
            self.ethermo.store(value.get("ethermo", 0.0))
            return
        self.mode.store("svr" if isinstance(value, PotentiostatSVR) else "langevin")
        self.tau.store(value.tau)
        self.temp.store(getattr(value, "_input_temp", value.temp))
        self.ethermo.store(value.ethermo)

    def fetch(self):
        super().fetch()
        tau = float(self.tau.fetch())
        temp = float(self.temp.fetch())
        ethermo = float(self.ethermo.fetch())
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("Potentiostat tau must be finite and positive.")
        if not math.isnan(temp) and (not np.isfinite(temp) or temp < 0.0):
            raise ValueError(
                "Potentiostat temperature must be non-negative or omitted."
            )
        mode = self.mode.fetch()
        result = (
            PotentiostatLangevin(tau=tau, ethermo=ethermo)
            if mode == "langevin"
            else PotentiostatSVR(tau=tau, ethermo=ethermo)
        )
        result._input_temp = temp
        return result


class InputElectrons(Input):
    """The ``<electrons>`` block nested inside ``<dynamics>``."""

    attribs = {
        "enabled": (
            InputAttribute,
            {"dtype": bool, "default": False, "help": "Enable electronic dynamics."},
        ),
        "charge_mixing": (
            InputAttribute,
            {
                "dtype": bool,
                "default": False,
                "help": "Mix two integer-electron socket endpoints.",
            },
        ),
        "Ne_doping": (
            InputAttribute,
            {
                "dtype": bool,
                "default": False,
                "help": "Use the Ne-electrode mapping in a compatible driver.",
            },
        ),
    }

    fields = {
        "target_fermi_level": (
            InputValue,
            {"dtype": float, "default": float("nan"), "dimension": "energy"},
        ),
        "initial_target_fermi_level": (
            InputValue,
            {"dtype": float, "default": float("nan"), "dimension": "energy"},
        ),
        "final_target_fermi_level": (
            InputValue,
            {"dtype": float, "default": float("nan"), "dimension": "energy"},
        ),
        "target_workfunction": (
            InputValue,
            {"dtype": float, "default": float("nan"), "dimension": "energy"},
        ),
        "initial_target_workfunction": (
            InputValue,
            {"dtype": float, "default": float("nan"), "dimension": "energy"},
        ),
        "final_target_workfunction": (
            InputValue,
            {"dtype": float, "default": float("nan"), "dimension": "energy"},
        ),
        "transition_steps": (InputValue, {"dtype": int, "default": -1}),
        "q_init": (InputValue, {"dtype": float, "default": 1.0}),
        "p_init": (InputValue, {"dtype": float, "default": 0.0}),
        "neutral_electrons": (InputValue, {"dtype": int, "default": -1}),
        "mass": (
            InputValue,
            {"dtype": float, "default": 1.0, "dimension": "mass"},
        ),
        "potential_average_axis": (
            InputValue,
            {
                "dtype": str,
                "default": "",
                "help": "Lattice axis (x, y, or z) normal to the planar-potential averaging region.",
            },
        ),
        "potential_average_region": (
            InputArray,
            {
                "dtype": float,
                "default": np.array([], float),
                "dimension": "length",
            },
        ),
        "potentiostat": (
            InputPotentiostat,
            {"default": input_default(factory=InputPotentiostat)},
        ),
        "solvation_update_stride": (InputValue, {"dtype": int, "default": 1}),
        "charge_mixing_span": (InputValue, {"dtype": int, "default": 1}),
        "charge_mixing_low": (
            InputValue,
            {
                "dtype": str,
                "default": "",
                "help": "Optional name of the lower integer-electron ffsocket endpoint.",
            },
        ),
        "charge_mixing_high": (
            InputValue,
            {
                "dtype": str,
                "default": "",
                "help": "Optional name of the upper integer-electron ffsocket endpoint.",
            },
        ),
        "workfunction_average_steps": (InputValue, {"dtype": int, "default": 50}),
        # Complete restart state. Compatibility with 3.1.x restart files is not required.
        "runtime_step": (InputValue, {"dtype": int, "default": 0}),
        "current_fermi_level": (
            InputValue,
            {"dtype": float, "default": 0.0, "dimension": "energy"},
        ),
        "current_workfunction": (
            InputValue,
            {"dtype": float, "default": 0.0, "dimension": "energy"},
        ),
        "workfunction_average": (
            InputValue,
            {"dtype": float, "default": 0.0, "dimension": "energy"},
        ),
        "workfunction_history": (
            InputArray,
            {
                "dtype": float,
                "default": np.array([], float),
                "dimension": "energy",
            },
        ),
        "workfunction_total_samples": (InputValue, {"dtype": int, "default": 0}),
        "protocol_work": (
            InputValue,
            {"dtype": float, "default": 0.0, "dimension": "energy"},
        ),
    }
    dynamic = {}
    default_help = "Electronic coordinate and potentiostat for constant-potential MD."
    default_label = "ELECTRONS"

    _simple_fields = (
        "target_fermi_level",
        "initial_target_fermi_level",
        "final_target_fermi_level",
        "target_workfunction",
        "initial_target_workfunction",
        "final_target_workfunction",
        "transition_steps",
        "q_init",
        "p_init",
        "neutral_electrons",
        "mass",
        "potential_average_axis",
        "solvation_update_stride",
        "charge_mixing_span",
        "charge_mixing_low",
        "charge_mixing_high",
        "workfunction_average_steps",
        "runtime_step",
        "current_fermi_level",
        "current_workfunction",
        "workfunction_average",
        "workfunction_total_samples",
        "protocol_work",
    )

    def store(self, value):
        if value is None:
            return
        if isinstance(value, InputElectrons):
            return
        if not isinstance(value, dict):
            raise TypeError("Electronic configuration must be a dictionary.")
        super().store()
        self.enabled.store(value.get("enabled", False))
        self.charge_mixing.store(value.get("charge_mixing", False))
        self.Ne_doping.store(value.get("Ne_doping", False))
        for name in self._simple_fields:
            if name in value:
                getattr(self, name).store(value[name])
        if "potential_average_region" in value:
            self.potential_average_region.store(
                np.asarray(value["potential_average_region"], float)
            )
        if "workfunction_history" in value:
            self.workfunction_history.store(
                np.asarray(value["workfunction_history"], float)
            )
        if "potentiostat" in value:
            self.potentiostat.store(value["potentiostat"])
        if "potentiostat_temp" in value:
            self.potentiostat.temp.store(value["potentiostat_temp"])

    def store_runtime_state(self, dynamics):
        config = getattr(dynamics, "electrons_config", None)
        state = getattr(dynamics, "electronic_state", None)
        if not isinstance(config, dict) or state is None:
            return
        self.store(config)
        self.q_init.store(state.q)
        self.p_init.store(state.p)
        self.runtime_step.store(state.current_step)
        self.current_fermi_level.store(state.current_ef)
        self.current_workfunction.store(state.current_workfunction)
        self.workfunction_average.store(state.current_workfunction_steps_average)
        self.workfunction_history.store(
            np.asarray(state._workfunction_average_history, float)
        )
        self.workfunction_total_samples.store(state._workfunction_average_total_samples)
        self.protocol_work.store(state.protocol_work)

    def fetch(self):
        super().fetch()
        config = {
            "enabled": self.enabled.fetch(),
            "charge_mixing": self.charge_mixing.fetch(),
            "Ne_doping": self.Ne_doping.fetch(),
            "potential_average_region": np.asarray(
                self.potential_average_region.fetch(), float
            ).reshape(-1),
            "workfunction_history": np.asarray(
                self.workfunction_history.fetch(), float
            ).reshape(-1),
            "potentiostat": self.potentiostat.fetch(),
            "potentiostat_temp": self.potentiostat.temp.fetch(),
        }
        for name in self._simple_fields:
            config[name] = getattr(self, name).fetch()
        if config["enabled"]:
            self._validate_enabled(config)
        return config

    @staticmethod
    def _validate_enabled(config):
        for name in ("q_init", "p_init", "mass"):
            if not np.isfinite(float(config[name])):
                raise ValueError(f"{name} must be finite.")
        if config["q_init"] <= 0.0 or config["mass"] <= 0.0:
            raise ValueError("q_init and electronic mass must be positive.")
        if config["neutral_electrons"] <= 0:
            raise ValueError("neutral_electrons must be a positive integer.")
        if config["charge_mixing_span"] not in (1, 2):
            raise ValueError("charge_mixing_span must be 1 or 2.")
        if config["solvation_update_stride"] <= 0:
            raise ValueError("solvation_update_stride must be a positive integer.")
        if config["workfunction_average_steps"] <= 0:
            raise ValueError("workfunction_average_steps must be positive.")
        if config["charge_mixing"] and config["Ne_doping"]:
            raise ValueError("charge_mixing and Ne_doping are mutually exclusive.")
        low = str(config["charge_mixing_low"]).strip()
        high = str(config["charge_mixing_high"]).strip()
        if bool(low) != bool(high):
            raise ValueError(
                "charge_mixing_low and charge_mixing_high must be set together."
            )
        if low and not config["charge_mixing"]:
            raise ValueError("Explicit endpoints require charge_mixing='true'.")
        if low and low == high:
            raise ValueError("Charge-mixing endpoint names must be distinct.")
        config["charge_mixing_low"] = low
        config["charge_mixing_high"] = high

        target_names = (
            "target_fermi_level",
            "target_workfunction",
            "initial_target_fermi_level",
            "final_target_fermi_level",
            "initial_target_workfunction",
            "final_target_workfunction",
        )
        present = {name: not math.isnan(float(config[name])) for name in target_names}
        if any(present.values()):
            for name in target_names:
                if present[name] and not np.isfinite(float(config[name])):
                    raise ValueError(f"{name} must be finite.")

        fermi_pair = (
            present["initial_target_fermi_level"]
            and present["final_target_fermi_level"]
        )
        workfunction_pair = (
            present["initial_target_workfunction"]
            and present["final_target_workfunction"]
        )
        if present["initial_target_fermi_level"] != present["final_target_fermi_level"]:
            raise ValueError("Both Fermi scan endpoints are required.")
        if (
            present["initial_target_workfunction"]
            != present["final_target_workfunction"]
        ):
            raise ValueError("Both workfunction scan endpoints are required.")
        if (fermi_pair or workfunction_pair) != (config["transition_steps"] > 0):
            raise ValueError(
                "Linear endpoints and positive transition_steps must be used together."
            )

        selected = sum(
            (
                present["target_fermi_level"],
                present["target_workfunction"],
                fermi_pair,
                workfunction_pair,
            )
        )
        if selected != 1:
            raise ValueError(
                "Exactly one constant or linear electronic target is required."
            )
        config["mode"] = (
            "workfunction"
            if present["target_workfunction"] or workfunction_pair
            else "fermi"
        )
        if config["Ne_doping"] and config["mode"] != "workfunction":
            raise ValueError("Ne_doping requires workfunction control.")
        if config["mode"] == "workfunction":
            axis = str(config["potential_average_axis"]).strip().lower()
            if axis not in ("x", "y", "z"):
                raise ValueError(
                    "Workfunction control requires potential_average_axis=x, y, or z."
                )
            config["potential_average_axis"] = axis
            region = config["potential_average_region"]
            if (
                region.size != 2
                or not np.all(np.isfinite(region))
                or region[1] <= region[0]
            ):
                raise ValueError(
                    "Workfunction control requires finite "
                    "potential_average_region=[min,max]."
                )

    def create_electronic_state(self, config=None):
        if config is None:
            config = self.fetch()
        if not config["enabled"]:
            return None
        linear = config["transition_steps"] > 0
        kwargs = {
            "q_init": config["q_init"],
            "p_init": config["p_init"],
            "mass": config["mass"],
            "mode": config["mode"],
        }
        if config["mode"] == "fermi":
            if linear:
                kwargs.update(
                    initial_target_ef=config["initial_target_fermi_level"],
                    final_target_ef=config["final_target_fermi_level"],
                    transition_steps=config["transition_steps"],
                )
            else:
                kwargs["target_ef"] = config["target_fermi_level"]
        elif linear:
            kwargs.update(
                initial_target_workfunction=config["initial_target_workfunction"],
                final_target_workfunction=config["final_target_workfunction"],
                transition_steps=config["transition_steps"],
            )
        else:
            kwargs["target_workfunction"] = config["target_workfunction"]
        state = ElectronicState(**kwargs)
        state.neutral_electrons = int(config["neutral_electrons"])
        state.Ne_doping = bool(config["Ne_doping"])
        state.workfunction_average_steps = (
            int(config["workfunction_average_steps"])
            if state.Ne_doping and state.mode == "workfunction"
            else 0
        )

        step = int(config["runtime_step"])
        samples = int(config["workfunction_total_samples"])
        history = np.asarray(config["workfunction_history"], float).reshape(-1)
        values = np.asarray(
            [
                config["current_fermi_level"],
                config["current_workfunction"],
                config["workfunction_average"],
            ],
            float,
        )
        if (
            step < 0
            or samples < 0
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(history))
        ):
            raise ValueError("Electronic restart state contains invalid values.")
        if state.use_workfunction_average:
            expected = min(samples, state.workfunction_average_steps)
            if history.size != expected:
                raise ValueError(
                    "Electronic restart workfunction history has the wrong length."
                )
            if history.size and not np.isclose(
                values[2], np.mean(history), rtol=1.0e-12, atol=1.0e-14
            ):
                raise ValueError(
                    "Electronic restart workfunction average is inconsistent."
                )
        elif samples or history.size:
            raise ValueError(
                "Workfunction history is only valid in Ne workfunction mode."
            )

        state.current_step = step
        if state.is_linear_mode:
            state.update_target_fermi_level(step, account_work=False)
        state.protocol_work = float(config["protocol_work"])
        state.current_ef = float(values[0])
        state.current_workfunction = float(values[1])
        state.current_workfunction_steps_average = float(values[2])
        state._workfunction_average_history = history.tolist()
        state._workfunction_average_total_samples = samples
        return state
