"""Strict helpers and a thin forcefield mixin for constant-potential MD."""

import re

import numpy as np

from ipi.utils.units import Constants, unit_to_user


def finite_values(extras, key, context="driver extras"):
    """Return every bead value for a required finite numerical extra."""

    if not isinstance(extras, dict):
        raise RuntimeError(f"{context} must be a dictionary.")
    if key not in extras:
        raise RuntimeError(f"{context} do not contain required key '{key}'.")
    try:
        values = np.asarray(extras[key], dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} key '{key}' is not numerical.") from exc
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError(f"{context} key '{key}' must contain finite values.")
    return values


def mean_extra(extras, key, context="driver extras"):
    """Average a shared electronic observable over all ring-polymer beads."""

    return float(np.mean(finite_values(extras, key, context)))


def append_init_token(pars, key, value):
    """Append one framework-owned INIT token and reject user collisions."""

    text = str(pars or " ").strip()
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(str(key))}\s*:"
    if re.search(pattern, text):
        raise ValueError(
            f"INIT parameter '{key}' is reserved by <electrons>; remove the manual value."
        )
    token = f"{key} : {value}"
    return f"{text} , {token}" if text else token


class ConstantPotentialSocketMixin:
    """Adds constant-potential request metadata to an ordinary FFSocket."""

    def _init_constant_potential(self):
        self.constant_potential_capable = True
        self.charge_enabled = False
        self.mixing_enabled = False
        self.Ne_doping = False
        self.current_nelect = None
        self.neutral_electrons = None
        self.electrons_config = None
        self._dynamics_ref = None
        self._charge_mixing_endpoint = False
        self._potential_average_region_A = None

    def _validate_transport(self):
        mode = str(getattr(self.socket, "mode", "inet"))
        batch_size = int(getattr(self.socket, "batch_size", 1))
        if mode not in ("inet", "unix"):
            raise ValueError(
                "Constant-potential sockets currently support only inet/unix; "
                f"got mode='{mode}'."
            )
        if batch_size != 1:
            raise ValueError(
                "Constant-potential sockets currently require batch_size=1."
            )

    def configure_electrons(self, electrons_config, dynamics=None, endpoint=False):
        if not isinstance(electrons_config, dict) or not electrons_config.get(
            "enabled", False
        ):
            return
        if dynamics is None:
            raise RuntimeError(
                "Constant-potential socket configuration requires an explicit Dynamics reference."
            )
        if self.charge_enabled and self._dynamics_ref is not dynamics:
            raise RuntimeError(
                f"Constant-potential socket '{self.name}' cannot be shared by multiple systems."
            )
        if electrons_config.get("charge_mixing", False) and not endpoint:
            raise RuntimeError(
                "Charge mixing must be configured through FFChargeMixTwoSockets."
            )
        self._validate_transport()
        self.charge_enabled = True
        self.mixing_enabled = False
        self.Ne_doping = bool(electrons_config.get("Ne_doping", False)) and not endpoint
        self.neutral_electrons = int(electrons_config["neutral_electrons"])
        self.electrons_config = electrons_config
        self._dynamics_ref = dynamics
        self._charge_mixing_endpoint = bool(endpoint)

    def set_electronic_state(self, q):
        if not self.charge_enabled:
            return
        value = float(q)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("The electronic coordinate must be finite and positive.")
        self.current_nelect = value

    def _require_dynamics(self):
        if self._dynamics_ref is None:
            raise RuntimeError("Constant-potential socket is not bound to Dynamics.")
        return self._dynamics_ref

    def _mode_flags(self):
        dynamics = self._require_dynamics()
        step = int(
            getattr(
                dynamics,
                "_constant_potential_force_step",
                dynamics.electronic_state.current_step,
            )
        )
        stride = int(self.electrons_config.get("solvation_update_stride", 1))
        flags = 1 if step % stride == 0 else 0
        if self.Ne_doping:
            flags |= 2
        return flags

    def _workfunction_region_A(self):
        if self._potential_average_region_A is not None:
            return self._potential_average_region_A
        region = np.asarray(
            self.electrons_config.get("potential_average_region", []), dtype=float
        ).reshape(-1)
        if (
            region.size != 2
            or not np.all(np.isfinite(region))
            or region[1] <= region[0]
        ):
            raise ValueError("Invalid workfunction potential_average_region.")
        self._potential_average_region_A = tuple(
            float(unit_to_user("length", "angstrom", value)) for value in region
        )
        return self._potential_average_region_A

    def queue(self, atoms, cell, reqid=-1, template=None):
        request = super().queue(atoms, cell, reqid=reqid, template=template)
        if not self.charge_enabled:
            return request
        if self.current_nelect is None:
            raise RuntimeError(
                "A constant-potential request was queued before q was synchronized."
            )
        dynamics = self._require_dynamics()
        state = dynamics.electronic_state
        pars = append_init_token(
            request.get("pars", " "), "NELECT", f"{self.current_nelect:.16g}"
        )
        pars = append_init_token(
            pars, "neutral_electrons", str(int(self.neutral_electrons))
        )
        pars = append_init_token(
            pars,
            "solvation_update_stride",
            str(int(self.electrons_config.get("solvation_update_stride", 1))),
        )
        if state.mode == "workfunction":
            axis = str(self.electrons_config["potential_average_axis"]).strip().lower()
            if axis not in ("x", "y", "z"):
                raise ValueError("Invalid workfunction potential_average_axis.")
            minimum, maximum = self._workfunction_region_A()
            pars = append_init_token(pars, "potential_average_axis", axis)
            pars = append_init_token(
                pars, "potential_average_min_A", f"{minimum:.16g}"
            )
            pars = append_init_token(
                pars, "potential_average_max_A", f"{maximum:.16g}"
            )
        request["pars"] = pars
        request["nelect"] = float(self.current_nelect)
        request["mode_flags"] = int(self._mode_flags())
        return request

    def validate_extras(self, extras, expected_nelect=None, context=None):
        context = context or f"FFSocket '{self.name}' extras"
        dynamics = self._require_dynamics()
        nbeads = int(getattr(getattr(dynamics, "beads", None), "nbeads", 1))
        fermi_values = finite_values(extras, "fermi_level_eV", context)
        returned = finite_values(extras, "nelect", context)
        if fermi_values.size != nbeads or returned.size != nbeads:
            raise RuntimeError(
                f"{context} must contain one scalar fermi_level_eV and nelect per bead."
            )
        fermi = float(np.mean(fermi_values))
        expected = self.current_nelect if expected_nelect is None else expected_nelect
        if expected is None or np.any(np.abs(returned - float(expected)) > 1.0e-8):
            raise RuntimeError(
                f"{context} report electron numbers inconsistent with CHGDATA "
                f"(expected {expected!r}, received {returned.tolist()})."
            )
        state = dynamics.electronic_state
        workfunction = None
        if state.mode == "workfunction":
            values = finite_values(extras, "workfunction_eV", context)
            if values.size != nbeads:
                raise RuntimeError(
                    f"{context} must contain one scalar workfunction_eV per bead."
                )
            workfunction = float(np.mean(values))
        return fermi, workfunction

    def update(self):
        super().update()
        if not self.charge_enabled or self._charge_mixing_endpoint:
            return
        dynamics = self._require_dynamics()
        extras = dynamics.forces.extras
        fermi, workfunction = self.validate_extras(extras)
        dynamics._cache_fermi_level(f"socket:{self.name}", fermi)
        if workfunction is not None:
            extras["workfunction_eV_mean"] = workfunction
            extras["workfunction_mean"] = workfunction / Constants.EV_PER_HARTREE
