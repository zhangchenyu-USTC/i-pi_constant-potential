"""Strict helpers and a thin forcefield mixin for constant-potential MD."""

import re
from dataclasses import dataclass, replace

import numpy as np

from ipi.utils.units import unit_to_user


@dataclass(frozen=True)
class ElectronicRequestContext:
    """Electronic inputs owned by one evaluation, never by the dispatch time."""

    nelect: float
    mode_flags: int
    mode: str
    owner_id: int
    bead_id: int = -1
    purpose: str = "production"
    neutral_electrons: int = 0
    potential_region: tuple = ()

    def at_charge(self, nelect, *, mode_flags=None):
        return replace(
            self,
            nelect=float(nelect),
            mode_flags=(self.mode_flags if mode_flags is None else int(mode_flags)),
        )


def electronic_request_context(
    forcefield, dynamics=None, q=None, bead_id=-1, purpose="production"
):
    """Capture every electronic input before publishing any part of a request."""
    dynamics = dynamics if dynamics is not None else forcefield._dynamics_ref
    if dynamics is None:
        raise RuntimeError("Electronic request requires an owning Dynamics.")
    state = dynamics.electronic_state
    if q is None:
        q = forcefield.current_nelect
    q = float(q) if q is not None else float("nan")
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError(
            "Electronic request requires a synchronized finite positive q."
        )
    config = getattr(forcefield, "electrons_config", None) or {}
    region = ()
    if state.mode == "workfunction":
        axis = str(config.get("potential_average_axis", "")).strip().lower()
        bounds = np.asarray(config.get("potential_average_region", []), float).reshape(
            -1
        )
        if (
            axis not in ("x", "y", "z")
            or bounds.size != 2
            or not np.all(np.isfinite(bounds))
            or bounds[1] <= bounds[0]
        ):
            raise ValueError("Invalid workfunction potential average axis/region.")
        region = (axis, *(float(unit_to_user("length", "angstrom", x)) for x in bounds))
    flags = (
        2
        if config.get("Ne_doping", False)
        and not getattr(forcefield, "_charge_mixing_endpoint", False)
        else 0
    )
    return ElectronicRequestContext(
        q,
        flags,
        state.mode,
        id(dynamics),
        bead_id,
        purpose,
        int(config.get("neutral_electrons", 0)),
        region,
    )


def validate_constant_potential_weights(dynamics):
    component = dynamics._electronic_provider_component
    weights = np.asarray(component.mts_weights, dtype=float)
    weight = float(component.weight)
    if (
        not np.isfinite(weight)
        or not np.all(np.isfinite(weights))
        or not np.isclose(weight * float(weights.sum()), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(
            "Constant-potential provider requires weight * sum(mts_weights) = 1."
        )
    hweights = np.asarray(getattr(dynamics.ensemble, "hweights", []), dtype=float)
    index = dynamics._electronic_provider_index
    if hweights.size and (
        index >= hweights.size
        or not np.isfinite(hweights[index])
        or not np.isclose(hweights[index], 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("Constant-potential provider requires Hamiltonian weight 1.")


def bind_constant_potential_provider(dynamics, component, provider):
    dynamics._electronic_provider_component = component
    dynamics._electronic_provider = provider
    dynamics._electronic_provider_index = next(
        i
        for i, candidate in enumerate(dynamics.forces.mforces)
        if candidate is component
    )
    component._electronic_owner = dynamics
    for bead in component._forces:
        bead.bind_electronic_context(dynamics)
    validate_constant_potential_weights(dynamics)


def read_constant_potential_feedback(dynamics):
    """Read the designated provider only; never advance electronic state/history."""
    validate_constant_potential_weights(dynamics)
    extras = dynamics._electronic_provider_component.extras
    context = "constant-potential provider extras"
    nelect = finite_values(extras, "nelect", context)
    fermi = finite_values(extras, "fermi_level_eV", context)
    nbeads = dynamics.beads.nbeads
    if nelect.size != nbeads or fermi.size != nbeads:
        raise RuntimeError(
            "Constant-potential provider needs one scalar nelect and Fermi level per bead."
        )
    if np.any(np.abs(nelect - dynamics.electronic_state.q) > 1.0e-8):
        raise RuntimeError(
            "Constant-potential provider electron numbers disagree with q."
        )
    workfunction = None
    if dynamics.electronic_state.mode == "workfunction":
        values = finite_values(extras, "workfunction_eV", context)
        if values.size != nbeads:
            raise RuntimeError(
                "Constant-potential provider needs one scalar workfunction per bead."
            )
        workfunction = float(values.mean())
    return float(fermi.mean()), workfunction


def validate_electronic_request_result(extras, snapshot):
    """Validate one reply against the request, not a subsequently advanced q."""
    returned = finite_values(extras, "nelect", "electronic request result")
    fermi = finite_values(extras, "fermi_level_eV", "electronic request result")
    if (
        returned.size != 1
        or fermi.size != 1
        or abs(returned[0] - snapshot.nelect) > 1.0e-8
    ):
        raise RuntimeError("Electronic reply disagrees with its request snapshot.")
    if snapshot.mode == "workfunction":
        if (
            finite_values(extras, "workfunction_eV", "electronic request result").size
            != 1
        ):
            raise RuntimeError("Electronic reply requires one scalar workfunction.")


def validate_request_result(result, request):
    """Check a complete CP reply in the full, single-bead coordinate space."""
    snapshot = request.get("electronic_context")
    if snapshot is None:
        return
    if result is None or len(result) < 4:
        raise RuntimeError("Constant-potential request returned an incomplete result.")
    energy = np.asarray(result[0], dtype=float)
    force = np.asarray(result[1], dtype=float)
    virial = np.asarray(result[2], dtype=float)
    if (
        energy.shape != ()
        or not np.isfinite(energy)
        or force.shape != (np.asarray(request["pos"]).size,)
        or not np.all(np.isfinite(force))
        or virial.shape != (3, 3)
        or not np.all(np.isfinite(virial))
    ):
        raise RuntimeError(
            "Constant-potential reply energy, force, or virial has invalid shape "
            "or non-finite values."
        )
    validate_electronic_request_result(result[3], snapshot)


def validate_constant_potential_ownership(simulation):
    """Reject incompatible backend consumers before opening any client socket."""
    consumers = {}

    def backends(forcefield):
        yield forcefield
        if getattr(forcefield, "_is_charge_mixing_wrapper", False):
            yield from backends(forcefield.endpoint_low)
            yield from backends(forcefield.endpoint_high)

    for system in simulation.syslist:
        for role, forces in (("forces", system.forces), ("bias", system.ensemble.bias)):
            for component in forces.mforces:
                # ScaledForceComponent reuses a main component, not a backend request.
                if not hasattr(component, "_forces"):
                    continue
                for backend in backends(component.ff):
                    consumers.setdefault(id(backend), (backend, []))[1].append(
                        (
                            system,
                            role,
                            component,
                            getattr(component, "_electronic_owner", None),
                        )
                    )
    for backend, uses in consumers.values():
        if not getattr(backend, "charge_enabled", False):
            continue
        owner = getattr(backend, "_dynamics_ref", None)
        if (
            any(
                role != "forces" or bound_owner is not owner
                for _, role, _, bound_owner in uses
            )
            or len({id(s) for s, _, _, _ in uses}) != 1
            or len({id(component) for _, _, component, _ in uses}) != 1
        ):
            raise ValueError(
                f"Constant-potential backend '{backend.name}' cannot be shared with another system or independent force/bias consumer."
            )


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
        if (
            isinstance(electrons_config, dict)
            and "solvation_update_stride" in electrons_config
        ):
            raise ValueError("solvation_update_stride has been removed from i-PI.")
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
        return 2 if self.Ne_doping else 0

    def _prepare_request(self, request, request_context):
        request = super()._prepare_request(request, request_context)
        if not self.charge_enabled:
            return request
        snapshot = request_context or electronic_request_context(self)
        if snapshot.owner_id != id(self._require_dynamics()):
            raise RuntimeError("Electronic request belongs to a different Dynamics.")
        if (
            not np.isfinite(snapshot.nelect)
            or snapshot.nelect <= 0
            or snapshot.mode_flags not in (0, 2)
        ):
            raise ValueError("Invalid electronic request snapshot.")
        if snapshot.neutral_electrons <= 0:
            raise ValueError("Electronic request requires positive neutral_electrons.")
        if re.search(
            r"(?<![A-Za-z0-9_])solvation_update_stride\s*:", request.get("pars", " ")
        ):
            raise ValueError("INIT parameter solvation_update_stride has been removed.")
        pars = append_init_token(
            request.get("pars", " "), "NELECT", f"{snapshot.nelect:.16g}"
        )
        pars = append_init_token(
            pars, "neutral_electrons", str(snapshot.neutral_electrons)
        )
        if snapshot.mode == "workfunction":
            axis, minimum, maximum = snapshot.potential_region
            pars = append_init_token(pars, "potential_average_axis", axis)
            pars = append_init_token(pars, "potential_average_min_A", f"{minimum:.16g}")
            pars = append_init_token(pars, "potential_average_max_A", f"{maximum:.16g}")
        request["pars"] = pars
        request["electronic_context"] = snapshot
        request["nelect"] = snapshot.nelect
        request["mode_flags"] = snapshot.mode_flags
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
        read_constant_potential_feedback(self._require_dynamics())
