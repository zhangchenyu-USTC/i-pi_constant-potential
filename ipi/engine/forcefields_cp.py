"""Two-socket integer-electron interpolation for constant-potential MD."""

import time

import numpy as np

from ipi.engine.constant_potential import finite_values, mean_extra
from ipi.engine.forcefields import ForceField, ForceRequest
from ipi.utils.depend import dstrip
from ipi.utils.softexit import softexit
from ipi.utils.units import Constants

__all__ = ["FFChargeMixTwoSockets"]


class FFChargeMixTwoSockets(ForceField):
    """Linearly interpolate two socket results at adjacent integer endpoints."""

    def __init__(
        self,
        endpoint_low,
        endpoint_high,
        latency=1.0e-4,
        offset=0.0,
        name="",
        pars=None,
        dopbc=True,
        active=np.array([-1]),
        threaded=True,
    ):
        super().__init__(latency, offset, name, pars, dopbc, active, threaded)
        self.endpoint_low = endpoint_low
        self.endpoint_high = endpoint_high
        self._is_charge_mixing_wrapper = True
        self._dynamics_ref = None
        self.electrons_config = None
        self.charge_enabled = True
        self.constant_potential_capable = True
        self.current_nelect = None
        self.neutral_electrons = None
        self.charge_mixing_span = 1

    def configure_electrons(self, electrons_config, dynamics=None):
        if not isinstance(electrons_config, dict) or not electrons_config.get(
            "enabled", False
        ):
            return
        if not electrons_config.get("charge_mixing", False):
            raise RuntimeError(
                "The charge-mixing wrapper requires charge_mixing='true'."
            )
        if dynamics is None:
            raise RuntimeError(
                "The charge-mixing wrapper requires an explicit Dynamics reference."
            )
        self.electrons_config = electrons_config
        self._dynamics_ref = dynamics
        self.charge_mixing_span = int(electrons_config["charge_mixing_span"])
        if self.charge_mixing_span not in (1, 2):
            raise ValueError("charge_mixing_span must be 1 or 2.")
        self.neutral_electrons = int(electrons_config["neutral_electrons"])
        for endpoint in (self.endpoint_low, self.endpoint_high):
            endpoint.configure_electrons(electrons_config, dynamics, endpoint=True)

    def set_electronic_state(self, q):
        value = float(q)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "Charge mixing requires a finite positive electron number."
            )
        span = self.charge_mixing_span
        low = span * int(np.floor(value / float(span)))
        high = low + span
        if low <= 0 or high <= low:
            raise ValueError(f"Invalid charge-mixing endpoints {low} and {high}.")
        fraction = (value - low) / float(span)
        if fraction < -1.0e-10 or fraction > 1.0 + 1.0e-10:
            raise ValueError("Charge-mixing interpolation fraction is outside [0, 1].")
        self.current_nelect = value
        self.q_low = low
        self.q_high = high
        self.lambda_val = min(max(fraction, 0.0), 1.0)
        self.endpoint_low.set_electronic_state(float(low))
        self.endpoint_high.set_electronic_state(float(high))

    def queue(self, atoms, cell, reqid=-1, template=None):
        if self.current_nelect is None:
            raise RuntimeError("Charge mixing was queued before q was synchronized.")
        pbcpos = dstrip(atoms.q).copy()
        if self.iactive is None:
            if self.active[0] == -1:
                self.iactive = np.arange(len(pbcpos))
            else:
                self.iactive = np.asarray(
                    [[3 * atom + i for i in range(3)] for atom in self.active]
                ).reshape(-1)
            if self.iactive.size > pbcpos.size or np.any(self.iactive >= pbcpos.size):
                raise ValueError("Charge-mixing active atom indices are out of range.")
        if self.dopbc:
            cell.array_pbc(pbcpos)

        child_low = self.endpoint_low.queue(atoms, cell, reqid=reqid)
        child_high = self.endpoint_high.queue(atoms, cell, reqid=reqid)
        fields = {
            "id": reqid,
            "pos": pbcpos,
            "active": self.iactive,
            "cell": (dstrip(cell.h).copy(), dstrip(cell.ih).copy()),
            "pars": " ",
            "result": None,
            "status": "Dispatched",
            "start": -1,
            "t_queued": time.time(),
            "t_dispatched": time.time(),
            "t_finished": 0,
            "child_low": child_low,
            "child_high": child_high,
            "lambda": self.lambda_val,
            "q_current": self.current_nelect,
            "q_low": self.q_low,
            "q_high": self.q_high,
        }
        if template is not None:
            template.update(fields)
            fields = template
        request = ForceRequest(fields)
        with self._threadlock:
            self.requests.append(request)
        return request

    def poll(self):
        with self._threadlock:
            requests = list(self.requests)
        for request in requests:
            if request["status"] not in ("Queued", "Dispatched"):
                continue
            low = request["child_low"]
            high = request["child_high"]
            if low["status"] in ("Exit", "Error") or high["status"] in (
                "Exit",
                "Error",
            ):
                self._fail(
                    request,
                    f"Charge-mixing endpoint failed: low={low['status']}, "
                    f"high={high['status']}.",
                )
                continue
            if low["status"] == "Done" and high["status"] == "Done":
                try:
                    request["result"] = self._mix_results(
                        low["result"], high["result"], request
                    )
                except Exception as exc:
                    self._fail(request, str(exc))
                    continue
                request["status"] = "Done"
                request["t_finished"] = time.time()
                request._event_done.set()

    @staticmethod
    def _fail(request, message):
        request["status"] = "Exit"
        request["error"] = message
        request["t_finished"] = time.time()
        request._event_done.set()
        softexit.trigger(status="bad", message=f"Charge mixing failed: {message}")

    def release(self, request, lock=True):
        if request is not None:
            self.endpoint_low.release(request.get("child_low"))
            self.endpoint_high.release(request.get("child_high"))
        super().release(request, lock=lock)

    @staticmethod
    def _endpoint_scalar(extras, key, label):
        values = finite_values(extras, key, f"{label} endpoint extras")
        if values.size != 1:
            raise RuntimeError(f"{label} endpoint result contains non-scalar '{key}'.")
        return float(values[0])

    def _mix_results(self, low, high, request):
        if low is None or high is None or len(low) < 4 or len(high) < 4:
            raise RuntimeError("Charge-mixing endpoint returned an incomplete result.")
        fraction = float(request["lambda"])
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise RuntimeError("Charge-mixing interpolation fraction is invalid.")
        wlow = 1.0 - fraction
        whigh = fraction
        energy_low = float(low[0])
        energy_high = float(high[0])
        force_low = np.asarray(low[1], float)
        force_high = np.asarray(high[1], float)
        virial_low = np.asarray(low[2], float)
        virial_high = np.asarray(high[2], float)
        if (
            not np.isfinite(energy_low)
            or not np.isfinite(energy_high)
            or force_low.ndim != 1
            or force_low.shape != force_high.shape
            or (
                "active" in request
                and force_low.size != np.asarray(request["active"]).size
            )
            or virial_low.shape != (3, 3)
            or virial_low.shape != virial_high.shape
            or not np.all(np.isfinite(force_low))
            or not np.all(np.isfinite(force_high))
            or not np.all(np.isfinite(virial_low))
            or not np.all(np.isfinite(virial_high))
        ):
            raise RuntimeError(
                "Charge-mixing endpoint energy, force, or virial is invalid."
            )

        extras_low = low[3]
        extras_high = high[3]
        nelect_low = self._endpoint_scalar(extras_low, "nelect", "low")
        nelect_high = self._endpoint_scalar(extras_high, "nelect", "high")
        if (
            abs(nelect_low - request["q_low"]) > 1.0e-8
            or abs(nelect_high - request["q_high"]) > 1.0e-8
        ):
            raise RuntimeError(
                "Charge-mixing endpoint electron number disagrees with CHGDATA."
            )
        fermi_low = self._endpoint_scalar(extras_low, "fermi_level_eV", "low")
        fermi_high = self._endpoint_scalar(extras_high, "fermi_level_eV", "high")
        extras = {
            "raw": "",
            "fermi_level_eV": wlow * fermi_low + whigh * fermi_high,
            "nelect": wlow * nelect_low + whigh * nelect_high,
            "charge_mixing_lambda": fraction,
            "charge_mixing_nelect_low": float(request["q_low"]),
            "charge_mixing_nelect_high": float(request["q_high"]),
            "endpoint_low_fermi_level_eV": fermi_low,
            "endpoint_high_fermi_level_eV": fermi_high,
            "endpoint_low_energy": energy_low,
            "endpoint_high_energy": energy_high,
        }
        state = self._dynamics_ref.electronic_state
        if state.mode == "workfunction":
            wf_low = self._endpoint_scalar(extras_low, "workfunction_eV", "low")
            wf_high = self._endpoint_scalar(extras_high, "workfunction_eV", "high")
            extras["workfunction_eV"] = wlow * wf_low + whigh * wf_high
            extras["endpoint_low_workfunction_eV"] = wf_low
            extras["endpoint_high_workfunction_eV"] = wf_high
        mixed = [
            wlow * energy_low + whigh * energy_high,
            wlow * force_low + whigh * force_high,
            wlow * virial_low + whigh * virial_high,
            extras,
        ]
        if (
            not np.isfinite(mixed[0])
            or not np.all(np.isfinite(mixed[1]))
            or not np.all(np.isfinite(mixed[2]))
        ):
            raise RuntimeError("Charge-mixing produced non-finite force data.")
        return mixed

    def update(self):
        super().update()
        if self._dynamics_ref is None:
            raise RuntimeError("Charge-mixing wrapper is not bound to Dynamics.")
        extras = self._dynamics_ref.forces.extras
        returned = finite_values(extras, "nelect", "mixed force extras")
        if np.any(np.abs(returned - self.current_nelect) > 1.0e-8):
            raise RuntimeError("Mixed electron number is inconsistent with q.")
        fermi = mean_extra(extras, "fermi_level_eV", "mixed force extras")
        self._dynamics_ref._cache_fermi_level("charge_mixing", fermi)
        if self._dynamics_ref.electronic_state.mode == "workfunction":
            workfunction = mean_extra(extras, "workfunction_eV", "mixed force extras")
            extras["workfunction_eV_mean"] = workfunction
            extras["workfunction_mean"] = workfunction / Constants.EV_PER_HARTREE
