"""Two-socket integer-electron interpolation for constant-potential MD."""

import time
import threading

import numpy as np

from ipi.engine.constant_potential import (
    electronic_request_context,
    validate_request_result,
    read_constant_potential_feedback,
)
from ipi.engine.forcefields import ForceField, ForceRequest
from ipi.utils.depend import dstrip

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

    def queue(self, atoms, cell, reqid=-1, template=None, request_context=None):
        if self._failure is not None:
            self._failure.reraise()
        snapshot = request_context or electronic_request_context(self)
        # Both endpoint evaluations must use the same geometry as well as q.
        atoms = atoms.clone()
        cell = cell.clone()
        span = self.charge_mixing_span
        q_low = span * int(np.floor(snapshot.nelect / float(span)))
        q_high = q_low + span
        if q_low <= 0:
            raise ValueError(
                "Charge-mixing endpoint electron numbers must be positive."
            )
        fraction = (snapshot.nelect - q_low) / float(span)
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

        # Validate both endpoints before either becomes visible to a worker.
        child_low = self.endpoint_low._build_request(
            atoms,
            cell,
            reqid=reqid,
            request_context=snapshot.at_charge(q_low, mode_flags=0),
        )
        child_high = self.endpoint_high._build_request(
            atoms,
            cell,
            reqid=reqid,
            request_context=snapshot.at_charge(q_high, mode_flags=0),
        )
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
            "lambda": fraction,
            "q_current": snapshot.nelect,
            "q_low": q_low,
            "q_high": q_high,
            "electronic_context": snapshot,
        }
        if template is not None:
            template.update(fields)
            fields = template
        request = ForceRequest(fields)
        try:
            self.endpoint_low._publish_request(child_low)
            self.endpoint_high._publish_request(child_high)
        except BaseException:
            self.endpoint_low.cancel_request(child_low)
            self.endpoint_high.cancel_request(child_high)
            raise
        self._publish_request(request)
        if request["status"] == "Exit":
            self.endpoint_low.cancel_request(child_low)
            self.endpoint_high.cancel_request(child_high)
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
                    RuntimeError(
                        f"Charge-mixing endpoint failed: low={low['status']}, "
                        f"high={high['status']}."
                    ),
                )
                continue
            if low["status"] == "Done" and high["status"] == "Done":
                try:
                    request["result"] = self._mix_results(
                        low["result"], high["result"], request
                    )
                except Exception as exc:
                    self._fail(request, exc)
                    continue
                request["status"] = "Done"
                request["t_finished"] = time.time()
                request._event_done.set()

    def _fail(self, request, exception):
        self.endpoint_low.cancel_request(request.get("child_low"))
        self.endpoint_high.cancel_request(request.get("child_high"))
        self.record_failure(exception, request)
        self._doloop[0] = False
        if threading.current_thread() is self._thread:
            self.notify_failure()

    def release(self, request, lock=True):
        if request is not None:
            for endpoint, key in (
                (self.endpoint_low, "child_low"),
                (self.endpoint_high, "child_high"),
            ):
                child = request.get(key)
                if child is None:
                    continue
                if child["status"] in ("Done", "Exit", "Error"):
                    endpoint.release(child)
                else:
                    endpoint.cancel_request(child)
        super().release(request, lock=lock)

    @staticmethod
    def _endpoint_scalar(extras, key):
        # Complete replies have already passed the shared scalar checks.
        return float(np.asarray(extras[key]).reshape(-1)[0])

    def _mix_results(self, low, high, request):
        validate_request_result(low, request["child_low"])
        validate_request_result(high, request["child_high"])
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
        extras_low = low[3]
        extras_high = high[3]
        nelect_low = self._endpoint_scalar(extras_low, "nelect")
        nelect_high = self._endpoint_scalar(extras_high, "nelect")
        fermi_low = self._endpoint_scalar(extras_low, "fermi_level_eV")
        fermi_high = self._endpoint_scalar(extras_high, "fermi_level_eV")
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
        mode = request["electronic_context"].mode
        if mode == "workfunction":
            wf_low = self._endpoint_scalar(extras_low, "workfunction_eV")
            wf_high = self._endpoint_scalar(extras_high, "workfunction_eV")
            extras["workfunction_eV"] = wlow * wf_low + whigh * wf_high
            extras["endpoint_low_workfunction_eV"] = wf_low
            extras["endpoint_high_workfunction_eV"] = wf_high
        mixed = [
            wlow * energy_low + whigh * energy_high,
            wlow * force_low + whigh * force_high,
            wlow * virial_low + whigh * virial_high,
            extras,
        ]
        validate_request_result(mixed, request)
        return mixed

    def update(self):
        super().update()
        if self._dynamics_ref is None:
            raise RuntimeError("Charge-mixing wrapper is not bound to Dynamics.")
        read_constant_potential_feedback(self._dynamics_ref)
