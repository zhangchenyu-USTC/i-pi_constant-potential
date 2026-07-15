"""Asymmetric double-well test PES with an electronic coordinate.

This is a lightweight example for FFDirect tests. Production socket coupling
does not import or depend on this module.
"""

import json

import numpy as np

from .dummy import Dummy_driver
from ipi.utils.units import unit_to_internal

__DRIVER_NAME__ = "asym_dw_constant_potential"
__DRIVER_CLASS__ = "AsymDwConstantPotential_driver"


class AsymDwConstantPotential_driver(Dummy_driver):
    """E(R,Q) = V(R) + Q*PZC + Q^2/(2*C), with Q=-N_electrons."""

    def __init__(
        self,
        a=1.0,
        b=3.0,
        d=1.0,
        x0=0.0,
        c=-10.0,
        q_default=1.0,
        pzc=5.0,
        cap=10.0,
        *args,
        **kwargs,
    ):
        angstrom = unit_to_internal("length", "angstrom", 1.0)
        electronvolt = unit_to_internal("energy", "electronvolt", 1.0)
        self.a = float(a) * electronvolt / angstrom**4
        self.b = float(b) * electronvolt / angstrom**2
        self.d = float(d) * electronvolt / angstrom
        self.x0 = float(x0) * angstrom
        self.c = float(c) * electronvolt
        self.pzc = float(pzc) * electronvolt
        self.cap = float(cap) / electronvolt
        if self.a <= 0.0 or self.b <= 0.0 or self.cap <= 0.0:
            raise ValueError("a, b, and cap must be positive.")
        self._electronvolt = electronvolt
        self.set_electronic_state(q_default)
        super().__init__(*args, **kwargs)

    def set_electronic_state(self, electron_number):
        value = float(electron_number)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Electron number must be finite and positive.")
        self.electron_number = value

    def __call__(self, cell, pos):
        positions = np.asarray(pos, float).reshape(-1, 3)
        displacement = positions[:, 0] - self.x0
        atomic_energy = np.sum(
            self.a * displacement**4
            - self.b * displacement**2
            + self.d * displacement
            + self.c
        )
        derivative = (
            4.0 * self.a * displacement**3 - 2.0 * self.b * displacement + self.d
        )
        forces = np.zeros_like(positions)
        forces[:, 0] = -derivative

        net_charge = -self.electron_number
        energy = (
            atomic_energy + net_charge * self.pzc + net_charge**2 / (2.0 * self.cap)
        )
        workfunction = self.pzc + net_charge / self.cap
        fermi = -workfunction
        extras = json.dumps(
            {
                "fermi_level_eV": fermi / self._electronvolt,
                "workfunction_eV": workfunction / self._electronvolt,
                "nelect": self.electron_number,
            }
        )
        return energy, forces, np.zeros((3, 3)), extras
