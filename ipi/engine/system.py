"""Contains the class that deals with storing the state of a physical system.

Contains code used to hold the information which represents the state of
a system, including the particle positions and momenta, and the
forcefields which govern the interaction potential.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import threading

import numpy as np

from ipi.utils.depend import dpipe
from ipi.utils.messages import verbosity, info
from ipi.engine.forces import Forces
from ipi.engine.properties import Properties, Trajectories

__all__ = ["System"]


class System:
    """Physical system object.

    Contains all the physical information. Also handles stepping and output.

    Attributes:
       beads: A beads object giving the atom positions.
       cell: A cell object giving the system box.
       fcomp: A list of force components that must act on each replica
       forces: A Forces object that actually compute energy and forces
       ensemble: An ensemble object giving the objects necessary for producing
          the correct ensemble.
       outputs: A list of output objects that should be printed during the run
       nm:  A helper object dealing with normal modes transformation
       properties: A property object for dealing with property output.
       trajs: A trajectory object for dealing with trajectory output.
       init: A class to deal with initializing the system.
       simul: The parent simulation object.
    """

    def __init__(
        self, init, beads, nm, cell, fcomponents, ensemble=None, motion=None, prefix=""
    ):
        """Initialises System class.

        Args:
           init: A class to deal with initializing the system.
           beads: A beads object giving the atom positions.
           cell: A cell object giving the system box.
           fcomponents: A list of force components that are active for each
              replica of the system.
           bcomponents: A list of force components that are considered as bias, and act on each
              replica of the system.
           ensemble: An ensemble object giving the objects necessary for
              producing the correct ensemble.
           nm: A class dealing with path NM operations.
           prefix: A string used to differentiate the output files of different
              systems.
        """

        info(" @system: Initializing system object ", verbosity.low)
        self.prefix = prefix
        self.init = init
        self.ensemble = ensemble
        self.motion = motion
        self.beads = beads
        self.cell = cell
        self.nm = nm

        self.fcomp = fcomponents
        self.forces = Forces()

        self.properties = Properties()
        self.trajs = Trajectories()

    def bind(self, simul):
        """Calls the bind routines for all the objects in the system."""

        self.simul = simul  # keeps a handle to the parent simulation object
        self._prepare_charge_mixing_forcefield()

        # binds important computation engines
        info(" @system.bind: Binding the forces ", verbosity.low)
        self.forces.bind(
            self.beads,
            self.cell,
            self.fcomp,
            self.simul.fflist,
            open_paths=self.nm.open_paths,
            output_maker=simul.output_maker,
        )
        self.nm.bind(self.ensemble, self.motion, beads=self.beads, forces=self.forces)
        self.ensemble.bind(
            self.beads,
            self.nm,
            self.cell,
            self.forces,
            self.simul.fflist,
            simul.output_maker,
        )
        self.motion.bind(
            self.ensemble,
            self.beads,
            self.nm,
            self.cell,
            self.forces,
            self.prng,
            simul.output_maker,
        )

        dpipe(self.nm._omegan2, self.forces._omegan2)

        self.init.init_stage2(self)

        # binds output management objects
        self._propertylock = threading.Lock()
        self.properties.bind(self)
        self.trajs.bind(self)

    def _prepare_charge_mixing_forcefield(self):
        """Replace the charge-dependent component with a two-socket wrapper."""

        config = getattr(self.motion, "electrons_config", None)
        if (
            not isinstance(config, dict)
            or not config.get("enabled", False)
            or not config.get("charge_mixing", False)
        ):
            return

        from ipi.engine.forcefields import FFSocket
        from ipi.engine.forcefields_cp import FFChargeMixTwoSockets

        fflist = self.simul.fflist
        configured_low = str(config.get("charge_mixing_low", "")).strip()
        configured_high = str(config.get("charge_mixing_high", "")).strip()
        socket_names = [
            name
            for name, forcefield in fflist.items()
            if isinstance(forcefield, FFSocket)
        ]
        if configured_low:
            endpoint_names = [configured_low, configured_high]
            missing = [name for name in endpoint_names if name not in fflist]
            if missing:
                raise ValueError(f"Undefined charge-mixing endpoints: {missing}.")
        else:
            if len(socket_names) != 2:
                raise ValueError(
                    "Without explicit endpoint names, charge mixing requires exactly two ffsockets."
                )
            referenced = [fc.ffield for fc in self.fcomp if fc.ffield in socket_names]
            if len(referenced) != 1:
                raise ValueError(
                    "Exactly one system force component must reference a charge-mixing socket."
                )
            endpoint_names = [
                referenced[0],
                next(name for name in socket_names if name != referenced[0]),
            ]

        low, high = (fflist[name] for name in endpoint_names)
        if not isinstance(low, FFSocket) or not isinstance(high, FFSocket):
            raise TypeError(
                "Both charge-mixing endpoints must be ffsocket forcefields."
            )
        if low.dopbc != high.dopbc or not np.array_equal(low.active, high.active):
            raise ValueError(
                "Charge-mixing endpoints must use identical dopbc and active settings."
            )

        candidates = [fc for fc in self.fcomp if fc.ffield in set(endpoint_names)]
        if len(candidates) != 1:
            raise ValueError(
                "Exactly one force component may reference the charge-mixing endpoints; "
                "other MTS components must use independent forcefields."
            )
        component = candidates[0]
        if component.nbeads not in (0, self.beads.nbeads):
            raise ValueError(
                "The constant-potential force component cannot use ring-polymer contraction."
            )

        label = self.prefix or component.name or "system"
        wrapper_name = (
            f"__charge_mixing_{label}_{endpoint_names[0]}_{endpoint_names[1]}"
        )
        if wrapper_name in fflist:
            raise ValueError(
                f"Charge-mixing wrapper name collision for '{wrapper_name}'; use unique system prefixes."
            )
        wrapper = FFChargeMixTwoSockets(
            low,
            high,
            latency=min(low.latency, high.latency),
            name=wrapper_name,
            dopbc=low.dopbc,
            active=low.active,
            threaded=True,
        )
        fflist[wrapper_name] = wrapper
        component._charge_mixing_original_ffield = component.ffield
        component._charge_mixing_secondary_ffield = (
            endpoint_names[1]
            if component.ffield == endpoint_names[0]
            else endpoint_names[0]
        )
        component.ffield = wrapper_name
