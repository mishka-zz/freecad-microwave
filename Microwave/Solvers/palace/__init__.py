# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The Palace adapter: a finite element solver in the frequency domain.

Palace takes a tetrahedral mesh whose pieces are integer attributes, and a JSON
configuration naming what each attribute is. So this adapter draws no grid of
its own. It asks ``Microwave.Gmsh`` for a mesh, and what it writes beside it is
the configuration - which is Palace's own input format and not an envelope of
ours, there being no process boundary here that a private description has to
cross.

Every adapter implements the same set, and the modules below carry those roles:

* :mod:`.capabilities` - what this adapter can express, as pure data readable
  where no solver is installed.
* :mod:`.config` - the configuration, as a value that can be built and written
  without Palace present.
* :mod:`.document` - the front door: a study becomes a :mod:`.problem`, and
  everything the study asks for that this backend cannot do is refused by name.
  :mod:`.policy` holds what the run is as against what the device is - the
  sweep, and what the mesher is asked for.
* :mod:`.write` - the stage that owns a directory: one file per shape, and the
  configuration beside them.
* :mod:`.attributes` - what each label became in the mesh, which is what a
  Palace condition is written against.
* :mod:`.run` - Palace as a subprocess, and the log read for the faults that
  finish cleanly and for what stopped a run that did not.
* :mod:`.read` - the scattering table, as a complex matrix and the port numbers
  its rows and columns stand for.
* :mod:`.pipeline` - those stages in the order they happen, which is what a
  caller holding a study and a directory asks for. It is also where the
  mesher's own refusals become the fault a caller catches to know that the
  document is what is wrong.

The tetrahedral mesher between them belongs to no adapter and is
``Microwave.Solvers.gmsh_meshing``, which runs ``Microwave.Gmsh`` in a process
of its own.

Units
-----

Lengths reach this adapter in millimetres, as they do everywhere in this
workbench, and the mesh is written in them. Palace is told the scale by
``Model.L0``. Frequencies reach it in Hertz and are written in gigahertz, which
is the unit every frequency in a Palace configuration is in.
"""
