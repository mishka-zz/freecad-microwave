# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The openEMS (FDTD) solver adapter.

Everything specific to openEMS lives here and nowhere else: the Yee-grid
mesher, the native input writer, the subprocess runner, and the result reader.
No other adapter imports from this package, and this package imports no other
adapter.

Import discipline
-----------------

The adapter straddles a process boundary. openEMS runs in a different Python
from FreeCAD's, so the modules here are grouped by which of them they run in.

* **In-process**, inside FreeCAD: capability declaration, pre-flight checks,
  planning the grid, writing the input, reading results back into document
  objects. These must never import openEMS or CSXCAD. They do not import
  FreeCAD either: a document object is an attribute bag, and :mod:`.document`
  reads one by duck typing, which is what keeps the whole translation testable
  with neither a kernel nor a solver.
* **Subprocess**, under the openEMS interpreter: everything that touches the
  solver - :mod:`.driver` and the port class it builds out of openEMS' own.
  These may import openEMS and CSXCAD, and must never import FreeCAD.

``tests/import_sweep.py`` imports every module in this package that is not
solver-side into its own interpreter, and CI runs it where neither FreeCAD nor
the engine is installed.

:mod:`.mesh` and :mod:`.model` are held to both rules at once, importing
nothing outside this package but numpy and the standard library, so that they
run on either side. The workbench can therefore show a mesh preview before
openEMS is installed, and the driver can read the envelope the workbench
wrote.

The pieces
----------

Every adapter implements the same set:

=====================  ==============================================
:mod:`.capabilities`   What this solver can express. Pure data.
:mod:`.preflight`      ``check`` - refuse, warn or substitute.
:mod:`.plan`           Where the grid spans, and what it resolves.
:mod:`.write`          The native input, on disk and read back.
:mod:`.run`            Launches the subprocess, follows its progress.
:mod:`.read`           Native output back into result objects.
=====================  ==============================================

:mod:`.driver` is the exception. It is the far side of :mod:`.run`, executing
under the solver's own interpreter.

Nothing here is imported at package level. Importing this package has to stay
free of both FreeCAD and openEMS, so that the capability declaration can be read
on a machine with neither.
"""
