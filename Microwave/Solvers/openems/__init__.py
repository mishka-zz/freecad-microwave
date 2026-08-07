# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The openEMS (FDTD) solver adapter.

Everything specific to openEMS lives here and nowhere else: the Yee-grid
mesher, the native input writer, the subprocess runner, and the result reader.
No other adapter imports from this package, and this package imports no other
adapter.

Import discipline
-----------------

The adapter straddles a process boundary. openEMS runs in a *different* Python
than FreeCAD does, so modules here fall into two groups:

* **In-process** (runs inside FreeCAD): capability declaration, pre-flight
  checks, writing the input, reading results back into document objects. May
  import FreeCAD, must never import openEMS or CSXCAD.
* **Subprocess** (runs under the openEMS interpreter): everything that touches
  the solver. May import openEMS and CSXCAD, must never import FreeCAD.

:mod:`.mesh` and :mod:`.model` belong to *neither*. They import nothing but
numpy, so they run on both sides - which is what lets the workbench show a mesh
preview before openEMS is installed, and lets the driver read the same envelope
the workbench wrote.

The pieces
----------

Every adapter implements the same set:

=====================  ==============================================
:mod:`.capabilities`   What this solver can express. Pure data.
:mod:`.preflight`      ``check`` - refuse, warn or substitute.
:mod:`.write`          Meshes, and produces the native input.
:mod:`.run`            Launches the subprocess, follows its progress.
:mod:`.read`           Native output back into result objects.
=====================  ==============================================

:mod:`.driver` is the odd one out: it is the *far side* of :mod:`.run`,
executing under the solver's own interpreter.

Nothing here is imported at package level, on purpose. Importing this package
must stay free of both FreeCAD and openEMS, so that the capability declaration
can be read on a machine with neither.
"""
