# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The Microwave workbench.

Importing this package makes the vendored third-party libraries in ``_vendor/``
importable under their own names, and does nothing else. It does not import
them. See ``_vendor/README.md`` for why that matters: scikit-rf imports pandas
and scipy, and most sessions never touch it.
"""

import os
import sys

#: The workbench's version, and the only place it is written down. Everything
#: that reports a version reads this. ``pyproject.toml`` restates it and a test
#: asserts the two agree: a package that is copied into FreeCAD's ``Mod/`` is
#: never installed, so it has no metadata to be asked for.
#:
#: The openEMS envelope does not carry this version. The envelope's digest is
#: taken over its serialised form, so a producer field there would move every
#: digest on every release and make each stored result read as stale. The
#: envelope carries ``SCHEMA_VERSION`` instead, which is the format's version
#: and moves when the format does. A result records this version, as
#: ``adapter_version``. That is where a number describing the producer belongs.
__version__ = "0.0.2"

#: Vendored pure-Python dependencies FreeCAD does not ship. See _vendor/README.md.
VENDOR = os.path.join(os.path.dirname(__file__), "_vendor")


def _extend_path() -> None:
    """Put ``_vendor`` on ``sys.path``, after everything already there.

    The path is appended rather than inserted. FreeCAD's own ``site-packages``
    then precedes it, so an interpreter that already provides one of these
    libraries keeps its copy, and the vendored copy never takes the name in the
    user's own scripts. ``_vendor`` is the fallback.

    The path comes from ``__file__`` rather than from FreeCAD's resource
    directories. This function runs during ``import Microwave``, which is
    ordinary module scope and has ``__file__``. FreeCAD execs ``Init.py`` and
    ``InitGui.py`` without it, and a bootstrap there would fail silently at
    startup.
    """
    if VENDOR not in sys.path:
        sys.path.append(VENDOR)


_extend_path()
