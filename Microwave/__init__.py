# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The Microwave workbench.

Importing this package makes the vendored third-party libraries in ``_vendor/``
importable under their own names, and does nothing else. In particular it does
not import them: see ``_vendor/README.md`` for why that matters - scikit-rf
1.13.0 costs about a second to import, and most sessions never touch it.
"""

import os
import sys

#: The workbench's version, and the only place it is written down. Everything
#: that reports one reads this: ``pyproject.toml`` restates it and a test
#: asserts the two agree, because a package that is copied into FreeCAD's
#: ``Mod/`` is never installed and so has no metadata to be asked for.
#:
#: Deliberately **not** in the openEMS envelope. Its digest is taken over the
#: serialised form, so a producer field there would move every digest on every
#: release and make each stored result read as stale. What the envelope carries
#: is ``SCHEMA_VERSION``, which is the format's version and moves when the
#: format does. A result records this one, as ``adapter_version``, which is
#: where a number describing the producer belongs.
__version__ = "0.0.1"

#: Vendored pure-Python dependencies FreeCAD does not ship. See _vendor/README.md.
VENDOR = os.path.join(os.path.dirname(__file__), "_vendor")


def _extend_path():
    """Put ``_vendor`` on ``sys.path``, after everything already there.

    **Appended, not inserted.** FreeCAD's own ``site-packages`` precedes it, so
    an interpreter that already provides one of these libraries keeps its copy
    and we never hijack the name in the user's own scripts. We are the fallback.

    ``__file__`` is used rather than FreeCAD's resource directories because this
    runs during ``import Microwave``, which is ordinary module scope and has it.
    ``Init.py`` and ``InitGui.py`` do *not* - FreeCAD execs those without it,
    and a bootstrap there would fail silently at startup.
    """
    if VENDOR not in sys.path:
        sys.path.append(VENDOR)


_extend_path()
