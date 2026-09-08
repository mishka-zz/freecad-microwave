# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The one place that imports scikit-rf.

Everything S-parameter-shaped rests on ``skrf.Network``, and every module that
needs it asks here rather than importing it directly. Routing the import through
this module defers it until results are wanted, settles the choice between the
installed copy and the vendored one once, and lets provenance record which copy
produced a number.

Nothing below this line knows about scikit-rf. The adapter, the mesher, the
translation and pre-flight are numpy and stdlib, and a test enforces that.
"""

from __future__ import annotations

import os
import sys
from typing import Any

#: Set by :func:`module` on first resolution to whatever :func:`_whose`
#: answers: "installed", "vendored", or "unknown".
_source: str | None = None
_module = None


class NoResultLibrary(RuntimeError):
    """scikit-rf could not be imported, from anywhere.

    Without this library the result layer cannot produce an answer. It raises
    this error and names what is missing rather than handing back an empty
    network.
    """


def _import_vendored() -> Any:
    """Import the copy in ``_vendor``, in preference to whatever else is found.

    ``Microwave/__init__.py`` appends ``_vendor`` to ``sys.path``, so a plain
    ``import skrf`` finds the interpreter's own copy first. That order is the
    wanted one until the interpreter's copy stops working. Putting the vendored
    directory in front for the duration of one import is the narrowest way to
    override it.

    The partially-imported modules from the failed attempt are deleted first. A
    failure part-way through leaves ``skrf`` and some of its submodules in
    ``sys.modules``, and the retry would find those instead of starting over.
    """
    from .. import VENDOR

    for name in [name for name in sys.modules if name == "skrf" or name.startswith("skrf.")]:
        del sys.modules[name]

    sys.path.insert(0, VENDOR)
    try:
        import skrf

        return skrf
    finally:
        # Removed by value rather than by position. ``remove`` takes the
        # earliest match, which is the entry inserted above; the one appended at
        # startup is at the far end. A positional check (``sys.path[0] is
        # VENDOR``) is wrong, because importing scikit-rf is slow and
        # anything inserting a path meanwhile shifts the entry off index
        # zero. The guard would then decline, stranding ``_vendor`` ahead
        # of site-packages for the life of the session. That is the hijack that
        # appending rather than inserting exists to prevent.
        #
        # The failure this trades into is milder. If the inserted entry has
        # already gone, ``remove`` takes the appended one, and ``_vendor``
        # leaves ``sys.path`` for the rest of the session: the vendored copy is
        # already imported by then, so what is lost is the fallback for anything
        # else under ``_vendor``.
        try:
            sys.path.remove(VENDOR)
        except ValueError:
            pass


def _whose(found: Any) -> str:
    """``"vendored"`` if this module came out of ``_vendor``, else ``"installed"``.

    This reads where the module actually is rather than which import attempt
    succeeded. ``_vendor`` is on ``sys.path``, so a plain ``import skrf`` finds
    the vendored copy whenever nothing else provides one. Labelling that
    "installed" would put a false claim in every result's provenance.
    """
    from .. import VENDOR

    origin = getattr(found, "__file__", None)
    if not origin:
        # A namespace package, or something synthesised onto sys.meta_path.
        # This function cannot tell where it came from, so it does not guess.
        return "unknown"
    root = os.path.realpath(VENDOR) + os.sep
    return "vendored" if os.path.realpath(origin).startswith(root) else "installed"


def module() -> Any:
    """The scikit-rf module, imported on first call.

    This prefers whatever the interpreter already provides and falls back to the
    vendored copy. The interpreter's copy wins only while it works. scikit-rf
    2.0.x raises ``AttributeError`` on numpy 1.x, which is what the official
    FreeCAD build ships, so a user with an up-to-date install of their own is
    an expected case. See ``_vendor/README.md``.

    Any failure of the interpreter's copy falls through to the vendored one: an
    ImportError because it is absent, an AttributeError because it is 2.0.x on
    numpy 1.x, or something not yet seen. A failure of the vendored copy is not
    caught. That failure is a bug in this workbench, and it should be loud.
    """
    global _module, _source
    if _module is not None:
        return _module

    try:
        import skrf

        _module, _source = skrf, _whose(skrf)
    except Exception as installed:
        try:
            found = _import_vendored()
            # The source is asked for rather than assumed. ``sys.meta_path``
            # outranks ``sys.path``, so an editable install or an import hook
            # can answer this import too, and calling whatever comes back
            # "vendored" would be the false claim in provenance that _whose
            # exists to prevent.
            _module, _source = found, _whose(found)
        except Exception as vendored:
            # The message points at the two reprs rather than naming libraries.
            # The vendored copy is always on disk, so a failure of both is
            # nearly always something scikit-rf imports rather than scikit-rf
            # itself, and which library that is belongs to the vendored version
            # rather than to this message. Telling the user to install scikit-rf
            # would be wrong. They have it, and the copy pip would fetch is the
            # release ``_vendor/README.md`` exists to avoid.
            raise NoResultLibrary(
                "S-parameter results need scikit-rf, and neither copy could be "
                f"loaded. This interpreter's: {installed!r}. The one bundled with "
                f"the workbench: {vendored!r}. A copy travels with the workbench, so "
                "what is missing is usually something scikit-rf itself imports: "
                "install whatever those two errors name into the Python that "
                "runs FreeCAD."
            ) from vendored
    return _module


def description() -> str:
    """``"scikit-rf 1.13.0 (vendored)"``, for provenance and the GUI.

    This resolves the library if it has not been resolved yet. A result that
    records where its maths came from therefore also settles which copy answers.
    """
    version = getattr(module(), "__version__", "unknown")
    return f"scikit-rf {version} ({_source})"


def source() -> str | None:
    """What :func:`_whose` said of the copy in use, or ``None`` before
    anything asked. See it for the third answer."""
    return _source
