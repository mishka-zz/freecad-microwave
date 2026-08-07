# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The one place that imports scikit-rf.

Everything S-parameter-shaped rests on ``skrf.Network``, and every module that
needs it asks here rather than importing it directly. That buys: the import is
deferred until results are actually wanted, the choice between the user's copy
and our vendored one is made once, and provenance can say which copy produced a
number.

Nothing below this line knows about scikit-rf. The adapter, the mesher,
``document.py`` and pre-flight are numpy and stdlib, and a test enforces that.
"""

from __future__ import annotations

import os
import sys

#: Set by :func:`module` on first resolution: "installed" or "vendored".
_source: str | None = None
_module = None


class NoResultLibrary(RuntimeError):
    """scikit-rf could not be imported, from anywhere.

    A silent no-op is never acceptable: without this library the result layer
    cannot produce an answer, so it says so and names what is missing rather
    than handing back an empty network.
    """


def _import_vendored():
    """Import the copy in ``_vendor``, in preference to whatever else is found.

    ``Microwave/__init__.py`` *appends* ``_vendor`` to ``sys.path``, so a plain
    ``import skrf`` finds the interpreter's own copy first - which is what we
    want, right up until that copy is one that does not work. Putting the
    vendored directory in front for the duration of one import is the narrowest
    way to say "no, this one".

    The partially-imported modules from the failed attempt have to go first:
    a failure part-way through leaves ``skrf`` and some of its submodules in
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
        # By value, and deliberately: ``remove`` takes the *earliest* match,
        # which is the entry inserted above - the copy this module appended at
        # startup is at the far end. A positional check (``sys.path[0] is ours``)
        # is wrong, because the import below takes over a second and anything
        # inserting a path meanwhile shifts ours off index zero. The guard would
        # then decline, stranding ``_vendor`` ahead of
        # site-packages for the life of the session: precisely the hijack that
        # appending rather than inserting exists to prevent.
        #
        # The failure this trades into is benign by comparison - if ours has
        # already gone, ``remove`` takes the appended entry and we lose a
        # fallback path that the next ``import Microwave`` puts back.
        try:
            sys.path.remove(VENDOR)
        except ValueError:
            pass


def _whose(found):
    """``"vendored"`` if this module came out of ``_vendor``, else ``"installed"``.

    Asked by looking at where the module actually is, not at which import
    attempt succeeded - ``_vendor`` is on ``sys.path``, so a plain
    ``import skrf`` finds our copy whenever nothing else provides one. Labelling
    that "installed" would put a false claim in every result's provenance.
    """
    from .. import VENDOR

    origin = getattr(found, "__file__", None)
    if not origin:
        # A namespace package, or something synthesised onto sys.meta_path. We
        # cannot tell whose it is, and provenance should not guess.
        return "unknown"
    root = os.path.realpath(VENDOR) + os.sep
    return "vendored" if os.path.realpath(origin).startswith(root) else "installed"


def module():
    """The scikit-rf module, imported on first call.

    Prefers whatever the interpreter already provides and falls back to the
    vendored copy, because "their copy wins" is only the right rule while their
    copy works. scikit-rf 2.0.x - the current release - raises
    ``AttributeError`` on numpy 1.x, which is what the official FreeCAD build
    ships, so a user with their own up-to-date install is a case we expect to
    hit rather than a hypothetical. See ``_vendor/README.md``.

    Any failure of the interpreter's copy sends us to ours: an ImportError
    because it is absent, an AttributeError because it is 2.0.x on numpy 1.x,
    or something we have not seen yet. A failure of *our* copy is not caught -
    that is our bug, and it should be loud.
    """
    global _module, _source
    if _module is not None:
        return _module

    try:
        import skrf

        _module, _source = skrf, _whose(skrf)
    except Exception as theirs:
        try:
            found = _import_vendored()
            # Asked, not assumed. ``sys.meta_path`` outranks ``sys.path``, so an
            # editable install or an import hook can answer this import too, and
            # calling whatever comes back "vendored" would be the same false
            # claim in provenance that _whose exists to prevent.
            _module, _source = found, _whose(found)
        except Exception as ours:
            # Pointing at the two reprs rather than naming libraries. Our copy
            # is always on disk, so a failure of *both* is nearly always
            # something scikit-rf imports rather than scikit-rf itself - and
            # which library that is belongs to the vendored version, not to this
            # message. Telling the user to install scikit-rf would be the one
            # instruction that is actively wrong: they have it, and the copy pip
            # would fetch is the release ``_vendor/README.md`` exists to avoid.
            raise NoResultLibrary(
                "S-parameter results need scikit-rf, and neither copy could be "
                f"loaded. This interpreter's: {theirs!r}. The one bundled with "
                f"the workbench: {ours!r}. A copy travels with the workbench, so "
                "what is missing is usually something scikit-rf itself imports: "
                "install whatever those two errors name into the Python that "
                "runs FreeCAD."
            ) from ours
    return _module


def description():
    """``"scikit-rf 1.13.0 (vendored)"`` - for provenance and the GUI.

    Resolves the library if it has not been resolved yet, so a result
    recording where its maths came from is also what settles which copy
    answers.
    """
    version = getattr(module(), "__version__", "unknown")
    return f"scikit-rf {version} ({_source})"


def source():
    """``"installed"``, ``"vendored"``, or ``None`` before anything asked."""
    return _source
