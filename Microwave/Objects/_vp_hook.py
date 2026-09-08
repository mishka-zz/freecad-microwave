# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where the document layer meets the GUI, without importing it.

Each hook here covers something a document object must not do itself: attaching
a view provider, and redrawing a mesh preview. The redraw reaches a solver
adapter, and a solver-neutral document object that reaches for openEMS breaks
the layering rule in one line.

The GUI layer registers a callable here at import, and the document objects call
whatever is registered. Unset is a supported state. Under ``freecadcmd``, in the
test suite, and on a machine with no GUI at all, nothing registers, nothing
redraws, and nothing breaks.

:class:`ViewProviderRestored` runs the other way. A document restored from a
file written headlessly has no view provider, and the mixin puts one back. It
puts back the property statuses ``Objects/staleness.py`` describes as well, for
the same reason: a file holds what its class declared on the day it was
written.
"""

from collections.abc import Collection

from .staleness import stop_quiet_properties_touching

VIEW_PROVIDER_INJECTOR = None

#: Set by the GUI layer to redraw a mesh preview after a display property
#: changes.
PREVIEW_REDRAW = None


def register_preview_redraw(redraw):
    global PREVIEW_REDRAW
    PREVIEW_REDRAW = redraw


def redraw_preview(obj):
    """Call the registered redraw, if the GUI layer registered one."""
    if PREVIEW_REDRAW:
        PREVIEW_REDRAW(obj)


def register_view_provider_injector(injector):
    global VIEW_PROVIDER_INJECTOR
    VIEW_PROVIDER_INJECTOR = injector


def inject_view_provider(obj, kind):
    """Give a freshly created object its view provider, if the GUI is here."""
    if VIEW_PROVIDER_INJECTOR:
        VIEW_PROVIDER_INJECTOR(obj, kind)


def restore_view_provider(obj, kind):
    """Re-attach a ViewProvider to a restored object that has none.

    Returns True when one was injected, so a caller sweeping a document can
    count what it actually did rather than what it attempted.

    A document written headlessly - by a script, or by ``freecadcmd`` - has
    no ViewObject at all, so opening it in the GUI leaves every object on
    FreeCAD's default view provider: no icon, no ``claimChildren``, and no
    ``doubleClicked``, which is the only thing that opens the simulation panel.
    The document is intact and translates correctly, and only its appearance is
    wrong.

    Objects saved from the GUI already carry their proxy, and are left alone.
    Replacing a live view provider mid-restore would drop whatever display state
    it holds.
    """
    view_object = getattr(obj, "ViewObject", None)
    if view_object is None:
        return False  # console mode: nothing to attach to, and nothing to fix
    # An object that already has a view provider is left alone. Anything else,
    # including whatever FreeCAD puts in an unset Proxy property, is a gap to
    # fill. Testing for "not None" would be a bet on what FreeCAD leaves in an
    # unset property.
    #
    # The test is on the suffix rather than the prefix, so it also matches a
    # provider from another workbench. Such a provider is as good a reason to
    # leave the object alone as one of this workbench's: replacing a live one
    # mid-session drops whatever display state it holds.
    if type(getattr(view_object, "Proxy", None)).__name__.endswith("ViewProvider"):
        return False
    inject_view_provider(obj, kind)
    return True


class ViewProviderRestored:
    """Mixin: what a restored document object here needs putting back.

    Both are repairs rather than behaviour: the view provider, and the status
    that keeps a quiet property off the dependency graph.

    ``onDocumentRestored`` is the hook that fires reliably. Measured on FreeCAD
    1.1: ``openDocument`` never emits the ``slotFinishRestoreDocument`` a
    document observer would hang this on.

    It may still run before the ViewObject exists, in which case
    :func:`restore_view_provider` declines and nothing happens. The workbench
    therefore sweeps open documents again on activation. Both routes are
    idempotent, so whichever gets there first wins and the other is a no-op.
    """

    #: Which of this class's properties move no cell. ``Objects/staleness.py``
    #: says what the declaration is for and which way round it fails. A class
    #: that starts declaring has to call
    #: :meth:`declare_what_moves_no_cell` from its ``__init__`` as well: this
    #: hook only repairs a file written before the declaration existed.
    MOVES_NO_CELL: Collection[str] = ()

    def declare_what_moves_no_cell(self, obj):
        stop_quiet_properties_touching(obj, type(self).MOVES_NO_CELL)

    def onDocumentRestored(self, obj):
        restore_view_provider(obj, type(self).__name__)
        self.declare_what_moves_no_cell(obj)
