# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where the document layer meets the GUI, without importing it.

Every hook here exists because a document object needs something done that it
must not know how to do itself: attaching a view provider, redrawing a mesh
preview, deciding whether a preview still describes its document. The last two
mean reading the model through a *solver adapter*, and a solver-neutral document
object that reaches for openEMS breaks the layering rule in one line.

So the GUI layer registers a callable here at import, and the document objects
call whatever is registered. **Unset is a supported state**, not a degraded one:
under ``freecadcmd``, in the test suite, and on a machine with no GUI at all,
nothing registers, nothing redraws, and nothing breaks.

:class:`ViewProviderRestored` is the other direction - a document restored from
a file written headlessly has no view provider, and the mixin puts one back.
"""

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


#: Set by the GUI layer: decides whether a preview still describes its document.
PREVIEW_STATUS = None


def register_preview_status(check):
    global PREVIEW_STATUS
    PREVIEW_STATUS = check


def preview_status(obj):
    """The registered verdict, or ``None`` when nothing registered one."""
    if PREVIEW_STATUS:
        return PREVIEW_STATUS(obj)
    return None


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
    The document is intact and translates fine; it just looks broken.

    Objects saved from the GUI already carry their proxy, and are left alone -
    replacing a live view provider mid-restore would drop whatever display state
    it holds.
    """
    view_object = getattr(obj, "ViewObject", None)
    if view_object is None:
        return False  # console mode: nothing to attach to, and nothing to fix
    # A view provider already? Leave it. Anything else - including whatever
    # FreeCAD puts in an unset Proxy property - is a gap to fill. Testing for
    # "not None" would be a bet on what FreeCAD leaves in an unset property.
    #
    # By suffix, not by prefix. The question is "is there a provider here at
    # all?", and a provider from another workbench is just as much a reason to
    # leave the object alone as this workbench's - replacing a live one mid-session
    # drops whatever display state it holds.
    if type(getattr(view_object, "Proxy", None)).__name__.endswith("ViewProvider"):
        return False
    inject_view_provider(obj, kind)
    return True


class ViewProviderRestored:
    """Mixin: give a restored object its view provider back.

    ``onDocumentRestored`` is the hook that reliably fires - measured, on
    FreeCAD 1.1: the ``slotFinishRestoreDocument`` a document observer would
    hang this on is never emitted by ``openDocument`` at all.

    It may nonetheless run before the ViewObject exists, in which case
    :func:`restore_view_provider` declines and nothing happens. That is why the
    workbench sweeps open documents again on activation; both are idempotent, so
    whichever gets there first wins and the other is a no-op.
    """

    def onDocumentRestored(self, obj):
        restore_view_provider(obj, type(self).__name__)
