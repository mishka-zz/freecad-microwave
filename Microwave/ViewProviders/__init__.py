# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How the workbench's objects look in the tree and the 3D view.

Nothing here decides anything about the model. A view provider owns an icon, a
display mode and, for the two objects that carry real geometry, deliberately
neither - see :class:`Provider` for why defining a display mode on a
``Part::FeaturePython`` makes its geometry disappear.

**No provider keeps state.** Everything a provider would want is on the document
object or derived from it, so ``__getstate__`` returns ``None`` and FreeCAD never
writes a pickled GUI class into a saved document.

This package is the only one that may import ``FreeCADGui`` - and it does not
import it at module scope either, because ``Commands`` and the document objects
both reach it for the icon path.
"""

import os

import FreeCAD

#: Where the icon set lives. One definition; every provider and command asks
#: here, and a missing file gives "" so a stale name is a plain icon rather
#: than a traceback during document restore.
RESOURCES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Resources")


def icon(name):
    """The path to an icon in the set, or ``""`` if it is not there."""
    path = os.path.join(RESOURCES, name)
    return path if os.path.exists(path) else ""


class Provider:
    """What every view provider here shares: an icon, and no state to keep.

    Separate from :class:`HasDisplayMode` rather than merged into it, because the
    two objects that carry real geometry - a port and the mesh preview, both
    ``Part::FeaturePython`` - must *not* have display modes. FreeCAD derives
    their provider from Part's own, which already knows how to draw a ``Shape``;
    defining ``getDisplayModes`` or ``setDisplayMode`` takes that over and the
    geometry stops appearing. So they take this and the rest take the subclass.
    """

    #: Icon file in ``Resources``. Set per subclass, with no default on purpose:
    #: a default would silently give a new kind somebody else's picture, and the
    #: tree is where a user tells one port from another at a glance.
    ICON = ""

    def __init__(self, vobj):
        vobj.Proxy = self

    def getIcon(self):
        return icon(self.ICON)

    # FreeCAD serialises a view provider through these. Nothing here is worth
    # keeping - every provider's state is either on the document object or
    # derived from it - and saying so explicitly is what stops FreeCAD writing
    # a pickle of a GUI class into the document.
    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class HasDisplayMode(Provider):
    """Give a non-geometric object a display mode, so it is not born hidden.

    On FreeCAD 1.1.1 an ``App::FeaturePython`` whose view provider offers no
    display mode reports ``Visibility`` as ``True`` and is still drawn greyed
    out in the tree, and Space does nothing - ``isShow()`` needs a mode to
    show, so without this the whole markup reads as disabled. FreeCAD's own FEM
    analysis carries a mode called ``Analysis`` for exactly this reason.

    The mode draws nothing: an empty ``SoGroup``. These objects have no
    geometry - what they need is not something to draw but the *right* to be
    shown, which is what makes hiding one meaningful.
    """

    #: One name for all of them. It is never shown: the display-mode combo only
    #: appears for objects with more than one, and these have exactly one.
    DISPLAY_MODE = "Default"

    def attach(self, vobj):
        self.Object = vobj.Object
        add_display_mode(vobj, self.DISPLAY_MODE)

    def getDisplayModes(self, vobj):
        return [self.DISPLAY_MODE]

    def getDefaultDisplayMode(self):
        return self.DISPLAY_MODE

    def setDisplayMode(self, mode):
        return mode


def add_display_mode(vobj, name="Default"):
    """Attach an empty display mode. Never raises.

    ``pivy`` is imported here rather than at module scope so these modules stay
    importable without a GUI, which a test enforces. A view provider that cannot
    build its mode is a cosmetic loss; one that raises during document restore
    costs the object its icon and its double-click.
    """
    try:
        from pivy import coin

        vobj.addDisplayMode(coin.SoGroup(), name)
    except Exception as error:  # pragma: no cover - needs a real GUI
        FreeCAD.Console.PrintLog(
            f"Microwave: no display mode for {getattr(vobj, 'Object', '?')}: {error}\n"
        )


#: Document kind to the module its view provider lives in. The class is always
#: ``<kind>ViewProvider``, so the name does not need a column of its own.
#:
#: A table rather than the twelve-branch ladder it replaces, because the branches
#: were mechanically identical and the thing that actually breaks - adding a
#: document object and forgetting its provider - is invisible in a ladder and a
#: missing row here. ``test_every_document_kind_has_a_view_provider`` reads it
#: against :func:`Objects.kinds.kinds`, which derives the kinds from the classes.
_PROVIDER_MODULES = {
    "EMAnalysis": "analysis",
    "EMSolverOpenEMS": "solver",
    "EMMeshPolicy": "mesh",
    "EMMeshRegion": "mesh",
    "EMMaterial": "materials",
    "EMMaterialBinding": "materials",
    "EMPortLumped": "ports",
    "EMPortMicrostrip": "ports",
    "EMPortRectWaveguide": "ports",
    "EMMeshPreview": "preview",
    "EMSParameters": "results",
}


def provider_class(kind):
    """The view provider class for a document kind, or ``None`` if we own none.

    Imported on demand, as the ladder did: these modules reach for ``PySide`` and
    ``pivy`` and must not be imported when nobody is drawing anything.
    """
    module = _PROVIDER_MODULES.get(kind)
    if module is None:
        return None
    import importlib

    return getattr(importlib.import_module(f".{module}", __package__), f"{kind}ViewProvider")


def inject_vp(obj, kind):
    """Attach the view provider for ``kind``, if this workbench has one.

    Called from ``Microwave.Objects`` when an object is created, through the
    injector hook it registers - a document object must not import the GUI.
    """
    provider = provider_class(kind)
    if provider is not None:
        provider(obj.ViewObject)

    # Assigning Proxy to a *restored* ViewObject does not make FreeCAD 1.1.1
    # call attach(); on a freshly created one it does. The distinction is the
    # object's restore state, not whether the ViewObject already exists. Every
    # provider above sets self.Object in attach and nowhere else, so without
    # this call the restore path injects a provider without it and anything
    # reading self.Object silently does nothing.
    #
    # Guarded on self.Object rather than called unconditionally: that attribute
    # is exactly "has attach run?", and attach is *not* idempotent everywhere.
    # ports.py rebuilds its coin subgraph and calls addDisplayMode again,
    # orphaning an SoSeparator in the mode switch; materials re-runs a
    # whole-document colour walk.
    proxy = getattr(getattr(obj, "ViewObject", None), "Proxy", None)
    attach = getattr(proxy, "attach", None)
    if attach is not None and getattr(proxy, "Object", None) is None:
        attach(obj.ViewObject)


def restore_view_providers(doc):
    """Give every object in one document its ViewProvider back. Returns the count.

    A document written headlessly - by a script, or by ``freecadcmd`` - was
    never given a ViewObject, so opening it in the GUI leaves each object on
    FreeCAD's default view provider: no icon, no display mode, and no
    ``doubleClicked``, which is the only thing that opens the simulation panel.
    The document is intact and translates fine. It just looks broken, and the
    one action that would prove otherwise is the action that is missing.

    Objects that already carry a view provider are skipped, so this only ever
    fills gaps and can be run as often as it likes.
    """
    from ..Objects._vp_hook import restore_view_provider
    from ..Objects.kinds import kind_of, kinds

    restored = 0
    for obj in doc.Objects:
        kind = kind_of(obj)
        if kind not in kinds():
            continue
        try:
            # One implementation of "does this already have a provider?", in
            # the lower module.
            if not restore_view_provider(obj, kind):
                continue
            restored += 1
            # The tree was built during restore, before this provider existed,
            # so it is still showing FreeCAD's default icon and does not ask
            # again on its own. signalChangeIcon is the cheapest nudge and, on
            # 1.1.1, leaves the document unmodified - which touch() plus
            # recompute() is not guaranteed to do. Whether the tree redraws
            # cannot be checked without Qt.
            view_object = getattr(obj, "ViewObject", None)
            signal = getattr(view_object, "signalChangeIcon", None)
            if signal is not None:
                signal()
        except Exception as error:
            # Per object, not per document. One object that cannot be given a
            # view provider must not cost the rest of the tree theirs - an
            # abort here leaves the simulation with no doubleClicked, which is
            # the only route to the panel.
            FreeCAD.Console.PrintWarning(
                f"Microwave: no view provider for {getattr(obj, 'Name', '?')!r}: {error}\n"
            )
    return restored


def restore_open_documents():
    """Sweep every open document. Called when the workbench is activated.

    The per-object ``onDocumentRestored`` hook does the work for documents opened
    while this workbench is loaded, but it can run before the ViewObject exists,
    and a document may well have been opened before the user ever switched here.
    Sweeping on activation covers both, and costs one pass over the object tree.
    """
    total = 0
    for doc in FreeCAD.listDocuments().values():
        try:
            total += restore_view_providers(doc)
        except Exception as error:  # a display fix must never break a document
            FreeCAD.Console.PrintWarning(
                f"Microwave: could not restore view providers on {doc.Name!r}: {error}\n"
            )
    return total


class _DocumentWatcher:
    """Sweep a document when it becomes active, however it got opened.

    ``restore_open_documents`` runs on workbench activation, which covers
    "open FreeCAD, open a file, switch to Microwave" and nothing else - a
    second document opened from inside the workbench got no sweep at all.

    It restores view providers and nothing else. **It must not touch
    visibility.** This runs on every switch between documents, so forcing
    geometry visible here would override a deliberate "hide this" every time
    the user comes back. A document generated by a script carries its own
    visibility; see ``examples/microstrip_50ohm.py``.

    ``slotActivateDocument`` is the hook. Observing every slot while
    ``openDocument`` runs, on FreeCAD 1.1, gives:

        slotCreatedDocument, slotBeforeChangeDocument, slotChangedDocument,
        slotRelabelDocument, slotCreatedObject, slotBeforeChangeObject,
        slotChangedObject, slotAppendDynamicProperty, slotActivateDocument

    ``slotFinishRestoreDocument``, the obvious candidate, is not in that list:
    it is never emitted. ``slotActivateDocument`` comes last, after every
    object exists, and also fires on every switch between open documents. That
    costs one pass over the object tree, because both sweeps only fill gaps.
    """

    def slotActivateDocument(self, doc):
        try:
            restore_view_providers(doc)
        except Exception as error:  # never break a document over cosmetics
            FreeCAD.Console.PrintWarning(
                f"Microwave: could not sweep {getattr(doc, 'Name', '?')!r}: {error}\n"
            )


#: One instance, kept alive here. FreeCAD stores observers by reference and does
#: not own them, so a local would be collected and the slots would stop firing
#: with no error anywhere.
_WATCHER = None


def install_document_observer():
    """Idempotent. Called from workbench initialisation."""
    global _WATCHER
    if _WATCHER is not None:
        return _WATCHER
    _WATCHER = _DocumentWatcher()
    FreeCAD.addDocumentObserver(_WATCHER)
    return _WATCHER
