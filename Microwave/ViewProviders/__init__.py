# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How the workbench's objects look in the tree and the 3D view.

Nothing here decides anything about the model. A view provider owns an icon and
a display mode. The objects that carry real geometry get the icon and no display
mode: defining a display mode on a ``Part::FeaturePython`` makes its geometry
disappear, as :class:`Provider` describes.

No provider keeps state. Everything a provider would want is on the document
object or derived from it, so ``__getstate__`` returns ``None`` and FreeCAD never
writes a pickled GUI class into a saved document.

This package is the only one that may import ``FreeCADGui``. It does not import
it at module scope either, because ``Commands`` and the document objects both
reach it for the icon path.
"""

import os

import FreeCAD

#: Where the icon set lives. Every provider and command asks here. A missing
#: file gives "", so a stale name gives a plain icon rather than a traceback
#: during document restore.
RESOURCES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Resources")


def icon(name):
    """The path to an icon in the set, or ``""`` if it is not there."""
    path = os.path.join(RESOURCES, name)
    return path if os.path.exists(path) else ""


class Provider:
    """An icon and no state to keep, which every view provider here shares.

    This class stays separate from :class:`HasDisplayMode` because the objects
    that carry real geometry - a port and the mesh preview, both
    ``Part::FeaturePython`` - must not have display modes. FreeCAD derives their
    provider from Part's own, which already knows how to draw a ``Shape``.
    Defining ``getDisplayModes`` or ``setDisplayMode`` takes that over and the
    geometry stops appearing. Those objects take this class, and the rest take
    the subclass.
    """

    #: Icon file in ``Resources``. Each subclass sets it, and there is no
    #: default: a default would give a new kind another kind's picture, and the
    #: tree is where a user tells one port from another at a glance.
    ICON = ""

    def __init__(self, vobj):
        vobj.Proxy = self

    def getIcon(self):
        return icon(self.ICON)

    # FreeCAD serialises a view provider through these. Every provider's state
    # is on the document object or derived from it, so there is nothing to
    # keep. Saying so here stops FreeCAD writing a pickle of a GUI class into
    # the document.
    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class HasDisplayMode(Provider):
    """Give a non-geometric object a display mode, so it is not born hidden.

    On FreeCAD 1.1.1 an ``App::FeaturePython`` whose view provider offers no
    display mode reports ``Visibility`` as ``True``, is still drawn greyed out
    in the tree, and does not respond to Space. ``isShow()`` needs a mode to
    show, so without this the whole markup reads as disabled. FreeCAD's own FEM
    analysis carries a mode called ``Analysis`` for the same reason.

    The mode draws nothing. It is an empty ``SoGroup``. These objects have no
    geometry, so they need the right to be shown rather than something to draw,
    and that right is what makes hiding one meaningful.
    """

    #: One name for all of them. Nothing shows it: the display-mode combo
    #: appears only for objects with more than one mode, and these have one.
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
    importable outside FreeCAD, which is where pivy ships. A test enforces it.
    A view provider that cannot build its mode costs only appearance. One that
    raises during document restore costs the object its icon and its
    double-click.
    """
    try:
        from pivy import coin

        vobj.addDisplayMode(coin.SoGroup(), name)
    except Exception as error:  # pragma: no cover - needs a real GUI
        FreeCAD.Console.PrintLog(
            f"Microwave: no display mode for {getattr(vobj, 'Object', '?')}: {error}\n"
        )


#: Document kind to the module its view provider lives in. The class is always
#: ``<kind>ViewProvider``, so the name needs no column of its own.
#:
#: This is a table rather than a branch per kind, because the branches are
#: mechanically identical. Adding a document object and forgetting its provider
#: is invisible in a ladder of branches, and shows here as a missing row.
#: ``test_every_document_kind_has_a_view_provider`` reads it
#: against :func:`Objects.kinds.kinds`, which derives the kinds from the classes.
_PROVIDER_MODULES = {
    "EMAnalysis": "analysis",
    "EMSolverOpenEMS": "solver",
    "EMMeshPolicy": "mesh",
    "EMMeshRegion": "mesh",
    "EMMaterial": "materials",
    "EMMaterialBinding": "materials",
    "EMPortCoaxial": "ports",
    "EMPortLumped": "ports",
    "EMPortMicrostrip": "ports",
    "EMPortRectWaveguide": "ports",
    "EMMeshPreview": "preview",
    "EMSParameters": "results",
}


def provider_class(kind):
    """The view provider class for a document kind, or ``None`` where there is none.

    The module is imported on demand. These modules reach for ``PySide`` and
    ``pivy``, so they must not be imported when nothing is drawing.
    """
    module = _PROVIDER_MODULES.get(kind)
    if module is None:
        return None
    import importlib

    return getattr(importlib.import_module(f".{module}", __package__), f"{kind}ViewProvider")


def inject_vp(obj, kind):
    """Attach the view provider for ``kind``, if this workbench has one.

    Called from ``Microwave.Objects`` when an object is created, through the
    injector hook it registers. A document object must not import the GUI.
    """
    provider = provider_class(kind)
    if provider is not None:
        provider(obj.ViewObject)

    # Assigning Proxy to a restored ViewObject does not make FreeCAD 1.1.1 call
    # attach(). On a freshly created one it does. The object's restore state
    # decides that, not whether the ViewObject already exists. Every provider
    # above sets self.Object in attach and nowhere else, so without this call
    # the restore path injects a provider that never ran attach, and anything
    # reading self.Object does nothing.
    #
    # The call is guarded on self.Object rather than made unconditionally.
    # That attribute records whether attach has run, and attach is not
    # idempotent everywhere: ports.py rebuilds its coin subgraph and calls
    # addDisplayMode again, orphaning an SoSeparator in the mode switch, and
    # materials re-runs a whole-document colour walk.
    proxy = getattr(getattr(obj, "ViewObject", None), "Proxy", None)
    attach = getattr(proxy, "attach", None)
    if attach is not None and getattr(proxy, "Object", None) is None:
        attach(obj.ViewObject)


def restore_view_providers(doc):
    """Give every object in one document its ViewProvider back. Returns the count.

    This sweeps a whole document.
    :func:`~..Objects._vp_hook.restore_view_provider` does one object and
    reports what a headlessly written document is missing. This function skips
    objects that already carry a view provider, so it only fills gaps and can
    be run as often as needed.
    """
    from ..Objects._vp_hook import restore_view_provider
    from ..Objects.kinds import kind_of, kinds

    restored = 0
    for obj in doc.Objects:
        kind = kind_of(obj)
        if kind not in kinds():
            continue
        try:
            # The lower module holds the one test for whether an object
            # already has a provider.
            if not restore_view_provider(obj, kind):
                continue
            restored += 1
            # The tree was built during restore, before this provider existed,
            # so it still shows FreeCAD's default icon and does not ask again
            # on its own. signalChangeIcon is the cheapest nudge, and on 1.1.1
            # it leaves the document unmodified. touch() plus recompute() is
            # not guaranteed to do that. Whether the tree redraws cannot be
            # checked without Qt.
            view_object = getattr(obj, "ViewObject", None)
            signal = getattr(view_object, "signalChangeIcon", None)
            if signal is not None:
                signal()
        except Exception as error:
            # The guard is per object rather than per document. One object
            # that cannot be given a view provider must not cost the rest of
            # the tree theirs. An abort here leaves the simulation with no
            # doubleClicked, and that is the only route to the panel.
            FreeCAD.Console.PrintWarning(
                f"Microwave: no view provider for {getattr(obj, 'Name', '?')!r}: {error}\n"
            )
    return restored


def restore_open_documents():
    """Sweep every open document. Called when the workbench is activated.

    The per-object ``onDocumentRestored`` hook does the work for documents opened
    while this workbench is loaded, but it can run before the ViewObject exists,
    and a document may well have been opened before the user ever switched here.
    Sweeping on activation covers both cases, and costs one pass over the object
    tree.
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
    """Sweep a document when it becomes active, however it was opened.

    ``restore_open_documents`` runs on workbench activation, which covers
    opening FreeCAD, opening a file and switching to Microwave. A second
    document opened from inside the workbench gets no sweep from it.

    This watcher restores view providers and nothing else. It must not touch
    visibility. It runs on every switch between documents, so forcing geometry
    visible here would override a deliberate "hide this" every time the user
    comes back. A document generated by a script carries its own visibility;
    see ``examples/microstrip_50ohm.py``.

    ``slotActivateDocument`` is the hook. Observing every slot while
    ``openDocument`` runs, on FreeCAD 1.1, gives:

        slotCreatedDocument, slotBeforeChangeDocument, slotChangedDocument,
        slotRelabelDocument, slotCreatedObject, slotBeforeChangeObject,
        slotChangedObject, slotAppendDynamicProperty, slotActivateDocument

    ``slotFinishRestoreDocument`` is not in that list. FreeCAD never emits it.
    ``slotActivateDocument`` comes last, after every object exists, and it also
    fires on every switch between open documents. That costs one pass over the
    object tree, because both sweeps only fill gaps.
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
