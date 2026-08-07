# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which document objects this workbench owns.

Asked when deciding what a refinement region may reference, which objects need
a view provider restored, and which the tree may claim.

Answered by looking, not by spelling. Testing the proxy class name for a prefix
makes renaming a class silently change which objects the workbench recognises,
and ``EM`` is one letter away from matching anything. The set is derived from
the classes themselves, which is the one form that cannot drift out of step with
them: add a document object and it is recognised, with nothing to remember.
"""

import functools


@functools.cache
def kinds():
    """Every proxy class name this workbench defines, as a frozenset.

    Cached because it is asked once per object during a document sweep and the
    answer is fixed at import time.
    """
    from . import _vp_hook, analysis, materials, mesh, ports, preview, results, solver

    modules = (analysis, materials, mesh, ports, preview, results, solver)
    found = {}
    for module in modules:
        for name, value in vars(module).items():
            if not isinstance(value, type):
                continue
            if issubclass(value, _vp_hook.ViewProviderRestored):
                found[name] = value

    # A class the others inherit from is shared behaviour, not a kind: nothing
    # carries EMPortBase as its Proxy, and counting it makes the view-provider
    # sweep accept an object it then has no provider for. Answered by looking,
    # like the rest of this module, rather than by spelling "Base".
    #
    # It follows that subclassing a *concrete* kind would drop the parent.
    # test_every_document_kind_has_a_view_provider catches that, comparing this
    # set against the injector's table.
    bases = {base for value in found.values() for base in value.__mro__[1:]}
    return frozenset(name for name, value in found.items() if value not in bases)


def kind_of(obj):
    """The document object's class, by its Python proxy, or ``""``.

    FreeCAD's ``App::FeaturePython`` gives every workbench object the same
    ``TypeId``, so the proxy class is the only thing that tells an
    ``EMPortMicrostrip`` from an ``EMAnalysis``.
    """
    proxy = getattr(obj, "Proxy", None)
    return type(proxy).__name__ if proxy is not None else ""


def is_ours(obj):
    """True when this document object is one of ours."""
    return kind_of(obj) in kinds()
