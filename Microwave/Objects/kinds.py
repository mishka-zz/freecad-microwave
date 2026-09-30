# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which document objects this workbench owns.

The answer decides what a refinement region may reference, which objects need a
view provider restored, and which objects the tree may claim.

The set is derived from the classes themselves rather than from their names.
Testing the proxy class name for a prefix makes renaming a class silently change
which objects the workbench recognises, and ``EM`` is one letter away from
matching anything. ``classes`` lists the modules it reads, so a class added to one
of them is recognised as it stands, and a class in a new module is not until that
module is listed.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


@functools.cache
def classes() -> Mapping[str, type]:
    """Every proxy class this workbench defines, by the name it carries.

    Cached because a document sweep reaches it once per object, and the answer
    is fixed at import time. Read-only, the cache handing every caller the one
    mapping.
    """
    from . import _vp_hook, analysis, materials, mesh, ports, preview, results, solver

    modules = (analysis, materials, mesh, ports, preview, results, solver)
    found: dict[str, type] = {}
    for module in modules:
        for name, value in vars(module).items():
            if not isinstance(value, type):
                continue
            if issubclass(value, _vp_hook.ViewProviderRestored):
                found[name] = value

    # A class the others inherit from is shared behaviour rather than a kind.
    # Nothing carries EMPortBase as its Proxy, and counting it makes the
    # view-provider sweep accept an object it then has no provider for. The base
    # classes are found from the classes themselves, like the rest of this
    # module, rather than by spelling "Base".
    #
    # Subclassing a concrete kind would therefore drop the parent.
    # test_every_document_kind_has_a_view_provider catches that, comparing this
    # set against the injector's table.
    bases = {base for value in found.values() for base in value.__mro__[1:]}
    return MappingProxyType({name: value for name, value in found.items() if value not in bases})


@functools.cache
def kinds() -> frozenset[str]:
    """Every proxy class name this workbench defines, as a frozenset.

    Cached like the mapping it is taken from: a document sweep asks once per
    object.
    """
    return frozenset(classes())


def solver_kinds() -> frozenset[str]:
    """Every kind that is a solver.

    Derived from the classes, so a backend added tomorrow answers here as it
    stands. ``Objects/analysis.py::solvers_in`` reads it to say what a study is
    for, and the check on ``Objects/analysis.py::NOT_MESHED_FROM`` reads it to
    hold a hand-written list to naming every backend but the one the mesh
    preview is laid from. The relation is the one :func:`kinds` has to the view
    provider table: a typed table, and a derivation to hold it to.
    """
    from .solver import EMSolverBase

    return frozenset(name for name, value in classes().items() if issubclass(value, EMSolverBase))


def recipe_kinds() -> frozenset[str]:
    """Every kind that is one meshing pipeline's own settings.

    Derived from the classes, the way :func:`solver_kinds` is, and read for the
    same job: the check on ``Objects/analysis.py::NOT_MESHED_FROM`` holds that
    hand-written list to naming every recipe but the one the mesh preview is
    laid from.
    """
    from .mesh import EMMeshRecipe

    return frozenset(name for name, value in classes().items() if issubclass(value, EMMeshRecipe))


def kind_of(obj: Any) -> str:
    """The document object's class, by its Python proxy, or ``""``.

    FreeCAD's ``App::FeaturePython`` gives every workbench object the same
    ``TypeId``, so the proxy class is the only thing that tells an
    ``EMPortMicrostrip`` from an ``EMAnalysis``.
    """
    proxy = getattr(obj, "Proxy", None)
    return type(proxy).__name__ if proxy is not None else ""


def is_ours(obj: Any) -> bool:
    """True when this document object belongs to this workbench."""
    return kind_of(obj) in kinds()
