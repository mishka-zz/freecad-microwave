# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import FreeCAD

from ._vp_hook import ViewProviderRestored


class EMMeshRegion(ViewProviderRestored):
    """Local sizing: change how big the elements around this geometry are.

    Sizes here are absolute lengths, where ``EMMeshPolicy`` is per wavelength.
    Global sizing resolves the wave, whose scale is lambda, and a region
    resolves a feature, whose scale is millimetres. openEMS spells it the same
    way: ``MSL_Losses.m`` sets its local overrides as lengths and never as a
    cells-per-wavelength figure.

    The object points in both directions, and the two are not mirror images.

    ``Refine`` covers a box - the bounding box of what it names - and asks for
    elements no larger than ``ElementSize`` anywhere inside it. Asking for
    something coarser than the mesher's own ceiling is refused by name. A
    refinement that coarsens is a contradiction rather than a request.

    ``MinElementsAcross`` beside it asks for a count, and a count is spent on
    each axis against that axis' own span. It therefore costs nothing along a
    direction the box is long in, and it is the way to resolve something narrow
    without refining everything level with it. It reaches every axis the box has
    extent on, a thickness included.

    ``Coarsen`` attaches to the object, and lets that object's own demands settle
    for ``ElementSize`` rather than the size they would otherwise ask for. It is
    not a box. A rectilinear grid is separable, so a box spends itself on a slab
    through the model along each axis. A refinement that overshoots that way
    hands out elements nobody asked for, and coarsening that way would take
    elements from whatever else lies level with the box, anywhere in the model.
    Naming the object cannot reach past it.

    ``Coarsen`` never moves where the elements are. Faces, port planes and
    sheets are pinned whatever the sizing says, so the solver is given the
    geometry that was drawn, and what changes is how many elements are spent
    following it. It also cannot coarsen past the global ceiling, which bounds
    every element in the model.

    It gives up everything that geometry was asking for, and not only the bulk
    size: a conductor's edge treatment, a dielectric's own element count, a
    curve's fidelity. A gap between the coarsened object and something else is
    the exception and stays. A separation belongs to both objects, so
    coarsening one does not coarsen the gap between them.

    This object is neutral, like every object in this layer. "Element" covers an
    FDTD cell, a MoM segment and an FEM tetrahedron, and all three meshers size
    locally. Each backend decides what to do with it.
    """

    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyEnumeration",
            "Mode",
            "Refinement",
            "Whether this makes the elements around its geometry finer, or lets"
            " that geometry settle for coarser ones",
        )
        obj.Mode = ["Refine", "Coarsen"]

        # LinkSubList, matching EMMaterialBinding.References. A refinement
        # region is as likely to be aimed at a face as at a whole solid.
        obj.addProperty(
            "App::PropertyLinkSubList",
            "References",
            "Refinement",
            "The geometry this is aimed at. Refining covers its bounding box,"
            " and on a rectilinear grid each axis is refined as a slab through"
            " the model; coarsening applies to the whole object instead",
        )

        obj.addProperty(
            "App::PropertyLength",
            "ElementSize",
            "Refinement",
            "Target element size. Refining, it must be finer than the global"
            " size set by ElementsPerWavelength; coarsening, it is the size"
            " this geometry settles for",
        )
        obj.ElementSize = 0.0

        # Zero inherits EMMeshPolicy.MinElementsAcross. It takes that value but
        # not that rule: the global count covers dielectrics only, so a region is
        # also how a conductor gets a count when one is wanted. A 50 um bond
        # layer and a 1.6 mm core also want different counts, which one global
        # integer cannot express.
        obj.addProperty(
            "App::PropertyInteger",
            "MinElementsAcross",
            "Refinement",
            "Fewest elements across this region, on every axis it has extent on"
            " (0 = inherit the global count). A count asks for resolution, so it"
            " belongs to Refine only",
        )
        obj.MinElementsAcross = 0

        obj.addProperty(
            "App::PropertyBool",
            "Enabled",
            "Refinement",
            "Uncheck to leave this in the tree but out of the mesh",
        )
        obj.Enabled = True

        obj.Proxy = self

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class EMMeshPolicy(ViewProviderRestored):
    """Mesh policy: solver-neutral intent, in the vocabulary every mesher shares.

    "Element" rather than "cell" throughout. A cell is FDTD's word, a segment is
    MoM's and a tetrahedron is FEM's, and all three backends read this object,
    so it is not named after any one of them. Inside ``Solvers/openems/`` the
    FDTD words are correct and are used.

    Sizing is per wavelength and never in millimetres. A remembered millimetre
    value silently under-resolves the moment the permittivity or the frequency
    is raised. Local refinement is absolute instead, because it resolves a
    feature, whose scale is millimetres, rather than the wave, whose scale is
    lambda. See ``EMMeshRegion``.

    The defaults are openEMS' own, from ``MSL_Losses.m``: bulk elements at
    lambda/20 in the slowest material in the model, conductor edges six times
    finer. The refinement is what resolves the field at a conductor edge, which
    is where a planar line's impedance is set and where the bulk size does not
    reach.

    How far a coarser refinement reaches is bounded by the mesher rather than
    here. A conductor's width is held to a share of itself whatever this asks
    for, so below some refinement the width rule sizes a narrow trace and
    coarsening further stops changing the mesh across it.
    ``tests/test_acceptance_microstrip.py`` solves one such line across the range
    this property is meant to be moved over, prints where the two rules part
    company and prints what that range costs against what the gate can see.
    """

    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyFloat",
            "ElementsPerWavelength",
            "Mesh",
            "Elements per wavelength in the slowest material, at the top of the band",
        )
        obj.ElementsPerWavelength = 20.0
        # A float rather than an integer, because bisecting the default during a
        # convergence study is ordinary work and the first bisection is not a
        # whole number.
        obj.addProperty(
            "App::PropertyFloat",
            "EdgeRefinement",
            "Mesh",
            "How many times finer than bulk the elements at conductor edges are",
        )
        obj.EdgeRefinement = 6.0

        # One ratio rather than three. Per-axis grading is rectilinear-specific
        # and does not belong on a neutral object. The mesher still takes a
        # triple, so this can be re-expanded if a model ever needs it.
        obj.addProperty(
            "App::PropertyFloat",
            "MaxGrowthRatio",
            "Mesh",
            "Largest size ratio between adjacent elements",
        )
        obj.MaxGrowthRatio = 1.3

        # MSL_Losses.m spans its substrate with linspace(0, thickness, 10). A
        # thin substrate carries the whole field, so one element across it
        # states a different problem rather than approximating this one.
        # Dielectrics only: a conductor has no field inside it to sample, and
        # spanning a foil with this count sets the smallest element in the model.
        obj.addProperty(
            "App::PropertyInteger",
            "MinElementsAcross",
            "Mesh",
            "Fewest elements through a dielectric with thickness",
        )
        obj.MinElementsAcross = 9

        # Zero asks for nothing. A curved surface reaches a solver as flat facets
        # whose chords lie off the arc, and the CAD kernel decides for itself how
        # close it comes to the drawing: wide bands of requests come back as one
        # mesh. Setting this holds the surface to a distance instead, and refuses
        # where that distance cannot be reached rather than quietly missing it.
        #
        # A length rather than a share of anything. How evenly the surface is
        # parameterised sets what leaving the kernel's own band costs, so one
        # share of a radius costs very different work on two shapes. The run also
        # reports the departure as a length, so what is asked for and what comes
        # back are comparable.
        #
        # This is not the fidelity the mesh is held to. That one is a share of a
        # radius, and it governs where elements go rather than what shape they
        # are laid on.
        obj.addProperty(
            "App::PropertyLength",
            "CurveTolerance",
            "Mesh",
            "How far a curved surface may be solved from where it was drawn"
            " (0 = whatever the CAD kernel gives unasked). Costs facets, and"
            " refuses if it cannot be reached",
        )
        obj.CurveTolerance = 0.0

        # Zero means derive it. This is a floor against degenerate geometry: a
        # 1 um sliver from a CAD boolean sets the timestep for the whole
        # simulation. It is not a way to shape the grid. Set anywhere near the
        # element sizes, it starts refusing features the user is entitled to
        # mesh.
        obj.addProperty(
            "App::PropertyLength",
            "MinElementSize",
            "Mesh",
            "Hard floor on element size (0 = derive it from the edge size)",
        )
        obj.MinElementSize = 0.0

        # 8 is also openems.plan.DEFAULT_PADDING. The element is the one air is
        # meshed at - the bulk size in vacuum - so eight of them is eight
        # ElementsPerWavelength-ths of a free-space wavelength at the top of the
        # band, and 0.4 of one at the 20 above, whatever the substrate is. The
        # calibration against openEMS' own tutorials sits on that constant. A
        # test asserts the two stay equal, because this layer may not import an
        # adapter.
        for axis in ["X", "Y", "Z"]:
            for side in ["Min", "Max"]:
                pname = f"AirCells{axis}{side}"
                obj.addProperty(
                    "App::PropertyInteger",
                    pname,
                    "Domain",
                    f"Air buffer elements outside the structure on {axis}{side},"
                    " when that face is Air",
                )
                setattr(obj, pname, 8)

        # Per-face, and not inferred. What Through does to the domain, and why a
        # line needs it, is in openems.policy._padding.
        for axis in ["X", "Y", "Z"]:
            for side in ["Min", "Max"]:
                pname = f"Padding{axis}{side}"
                obj.addProperty(
                    "App::PropertyEnumeration",
                    pname,
                    "Domain",
                    f"Whether the structure ends inside the domain on {axis}{side}"
                    " or runs out through the absorber",
                )
                setattr(obj, pname, ["Air", "Through"])

        obj.Proxy = self

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def createEMMeshPolicy(doc: Any = None) -> Any:
    """The mesh policy, unowned. The analysis files it away."""
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::FeaturePython", "EMMeshPolicy")
    EMMeshPolicy(obj)
    obj.Label = "Mesh Policy"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMeshPolicy")
    return obj


def references_from(selection: Iterable[Any]) -> list[tuple[Any, list[str]]]:
    """What a link may be aimed at, out of whatever the user had picked.

    Read by both objects that point at geometry: a refinement region and a
    material binding.

    Geometry only. A refinement region resolves a feature, and a material is
    what a solid is made of, so pointing either at a mesh policy or at a study
    means nothing. Pointing one at its own analysis closes a cycle in FreeCAD's
    dependency graph, because group membership is itself a link: container to
    member, and member back to container. FreeCAD then prints "The graph must be
    a DAG" and cannot order the recompute. Nothing warns, and the model quietly
    stops settling.

    A whole object is written ``[""]`` rather than ``[]``. FreeCAD silently
    drops a ``LinkSubList`` entry with an empty sub-element list on assignment,
    leaving the object referencing nothing.
    """
    references: list[tuple[Any, list[str]]] = []
    for chosen in selection:
        obj = getattr(chosen, "Object", chosen)
        if not _is_geometry(obj):
            continue
        references.append((obj, list(getattr(chosen, "SubElementNames", ())) or [""]))
    return references


def _is_geometry(obj: Any) -> bool:
    """Something a user drew, rather than something this workbench made.

    The ``Shape`` test alone is not enough. ``EMMeshPreview`` is a
    ``Part::FeaturePython``, so it has a shape, and it is a group member like
    any other.
    """
    from .kinds import is_ours

    if is_ours(obj):
        return False
    if getattr(obj, "Group", None) is not None:
        return False
    return getattr(obj, "Shape", None) is not None


def createEMMeshRegion(doc: Any = None) -> Any:
    doc = doc or FreeCAD.ActiveDocument
    obj = doc.addObject("App::FeaturePython", "EMMeshRegion")
    EMMeshRegion(obj)
    obj.Label = "Mesh Refinement"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMeshRegion")
    return obj
