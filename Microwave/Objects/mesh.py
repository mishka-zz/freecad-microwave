# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

import FreeCAD

from ._vp_hook import ViewProviderRestored


class EMMeshRegion(ViewProviderRestored):
    """Local refinement: make the elements around this geometry smaller.

    Absolute lengths, where ``EMMeshPolicy`` is per wavelength. Global
    sizing resolves the *wave*, whose scale is lambda; a refinement region
    resolves a *feature*, whose scale is millimetres. Upstream openEMS agrees:
    every local override in its tutorials is a length or a divisor of the bulk,
    never a cells-per-wavelength figure.

    It **refines only**. A region coarser than the coarsest cell the mesher
    would produce anyway is refused by name when the mesh is built, because
    coarsening past the bulk target is numerical dispersion - an error that
    appears as a wrong answer rather than as a visibly bad grid. A region
    between that ceiling and the local material's own size is accepted and
    simply does not bite: the sizing field takes the finer of the two.

    Neutral, like every object in this layer: "element" covers an FDTD cell, a
    MoM segment and an FEM tetrahedron, and all three meshers have a local
    refinement region. What each backend does with the box is its own business.
    """

    def __init__(self, obj):
        # LinkSubList, matching EMMaterialBinding.References: a refinement
        # region is just as likely to be aimed at a face as at a whole solid.
        obj.addProperty(
            "App::PropertyLinkSubList",
            "References",
            "Refinement",
            "The geometry this refinement is aimed at. Its bounding box is what"
            " gets refined, and on a rectilinear grid each axis is refined as a"
            " slab through the model",
        )

        obj.addProperty(
            "App::PropertyLength",
            "ElementSize",
            "Refinement",
            "Target element size inside this region; must be finer than the"
            " global size, which is set by ElementsPerWavelength",
        )
        obj.ElementSize = 0.0

        # Zero inherits EMMeshPolicy.MinElementsAcross, whose value this takes
        # but whose rule it does not: the global count is dielectrics only, so
        # a region is also how a conductor gets a count when one is genuinely
        # wanted. And a 50 um bond layer and a 1.6 mm core want different
        # counts, which one global integer cannot express either.
        obj.addProperty(
            "App::PropertyInteger",
            "MinElementsAcross",
            "Refinement",
            "Fewest elements across this region (0 = inherit the global count)",
        )
        obj.MinElementsAcross = 0

        obj.addProperty(
            "App::PropertyBool",
            "Enabled",
            "Refinement",
            "Uncheck to leave the region in the tree but out of the mesh",
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

    "Element" rather than "cell" throughout, deliberately. A cell is FDTD's
    word, a segment is MoM's and a tetrahedron is FEM's, and this object is
    read by all three - naming it after the first backend is exactly what the
    adapter architecture exists to avoid. Inside ``Solvers/openems/`` the FDTD
    words are correct and are used.

    Sizing is **per wavelength, never in millimetres**, and that is load-
    bearing. A remembered millimetre value silently under-resolves the moment
    anyone raises the permittivity or the frequency. Local refinement is the
    opposite - absolute - because it resolves a *feature*, whose scale is
    millimetres, rather than the *wave*, whose scale is lambda. See
    ``EMMeshRegion``.

    The defaults are openEMS' own, from ``MSL_Losses.m``: bulk elements at
    lambda/20 in the slowest material in the model, conductor edges six times
    finer. The *refinement* is the load-bearing part - at the same element
    count, coarsening it to a third moves the microstrip gate's extracted
    impedance by more than the whole tolerance. See ``tests/test_solver.py``.
    """

    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyFloat",
            "ElementsPerWavelength",
            "Mesh",
            "Elements per wavelength in the slowest material, at the top of the band",
        )
        obj.ElementsPerWavelength = 20.0
        # A float, not an integer: bisecting to 3 or 4.5 during a convergence
        # study is ordinary work, and a whole step is a large move on the knob
        # this class's docstring calls the load-bearing one.
        obj.addProperty(
            "App::PropertyFloat",
            "EdgeRefinement",
            "Mesh",
            "How many times finer than bulk the elements at conductor edges are",
        )
        obj.EdgeRefinement = 6.0

        # One ratio, not three. Per-axis grading is rectilinear-specific and so
        # does not belong on a neutral object; the mesher still takes a triple,
        # so this can be re-expanded if a model ever needs it.
        obj.addProperty(
            "App::PropertyFloat",
            "MaxGrowthRatio",
            "Mesh",
            "Largest size ratio between adjacent elements",
        )
        obj.MaxGrowthRatio = 1.3

        # MSL_Losses.m spans its substrate with linspace(0, thickness, 10). A
        # thin substrate carries the whole field, so one element across it is
        # not an approximation, it is a different problem. Dielectrics only:
        # a conductor has no field inside it to sample, and spanning a foil
        # with this count sets the smallest element in the model.
        obj.addProperty(
            "App::PropertyInteger",
            "MinElementsAcross",
            "Mesh",
            "Fewest elements through a dielectric with thickness",
        )
        obj.MinElementsAcross = 9

        # Zero means "derive it". This is a floor against degenerate geometry -
        # a 1 um sliver from a CAD boolean sets the timestep for the whole
        # simulation - and not a way to shape the grid: set anywhere near the
        # element sizes it starts refusing features the user is entitled to mesh.
        obj.addProperty(
            "App::PropertyLength",
            "MinElementSize",
            "Mesh",
            "Hard floor on element size (0 = derive it from the edge size)",
        )
        obj.MinElementSize = 0.0

        # 8 is also openems.write.DEFAULT_PADDING, and the element is the one
        # air is meshed at - the bulk size in vacuum - so eight of them is
        # eight ElementsPerWavelength-ths of a free-space wavelength at the top
        # of the band, 0.4 of one at the 20 above, whatever the substrate is.
        # The calibration against openEMS' own tutorials is on that constant; a
        # test asserts the two stay equal, since this layer may not import an
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

        # Per-face: does the model stop inside the domain with air around it, or
        # run out *through* the absorber? "Through" pulls the domain in so the
        # absorber lands on the structure, which is what makes a transmission
        # line infinite. Give a line air at its ends instead and it radiates off
        # an open circuit, and every impedance read from it is contaminated by
        # the reflection - so this is a choice, not something to infer.
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


def createEMMeshPolicy(doc=None):
    """The mesh policy, unowned. Filing it away is the analysis's job."""
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::FeaturePython", "EMMeshPolicy")
    EMMeshPolicy(obj)
    obj.Label = "Mesh Policy"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMeshPolicy")
    return obj


def references_from(selection):
    """What a link may be aimed at, out of whatever the user had picked.

    Read by both objects that point at geometry: a refinement region and a
    material binding.

    Geometry only. A refinement region resolves a *feature*, and a material is
    what a solid is made of, so pointing either at a mesh policy or at a study
    is meaningless - and pointing one at its own analysis closes a cycle in
    FreeCAD's dependency graph: group membership is itself a link, so container
    to member and member back to container. FreeCAD then prints *"The graph
    must be a DAG"* and stops being able to order the recompute. Nothing warns;
    the model just quietly stops settling.

    ``[""]`` and not ``[]`` for a whole object. A ``LinkSubList`` entry with an
    empty sub-element list is dropped by FreeCAD on assignment, silently, and
    the object is then left referencing nothing.
    """
    references = []
    for chosen in selection:
        obj = getattr(chosen, "Object", chosen)
        if not _is_geometry(obj):
            continue
        references.append((obj, list(getattr(chosen, "SubElementNames", ())) or [""]))
    return references


def _is_geometry(obj):
    """Something a user drew, rather than something this workbench made.

    The ``Shape`` test alone is not enough: ``EMMeshPreview`` is a
    ``Part::FeaturePython`` and has one, and it is a group member like any other.
    """
    from .kinds import is_ours

    if is_ours(obj):
        return False
    if getattr(obj, "Group", None) is not None:
        return False
    return getattr(obj, "Shape", None) is not None


def createEMMeshRegion(doc=None):
    doc = doc or FreeCAD.ActiveDocument
    obj = doc.addObject("App::FeaturePython", "EMMeshRegion")
    EMMeshRegion(obj)
    obj.Label = "Mesh Refinement"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMeshRegion")
    return obj
