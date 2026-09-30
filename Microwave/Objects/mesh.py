# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import FreeCAD

from ._vp_hook import ViewProviderRestored

#: What ``EMMeshPolicy.Padding{axis}{side}`` may say about a face of the domain.
#: Each is a statement about the problem rather than about a method: what lies
#: beyond the structure there. A backend that cannot build free space refuses
#: :data:`AIR` and takes the other two. The adapters repeat these strings rather
#: than importing them, because this module sits behind ``import FreeCAD``, and
#: ``tests/test_openems_document_translation.py`` holds the copies to these.
AIR = "Air"
THROUGH = "Through"
ENDS = "Ends"


class EMMeshRegion(ViewProviderRestored):
    """Local sizing: change how big the elements around this geometry are.

    Sizes here are absolute lengths, where a recipe sizes per wavelength.
    Global sizing resolves the wave, whose scale is lambda, and a region
    resolves a feature, whose scale is millimetres. openEMS spells it the same
    way: ``MSL_Losses.m`` sets its local overrides as lengths and never as a
    cells-per-wavelength figure.

    The object points in both directions, and the two are not mirror images.

    ``Refine`` states where the drawing needs resolving: elements no larger than
    ``ElementSize`` at what it names. That is one meaning, and each backend
    realises it the way its mesh can. A rectilinear grid places whole lines, so
    it refines every line across the bounding box of what the region names, and
    each axis is refined as a slab through the model. A tetrahedral mesh sizes
    the geometry where it is: along a face, an edge or a point the region names,
    and throughout a body it names, growing away from there, so the space a
    curved shape goes round is not refined with it. Asking for something
    coarser than the mesher's own ceiling is a contradiction rather than a
    request: a rectilinear grid refuses it by name, and a tetrahedral mesh lays
    nothing for it and says so.

    ``MinElementsAcross`` beside it asks for a count, and on a rectilinear grid
    a count is spent on each axis against that axis' own span. It therefore
    costs nothing along a direction the box is long in, and it is the way to
    resolve something narrow without refining everything level with it. It
    reaches every axis the box has extent on, a thickness included. An element
    size on tetrahedra is the same along every axis, and a backend meshing
    tetrahedra says so rather than laying the count.

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
    curve's fidelity. On a rectilinear grid a gap between the coarsened object
    and something else is the exception and stays. A separation belongs to both
    objects, so coarsening one does not coarsen the gap between them. A
    tetrahedral mesh sizes metal only along its edges, so there the coarsening
    is laid at those edges where every face the metal's bindings name is
    coarsened, and a gap along them is meshed as the edges are.

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
            "The geometry this is aimed at. Refining sizes it: a rectilinear"
            " grid refines every line across its bounding box, and a tetrahedral"
            " mesh sizes the geometry itself and grows away from it. Coarsening"
            " lets what a material binding asks of it settle for ElementSize,"
            " where the coarsenings name everything the bindings of that metal name",
        )

        obj.addProperty(
            "App::PropertyLength",
            "ElementSize",
            "Refinement",
            "Target element size. Refining, it must be finer than the global"
            " size the backend's own recipe sets with ElementsPerWavelength;"
            " coarsening, it is the size this geometry settles for",
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
    """Mesh policy: what the device asks of any mesh, whatever lays it.

    Everything here is a demand about the problem. A count across a drawn
    feature, a distance a curved surface may be solved from, a floor against a
    sliver, and what lies beyond the structure on each face of the domain. None
    of them names a primitive of a method's discretisation, so each is
    answerable by every backend reading this object alone, in the unit it was
    stated in, and each is refused or reported by name where a backend cannot
    meet it.

    "Element" rather than "cell" throughout. A cell is FDTD's word, a segment is
    MoM's and a tetrahedron is FEM's, and all three backends read this object,
    so it is not named after any one of them. Inside ``Solvers/openems/`` the
    FDTD words are correct and are used.

    What one meshing pipeline does about it is that pipeline's own object:
    ``EMYeeGrid`` for the rectilinear grid openEMS is solved on, ``EMGmshMesh``
    for the tetrahedra Palace is solved on. Elements per wavelength, an edge
    refinement factor and a growth ratio each name a primitive of one method,
    and one number of them is a different accuracy on each, so each pipeline
    states its own.

    How far the open surface stands from the structure is here rather than on a
    recipe. It is a length, and a count of cells was one pipeline standing in
    for it.

    Local refinement is absolute rather than per wavelength, because it resolves
    a feature, whose scale is millimetres, rather than the wave, whose scale is
    lambda. See ``EMMeshRegion``.
    """

    def __init__(self, obj):
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

        # Per-face, and not inferred. What each value does to a domain made of
        # cells, and why a line needs Through, is in openems.policy._padding.
        #
        # Each says what is beyond the drawing on that face, and each backend
        # answers it: Air is the medium reserved beyond the face, Through a
        # structure running on without end, and Ends a perfect wall on the drawn
        # face. Air is first, so it is the default, because a structure nobody
        # has said anything about is one with room around it.
        for axis in ["X", "Y", "Z"]:
            for side in ["Min", "Max"]:
                pname = f"Padding{axis}{side}"
                obj.addProperty(
                    "App::PropertyEnumeration",
                    pname,
                    "Domain",
                    f"What lies beyond the structure on {axis}{side}: Air for the"
                    " medium running out to an absorber, Through for the structure"
                    " running out through the absorber, Ends for a domain that stops"
                    " where the structure does with a perfect wall on that face",
                )
                setattr(obj, pname, [AIR, THROUGH, ENDS])

        # A length rather than a count of anything, because the distance is the
        # problem's and every backend that builds open space pads by it. Zero
        # derives it from the band, and Solvers/properties.py::clearance is the
        # rule and the calibration behind it. A face that wants no room at all
        # is Ends, so zero is free to mean "derived".
        obj.addProperty(
            "App::PropertyLength",
            "Clearance",
            "Domain",
            "How far the medium reaches past the structure on each Air face"
            " (0 = derived: 0.4 of the wavelength in the medium at the top of the"
            " band). The derived value stands the same share of a wavelength off"
            " at the top of every band, so it reflects a radiated wave by a fixed"
            " amount whatever the band - a radiator wants more",
        )
        obj.Clearance = 0.0

        # A link rather than numbers typed here, so the medium is a material of
        # the catalog like any other and a study states it by name. Empty is
        # vacuum, which is what fills undrawn room when nothing says otherwise.
        # It fills the domain out to its faces and through the absorber, so it is
        # one material for the whole study rather than one per face.
        obj.addProperty(
            "App::PropertyLink",
            "Medium",
            "Domain",
            "The material filling every space no bound body fills, out to the"
            " domain's faces and through the absorber (empty = vacuum). A"
            " Dielectric; a body drawn and bound stands over it",
        )

        obj.Proxy = self

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class EMMeshRecipe(ViewProviderRestored):
    """Base for one meshing pipeline's own settings. Not a kind of its own.

    A recipe holds the primitives of one method's discretisation: what only
    that pipeline's mesher consumes, in the words that pipeline uses. The
    device's own demands are ``EMMeshPolicy``, which every backend reads.

    A study holds one recipe per pipeline, in its group beside the policy, and
    each adapter finds its own by kind. So a drawing marked up once is answered
    by each backend without either erasing the other's settings.

    ``Objects/kinds.py::recipe_kinds`` derives the kinds from this class, and
    ``Objects/analysis.py::NOT_MESHED_FROM`` is held to naming every recipe but
    the one the mesh preview is laid from.

    Where two recipes ask for the same thing they spell it the same way, and
    that is on purpose. A user comparing two backends on one drawing states the
    same numbers to both, and what differs is the accuracy each reaches at them
    rather than the words. Each tooltip names the pipeline that reads it.
    """

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class EMYeeGrid(EMMeshRecipe):
    """The rectilinear grid openEMS is solved on.

    ``Solvers/openems/mesh.py`` lays it. A cell is a box, the grid is
    separable, and a line laid for one feature runs through the whole model, so
    what a cell size costs is decided per axis.

    The defaults are openEMS' own, from ``MSL_Losses.m``: bulk cells at
    lambda/20 in the slowest material in the model, conductor edges six times
    finer. The refinement is what resolves the field at a conductor edge, which
    is where a planar line's impedance is set and where the bulk size does not
    reach.

    How far a coarser refinement reaches is bounded by the mesher rather than
    here. A conductor's width is held to a share of itself whatever this asks
    for, so below some refinement the width rule sizes a narrow trace and
    coarsening further stops changing the mesh across it.
    ``tests/test_acceptance_openems_microstrip.py`` solves one such line across
    the range this property is meant to be moved over, prints where the two rules
    part company and prints what that range costs against what the gate can see.
    """

    #: Nothing. Every property here sizes or places a cell, and the mesh
    #: preview links this object, so each edit ages the drawing.
    MOVES_NO_CELL: tuple[str, ...] = ()

    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyFloat",
            "ElementsPerWavelength",
            "Mesh",
            "Cells per wavelength in the slowest material, at the top of the band",
        )
        obj.ElementsPerWavelength = 20.0
        # A float rather than an integer, because bisecting the default during a
        # convergence study is ordinary work and the first bisection is not a
        # whole number.
        obj.addProperty(
            "App::PropertyFloat",
            "EdgeRefinement",
            "Mesh",
            "How many times finer than bulk the cells at conductor edges are",
        )
        obj.EdgeRefinement = 6.0

        # One ratio rather than three. The mesher still takes a triple, so this
        # can be re-expanded if a model ever needs it.
        obj.addProperty(
            "App::PropertyFloat",
            "MaxGrowthRatio",
            "Mesh",
            "Largest size ratio between adjacent cells of the grid",
        )
        obj.MaxGrowthRatio = 1.3

        self.declare_what_moves_no_cell(obj)

        obj.Proxy = self


class EMGmshMesh(EMMeshRecipe):
    """The tetrahedral mesh Gmsh lays, which Palace reads.

    ``Microwave/Gmsh/`` builds the request. An element is sized the same along
    every axis and the size is laid as a field over the drawing, so a size asked
    for at one place reaches only as far as the growth below carries it.

    No air. How far the medium stands off is ``Clearance`` on the policy, and
    the Palace adapter refuses a face stated ``Air`` by name.

    A count across a thickness is not here either, and it is on the policy
    rather than gone: an element size here is the same along every axis, so the
    run states the count it did not lay with the size the mesh reached.

    The refinement and the growth below are chosen together, and the other
    recipe's are not these: one number means a cell's neighbour there and the
    slope of a size field here. A rim is asked for at the bulk size over the
    refinement, and grows back over a ramp as wide as that difference divided by
    the growth less one - so what the refinement costs is set by the ramp, which
    is the volume that is paid for, while what it buys is the element left
    standing on the rim, which the growth coarsens whatever the size along the
    rim. A steeper growth with a finer ask therefore reaches the same element at
    the rim over a narrower ramp.
    """

    #: Nothing, and the declaration reaches nothing either: this object is on
    #: ``Objects/analysis.py::NOT_MESHED_FROM``, so the openEMS mesh preview
    #: neither links it nor counts it as membership. That list is where "none of
    #: this moves a Yee cell" is stated.
    MOVES_NO_CELL: tuple[str, ...] = ()

    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyFloat",
            "ElementsPerWavelength",
            "Mesh",
            "Elements per wavelength in the slowest material, at the top of the band",
        )
        obj.ElementsPerWavelength = 20.0
        obj.addProperty(
            "App::PropertyFloat",
            "EdgeRefinement",
            "Mesh",
            "How many times finer than bulk the elements are at an edge of metal the field is "
            "singular along",
        )
        obj.EdgeRefinement = 8.0

        # A second-order element follows a curved surface only across a limited
        # arc, and one laid across a tighter curve is turned inside out. So each
        # curved surface is sized by how sharply it bends as well as by the
        # wavelength, whichever is finer, and a thin wire, a pin or a fillet is
        # meshed at its own radius. On a surface the size is one length at each
        # point, so it holds along a wire's length too, and that is where the
        # cost of a thin one goes. Six puts a sixth of a turn under each element.
        obj.addProperty(
            "App::PropertyInteger",
            "ElementsPerTurn",
            "Mesh",
            "Elements round a full turn of a curved surface, where that is finer than the "
            "wavelength asks (0 = size by the wavelength alone)",
        )
        obj.ElementsPerTurn = 6

        # A trend rather than a bound. Gmsh carries no option for the ratio
        # between neighbouring tetrahedra: two that share a face differ by their
        # shape as much as by the field, so what this sets is the slope of the
        # size field away from a refined place.
        #
        # What bounds the slope is arithmetic rather than a cliff. A regular
        # tetrahedron of edge a with one edge on a curve sized s has its other
        # two vertices sqrt(3)/2 a off that curve, where a ramp of slope g - 1
        # has already sized them at s + (g - 1) sqrt(3)/2 a; its target is the
        # mean of its four vertex sizes, s + sqrt(3)/4 (g - 1) a, and the mesher
        # stops refining once its circumradius, sqrt(6)/4 a, falls under that
        # target. Solving for a leaves the edge at
        # 4/sqrt(6) s / (1 - (g - 1)/sqrt(2)), which grows without bound at a
        # growth of 1 + sqrt(2), about 2.41.
        #
        # Below that, a steeper growth buys a coarser tetrahedron rather than a
        # request that goes unanswered: the size along the curve itself is met
        # well above the value here. So what sits above this number is a caution
        # and not a measured edge, and every run states the element it left
        # standing at each place it sized, which is where to look before turning
        # it.
        obj.addProperty(
            "App::PropertyFloat",
            "MaxGrowthRatio",
            "Mesh",
            "How much the element size grows per element away from a refined place",
        )
        obj.MaxGrowthRatio = 1.6

        self.declare_what_moves_no_cell(obj)

        obj.Proxy = self


def createEMMeshPolicy(doc: Any = None) -> Any:
    """The mesh policy, unowned. The analysis files it away."""
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::FeaturePython", "EMMeshPolicy")
    EMMeshPolicy(obj)
    obj.Label = "Mesh Policy"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMeshPolicy")
    return obj


def createEMYeeGrid(doc: Any = None) -> Any:
    """openEMS' grid settings, unowned. The analysis files it away."""
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::FeaturePython", "EMYeeGrid")
    EMYeeGrid(obj)
    obj.Label = "Yee Grid"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMYeeGrid")
    return obj


def createEMGmshMesh(doc: Any = None) -> Any:
    """The tetrahedral mesh settings, unowned. The analysis files it away."""
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::FeaturePython", "EMGmshMesh")
    EMGmshMesh(obj)
    obj.Label = "Gmsh Mesh"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMGmshMesh")
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
