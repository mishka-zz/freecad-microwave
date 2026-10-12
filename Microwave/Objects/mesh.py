# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import FreeCAD

from ._vp_hook import SharesLabel, ViewProviderRestored

#: What ``EMMeshPolicy.Boundary{axis}{side}`` may say about a face of the domain.
#: Each is a statement about the problem rather than about a method: what lies
#: beyond the structure there. A backend that cannot build free space refuses
#: :data:`OPEN` and takes the other two. The adapters repeat these strings rather
#: than importing them, because this module sits behind ``import FreeCAD``, and
#: ``tests/test_mesh_recipes.py`` holds the copies to these.
OPEN = "Open"
THROUGH = "Through"
PEC_WALL = "PEC Wall"


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

    A region amends one pipeline's recipe, the one ``MeshSettings`` names, and
    only that pipeline reads it. One size is a different accuracy on each
    method, so the same place refined for two backends is two regions, each
    sized for its own. "Element" covers an FDTD cell, a MoM segment and an FEM
    tetrahedron, so the words here are the same whichever recipe it amends.
    """

    def __init__(self, obj):
        # A link from the region rather than a list on the recipe, so a region
        # deleted takes nothing else with it. A recipe deleted leaves the link
        # empty, and each adapter refuses a region that amends nothing.
        obj.addProperty(
            "App::PropertyLink",
            "MeshSettings",
            "Refinement",
            "Mesh settings this region amends. Only the solver that reads them uses the region",
        )

        obj.addProperty(
            "App::PropertyEnumeration",
            "Mode",
            "Refinement",
            "Refine: elements at References no larger than ElementSize. Coarsen:"
            " References settles for ElementSize in place of the finer size it asks for",
        )
        obj.Mode = ["Refine", "Coarsen"]

        # LinkSubList, matching EMMaterialBinding.References. A refinement
        # region is as likely to be aimed at a face as at a whole solid.
        #
        # A coarsening on a tetrahedral mesh is laid at the edges of a metal
        # only where the coarsenings name every face that metal's bindings
        # name, so the tooltip asks for every such face.
        obj.addProperty(
            "App::PropertyLinkSubList",
            "References",
            "Refinement",
            "Geometry this region applies to. On openEMS, Refine refines every"
            " grid line across its bounding box; on Palace it sizes the geometry"
            " itself and grows away from it. To coarsen a metal on Palace, name"
            " every face its material bindings name",
        )

        obj.addProperty(
            "App::PropertyLength",
            "ElementSize",
            "Refinement",
            "Element size at References. Refine: must be smaller than the size"
            " ElementsPerWavelength sets in the mesh settings. Coarsen: the size"
            " References settles for",
        )
        obj.ElementSize = 0.0

        # Zero inherits MinElementsAcross from MeshSettings. It takes that value but
        # not that rule: the settings' count covers dielectrics only, so a region is
        # also how a conductor gets a count when one is wanted. A 50 um bond
        # layer and a 1.6 mm core also want different counts, which the one
        # count on the settings cannot express.
        obj.addProperty(
            "App::PropertyInteger",
            "MinElementsAcross",
            "Refinement",
            "Fewest elements across References on each axis it spans (0 = the"
            " MinElementsAcross of the mesh settings). Refine only: leave it at 0"
            " to Coarsen",
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

    def onChanged(self, obj, prop):
        """Ask the tree to show the region under the settings it now amends.

        The tree reads an object's children again only when a property of that
        object changes, and this one changes the region alone.
        """
        if prop != "MeshSettings":
            return
        from .analysis import analysis_of, renest

        study = analysis_of(obj)
        if study is not None:
            renest(study)

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class EMMeshPolicy(SharesLabel, ViewProviderRestored):
    """Mesh policy: the problem every backend of the study solves.

    Everything here is a statement about the problem: what lies beyond the
    structure on each face of the domain, how far the medium reaches past it,
    what fills it, and a distance a curved surface may be solved from. None
    names a primitive of a method's discretisation, so each is answerable by
    every backend reading this object alone, in the unit it was stated in, and
    each is refused or reported by name where a backend cannot meet it. Every
    backend of a study reads this one object, so two backends solve one problem
    and their answers compare.

    What one meshing pipeline does about it is that pipeline's own object:
    ``EMYeeGrid`` for the rectilinear grid openEMS is solved on, ``EMGmshMesh``
    for the tetrahedra Palace is solved on. Elements per wavelength, an edge
    refinement factor and a growth ratio each name a primitive of one method,
    and one number of them is a different accuracy on each, so each pipeline
    states its own.

    How far the open surface stands from the structure is here rather than on a
    recipe. It is a length, and a count of cells was one pipeline standing in
    for it.

    A count of elements across a dielectric, a floor under every element and
    the local refinements are each pipeline's own, on its recipe: each names
    elements, and one number of elements is a different accuracy on each
    method.
    """

    def __init__(self, obj):
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
            "Largest distance a curved surface may be solved from where it was"
            " drawn (0 = FreeCAD's own faceting). A smaller value costs facets, and"
            " a run refuses one it cannot reach",
        )
        obj.CurveTolerance = 0.0

        # Per-face, and not inferred. What each value does to a domain made of
        # cells, and why a line needs Through, is in openems.policy._padding.
        #
        # Each says what is beyond the drawing on that face, and each backend
        # answers it: Open is the medium reserved beyond the face, Through a
        # structure running on without end, and PEC Wall a perfect wall on the
        # drawn face. Open is first, so it is the default, because a structure nobody
        # has said anything about is one with room around it.
        for axis in ["X", "Y", "Z"]:
            for side in ["Min", "Max"]:
                pname = f"Boundary{axis}{side}"
                obj.addProperty(
                    "App::PropertyEnumeration",
                    pname,
                    "Domain",
                    f"What lies beyond the structure on its {axis}{side} side. Open:"
                    " the medium, out to an absorber. Through: the structure runs on"
                    " through the absorber. PEC Wall: the domain ends at the"
                    " structure, with a perfect conductor on that face",
                )
                setattr(obj, pname, [OPEN, THROUGH, PEC_WALL])

        # A length rather than a count of anything, because the distance is the
        # problem's and every backend that builds open space pads by it. Zero
        # derives it from the band, and Solvers/properties.py::clearance is the
        # rule and the calibration behind it. A face that wants no room at all
        # is PEC Wall, so zero is free to mean "derived". The derived distance
        # stands the same share of a wavelength off at the top of every band, so
        # it reflects a radiated wave by a fixed amount whatever the band, and a
        # radiator wants more.
        obj.addProperty(
            "App::PropertyLength",
            "Clearance",
            "Domain",
            "Distance from the structure to each Open face of the domain (0 = 0.4"
            " of the wavelength in the medium at the top of the band). Set a larger"
            " value for a radiator",
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
            "Dielectric that fills all space no bound body fills, out to the"
            " domain faces and through the absorber (empty = vacuum)",
        )

        obj.Proxy = self

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class EMMeshRecipe(SharesLabel, ViewProviderRestored):
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

    A recipe holds in the tree the refinement regions that name it in
    ``MeshSettings``, since each amends what the recipe lays.
    """

    def add_element_bounds(self, obj, across, floor):
        """The count across a dielectric and the floor under every element,
        which every pipeline states and each reads as its own elements.
        ``across`` and ``floor`` are the two tooltips, which say what this
        pipeline does with each."""
        # MSL_Losses.m spans its substrate with linspace(0, thickness, 10). A
        # thin substrate carries the whole field, so one element across it
        # misses that field rather than approximating it.
        # Dielectrics only: a conductor has no field inside it to sample, and
        # spanning a foil with this count sets the smallest element in the model.
        obj.addProperty("App::PropertyInteger", "MinElementsAcross", "Mesh", across)
        obj.MinElementsAcross = 9

        # Zero means derive it. This is a floor against degenerate geometry: a
        # sliver from a CAD boolean would otherwise set the smallest element in
        # the model. It is not a way to shape the mesh. Set anywhere near the
        # element sizes, it starts refusing features the user is entitled to
        # mesh.
        obj.addProperty("App::PropertyLength", "MinElementSize", "Mesh", floor)
        obj.MinElementSize = 0.0

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

        self.add_element_bounds(
            obj,
            "Fewest cells across the thickness of each dielectric",
            "Smallest cell allowed, so a sliver left by a boolean cannot set the cell"
            " size (0 = derived from the cell size at conductor edges)",
        )
        self.declare_what_moves_no_cell(obj)

        obj.Proxy = self


class EMGmshMesh(EMMeshRecipe):
    """The tetrahedral mesh Gmsh lays, which Palace reads.

    ``Microwave/Gmsh/`` builds the request. An element is sized the same along
    every axis and the size is laid as a field over the drawing, so a size asked
    for at one place reaches only as far as the growth below carries it.

    No air. How far the medium stands off is ``Clearance`` on the policy.

    ``MinElementsAcross`` asks for a count across a thickness, and an element
    size here is the same along every axis, so the run states a count it did not
    lay with the size the mesh reached.

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
        # Fewer than the grid's, because an element carries a polynomial of the
        # solver's order across itself where a Yee cell carries a constant, so
        # its error falls with the order as well as with its size.
        obj.ElementsPerWavelength = 4.0
        obj.addProperty(
            "App::PropertyFloat",
            "EdgeRefinement",
            "Mesh",
            "How many times smaller than bulk the elements are at the sharp edges of metal",
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
            "Elements per full turn of a curved surface, where that is finer than"
            " ElementsPerWavelength asks (0 = size by the wavelength alone)",
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

        self.add_element_bounds(
            obj,
            "Fewest elements across the thickness of each dielectric. The run does"
            " not lay this count, and reports it beside the element size reached",
            "Smallest element allowed, so a sliver left by a boolean cannot set the"
            " element size (0 = derived from the element size in the slowest material)",
        )
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
    obj.Label = "Mesh Settings (openEMS)"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMYeeGrid")
    return obj


def createEMGmshMesh(doc: Any = None) -> Any:
    """The tetrahedral mesh settings, unowned. The analysis files it away."""
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::FeaturePython", "EMGmshMesh")
    EMGmshMesh(obj)
    obj.Label = "Mesh Settings (Palace)"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMGmshMesh")
    return obj


def references_from(selection: Iterable[Any], study: Any) -> list[tuple[Any, list[str]]]:
    """What a link from inside ``study`` may be aimed at, out of whatever the
    user had picked.

    Read by both objects that point at geometry: a refinement region and a
    material binding.

    Geometry only. A refinement region resolves a feature, and a material is
    what a solid is made of, so pointing either at a mesh policy or at a study
    means nothing.

    Nothing that depends on ``study``. Group membership is itself a link, from
    the study to each member, so a link from a member to an object that reaches
    the study closes a cycle in FreeCAD's dependency graph. FreeCAD then prints
    "The graph must be a DAG" and cannot order the recompute. Nothing warns,
    and the model quietly stops settling. :func:`reaching` names what this
    leaves out for that reason.

    The picks of one object are gathered into one entry.

    A whole object is written ``[""]`` rather than ``[]``. FreeCAD silently
    drops a ``LinkSubList`` entry with an empty sub-element list on assignment,
    leaving the object referencing nothing.
    """
    gathered: dict[int, tuple[Any, list[str]]] = {}
    for chosen in selection:
        obj = getattr(chosen, "Object", chosen)
        if not _is_geometry(obj) or _reaches(obj, study):
            continue
        names = gathered.setdefault(id(obj), (obj, []))[1]
        names.extend(getattr(chosen, "SubElementNames", ()) or [""])
    return [(obj, _whole_or_named(names)) for obj, names in gathered.values()]


def _whole_or_named(names: list[str]) -> list[str]:
    """One object's picks as one entry: ``[""]`` where it was picked whole,
    which covers every element of it, and its elements otherwise."""
    return [""] if "" in names else list(dict.fromkeys(names))


def reaching(selection: Iterable[Any], study: Any) -> list[str]:
    """The label of each object picked that reaches ``study``, which
    :func:`references_from` leaves out, so a command can say why."""
    return list(
        dict.fromkeys(
            str(obj.Label)
            for obj in (getattr(chosen, "Object", chosen) for chosen in selection)
            if _is_geometry(obj) and _reaches(obj, study)
        )
    )


def cycle_note(labels: Iterable[str]) -> str:
    """What a command says of the picks :func:`reaching` names."""
    labels = list(labels)
    named = ", ".join(repr(name) for name in labels)
    if len(labels) == 1:
        depends, it, was = "depends", "it", "it was"
    else:
        depends, it, was = "depend", "them", "they were"
    return (
        f"{named} {depends} on the study, so {was} left out of References: a link to "
        f"{it} from inside the study makes a cycle FreeCAD cannot recompute"
    )


def _reaches(obj: Any, study: Any) -> bool:
    """Whether ``obj`` depends on ``study``.

    An object reaches a study through more than a group: an ``App::LinkGroup``
    holds its members as links, a ``Part::Compound`` names its parts, and an
    expression reads a property. ``OutListRecursive`` lists everything an
    object reaches through any of those, in other documents as well.
    """
    return any(each is study for each in getattr(obj, "OutListRecursive", None) or ())


def _is_geometry(obj: Any) -> bool:
    """Something a user drew, rather than something this workbench made.

    The ``Shape`` test alone is not enough. ``EMMeshPreview`` is a
    ``Part::FeaturePython``, so it has a shape, and it is a group member like
    any other. A ``PartDesign::Body`` and an ``App::Part`` carry a ``Group``
    beside their ``Shape``, and are drawn geometry.
    """
    from .kinds import is_ours

    return not is_ours(obj) and getattr(obj, "Shape", None) is not None


def createEMMeshRegion(doc: Any = None, settings: Any = None) -> Any:
    """A refinement region amending ``settings``, unowned. The analysis files it
    away."""
    doc = doc or FreeCAD.ActiveDocument
    obj = doc.addObject("App::FeaturePython", "EMMeshRegion")
    EMMeshRegion(obj)
    obj.Label = "Mesh Refinement"
    obj.MeshSettings = settings

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMeshRegion")
    return obj
