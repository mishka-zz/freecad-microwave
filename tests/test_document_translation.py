# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the adapter reads out of a FreeCAD document.

The fakes here are deliberately not FreeCAD stubs. The translation imports no
FreeCAD, so the contract it actually depends on is small - properties by
attribute, ``Shape.BoundBox``, ``Shape.Volume``, ``Shape.getElement`` - and
these objects implement exactly that and nothing else. A stub that reimplemented
FreeCAD would let the translation start depending on things this module has no
business knowing, and the clean-interpreter test would be the only thing left
holding the line.

The geometry is a real 50 ohm microstrip, the same one the acceptance gate
solves. That matters: assertions here compare against numbers with a physical
meaning rather than against whatever the code produced when it was written.
"""

from __future__ import annotations

import dataclasses
import math
import sys

import numpy as np
import pytest

from Microwave import portbox, units
from Microwave.Objects.ports import FIXED_IMPEDANCE, PORT_IMPEDANCE
from Microwave.Solvers.openems import document, geometry, materials, ports, properties
from Microwave.Solvers.openems.model import SPEED_OF_LIGHT, THROUGH, origin_offset
from Microwave.Solvers.openems.sizing import fits_inside

# The acceptance line: a 3 mm trace on 1.6 mm FR4, ground underneath.
WIDTH = 3.0
HEIGHT = 1.6
LENGTH = 100.0
BOARD = 30.0
EPS_R = 4.4


# ---------------------------------------------------------------------------
# Fakes: the whole contract the translation has with FreeCAD
# ---------------------------------------------------------------------------


class BoundBox:
    def __init__(self, lower, upper):
        (self.XMin, self.YMin, self.ZMin) = lower
        (self.XMax, self.YMax, self.ZMax) = upper


class Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class Vertex:
    def __init__(self, point):
        self.Point = point


class Edge:
    """One edge, measured. ``Length`` is the arc, so a fake curve is a fake
    whose ``Length`` exceeds its chord - which is the only thing that tells the
    translation a curve from a straight line."""

    def __init__(self, start, end, length=None):
        self.Vertexes = [Vertex(start), Vertex(end)]
        chord = math.dist((start.x, start.y, start.z), (end.x, end.y, end.z))
        self.Length = chord if length is None else length


class _Circle:
    """One circular boundary, in the shape ``Microwave.annulus`` reads it."""

    def __init__(self, radius, centre):
        self.Radius = radius
        self.Center = Point(*centre)

    def wire(self):
        return type("Wire", (), {"Edges": [type("Edge", (), {"Curve": self})()]})()


class Shape:
    """A box-shaped shape. ``fill`` below 1 makes it something else.

    ``rings`` gives it *corners*: closed rings of ``(u, v)`` in the plane of a
    flat shape, from which ``Area`` and ``Edges`` are both derived, so the two
    cannot disagree the way ``fill`` and a hand-written area could. A shape that
    has to be cut up needs corners, because a fill fraction says how much of the
    box is covered and never says where.

    ``area`` overrides what the rings add up to, and exists for the one shape
    they cannot describe between them: a face whose rings lie over one another,
    where the kernel reports the area they cover in total and the outline
    encloses the shared part only once.
    """

    def __init__(
        self,
        lower,
        upper,
        fill=1.0,
        rings=None,
        area=None,
        coarsens=0.0,
        stubborn=False,
        wobble=0.0,
        solid=None,
        unclosed=False,
        offsets=True,
        swells=1.0,
    ):
        self.offsets = offsets
        self.swells = swells
        self.BoundBox = BoundBox(lower, upper)
        self.coarsens = coarsens
        self.stubborn = stubborn
        self.wobble = wobble
        self.unclosed = unclosed
        # Shared with every copy, so a test can see what the translator asked
        # for even though it asks a copy rather than this shape.
        self._record = {"deflection": None}
        self._last_deflection = 0.0
        self._faces = {}
        self.Edges = []
        extents = [b - a for a, b in zip(lower, upper)]
        flat = [d for d in range(3) if extents[d] <= geometry.FLATNESS]
        if rings is not None:
            self.Volume = 0.0
            enclosed = sum(
                abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:] + ring[:1]))) / 2.0
                for ring in rings
            )
            self.Area = enclosed if area is None else area
            axes = [d for d in range(3) if d != flat[0]]
            for ring in rings:
                for start, end in zip(ring, ring[1:] + ring[:1]):
                    corners = []
                    for point in (start, end):
                        place = [0.0, 0.0, 0.0]
                        place[flat[0]] = lower[flat[0]]
                        place[axes[0]], place[axes[1]] = point
                        corners.append(Point(*place))
                    self.Edges.append(Edge(*corners))
        elif flat and not solid:
            self.Area = math.prod(e for d, e in enumerate(extents) if d not in flat) * fill
            self.Volume = 0.0
        else:
            self.Volume = math.prod(extents) * fill
            self.Area = 2 * sum(extents[a] * extents[b] for a, b in ((0, 1), (0, 2), (1, 2)))

        # Whether a shape bounds a volume is its topology, which the kernel
        # answers separately from any measurement: a face carries no solid
        # however wide it is, and a solid rolled down to a foil still carries
        # one. ``solid`` is how a test says so for a shape too thin to be told
        # apart from a sheet by its extents alone.
        self.Solids = [] if (self.Volume == 0.0 and not solid) else [self]
        # No faces, because these shapes carry no surface: a face here would be
        # asked for its parameters the moment a conductor was bound to it, and
        # answering that is the kernel's job. What reads this is the count a
        # compound compares against its solids' to find a face belonging to
        # none of them, and two empty counts answer it correctly for a compound
        # made of nothing else. A compound with a loose face in it is beyond
        # what these can say, and is a corpus specimen.
        self.Faces = []
        # The boundary the area sits inside, which a thickness is judged
        # against. A box's twelve edges, since that is the shape this is.
        self.Length = 4.0 * sum(extents)

    @property
    def asked_deflection(self):
        return self._record["deflection"]

    def makeOffsetShape(self, offset, tolerance, inter=False, join=0, fill=False):
        """The kernel's thicken, as a solid holding a skin's worth of material.

        A real one sweeps the surface through ``offset``, so what it encloses is
        the area times the thickness, give or take what the surface's own
        curvature adds at the rim. That figure is the whole of what the
        translator reads back off the result, so it is what this carries; the
        bounds grow on every side because a box has no side to prefer, and the
        direction a real offset runs in is scored against the kernel in the
        corpus rather than here.

        ``swells`` is how a test asks for an offset that ran away with the shape,
        and ``offsets`` for a kernel that would not build one at all - a fold
        tighter than the thickness, which nothing built out of boxes can be
        shaped like.
        """
        if not self.offsets:
            raise RuntimeError("BRepOffsetAPI_MakeOffsetShape not done")
        self._record["offset"] = offset
        grown = Shape(
            tuple(getattr(self.BoundBox, low) - offset for low in ("XMin", "YMin", "ZMin")),
            tuple(getattr(self.BoundBox, high) + offset for high in ("XMax", "YMax", "ZMax")),
            solid=True,
        )
        grown.Volume = self.Area * offset * self.swells
        return grown

    @property
    def asked_offset(self):
        return self._record.get("offset")

    def hashCode(self):
        """What the kernel offers to group shapes by identity. Object identity
        here, which is exact - so the confirming ``isSame`` a real hash needs is
        never reached, and the shells that would reach it are a corpus
        specimen."""
        return id(self)

    def copy(self):
        """A shape carrying no triangulation of its own.

        The real one matters: a shape that has been displayed hands back its
        view provider's mesh from ``tessellate``, so the translator works on a
        copy and the geometry that gets solved stops depending on a display
        setting. Here the copy shares this shape's record, so what it was asked
        for is still observable.
        """
        twin = Shape.__new__(Shape)
        twin.__dict__.update(self.__dict__)
        return twin

    def distToShape(self, other):
        """The gap between two boxes, with the pair of points that realise it.

        Between boxes this is what the kernel returns, so a stand-in can be
        exact rather than approximate: the nearest points differ per axis by
        whichever way the boxes miss each other, and coincide on any axis where
        they overlap.
        """
        near, far = [], []
        for low, high in (("XMin", "XMax"), ("YMin", "YMax"), ("ZMin", "ZMax")):
            mine = (getattr(self.BoundBox, low), getattr(self.BoundBox, high))
            theirs = (getattr(other.BoundBox, low), getattr(other.BoundBox, high))
            if mine[1] < theirs[0]:
                near.append(mine[1])
                far.append(theirs[0])
            elif theirs[1] < mine[0]:
                near.append(mine[0])
                far.append(theirs[1])
            else:
                overlap = (max(mine[0], theirs[0]) + min(mine[1], theirs[1])) / 2
                near.append(overlap)
                far.append(overlap)
        return math.dist(near, far), [(Point(*near), Point(*far))], []

    def _flat_tessellation(self, low, high, flat):
        """A flat shape's triangles, covering exactly its own area.

        Two triangles over a rectangle scaled to match ``Area``, laid in the
        shape's own plane. A real tessellator returns the area the outline
        encloses with any holes left out; what matters to the translator is that
        the triangles are coplanar and add up to the area, which this gives
        exactly and a box's surface does not - a box's would count both sides.
        """
        across = [d for d in range(3) if d != flat]
        width = high[across[0]] - low[across[0]]
        asked = self._record["deflection"] if self.stubborn else self._last_deflection
        area = self.Area * (1.0 - self.coarsens * asked)
        height = area / width if width > 0 else 0.0
        middle = 0.5 * (low[across[1]] + high[across[1]])

        def at(u, v):
            place = [0.0, 0.0, 0.0]
            # ``wobble`` puts the triangles a hair off the plane, which is what a
            # kernel returns for a face that is flat to within its tolerance
            # rather than exactly. A sheet is modelled at one plane, so what the
            # translator does with that hair is the whole question.
            place[flat] = low[flat] + self.wobble
            place[across[0]] = u
            place[across[1]] = v
            return Point(*place)

        corners = [
            at(low[across[0]], middle - height / 2),
            at(high[across[0]], middle - height / 2),
            at(high[across[0]], middle + height / 2),
            at(low[across[0]], middle + height / 2),
        ]
        return corners, [(0, 1, 2), (0, 2, 3)]

    def tessellate(self, deflection):
        """A closed triangulation of this shape, at the fineness asked for.

        A box shrunk about its own centre until its volume matches, which is a
        genuine closed surface with a genuine volume - so the translator's
        closedness check and its volume comparison both see something real
        rather than a value handed to them. What it is not is a triangulation of
        the shape ``fill`` describes, because a fill fraction says how much of
        the box is covered and never says where.

        ``coarsens`` makes it lose volume in proportion to the fineness it was
        asked for, the way a real tessellator does. Without it the requested
        deflection would reach nothing and the volume comparison could not fail,
        so neither could be tested at all. ``stubborn`` is the other half of the
        same subject: it answers every later request with the first one, which
        is what a tessellator treating the request as a hint does, and is the
        only way the refusal is reachable.
        """
        low = [self.BoundBox.XMin, self.BoundBox.YMin, self.BoundBox.ZMin]
        high = [self.BoundBox.XMax, self.BoundBox.YMax, self.BoundBox.ZMax]
        if self._record["deflection"] is None or not self.stubborn:
            self._record["deflection"] = deflection
        self._last_deflection = deflection
        flat = [d for d in range(3) if high[d] - low[d] <= geometry.FLATNESS]
        if flat:
            return self._flat_tessellation(low, high, flat[0])
        box = math.prod(b - a for a, b in zip(low, high))
        # ``stubborn`` returns the same coarse mesh whatever is asked for,
        # which is what a tessellator that treats the request as a hint does.
        asked = self._record["deflection"] if self.stubborn else deflection
        volume = self.Volume * (1.0 - self.coarsens * asked)
        # By magnitude, because a shape bounding no region reports whatever the
        # kernel computes across its faces and that has a sign of its own. A
        # tessellator is handed the faces and never the figure, so the sign
        # cannot reach the triangles it returns.
        scale = (abs(volume) / box) ** (1 / 3) if box > 0 else 1.0
        middle = [(a + b) / 2 for a, b in zip(low, high)]
        corners = [
            Point(*[m + scale * (c - m) for m, c in zip(middle, corner)])
            for corner in (
                (low[0], low[1], low[2]),
                (high[0], low[1], low[2]),
                (high[0], high[1], low[2]),
                (low[0], high[1], low[2]),
                (low[0], low[1], high[2]),
                (high[0], low[1], high[2]),
                (high[0], high[1], high[2]),
                (low[0], high[1], high[2]),
            )
        ]
        facets = [
            (0, 3, 2),
            (0, 2, 1),  # bottom
            (4, 5, 6),
            (4, 6, 7),  # top
            (0, 1, 5),
            (0, 5, 4),  # front
            (1, 2, 6),
            (1, 6, 5),  # right
            (2, 3, 7),
            (2, 7, 6),  # back
            (3, 0, 4),
            (3, 4, 7),  # left
        ]
        # ``unclosed`` drops a triangle, leaving the surface open along the
        # edges it held. What that stands in for is a shell somebody left open:
        # the hole is in the shape's own surface rather than in the mesh taken
        # from it, which is the distinction the refusal turns on.
        return corners, facets[:-1] if self.unclosed else facets

    def face(self, name, lower, upper, area=True):
        """Register a named sub-shape. A face is a Shape too: the translation
        validates whatever a binding names, so it needs area and volume.

        It answers ``Faces`` with itself, as the kernel does for a face picked
        by name, since what a pick encloses is read off that. ``area=False``
        registers an edge instead, which encloses nothing and answers none.
        """
        picked = Shape(lower, upper)
        picked.Faces = [picked] if area else []
        self._faces[name] = picked
        return self

    def ring(self, name, centre, inner, outer, flat):
        """Register an annular face: a box-shaped bound plus two circles.

        The circles are what :mod:`Microwave.annulus` reads, and they are the
        only place in these fakes where anything below a bounding box is
        offered - a coaxial port's inner radius is the one quantity a box
        cannot carry.
        """
        lower = list(centre)
        upper = list(centre)
        for axis in range(3):
            if axis == flat:
                continue
            lower[axis] -= outer
            upper[axis] += outer
        face = Shape(tuple(lower), tuple(upper))
        face.Wires = [_Circle(outer, centre).wire(), _Circle(inner, centre).wire()]
        self._faces[name] = face
        return self

    def getElement(self, name):
        return self._faces[name]


class Compound:
    """Several shapes in one, which is what a user drawing two lumps gets.

    The translation asks a compound what it holds - solids where the lumps are
    volumes, faces where they are sheets - and each lump is then decomposed and
    measured on its own.

    ``BoundBox`` spans them all, which is what makes a compound handed to the
    mesher *look* like a body: it has an extent, it answers a distance query,
    and the distance it answers between itself and itself is zero.
    """

    def __init__(self, *lumps):
        self.lumps = list(lumps)
        self.Area = sum(lump.Area for lump in lumps)
        self.Volume = sum(lump.Volume for lump in lumps)
        self.Solids = [solid for lump in lumps for solid in lump.Solids]
        # A lump carrying no solid is a face of this compound. Said here rather
        # than by the lump, so a shape still answers for no face of its own -
        # which is what keeps it from being asked a face's questions.
        self.Faces = [lump for lump in lumps if not lump.Solids] + [
            face for lump in lumps for face in lump.Faces
        ]
        corners = [lump.BoundBox for lump in lumps]
        self.BoundBox = BoundBox(
            tuple(min(getattr(box, low) for box in corners) for low in ("XMin", "YMin", "ZMin")),
            tuple(max(getattr(box, high) for box in corners) for high in ("XMax", "YMax", "ZMax")),
        )

    def distToShape(self, other):
        """The nearest approach over every pair of lumps.

        Which against *itself* is a lump against itself, and so zero - the
        answer that makes handing a compound to the mesher silent rather than
        wrong-looking.
        """
        theirs = getattr(other, "lumps", None) or [other]
        return min(
            (mine.distToShape(yours) for mine in self.lumps for yours in theirs),
            key=lambda answer: answer[0],
        )


class Obj:
    """A document object: a bag of properties, a proxy class, maybe a shape."""

    def __init__(self, kind, name, shape=None, **properties):
        self.Proxy = type(kind, (), {})()
        self.Name = name
        self.Label = name
        self.Document = None
        self.Shape = shape
        for key, value in properties.items():
            setattr(self, key, value)


class Analysis(Obj):
    """A study whose ``Group`` is a live view of what the document holds.

    A property rather than a list, so every fixture and test below goes on
    adding and removing objects through ``Document.Objects`` exactly as it did
    when ownership was by document scan - the change under test is that
    ``contents()`` now reads a *group*, not that these particular fixtures
    changed shape.

    Membership itself is tested where it belongs: ``TestOwnershipIsMembership``
    builds explicit groups, because that is the behaviour a live view cannot
    demonstrate.
    """

    OWNED = (
        "EMSolverOpenEMS",
        "EMMeshPolicy",
        "EMMaterialBinding",
        "EMMeshRegion",
        "EMMeshPreview",
    )

    def addObject(self, obj):
        """Group membership, the FreeCAD way. Idempotent, as FreeCAD's is."""
        if obj not in self.Document.Objects:
            obj.Document = self.Document
            self.Document.Objects.append(obj)

    @property
    def Group(self):
        document = self.Document
        if document is None:
            return []
        return [
            obj
            for obj in document.Objects
            if obj is not self
            and (
                type(obj.Proxy).__name__ in self.OWNED
                or type(obj.Proxy).__name__.startswith("EMPort")
            )
        ]


class Document:
    def __init__(self, *objects):
        self.Objects = list(objects)
        for obj in self.Objects:
            obj.Document = self
        #: As on ``conftest.DocumentStub``: a real document has these, and
        #: anything that groups an edit into one undo step calls them. Recording
        #: rather than ignoring, so "was this one transaction?" is answerable.
        self.transactions = []
        self._open = None

    def openTransaction(self, label):
        if self._open is not None:
            self.transactions.append((self._open, "commit"))
        self._open = label

    def commitTransaction(self):
        if self._open is not None:
            self.transactions.append((self._open, "commit"))
            self._open = None

    def abortTransaction(self):
        if self._open is not None:
            self.transactions.append((self._open, "abort"))
            self._open = None


# ---------------------------------------------------------------------------
# A complete, valid model
# ---------------------------------------------------------------------------


def substrate():
    return Obj(
        "Part::Box",
        "Substrate",
        Shape((-LENGTH / 2, -BOARD / 2, 0.0), (LENGTH / 2, BOARD / 2, HEIGHT)),
    )


def ground():
    plane = Shape((-LENGTH / 2, -BOARD / 2, 0.0), (LENGTH / 2, BOARD / 2, 0.0))
    plane.face("Face1", (-LENGTH / 2, -BOARD / 2, 0.0), (LENGTH / 2, BOARD / 2, 0.0))
    return Obj("Part::Plane", "Ground", plane)


def trace():
    strip = Shape((-LENGTH / 2, -WIDTH / 2, HEIGHT), (LENGTH / 2, WIDTH / 2, HEIGHT))
    # The end face the wave enters through: flat in x, spanning the trace width.
    strip.face("Face1", (-LENGTH / 2, -WIDTH / 2, HEIGHT), (-LENGTH / 2, WIDTH / 2, HEIGHT))
    return Obj("Part::Plane", "Trace", strip)


def fr4():
    return Obj(
        "EMMaterial",
        "FR4",
        MaterialType="Dielectric",
        Permittivity=EPS_R,
        Permeability=1.0,
        Conductivity=0.0,
        LossTangent=0.0,
        MeasuredAt=0.0,
        Thickness=0.0,
    )


def copper():
    return Obj(
        "EMMaterial",
        "Copper",
        MaterialType="ConductingSheet",
        Permittivity=1.0,
        Permeability=1.0,
        Conductivity=5.8e7,
        LossTangent=0.0,
        MeasuredAt=0.0,
        Thickness=0.035,
    )


def binding(name, material, *references):
    """Bind a material to whole objects. Empty sub-element list = the solid."""
    return Obj(
        "EMMaterialBinding",
        name,
        Material=material,
        References=[(obj, []) for obj in references],
    )


#: The acceptance gate's geometry, which is what the default fixture carries:
#: the source 20 mm in from the trace's end face, clear of the absorber, and the
#: probes 30 mm downstream of *it*, landing at the middle of the board. 30 mm is
#: 0.21 wavelengths at the bottom of this band in FR4 - see
#: ``portbox.CLEARANCE``.
FEED_OFFSET = 20.0
MEASUREMENT_DISTANCE = 30.0


def microstrip_port(number, trace_obj, ground_obj, **overrides):
    properties = dict(
        Number=number,
        Excitation=True,
        FeedResistance=0.0,
        FeedOffset=FEED_OFFSET,
        MeasurementDistance=MEASUREMENT_DISTANCE,
        ReferenceImpedance=50.0,
        ReferencedTo=FIXED_IMPEDANCE,
        Length=0.0,
        TraceEnd=(trace_obj, ["Face1"]),
        GroundReference=(ground_obj, ["Face1"]),
        PropagationAxis="X",
        ExcitationAxis="-Z",
    )
    properties.update(overrides)
    return Obj("EMPortMicrostrip", f"Port{number}", **properties)


def mesh_settings(**overrides):
    properties = dict(
        ElementsPerWavelength=20.0,
        EdgeRefinement=6.0,
        MaxGrowthRatio=1.3,
        MinElementsAcross=9,
        MinElementSize=0.0,
        **{f"AirCells{a}{s}": 8 for a in "XYZ" for s in ("Min", "Max")},
        PaddingXMin="Through",
        PaddingXMax="Through",
        PaddingYMin="Air",
        PaddingYMax="Air",
        PaddingZMin="Air",
        PaddingZMax="Air",
    )
    properties.update(overrides)
    return Obj("EMMeshPolicy", "MeshSettings", **properties)


def analysis(**overrides):
    """The study: the band, and ownership. No solver settings."""
    properties = dict(
        FrequencyStart=1e9,
        FrequencyStop=10e9,
        NumFrequencyPoints=201,
        Waveform="Gaussian",
        SmallestResponse=0.0,
    )
    properties.update(overrides)
    return Analysis("EMAnalysis", "Analysis", **properties)


def simulation(**overrides):
    """The openEMS solver: this backend's settings, and nothing neutral."""
    properties = dict(
        PMLCells=8,
        MaxTimesteps=14000,
        EnergyDecay=0.0,
        TimestepFactor=1.0,
        Threads=0,
    )
    for axis in ("X", "Y", "Z"):
        for side in ("Min", "Max"):
            properties[f"Boundary{axis}{side}"] = "PML"
    properties.update(overrides)
    return Obj("EMSolverOpenEMS", "Simulation", **properties)


def model(**overrides):
    """A complete microstrip document. Overrides replace whole objects.

    ``Objects[0]`` is the analysis, because that is the front door now, and the
    indices after it are unchanged so the positional edits below still name what
    they used to. The solver sits at the end for the same reason.
    """
    board, plane, strip = substrate(), ground(), trace()
    settings = overrides.pop("settings", None) or mesh_settings()
    port = overrides.pop("port", None) or microstrip_port(1, strip, plane)
    solver_obj = overrides.pop("simulation", None) or simulation()
    study = overrides.pop("analysis", None) or analysis()
    return Document(
        study,
        settings,
        board,
        plane,
        strip,
        binding("DielectricBinding", fr4(), board),
        binding("GroundBinding", copper(), plane),
        binding("TraceBinding", copper(), strip),
        port,
        solver_obj,
    )


def refinement(name, *references, **overrides):
    """An ``EMMeshRegion`` aimed at whole objects."""
    properties = dict(
        References=[(obj, []) for obj in references],
        Mode="Refine",
        ElementSize=0.05,
        MinElementsAcross=0,
        Enabled=True,
    )
    properties.update(overrides)
    return Obj("EMMeshRegion", name, **properties)


def part(doc, name):
    """The one object in ``doc`` called ``name``.

    Tests reached into ``part(doc, "GroundBinding")`` for the ground binding and
    ``Objects[4]`` for the trace, so inserting an object into ``model()`` - or
    reordering it - silently repointed a hundred assertions at their
    neighbours, and the failure would read as physics rather than as a shifted
    index. ``Objects[0]`` is left as it is: the analysis is the front door and
    ``model()`` says so.
    """
    for obj in doc.Objects:
        if obj.Name == name:
            return obj
    raise AssertionError(f"no object called {name!r} in {[obj.Name for obj in doc.Objects]}")


def removes(doc, name):
    """Take the object called ``name`` out of the document."""
    doc.Objects.remove(part(doc, name))


def replaces(doc, name, obj):
    """Swap the object called ``name`` for ``obj``, keeping its place."""
    doc.Objects[doc.Objects.index(part(doc, name))] = obj
    obj.Document = doc
    return obj


def l_shaped_trace():
    """A document whose trace is an L - flat, Manhattan, short of its own box.

    Three quarters of the bounding box, with one corner missing, so it is two
    rectangles and no fewer. A real L is measured against the kernel outside
    this suite.
    """
    doc = model()
    strip = part(doc, "Trace")
    corner = Shape(
        (-LENGTH / 2, -WIDTH / 2, HEIGHT),
        (LENGTH / 2, WIDTH / 2, HEIGHT),
        rings=[
            [
                (-LENGTH / 2, -WIDTH / 2),
                (LENGTH / 2, -WIDTH / 2),
                (LENGTH / 2, 0.0),
                (0.0, 0.0),
                (0.0, WIDTH / 2),
                (-LENGTH / 2, WIDTH / 2),
            ]
        ],
    )
    corner.face("Face1", (-LENGTH / 2, -WIDTH / 2, HEIGHT), (-LENGTH / 2, WIDTH / 2, HEIGHT))
    strip.Shape = corner
    return doc


def diagonal_trace():
    """A flat trace with a corner cut off it - axis-aligned everywhere but one
    edge, which is the whole of what cannot be laid on a rectilinear grid."""
    doc = model()
    strip = part(doc, "Trace")
    cut = Shape(
        (-LENGTH / 2, -WIDTH / 2, HEIGHT),
        (LENGTH / 2, WIDTH / 2, HEIGHT),
        rings=[
            [
                (-LENGTH / 2, -WIDTH / 2),
                (LENGTH / 2, -WIDTH / 2),
                (LENGTH / 2, 0.0),
                (0.0, WIDTH / 2),
                (-LENGTH / 2, WIDTH / 2),
            ]
        ],
    )
    cut.face("Face1", (-LENGTH / 2, -WIDTH / 2, HEIGHT), (-LENGTH / 2, WIDTH / 2, HEIGHT))
    strip.Shape = cut
    return doc


def _add(doc, obj):
    """Put another object in an existing document, as the toolbar would."""
    obj.Document = doc
    doc.Objects.append(obj)
    return obj


def translated_microstrip(number=1, **overrides):
    """The one translated port of a document whose microstrip port carries
    ``overrides``.

    ``strip, plane = trace(), ground()`` followed by
    ``document.problem(model(port=port).Objects[0])`` appeared twenty-one times
    in this file. The repetition is the *preamble*, not the assertions: what
    each of those tests is about is the one keyword it overrides and the one
    thing it then checks, and both were three lines down.
    """
    strip, plane = trace(), ground()
    port = microstrip_port(number, strip, plane, **overrides)
    return document.problem(model(port=port).Objects[0]).ports[0]


def refuses_a_microstrip(match, number=1, **overrides):
    """Assert that a microstrip port carrying ``overrides`` is refused by name.

    ``match`` is a phrase from the refusal, and it is the point: a refusal names
    the object and says what to do, so a test that only
    asserted "it raised" would accept a refusal that had stopped saying which.
    """
    strip, plane = trace(), ground()
    port = microstrip_port(number, strip, plane, **overrides)
    with pytest.raises(document.TranslationError, match=match):
        document.problem(model(port=port).Objects[0])


def _far_end_port(doc, number, **overrides):
    """A second port at the other end of the same line, facing back along it."""
    strip, plane = part(doc, "Trace"), part(doc, "Ground")
    strip.Shape.face("Face2", (LENGTH / 2, -WIDTH / 2, HEIGHT), (LENGTH / 2, WIDTH / 2, HEIGHT))
    return microstrip_port(
        number,
        strip,
        plane,
        TraceEnd=(strip, ["Face2"]),
        PropagationAxis="-X",
        **overrides,
    )


@pytest.fixture
def doc():
    return model()


@pytest.fixture
def study(doc):
    return doc.Objects[0]


def policy(study):
    """The mesh policy in a study, found the way ``contents()`` finds it."""
    return _member(study, "EMMeshPolicy")


def solver_of(study):
    """The openEMS solver in a study."""
    return _member(study, "EMSolverOpenEMS")


def _member(study, kind):
    return next(o for o in study.Group if type(o.Proxy).__name__ == kind)


# ---------------------------------------------------------------------------


class TestATranslatedProblemIsRunnable:
    """The output has to satisfy everything the adapter asserts downstream."""

    def test_the_whole_document_becomes_one_problem(self, study):
        problem = document.problem(study)

        assert {m.name for m in problem.materials} == {"FR4", "Copper"}
        assert {s.label for s in problem.solids} == {"Substrate", "Ground", "Trace"}
        assert problem.excited_port.number == 1
        assert problem.frequency.start == 1e9
        assert problem.boundary == ("PML_8",) * 6

    def test_the_grid_resolves_the_trace(self, study):
        """The mesh must actually be finer at the strip than in the bulk.

        Asserted against the wavelength the policy is written in terms of, not
        against a remembered millimetre value - the same discipline the
        acceptance gate follows, and for the same reason.
        """
        problem = document.problem(study)
        wavelength = SPEED_OF_LIGHT / 10e9 / math.sqrt(EPS_R) * 1e3

        assert problem.grid.params["dielectric_res"] == pytest.approx(wavelength / 20)
        assert problem.grid.params["metal_res"] == pytest.approx(wavelength / 120)
        assert problem.grid.cell_count > 0

    def test_it_survives_a_round_trip_through_json(self, study):
        """The envelope crosses a process boundary; a Problem that cannot be
        serialised and read back is not runnable however valid it looks."""
        from Microwave.Solvers.openems.model import Problem

        problem = document.problem(study)
        again = Problem.from_json(problem.to_json())
        assert again.digest() == problem.digest()

    def test_preflight_finds_nothing_to_refuse(self, study):
        """The strongest statement available without a solver.

        Pre-flight is where silent-wrong-answer bugs are caught, and it knows
        about grid coverage, absorber overlap, sheet thickness and probe
        spacing. A document that translates cleanly and then refuses itself
        would mean the translation produced something merely well-formed.
        """
        from Microwave.Solvers.openems import preflight

        findings = preflight.check(document.problem(study))
        assert preflight.refusals(findings) == []

    def test_a_feed_inside_the_absorber_is_caught(self):
        """Not by the translation - by pre-flight, which is where it belongs.

        With Through padding the domain is pulled in by the absorber's depth, so
        the first several millimetres of the trace lie inside the PML: the
        interior runs from -44.28 while the board starts at -50. A source 4 mm
        in from the end face lands there, and the field it would drive is being
        attenuated on purpose.

        The interior depends on what a THROUGH face reserves in - the
        substrate's cell, which is what the absorber is laid in, not a vacuum
        one. Change that and this offset moves with it; the case being tested
        does not.

        This is the case FeedOffset exists for. It is the one number no geometry
        can supply - how far past the absorber the source has to sit depends
        on the grid, which does not exist yet when the port is drawn - so the
        port states it and pre-flight, which does have the grid, checks it.
        """
        from Microwave.Solvers.openems import preflight

        strip, plane = trace(), ground()
        port = microstrip_port(1, strip, plane, FeedOffset=4.0, MeasurementDistance=27.2)
        findings = preflight.check(document.problem(model(port=port).Objects[0]))

        refused = preflight.refusals(findings)
        assert refused and "inside the absorber" in refused[0].message


class TestThePortCarriesItsDirection:
    """Corner ordering is the sign of the excitation, not a bounding box."""

    def test_the_excitation_runs_from_trace_to_ground(self, study):
        port = document.problem(study).ports[0]
        assert port.start[2] == pytest.approx(HEIGHT)
        assert port.stop[2] == pytest.approx(0.0)
        assert port.excite_sign == -1

    def test_the_port_reaches_into_the_structure(self, study):
        port = document.problem(study).ports[0]
        assert port.start[0] == pytest.approx(-LENGTH / 2)
        assert port.direction == 1

    def test_the_offsets_arrive_as_millimetres_from_the_picked_face(self, study):
        """The envelope wants both from the box start; only the document holds
        the probe distance relative to the source."""
        port = document.problem(study).ports[0]
        assert port.feed_shift == pytest.approx(FEED_OFFSET)
        assert port.measurement_shift == pytest.approx(FEED_OFFSET + MEASUREMENT_DISTANCE)

    def test_moving_the_feed_carries_the_probes_with_it(self):
        """The separation is the requirement, so it is what the property holds.

        Measured from the picked face instead, setting FeedOffset silently eats
        into the very distance the port exists to provide - pre-flight warns,
        but the filled-in default has stopped being the default and nothing says
        so.
        """
        port = translated_microstrip(FeedOffset=FEED_OFFSET + 10.0)
        assert port.measurement_shift - port.feed_shift == pytest.approx(MEASUREMENT_DISTANCE)

    def test_an_unset_length_ends_the_box_at_the_measurement_plane(self):
        """Not at the end of the trace.

        A box that stopped where the copper stopped could never show that the
        copper was too short - it moved to fit, every time. Nothing past the
        probes is ever read, so there is nothing to reach for beyond them.
        """
        span = FEED_OFFSET + MEASUREMENT_DISTANCE
        translated = translated_microstrip(Length=0.0)

        assert translated.length == pytest.approx(span)
        assert translated.start[0] == pytest.approx(-LENGTH / 2)
        assert translated.stop[0] == pytest.approx(-LENGTH / 2 + span)

    def test_the_box_is_the_requirement_not_a_reading_of_the_trace(self):
        """A port needing more line than was drawn keeps its size and says so
        by sticking out - the whole reason the box is drawn at all."""
        translated = translated_microstrip(MeasurementDistance=LENGTH)

        assert translated.length == pytest.approx(FEED_OFFSET + LENGTH)

    def test_a_measurement_distance_of_zero_is_refused_with_the_number(self):
        """The band knows what it should be; the drawn box does not, which is
        why this is a refusal rather than a default filled in here."""
        refuses_a_microstrip(r"least 30 mm", MeasurementDistance=0.0)

    def test_a_cramped_port_is_warned_about_rather_than_refused(self):
        """The near-field error is real but its boundary is only bracketed.

        A refusal would claim a threshold two measurements do not establish; a
        silent pass would let a 1.7% error through as an answer. The finding
        carries the separation it measured, so the user can judge.
        """
        from Microwave.Solvers.openems import preflight

        strip, plane = trace(), ground()
        port = microstrip_port(1, strip, plane, FeedOffset=13.6, MeasurementDistance=10.2)
        findings = preflight.check(document.problem(model(port=port).Objects[0]))

        warnings = [f for f in findings if f.severity == preflight.WARN]
        assert any("near field has not decayed" in f.message for f in warnings)
        assert preflight.refusals(findings) == []

    def test_the_trace_takes_its_conductor_from_the_document(self, study):
        """Not from a property on the port. Two places to state the same fact is
        two places for it to disagree."""
        assert document.problem(study).ports[0].metal == "Copper"


class TestItRefusesRatherThanGuesses:
    def shape_filling(self, fraction):
        """A document whose substrate covers ``fraction`` of its bounding box."""
        board = substrate()
        board.Shape = Shape(
            (-LENGTH / 2, -BOARD / 2, 0.0), (LENGTH / 2, BOARD / 2, HEIGHT), fill=fraction
        )
        doc = model()
        replaces(doc, "Substrate", board)
        part(doc, "DielectricBinding").References = [(board, [])]
        return doc

    def test_a_rotated_solid_is_carried_as_its_own_surface(self):
        """Judged by volume, so nothing depends on recognising a type.

        A rotated box has a bounding box like any other shape, and taking that
        box would solve a different object without saying so. What happens
        instead is that the shape's own boundary is sent, and the grid - which
        stays rectilinear - is what staircases.
        """
        problem = document.problem(self.shape_filling(0.71).Objects[0])
        board = next(s for s in problem.solids if s.label == "Substrate")
        assert board.is_mesh
        assert board.faces

    def test_a_cylinder_is_carried_the_same_way(self):
        """pi/4 of its bounding box, and nothing about its type says so."""
        problem = document.problem(self.shape_filling(math.pi / 4).Objects[0])
        assert next(s for s in problem.solids if s.label == "Substrate").is_mesh

    def test_a_box_is_still_a_box(self):
        """The cheap path stays cheap: a shape that fills its box carries no
        triangles, and openEMS gets the primitive it has always got."""
        problem = document.problem(self.shape_filling(1.0).Objects[0])
        board = next(s for s in problem.solids if s.label == "Substrate")
        assert not board.is_mesh
        assert board.faces == ()

    def test_a_solid_thinner_than_the_flatness_floor_is_judged_as_a_volume(self):
        """A foil is a solid, and a solid's area counts both of its faces.

        Judging one by area therefore compares two faces against the extent of a
        single face and cannot agree however thin the shape gets, so the shape
        would be refused for a fault it does not have. What decides is whether
        the kernel says it bounds a volume.
        """
        board = substrate()
        board.Shape = Shape(
            (-LENGTH / 2, -BOARD / 2, 0.0),
            (LENGTH / 2, BOARD / 2, geometry.FLATNESS),
            solid=True,
        )
        pieces = geometry.solid_boxes(board)

        assert len(pieces) == 1
        piece = pieces[0]
        assert not piece.faces, "a shape the grid holds exactly was triangulated"
        assert piece.sheet_normal is None, "a thin solid was flattened into a sheet"
        assert piece.box.upper[2] - piece.box.lower[2] == pytest.approx(
            geometry.FLATNESS, rel=1e-9, abs=0.0
        )

    def test_the_surface_that_is_sent_is_the_one_that_was_measured(self):
        """The corners follow the triangulation, not the exact shape.

        A ``BoundBox`` bounds the true surface, which lies marginally outside a
        triangulation that chords across every curve - and the corners are what
        the domain, the absorber's reservation and every pre-flight check are
        measured against, so they have to describe what is actually sent.
        """
        problem = document.problem(self.shape_filling(0.71).Objects[0])
        board = next(s for s in problem.solids if s.label == "Substrate")
        for dim in range(3):
            assert board.lower[dim] == pytest.approx(min(v[dim] for v in board.vertices))
            assert board.upper[dim] == pytest.approx(max(v[dim] for v in board.vertices))

    def test_an_l_is_cut_into_rectangles_rather_than_refused(self):
        """An L has no rotation in it and every edge axis-aligned, so a
        rectilinear grid holds it exactly - once it is in two pieces. Cutting it
        here is what lets a layout be drawn as a layout instead of assembled
        out of primitives.

        Two, and the count is the contract rather than an incidental figure: a
        third rectangle is a third seam, and seams are grid the user pays for on
        every timestep.
        """
        problem = document.problem(l_shaped_trace().Objects[0])
        pieces = [solid for solid in problem.solids if solid.label.startswith("Trace")]
        assert len(pieces) == 2

        laid = sum(
            (solid.upper[0] - solid.lower[0]) * (solid.upper[1] - solid.lower[1])
            for solid in pieces
        )
        assert laid == pytest.approx(0.75 * LENGTH * WIDTH, rel=1e-9, abs=0.0)

    def test_every_rectangle_of_a_cut_names_the_outline_it_came_from(self):
        """They tile it, so the shape they were cut from is the geometry each of
        them is nearest to everything outside - and a piece naming nothing is a
        body the mesher cannot ask a distance of."""
        trace = part(l_shaped_trace(), "Trace")
        pieces = geometry.solid_boxes(trace)
        assert [piece.shape for piece in pieces] == [trace.Shape] * len(pieces)

    def test_the_pieces_of_one_shape_are_told_apart_by_name(self):
        """They reach the user in mesh reports and in refusals, and two of them
        under one name reads as one object bound twice - which is a fault the
        pre-flight check for coincident solids exists to catch."""
        problem = document.problem(l_shaped_trace().Objects[0])
        labels = [solid.label for solid in problem.solids if solid.label.startswith("Trace")]
        assert sorted(labels) == ["Trace#1", "Trace#2"]

    def test_pieces_that_overlap_are_refused_by_the_area_they_do_not_add_up_to(self):
        """Even-odd reads an overlap as *outside*, so a compound of two faces
        laid over each other cuts into less metal than either the drawing or the
        kernel says is there. Nothing about the edges is wrong, which is why the
        area has to be the thing that catches it.
        """
        doc = model()
        strip = part(doc, "Trace")
        strip.Shape = Shape(
            (-LENGTH / 2, -WIDTH / 2, HEIGHT),
            (LENGTH / 2, WIDTH / 2, HEIGHT),
            rings=[
                [
                    (-LENGTH / 2, -WIDTH / 2),
                    (LENGTH / 4, -WIDTH / 2),
                    (LENGTH / 4, WIDTH / 2),
                    (-LENGTH / 2, WIDTH / 2),
                ],
                [
                    (-LENGTH / 4, -WIDTH / 2),
                    (LENGTH / 2, -WIDTH / 2),
                    (LENGTH / 2, WIDTH / 2),
                    (-LENGTH / 4, WIDTH / 2),
                ],
            ],
        )
        strip.Shape.face(
            "Face1", (-LENGTH / 2, -WIDTH / 2, HEIGHT), (-LENGTH / 2, WIDTH / 2, HEIGHT)
        )
        with pytest.raises(document.TranslationError) as refusal:
            document.problem(doc.Objects[0])
        assert "laid over one another" in str(refusal.value)

    def test_a_flat_shape_with_one_diagonal_edge_is_carried_as_an_area(self):
        """The cut into rectangles is exact or it does not happen - a best fit
        there is the silent approximation that check exists to prevent. What an
        outline off the axes gets instead is its own area, as flat polygons, so
        nothing is approximated and nothing is refused.
        """
        problem = document.problem(diagonal_trace().Objects[0])
        trace = next(s for s in problem.solids if s.label.startswith("Trace"))
        assert trace.is_sheet
        assert trace.faces

    def test_and_it_is_flat_on_the_axis_it_was_drawn_flat_on(self):
        """Which is the plane openEMS will look for it at. A sheet found at no
        plane is not modelled at all."""
        problem = document.problem(diagonal_trace().Objects[0])
        trace = next(s for s in problem.solids if s.label.startswith("Trace"))
        assert trace.lower[trace.sheet_normal] == trace.upper[trace.sheet_normal]

    def test_a_shape_a_percent_short_of_its_box_is_not_squared_off(self):
        """The tolerance is for float arithmetic, not for a close enough fit.

        A shape that misses its box by a per cent is a per cent of geometry
        nobody drew, and rounding it up to the box is the silent change this
        check exists to prevent. It goes down the same path as a sphere.
        """
        problem = document.problem(self.shape_filling(0.99).Objects[0])
        assert next(s for s in problem.solids if s.label == "Substrate").is_mesh

    def test_a_port_kind_this_adapter_cannot_build_is_refused_by_name(self):
        """The refusal that names the kind, and it is reachable.

        An empty port class that exists only to be refused here is not worth
        keeping - no command can create one and every property on it reaches
        nothing. What has to survive without them is this: the document may hold
        a port kind the *openEMS* adapter cannot build, and the day a second
        adapter grows one, this is the message that says so.
        """
        port = Obj(
            "EMPortSomethingElse",
            "Exotic",
            Number=1,
            Excitation=True,
            ReferenceImpedance=50.0,
            ReferencedTo=FIXED_IMPEDANCE,
            Length=0.0,
        )
        with pytest.raises(document.TranslationError, match="no openEMS builder"):
            document.problem(model(port=port).Objects[0])

    @pytest.mark.parametrize("impedance", [0.0, -50.0])
    def test_a_port_impedance_the_envelope_refuses_names_the_port(self, impedance):
        """As a TranslationError, not the EnvelopeError underneath.

        ``model.py`` refuses ``ReferenceImpedance = 0`` on its own, as an
        EnvelopeError. Left as one it reaches the panel through "Internal
        error", which catches TranslationError and MeshError and sends
        everything else there - so a number the user can go and change arrives
        as a traceback, and a traceback reads as a bug in the workbench. The
        refusal being right and the channel wrong is the worse half.
        """
        doc = model()
        part(doc, "Port1").ReferenceImpedance = impedance
        with pytest.raises(document.TranslationError) as raised:
            document.problem(doc.Objects[0])
        message = str(raised.value)
        assert "Port1" in message, "it should name the object to fix"
        # The property editor's spelling, not the envelope's. A message saying
        # "reference_impedance" sends a user looking for a property that is not
        # there; this module already carries one scar from that - see the
        # AXIS_NAMES import comment.
        assert "ReferenceImpedance" in message
        assert "reference_impedance" not in message

    def test_a_model_drawn_below_the_origin_is_translated_rather_than_refused(self):
        """Where a device sits in the CAD document is not a fault in it.

        The whole model here is below the origin on every axis, with a solid in
        it that reaches openEMS as triangles. It translates, and what the driver
        will hand the engine has no negative coordinate in it.
        """
        doc = model()
        board = part(doc, "Substrate")
        board.Shape = Shape((-LENGTH, -BOARD, -HEIGHT), (-1.0, -1.0, -HEIGHT / 2), fill=0.5)
        problem = document.problem(doc.Objects[0])
        assert any(solid.is_mesh for solid in problem.solids)
        assert min(min(solid.lower) for solid in problem.solids) < 0.0, (
            "the envelope keeps the coordinates the user drew in"
        )
        placed, offset = problem.at_the_origin()
        assert offset != (0.0, 0.0, 0.0)
        assert min(min(placed.grid[dim]) for dim in range(3)) == pytest.approx(0.0, abs=0.0)
        assert min(min(solid.lower) for solid in placed.solids) >= 0.0

    def test_a_port_referenced_to_itself_states_no_impedance(self):
        """``None`` in the envelope, which is what this adapter's absence means:
        it renormalises nothing, so an unstated reference is the basis the
        probes already measure in."""
        doc = model()
        part(doc, "Port1").ReferencedTo = PORT_IMPEDANCE

        (port,) = document.problem(doc.Objects[0]).ports
        assert port.reference_impedance is None

    def test_the_hidden_number_is_not_read_even_when_it_is_nonsense(self):
        """The editor hides ``ReferenceImpedance`` in this mode, so refusing on
        it would name a property the user cannot see - and a port whose number
        was left at zero before switching would be unusable for good."""
        doc = model()
        part(doc, "Port1").ReferencedTo = PORT_IMPEDANCE
        part(doc, "Port1").ReferenceImpedance = 0.0

        (port,) = document.problem(doc.Objects[0]).ports
        assert port.reference_impedance is None

    def test_the_adapter_and_the_document_spell_it_the_same_way(self):
        """The value is written out twice: this module imports no FreeCAD and
        ``Objects.ports`` is a document object, so neither can import the other.
        A disagreement would leave every port silently referenced to a number.
        """
        assert ports._PORT_IMPEDANCE == PORT_IMPEDANCE

    @pytest.mark.parametrize(
        "obj_name, prop, value, names",
        [
            ("Analysis", "NumFrequencyPoints", 1, "Analysis"),
            ("Simulation", "MaxTimesteps", 0, "Simulation"),
            ("Simulation", "Threads", -1, "Threads"),
            ("DielectricBinding", "Permittivity", 0.5, "FR4"),
            ("DielectricBinding", "Permeability", 0.0, "FR4"),
            ("TraceBinding", "Conductivity", -1.0, "Copper"),
            ("TraceBinding", "Thickness", -0.1, "Copper"),
            ("Port1", "FeedResistance", -1.0, "FeedResistance"),
            ("Port1", "FeedResistance", float("nan"), "FeedResistance"),
            ("Port1", "FeedResistance", float("inf"), "FeedResistance"),
        ],
    )
    def test_every_property_that_reaches_the_envelope_is_named(self, obj_name, prop, value, names):
        """One traceback fixed is one property fixed, and there are six more.

        Every row here is a separate route, so a fix covering three of them
        cannot claim the rest. Two of the material rows reach the user through
        *Update Mesh* rather than
        Check, which is a path with an even thinner catch: the toolbar command
        has no catch-all at all.
        """
        doc = model()
        target = doc.Objects[0] if obj_name == "Analysis" else part(doc, obj_name)
        # A material is reached through its binding; it is not a top-level
        # object in the document the way a port or the solver is.
        target = getattr(target, "Material", target)
        setattr(target, prop, value)
        with pytest.raises(document.TranslationError) as raised:
            document.problem(doc.Objects[0])
        message = str(raised.value)
        assert names in message, "it should name the object to fix"
        assert message.count(names) == 1, (
            "the envelope's own message already names a material, so prefixing "
            "one gave \"'FR4': material 'FR4': ...\""
        )

    @pytest.mark.parametrize("prop", ["Permittivity", "Permeability"])
    def test_a_permittivity_on_a_conductor_is_named_not_ignored(self, prop):
        """The QA step that found this set ``Permittivity = 20`` on Copper and
        pressed Update Mesh. It meant nothing to openEMS - ``build_material``
        gives a conductor its conductivity and nothing else - but it reached
        ``_wavelength``, which sizes every cell in the model from
        ``max(epsilon * mu)``, so the grid moved +30.5% and the run was then
        refused by a THROUGH guard for a reason that named neither the material
        nor the property. It has to arrive as this material's problem.
        """
        doc = model()
        # Both of them: ``model()`` calls ``copper()`` twice, so editing one
        # leaves two materials labelled 'Copper' that differ, and the duplicate
        # -label check fires first - with a message that also says "Copper".
        # Matched on text only this refusal has, for the same reason.
        for binding in ("GroundBinding", "TraceBinding"):
            setattr(part(doc, binding).Material, prop, 20.0)
        with pytest.raises(document.TranslationError, match="conductor carries relative"):
            document.mesh(doc.Objects[0])

    def test_a_material_fault_is_named_on_the_mesh_path_too(self):
        """``document.mesh`` never builds a Problem, so nothing downstream of
        it would have converted this. It is the *Update Mesh* button."""
        doc = model()
        part(doc, "DielectricBinding").Material.Permittivity = 0.5
        with pytest.raises(document.TranslationError, match="FR4"):
            document.mesh(doc.Objects[0])

    def test_only_the_envelope_s_own_errors_are_converted(self):
        """The catch is narrow on purpose, and nothing else pinned it.

        Widening ``except EnvelopeError`` to ``except Exception`` passed the
        whole suite twice: once when this test did not exist, and again when a
        raising it from a *port builder* would no longer be
        inside a ``_model_fault`` block at all. It has to raise from inside one,
        so ``Material`` is the lever. Widened, this comes back as "cannot
        translate this model" with the traceback discarded - the same failure
        this change exists to stop, pointing the other way.
        """
        doc = model()
        original = materials.Material

        def explode(*args, **kwargs):
            raise RuntimeError("an adapter bug, not a property")

        materials.Material = explode
        try:
            with pytest.raises(RuntimeError, match="an adapter bug"):
                document.problem(doc.Objects[0])
        finally:
            materials.Material = original

    @pytest.mark.parametrize("mode", ["TM11", "TE00", "", "rubbish"])
    def test_a_waveguide_mode_the_adapter_cannot_run_names_the_port(self, mode):
        """The third of the three values a user can put into a port. Checked in
        the translator as well as the envelope, because only the translator's
        copy reaches the panel as the model's problem rather than a traceback;
        ``model.check_mode`` is the one rule both of them ask."""
        doc = waveguide_model()
        part(doc, "WG1").Mode = mode
        with pytest.raises(document.TranslationError) as raised:
            document.problem(doc.Objects[0])
        assert "WG1" in str(raised.value), "it should name the object to fix"

    def test_a_dispersive_material_is_refused_not_flattened(self):
        doc = model()
        part(doc, "DielectricBinding").Material.MaterialType = "FrequencyDependentDielectric"
        with pytest.raises(document.TranslationError, match="dispersive"):
            document.problem(doc.Objects[0])

    def test_an_empty_binding_is_refused(self):
        doc = model()
        part(doc, "TraceBinding").References = []
        with pytest.raises(document.TranslationError, match="binds .* to nothing"):
            document.problem(doc.Objects[0])

    def test_a_trace_with_no_material_is_refused(self):
        """A port whose conductor the document never states would otherwise be
        laid in whatever material happened to be first.

        The binding is *removed*, not emptied. Emptying it trips the binding
        check first, so the port's own refusal is never reached - which is
        exactly how this branch went untested while looking covered.
        """
        doc = model()
        removes(doc, "TraceBinding")
        with pytest.raises(document.TranslationError, match="no material bound"):
            document.problem(doc.Objects[0])

    def test_a_trace_bound_to_a_dielectric_is_refused(self):
        doc = model()
        part(doc, "TraceBinding").Material = fr4()
        with pytest.raises(document.TranslationError, match="carries no current"):
            document.problem(doc.Objects[0])

    def test_a_reversed_excitation_axis_is_reported(self):
        """The failure this prevents is not a crash. A port driven backwards
        solves cleanly and returns the phase inverted."""
        refuses_a_microstrip("points the other way", ExcitationAxis="Z")

    def test_a_port_pointing_out_of_the_structure_is_reported(self):
        refuses_a_microstrip("out of the structure", PropagationAxis="-X")

    def test_the_wrong_face_of_the_trace_is_reported(self):
        """A face running along the trace instead of across its end."""
        strip, plane = trace(), ground()
        strip.Shape.face(
            "Side", (-LENGTH / 2, -WIDTH / 2, HEIGHT), (LENGTH / 2, -WIDTH / 2, HEIGHT)
        )
        port = microstrip_port(1, strip, plane, TraceEnd=(strip, ["Side"]))
        with pytest.raises(document.TranslationError, match="end face of the trace"):
            document.problem(model(port=port).Objects[0])

    def test_two_ports_with_one_number_are_refused(self):
        doc = model()
        doc.Objects.append(microstrip_port(1, part(doc, "Trace"), part(doc, "Ground")))
        with pytest.raises(document.TranslationError, match="ambiguous"):
            document.problem(doc.Objects[0])

    def test_a_port_with_no_number_is_refused(self):
        refuses_a_microstrip("number is unset", number=0)

    def test_no_excited_port_is_refused(self):
        refuses_a_microstrip("nothing would drive", Excitation=False)

    def test_a_second_solver_in_one_analysis_is_refused(self):
        """Two openEMS solvers in one study and nothing says which one runs.
        Refused rather than guessed at - picking the first would run settings
        the user did not choose and report them as theirs."""
        doc = model()
        doc.Objects.append(simulation())
        with pytest.raises(document.TranslationError, match="2 openEMS solvers"):
            document.problem(doc.Objects[0])

    def test_an_unsupported_boundary_is_refused(self):
        solver_obj = simulation(BoundaryZMax="Periodic")
        with pytest.raises(document.TranslationError, match="Periodic"):
            document.problem(model(simulation=solver_obj).Objects[0])

    def test_a_waveform_this_adapter_cannot_produce_is_refused(self):
        """``EMAnalysis`` offers one waveform, so this is unreachable from the
        property editor and reachable from a file: FreeCAD stores an
        enumeration's choices in the document and restores them from there, so
        a file written with a longer list goes on offering it.

        What it prevents is the quiet kind. Nothing carries a waveform into the
        envelope, ``SetGaussExcite`` is called either way, and the run would
        finish with an S-matrix that answers a question nobody asked.
        """
        with pytest.raises(document.TranslationError, match="Sinusoid"):
            document.problem(model(analysis=analysis(Waveform="Sinusoid")).Objects[0])


class TestPolicyMapsOntoThePhysics:
    def test_through_padding_pulls_the_domain_in(self, study):
        """The x faces run out through the absorber, so the grid must stop
        inside the board rather than beyond it. That is what makes the line
        infinite; without it the line ends in an open circuit and rings."""
        problem = document.problem(study)
        assert problem.grid.params["padding"][0] == [THROUGH, THROUGH]
        assert problem.grid.x[0] > -LENGTH / 2
        assert problem.grid.x[-1] < LENGTH / 2

    def test_air_padding_grows_the_domain_outward(self, study):
        problem = document.problem(study)
        assert problem.grid.y[0] < -BOARD / 2
        assert problem.grid.z[-1] > HEIGHT

    def test_a_conductor_walled_axis_gets_no_absorber_cells(self):
        """A waveguide's side walls are PEC, and meshing absorber cells outside
        them would move the walls - which moves the cutoff frequency."""
        solver_obj = simulation(
            BoundaryXMin="PEC",
            BoundaryXMax="PEC",
            BoundaryYMin="PEC",
            BoundaryYMax="PEC",
        )
        problem = document.problem(model(simulation=solver_obj).Objects[0])
        assert problem.grid.params["pml_cells"] == [0, 0, 8]

    def test_a_loss_tangent_becomes_a_conductivity(self):
        doc = model()
        part(doc, "DielectricBinding").Material.LossTangent = 0.02
        problem = document.problem(doc.Objects[0])

        fr4_material = next(m for m in problem.materials if m.name == "FR4")
        expected = (
            2 * math.pi * problem.frequency.center * materials.VACUUM_PERMITTIVITY * EPS_R * 0.02
        )
        assert fr4_material.kind == "lossy_dielectric"
        assert fr4_material.kappa == pytest.approx(expected)

    def test_where_the_loss_was_measured_travels_with_the_conductivity(self):
        """``kappa`` reproduces that loss tangent at band centre and nowhere
        else, so whether the two agree is a question about ``MeasuredAt``.

        It has to be asked below the GUI - the driver is handed an envelope
        and runs pre-flight on it - and the envelope is the only thing that
        crosses. A conductivity travelling without it is an approximation with
        no way left to say how good it is.
        """
        doc = model()
        material = part(doc, "DielectricBinding").Material
        material.LossTangent = 0.02
        material.MeasuredAt = 1e6
        problem = document.problem(doc.Objects[0])
        assert next(m for m in problem.materials if m.name == "FR4").measured_at == 1e6

    def test_a_loss_tangent_on_a_conductor_is_refused_rather_than_dropped(self):
        """A conductor's loss is its conductivity and its thickness, so only a
        dielectric's loss tangent becomes anything at all.

        Left alone it translates to a clean zero: the property the user typed
        reaches neither the envelope nor the engine, and nothing says so. That
        is a silent no-op, and switching MaterialType after
        typing one is how a document arrives in that state.
        """
        doc = model()
        part(doc, "GroundBinding").Material.LossTangent = 0.02
        with pytest.raises(document.TranslationError, match="loss tangent"):
            document.problem(doc.Objects[0])

    @pytest.mark.parametrize("value", [-0.02, float("nan"), float("-inf")])
    def test_a_loss_tangent_that_is_not_a_loss_is_refused(self, value):
        """The failure this prevents is silent and digest-identical.

        ``kappa`` is only computed when the loss tangent is above zero, so any
        other value left it at 0.0 - a lossless board, indistinguishable from
        one the user meant to be lossless, with plausible S-parameters and
        nothing to say a minus sign had slipped.
        """
        doc = model()
        part(doc, "DielectricBinding").Material.LossTangent = value
        with pytest.raises(document.TranslationError, match="loss tangent"):
            document.problem(doc.Objects[0])

    def test_energy_termination_stays_off_by_default(self, study):
        """Zero dB reads as a disabled feature and is the only reproducible
        setting."""
        assert document.problem(study).termination.reproducible

    def test_a_negative_decay_switches_it_on(self):
        solver_obj = simulation(EnergyDecay=-50.0)
        problem = document.problem(model(simulation=solver_obj).Objects[0])
        assert problem.termination.end_criteria == pytest.approx(1e-5)
        assert not problem.termination.reproducible

    def test_the_timestep_factor_reaches_the_envelope(self):
        """A property reaching nothing is a silent no-op,
        and this is the step of the journey with nothing at either end of
        it."""
        solver_obj = simulation(TimestepFactor=0.6)
        problem = document.problem(model(simulation=solver_obj).Objects[0])
        assert problem.timestep_factor == pytest.approx(0.6)

    @pytest.mark.parametrize("factor", [0.0, -1.0, 2.0])
    def test_a_factor_the_engine_would_ignore_is_refused_by_name(self, factor):
        """As a TranslationError, not the EnvelopeError underneath it: the panel
        shows one as the model's problem and the other as an internal error with
        a traceback, and typing 2 into a property field is not a crash.

        2.0 is here because it is the quiet one. Below zero openEMS at least
        prints "invalid timestep factor, skipping!"; above one it prints nothing
        at all and writes the factor it did not use into its XML.
        """
        solver_obj = simulation(TimestepFactor=factor)
        with pytest.raises(document.TranslationError) as raised:
            document.problem(model(simulation=solver_obj).Objects[0])
        assert "timestep_factor" in str(raised.value)
        assert "Simulation" in str(raised.value), "it should name the object to fix"

    def test_a_study_reads_down_to_full_scale_until_it_says_otherwise(self, study):
        """Full scale is what a study that names no smaller response is held
        to, which is every document nobody has thought about."""
        assert document.problem(study).smallest_response == 1.0

    @pytest.mark.parametrize("declared, magnitude", [(-20.0, 0.1), (-40.0, 0.01)])
    def test_the_response_a_study_reads_reaches_the_envelope_as_a_magnitude(
        self, declared, magnitude
    ):
        """Decibels where it is declared and plotted, a magnitude in S where it
        is compared against one. Amplitude, so twenty a decade - halving it
        would make every declared floor the square of what was asked for."""
        analysis_obj = analysis(SmallestResponse=declared)
        problem = document.problem(model(analysis=analysis_obj).Objects[0])
        assert problem.smallest_response == pytest.approx(magnitude)

    @pytest.mark.parametrize("declared", [0.5, 20.0, float("inf"), float("-inf"), float("nan")])
    def test_a_response_that_is_not_a_depth_is_refused_by_name(self, declared):
        """A passive device answers no more than one, so anything above zero is
        a typing slip - and one that silently became "full scale" would leave
        the property reading as though it had been honoured.

        The two that are not above zero are here because the comparison alone
        lets them through: ``nan`` fails every comparison, and ``-inf`` converts
        to a magnitude of zero, which is a bar nothing can be judged against."""
        analysis_obj = analysis(SmallestResponse=declared)
        with pytest.raises(document.TranslationError) as raised:
            document.problem(model(analysis=analysis_obj).Objects[0])
        assert "SmallestResponse" in str(raised.value)
        assert "Analysis" in str(raised.value), "it should name the object to fix"

    def test_the_factor_is_refused_before_anything_is_meshed(self):
        """It is read on the way in, not caught on the way out.

        Wrapping the whole ``Problem(...)`` construction in the same conversion
        would put that ``try`` across ``plan_grid`` and every invariant in
        ``Problem.__post_init__``, so a non-monotonic grid - an adapter bug, not
        a user's - would be reported as this solver object's fault with its
        traceback swallowed. Reading the property through its own function costs
        nothing and catches only itself.
        """
        solver_obj = simulation(TimestepFactor=3.0)
        meshed = []
        original = document.plan_grid
        try:
            document.plan_grid = lambda *a, **kw: meshed.append(1) or original(*a, **kw)
            with pytest.raises(document.TranslationError):
                document.problem(model(simulation=solver_obj).Objects[0])
        finally:
            document.plan_grid = original
        assert meshed == [], "a factor openEMS would ignore should not cost a mesh"


class TestTheMeshPolicyReachesTheMesher:
    """The properties are user-facing names; these pin what they *mean*."""

    def test_elements_per_wavelength_sets_the_bulk_size(self, study):
        """Twenty elements per wavelength means the cap is one twentieth of it."""
        problem = document.problem(study)
        wavelength = SPEED_OF_LIGHT / problem.frequency.stop / math.sqrt(EPS_R) * 1e3
        assert problem.grid.params["dielectric_res"] == pytest.approx(wavelength / 20.0, rel=1e-9)

    def test_edge_refinement_is_a_ratio_of_the_bulk_size(self, study):
        params = document.problem(study).grid.params
        assert params["dielectric_res"] / params["metal_res"] == pytest.approx(6.0)

    @pytest.mark.parametrize("refinement", [1.0, 2.5, 6.0, 12.0])
    def test_a_fractional_refinement_is_honoured(self, refinement):
        """Bisecting during a convergence study is ordinary work, so this is a
        float. An integer property would forbid the 4.5 between 3 and 6."""
        settings = mesh_settings(EdgeRefinement=refinement)
        params = document.problem(model(settings=settings).Objects[0]).grid.params
        assert params["dielectric_res"] / params["metal_res"] == pytest.approx(refinement)

    def test_the_growth_ratio_applies_to_every_axis(self):
        settings = mesh_settings(MaxGrowthRatio=1.15)
        params = document.problem(model(settings=settings).Objects[0]).grid.params
        assert params["max_ratio"] == [1.15, 1.15, 1.15]

    def test_a_zero_floor_means_derive_it(self, study):
        params = document.problem(study).grid.params
        assert params["min_cell"] == pytest.approx(params["metal_res"] / 1000.0)

    def test_an_explicit_floor_is_used_verbatim(self):
        settings = mesh_settings(MinElementSize=0.004)
        params = document.problem(model(settings=settings).Objects[0]).grid.params
        assert params["min_cell"] == pytest.approx(0.004)

    def test_air_cells_are_read_per_face(self):
        """Six named properties, not one list - and each face is independent."""
        settings = mesh_settings(
            AirCellsYMin=3,
            AirCellsYMax=17,
            **{f"Padding{a}{s}": "Air" for a in "XYZ" for s in ("Min", "Max")},
        )
        problem = document.problem(model(settings=settings).Objects[0])
        assert problem.grid.params["padding"][1] == ["3", "17"]

    def test_refinement_below_one_is_refused_by_name(self):
        settings = mesh_settings(EdgeRefinement=0.5)
        with pytest.raises(document.TranslationError, match="EdgeRefinement is 0.5"):
            document.problem(model(settings=settings).Objects[0])

    def test_zero_elements_per_wavelength_is_refused(self):
        settings = mesh_settings(ElementsPerWavelength=0.0)
        with pytest.raises(document.TranslationError, match="ElementsPerWavelength is 0"):
            document.problem(model(settings=settings).Objects[0])

    def test_a_negative_air_buffer_says_to_use_through_instead(self):
        settings = mesh_settings(
            AirCellsZMin=-4,
            **{f"Padding{a}{s}": "Air" for a in "XYZ" for s in ("Min", "Max")},
        )
        with pytest.raises(document.TranslationError, match="AirCellsZMin is -4"):
            document.problem(model(settings=settings).Objects[0])


class TestThePreviewShowsWhatGetsSolved:
    """A preview of a mesh nobody solves is worse than no preview.

    ``mesh()`` and ``problem()`` go through one ``_translate``, so they cannot
    disagree. These assert the property rather than the implementation - the
    two could drift back apart under any future refactor that splits them.
    """

    def test_the_plan_and_the_envelope_are_the_same_grid(self, study):
        plan = document.mesh(study)
        grid = document.problem(study).grid
        for dim in range(3):
            assert list(plan.lines[dim]) == list(grid[dim])

    def test_the_plan_carries_provenance_the_envelope_throws_away(self, study):
        """Why a line exists is not solver input, so it stays out of the
        envelope - putting it in would move every digest."""
        plan = document.mesh(study)
        assert any(pin.required for pin in plan.lines.fixed[2])
        assert not hasattr(document.problem(study).grid, "fixed")

    def test_the_plan_carries_the_regions_the_report_names(self, study):
        labels = {region.name for region in document.mesh(study).regions}
        assert {"Substrate", "Ground", "Trace"} <= labels

    def test_a_preview_does_not_need_an_excitation(self):
        """Every run in a sweep shares one grid, so which port is driven is not
        a question a preview has to answer - and refusing to draw a mesh until
        a port is marked would hide the mesh exactly when it is being set up."""
        port = microstrip_port(1, trace(), ground(), Excitation=False)
        doc = model(port=port)
        assert document.mesh(doc.Objects[0]).lines.cell_count > 0
        with pytest.raises(document.TranslationError, match="no port has Excitation"):
            document.problem(doc.Objects[0])

    def test_a_preview_still_refuses_a_model_it_cannot_mesh(self, study):
        """It stops short of the envelope, not short of the checks."""
        policy(study).EdgeRefinement = 0.25
        with pytest.raises(document.TranslationError, match="EdgeRefinement"):
            document.mesh(study)


class TestTheCapIsTheVacuumWavelength:
    """Air around a board is meshed for a wave travelling in air.

    ``ElementsPerWavelength`` counts elements across a wavelength, and which
    wavelength depends on where you are. One taken from the slowest material in
    the model is safe and expensive: a patch on alumina gets air meshed 3.1x
    finer than anything there needs, for roughly 30x the cells.
    """

    def test_the_cap_is_the_bulk_size_in_vacuum(self, study):
        params = document.mesh(study).params
        wavelength_mm = SPEED_OF_LIGHT / 10e9 * 1e3
        assert params.cap == pytest.approx(wavelength_mm / 20.0)

    def test_the_bulk_size_is_still_the_slowest_material(self, study):
        """It sizes every dielectric. The domain is the ceiling's, not its."""
        params = document.mesh(study).params
        assert params.cap / params.dielectric_res == pytest.approx(math.sqrt(EPS_R))

    def test_the_dielectric_carries_its_own_size(self, study):
        params = document.mesh(study).params
        substrate = next(
            region for region in document.mesh(study).regions if region.name == "Substrate"
        )
        assert substrate.size == pytest.approx(params.cap / math.sqrt(EPS_R))

    def test_air_is_coarser_than_the_board(self, study):
        """The whole point, measured on the grid rather than on the inputs."""
        lines = document.mesh(study).lines
        inside = self._cell_at(lines[2], HEIGHT / 2)
        above = self._cell_at(lines[2], 8.0)
        assert above > inside * 1.5

    def test_raising_permittivity_refines_the_board_and_nothing_else(self, study):
        """The scaling debt this pays off, stated where it is exact.

        A denser board must cost cells inside the board. Before this, one
        wavelength served the whole model, so alumina in place of FR4 refined
        the air too - the ceiling itself moved.

        A real air cell is asserted alongside the ceiling because padding the
        domain in cells of the *slowest material* refines it anyway: a denser
        board gets a shorter gap, and the cells in it have less room to grade
        out to a ceiling that has not moved.
        Judged against the ceiling rather than against each other - where
        exactly a line lands in a span is placement arithmetic and wobbles by a
        few percent, while the fault took the air to 62% of the ceiling.
        """
        binding = next(o for o in study.Document.Objects if o.Name == "DielectricBinding")
        before = document.mesh(study)
        air_before = self._cell_at(before.lines[2], 8.0)
        binding.Material.Permittivity = 9.8
        after = document.mesh(study)
        air_after = self._cell_at(after.lines[2], 8.0)

        assert after.params.cap == pytest.approx(before.params.cap), "ceiling moved"
        assert min(air_before, air_after) > 0.9 * after.params.ceiling, "air refined"
        assert after.params.dielectric_res < before.params.dielectric_res

    @staticmethod
    def _cell_at(lines, position):
        index = int(np.searchsorted(lines, position))
        return float(lines[index] - lines[index - 1])


class TestLocalRefinementReachesTheMesher:
    """``EMMeshRegion`` was a stub the toolbar could create and nothing read.

    The tests that matter are the two ends: that drawing one changes the grid
    at all, and that it changes it *only* the way a sizing region is allowed to
    - no lines pinned at the box the user drew.
    """

    def _pad(self, doc):
        """A small box over the middle of the line, well inside the domain."""
        block = Obj(
            "Part::Box",
            "Pad",
            Shape((-4.0, -WIDTH / 2, 0.0), (4.0, WIDTH / 2, HEIGHT)),
        )
        return _add(doc, block)

    def _cell_at(self, lines, position):
        index = int(np.searchsorted(lines, position))
        return float(lines[index] - lines[index - 1])

    def test_a_refinement_makes_the_cells_there_smaller(self, doc):
        study = doc.Objects[0]
        before = self._cell_at(document.mesh(study).lines[0], 0.0)
        _add(doc, refinement("Fine", self._pad(doc), ElementSize=0.05))
        after = self._cell_at(document.mesh(study).lines[0], 0.0)
        assert after <= 0.05 * (1 + 1e-9)
        assert after < before

    def test_disabling_it_leaves_the_grid_exactly_as_it_was(self, doc):
        study = doc.Objects[0]
        before = list(document.mesh(study).lines[0])
        _add(doc, refinement("Fine", self._pad(doc), Enabled=False))
        assert list(document.mesh(study).lines[0]) == before

    def test_the_envelope_gets_the_same_refined_grid_as_the_preview(self, doc):
        study = doc.Objects[0]
        _add(doc, refinement("Fine", self._pad(doc)))
        plan = document.mesh(study)
        grid = document.problem(study).grid
        for dim in range(3):
            assert list(plan.lines[dim]) == list(grid[dim])

    def test_a_shape_that_is_not_a_box_is_refined_by_its_bounding_box(self, doc):
        """Unlike a material region, which is refused.

        Refining around a cylinder is a sensible thing to want, and the
        rectilinear answer is the box it sits in. That is a surprise the
        tooltip names rather than a refusal.
        """
        study = doc.Objects[0]
        round_thing = _add(
            doc,
            Obj(
                "Part::Cylinder",
                "Via",
                Shape((-1.0, -1.0, 0.0), (1.0, 1.0, HEIGHT), fill=math.pi / 4),
            ),
        )
        _add(doc, refinement("Fine", round_thing, ElementSize=0.05))
        assert self._cell_at(document.mesh(study).lines[0], 0.0) <= 0.05 * (1 + 1e-9)

    def test_it_can_be_aimed_at_a_face(self, doc):
        """The face must be *smaller* than the solid in the measured axis.

        Given the same span, deleting the entire sub-element lookup - taking
        the owning solid's box instead of the face's - passes, and the test
        proves only that naming a face does not crash.
        """
        study = doc.Objects[0]
        block = self._pad(doc)
        block.Shape.face("Face1", (-1.0, -WIDTH / 2, HEIGHT), (1.0, WIDTH / 2, HEIGHT))
        _add(doc, refinement("Fine", References=[(block, ["Face1"])]))
        lines = document.mesh(study).lines[0]
        assert self._cell_at(lines, 0.0) <= 0.05 * (1 + 1e-9)
        assert self._cell_at(lines, 3.0) > 0.05 * 2, (
            "the whole pad was refined, so the face was never read"
        )

    def test_a_region_coarser_than_the_global_size_is_refused_by_name(self, doc):
        study = doc.Objects[0]
        _add(doc, refinement("Coarsener", self._pad(doc), ElementSize=50.0))
        with pytest.raises(document.MeshError) as excinfo:
            document.mesh(study)
        assert "Coarsener" in str(excinfo.value)

    @pytest.mark.parametrize(
        "overrides, message",
        [
            (dict(ElementSize=0.0), "no element size"),
            (dict(MinElementsAcross=-1), "MinElementsAcross"),
        ],
    )
    def test_an_unusable_region_is_refused_by_name(self, doc, overrides, message):
        study = doc.Objects[0]
        _add(doc, refinement("Broken", self._pad(doc), **overrides))
        with pytest.raises(document.TranslationError, match=message) as excinfo:
            document.mesh(study)
        assert "Broken" in str(excinfo.value)

    def test_a_region_referencing_nothing_is_refused_by_name(self, doc):
        study = doc.Objects[0]
        _add(doc, refinement("Empty"))
        with pytest.raises(document.TranslationError, match="refines nothing"):
            document.mesh(study)


class TestCoarseningReachesTheMesher:
    """The other direction: geometry told to stop driving the grid.

    Aimed at the trace, which is bound to a material and so is a region the
    mesher sizes. That is the difference from a refinement, which covers a box
    and needs no material at all: coarsening is carried *by* the object, so an
    object the mesher never sizes has nothing to relax.
    """

    def coarsen(self, doc, name="Loose", target="Trace", **overrides):
        overrides.setdefault("Mode", "Coarsen")
        return _add(doc, refinement(name, part(doc, target), **overrides))

    def test_it_costs_fewer_cells_than_the_drawing_alone(self, doc):
        study = doc.Objects[0]
        before = document.mesh(study).lines
        self.coarsen(doc, ElementSize=2.0)
        after = document.mesh(study).lines
        assert np.prod([len(a) for a in after]) < np.prod([len(b) for b in before])

    def test_the_substrate_keeps_its_own_resolution(self, doc):
        """Coarsening the trace must not coarsen the board underneath it."""
        study = doc.Objects[0]
        before = list(document.mesh(study).lines[2])
        self.coarsen(doc, ElementSize=2.0)
        after = document.mesh(study).lines[2]
        through = [z for z in after if -1e-9 <= z <= HEIGHT + 1e-9]
        was = [z for z in before if -1e-9 <= z <= HEIGHT + 1e-9]
        assert len(through) >= len(was)

    def test_the_envelope_and_the_preview_agree(self, doc):
        study = doc.Objects[0]
        self.coarsen(doc, ElementSize=2.0)
        plan = document.mesh(study)
        grid = document.problem(study).grid
        for dim in range(3):
            assert list(plan.lines[dim]) == list(grid[dim])

    def test_the_solid_carries_it_across_the_envelope(self, doc):
        study = doc.Objects[0]
        self.coarsen(doc, ElementSize=2.0)
        problem = document.problem(study)
        relaxed = {s.label: s.relaxed_to for s in problem.solids}
        assert relaxed["Trace"] == pytest.approx(2.0)
        assert relaxed["Substrate"] == 0.0

    def test_disabling_it_leaves_the_grid_exactly_as_it_was(self, doc):
        study = doc.Objects[0]
        before = list(document.mesh(study).lines[0])
        self.coarsen(doc, ElementSize=2.0, Enabled=False)
        assert list(document.mesh(study).lines[0]) == before

    def test_two_coarsenings_on_one_object_leave_the_finer(self, doc):
        study = doc.Objects[0]
        self.coarsen(doc, name="Loose", ElementSize=2.0)
        self.coarsen(doc, name="Looser", ElementSize=8.0)
        relaxed = {s.label: s.relaxed_to for s in document.problem(study).solids}
        assert relaxed["Trace"] == pytest.approx(2.0)

    def _plated(self, doc):
        """Copper bound to the *top face* of the board, and nothing else.

        The ordinary way a plane is drawn - ``geometry._reference_boxes`` says
        so - and the case that decides whether a coarsening can reach past what
        it names. The board and the copper on it are one FreeCAD object, so
        keying a relaxation on the object alone relaxes both.
        """
        board = part(doc, "Substrate")
        top = ((-LENGTH / 2, -BOARD / 2, HEIGHT), (LENGTH / 2, BOARD / 2, HEIGHT))
        board.Shape.face("Face9", *top)
        plating = Obj(
            "EMMaterialBinding",
            "Plating",
            Material=copper(),
            References=[(board, ["Face9"])],
        )
        return _add(doc, plating)

    def test_a_reference_only_partly_coarsened_is_not_relaxed_at_all(self):
        """Its pieces are not paired back to the elements that made them.

        So a binding naming two faces with one of them coarsened has to relax
        both or neither, and neither is the direction that cannot take
        resolution off geometry nobody gave up.
        """
        doc = model()
        board = part(doc, "Substrate")
        top = ((-LENGTH / 2, -BOARD / 2, HEIGHT), (LENGTH / 2, BOARD / 2, HEIGHT))
        side = ((-LENGTH / 2, -BOARD / 2, 0.0), (-LENGTH / 2, BOARD / 2, HEIGHT))
        board.Shape.face("Face9", *top)
        board.Shape.face("Face10", *side)
        plating = _add(
            doc,
            Obj(
                "EMMaterialBinding",
                "Plating",
                Material=copper(),
                References=[(board, ["Face9", "Face10"])],
            ),
        )
        region = refinement(
            "Loose", Mode="Coarsen", References=[(board, ["Face9"])], ElementSize=8.0
        )
        _, solids, _, _ = document._geometry(
            [plating], 1.5e9, 0.035, document._relaxations([region])
        )
        assert {s.relaxed_to for s in solids} == {0.0}

    def _plated_solids(self, doc, region):
        """The solids a plated board translates to, with ``region`` coarsening.

        Translated rather than meshed. What is under test is which geometry a
        coarsening reaches, which is settled before a line is placed - and a
        board relaxed far enough to show it is also a board whose THROUGH face
        no longer lands on the structure, so meshing would refuse it for an
        unrelated and correct reason.
        """
        plating = self._plated(doc)
        bindings = [part(doc, "DielectricBinding"), plating]
        _, solids, _, _ = document._geometry(
            bindings, 1.5e9, 0.035, document._relaxations([region])
        )
        return {solid.label: solid.relaxed_to for solid in solids}

    def test_coarsening_a_board_leaves_the_conductor_drawn_on_its_face_alone(self, doc):
        """The claim the whole design rests on, at the one drawing that tests it."""
        region = refinement("Loose", part(doc, "Substrate"), Mode="Coarsen", ElementSize=8.0)
        relaxed = self._plated_solids(doc, region)
        assert relaxed["Substrate"] == pytest.approx(8.0)
        assert relaxed["Substrate:Face9"] == 0.0, "the coarsening reached past what it named"

    def test_a_conductor_on_a_face_can_be_coarsened_on_its_own(self, doc):
        """And the other way round, or that geometry could never be coarsened."""
        board = part(doc, "Substrate")
        region = refinement(
            "Loose", Mode="Coarsen", References=[(board, ["Face9"])], ElementSize=8.0
        )
        relaxed = self._plated_solids(doc, region)
        assert relaxed["Substrate:Face9"] == pytest.approx(8.0)
        assert relaxed["Substrate"] == 0.0

    def test_coarsening_geometry_no_material_names_is_refused_by_name(self, doc):
        """Only bound geometry has an element size to settle for."""
        study = doc.Objects[0]
        shape = Shape((20.0, 20.0, 0.0), (24.0, 24.0, 4.0))
        bracket = _add(doc, Obj("Part::Box", "Bracket", shape))
        _add(doc, refinement("Loose", bracket, Mode="Coarsen", ElementSize=2.0))
        with pytest.raises(document.TranslationError, match="no material binding") as excinfo:
            document.mesh(study)
        assert "Loose" in str(excinfo.value)
        assert "Bracket" in str(excinfo.value)

    def test_asking_for_a_count_as_well_is_refused_by_name(self, doc):
        """A count demands resolution, which is the opposite of what this asks."""
        study = doc.Objects[0]
        self.coarsen(doc, ElementSize=2.0, MinElementsAcross=8)
        with pytest.raises(document.TranslationError, match="elements across") as excinfo:
            document.mesh(study)
        assert "Loose" in str(excinfo.value)

    def test_a_mode_this_adapter_has_no_behaviour_for_is_refused(self, doc):
        """FreeCAD restores an enumeration's list from the file, not the class."""
        study = doc.Objects[0]
        self.coarsen(doc, Mode="Ignore")
        with pytest.raises(document.TranslationError, match="Mode is 'Ignore'"):
            document.mesh(study)

    def test_a_document_saved_before_the_property_existed_still_refines(self, doc):
        """The property is new, and a stored object carries only what it had."""
        study = doc.Objects[0]
        region = _add(doc, refinement("Fine", part(doc, "Trace"), ElementSize=0.05))
        del region.Mode
        assert {s.relaxed_to for s in document.problem(study).solids} == {0.0}


def explicit_analysis(*members, **overrides):
    """An analysis whose ``Group`` is exactly what you put in it.

    The fixtures above use a live view of the document so the older tests could
    stay as they were. Scoping is the one thing that view cannot demonstrate, so
    these say what is in the group and nothing else.
    """
    properties = dict(
        FrequencyStart=1e9,
        FrequencyStop=10e9,
        NumFrequencyPoints=201,
        Waveform="Gaussian",
        SmallestResponse=0.0,
        Group=list(members),
    )
    name = overrides.pop("name", "Analysis")
    properties.update(overrides)
    return Obj("EMAnalysis", name, **properties)


class TestOwnershipIsMembership:
    """What a study owns is what is in its group. Nothing else in the document.

    A document scan makes a second simulation unresolvable - it inherits every
    port and binding - so two would have to be refused outright, and that
    refusal is what would stand between a user and two studies over one board.
    """

    def parts(self):
        """The pieces of a working microstrip, unassembled."""
        board, plane, strip = substrate(), ground(), trace()
        return dict(
            board=board,
            plane=plane,
            strip=strip,
            settings=mesh_settings(),
            solver=simulation(),
            bindings=[
                binding("DielectricBinding", fr4(), board),
                binding("GroundBinding", copper(), plane),
                binding("TraceBinding", copper(), strip),
            ],
            port=microstrip_port(1, strip, plane),
        )

    def test_an_object_outside_the_group_is_not_in_the_study(self):
        """A binding sitting loose at document root belongs to no study, and a
        study that swept it up would be the old document scan by another name."""
        parts = self.parts()
        study = explicit_analysis(
            parts["solver"], parts["settings"], *parts["bindings"][:2], parts["port"]
        )
        Document(
            study,
            parts["board"],
            parts["plane"],
            parts["strip"],
            *parts["bindings"],
            parts["solver"],
            parts["settings"],
            parts["port"],
        )

        # The trace binding is in the document but not in the group.
        with pytest.raises(document.TranslationError, match="Trace"):
            document.problem(study)

    def test_two_studies_over_one_board_each_see_their_own(self):
        """An openEMS study and, later, a NEC2 one over the same geometry."""
        first, second = self.parts(), self.parts()
        # One board, drawn once, marked up twice.
        for name in ("board", "plane", "strip"):
            second[name] = first[name]

        studies = []
        for index, parts in enumerate((first, second), start=1):
            studies.append(
                explicit_analysis(
                    parts["solver"],
                    parts["settings"],
                    *parts["bindings"],
                    parts["port"],
                    name=f"Analysis{index}",
                )
            )
        Document(*studies, first["board"], first["plane"], first["strip"])

        # Each translates, and each sees exactly one port - not both.
        for study in studies:
            assert len(document.problem(study).ports) == 1

    def test_a_nested_group_is_still_part_of_the_study(self):
        """Sorting twenty ports into a folder is housekeeping. Dropping them
        out of the study for it would be a silent wrong answer."""
        parts = self.parts()
        folder = Obj("App::DocumentObjectGroup", "Ports", Group=[parts["port"]])
        study = explicit_analysis(parts["solver"], parts["settings"], *parts["bindings"], folder)
        Document(study, parts["board"], parts["plane"], parts["strip"])

        assert len(document.problem(study).ports) == 1

    def test_a_group_that_contains_itself_does_not_hang(self):
        """A document is a file a user can edit. An infinite walk here would
        freeze the GUI with no message at all."""
        parts = self.parts()
        study = explicit_analysis(parts["solver"], parts["settings"])
        study.Group.append(study)
        Document(study)

        with pytest.raises(document.TranslationError, match="no ports"):
            document.problem(study)


class TestARunStartsFromAStudy:
    """What ``problem()`` refuses when handed the wrong object.

    Three classes of migration refusal stood here - pre-analysis documents,
    property names that exist nowhere. Nothing outside this repository ever held
    such a document and nothing has been released, so there is no migration to
    test. What survives is every refusal that can still fire for a document made
    today.
    """

    def test_a_solver_is_not_a_study(self):
        """An ordinary mistake, not a stale document: the solver is the object
        called "openEMS" and it is where the run settings are, so it is what a
        user reaches for. The band and the ports live one level up."""
        doc = model()
        solver = solver_of(doc.Objects[0])
        with pytest.raises(document.TranslationError, match="is a solver, not a study"):
            document.problem(solver)

    def test_something_that_is_no_study_at_all_says_that_instead(self):
        with pytest.raises(document.TranslationError, match="not an EM analysis"):
            document.problem(substrate())

    def test_an_object_that_is_no_mesh_policy_says_so(self):
        """Reached through ``contents()``, which is the one place every route
        passes, because ``_Context`` divides by ElementsPerWavelength to size
        ports before mesh parameters exist."""
        doc = model(settings=Obj("EMMeshPolicy", "MeshSettings"))
        with pytest.raises(document.TranslationError, match="does not look like a mesh policy"):
            document.problem(doc.Objects[0])


class TestTheSweepIsOneGeometry:
    def test_every_run_shares_a_grid(self):
        """An S-matrix whose columns came from different meshes is not one
        matrix. Built by re-exciting a single translation, so they cannot."""
        doc = model()
        doc.Objects.append(_far_end_port(doc, 2))

        problems = document.sweep(doc.Objects[0])
        assert [p.excited_port.number for p in problems] == [1, 2]
        assert all((p.grid.x == problems[0].grid.x).all() for p in problems)
        assert all((p.grid.z == problems[0].grid.z).all() for p in problems)

    def test_an_inactive_port_is_still_measured(self):
        """It just is not driven. A two-port with one source still reports S21."""
        doc = model()
        doc.Objects.append(_far_end_port(doc, 2, Excitation=False))

        problems = document.sweep(doc.Objects[0])
        assert len(problems) == 1
        assert len(problems[0].ports) == 2


@pytest.mark.slow
def test_a_document_reproduces_the_acceptance_gate(interpreter, tmp_path):
    """The wiring, judged against physics rather than against itself.

    Everything else in this file checks that the translation produces something
    well-formed. Well-formed is not the bar: a reversed excitation, a trace
    collapsed onto the wrong plane, or a measurement plane in the wrong place
    all produce a perfectly valid Problem that solves cleanly and answers
    wrongly. The only way to tell is to solve it and compare against a closed
    form.

    The reference is Hammerstad's, quoted to about 1%, so the tolerance cannot
    honestly be tighter. What makes this a strong test anyway is that it is the
    *same* line ``test_acceptance_microstrip.py`` builds by hand, on a grid that
    is identical cell for cell - so the two GATE lines must agree with each
    other far more closely than either agrees with Hammerstad. A translation
    defect large enough to matter shows up as a gap between them long before it
    reaches the tolerance.

    Both figures are printed rather than quoted here. A figure written into a
    docstring is not re-measured when the thing it describes moves, and goes on
    reading as current long after it has stopped being either route's.
    """
    from Microwave.Solvers.openems import preflight, read, run, write
    from tests.analytic import reference

    problem = document.problem(model().Objects[0])
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(write.write(problem, tmp_path), interpreter=interpreter)
    results = read.read(tmp_path)

    import numpy as np

    quasi_static = results.band(1e9, 2e9)
    measured = float(np.mean(results.port(1).impedance[quasi_static]))
    expected = reference.characteristic_impedance(WIDTH, HEIGHT, EPS_R)
    error = (measured - expected) / expected

    print(
        f"\nGATE microstrip Z0 = {measured:.4f} ohm, Hammerstad "
        f"{expected:.4f} ohm, {error * 100:+.4f}% [through the document]"
    )
    assert abs(error) < 0.01, (
        f"a document-derived run gives Z0 = {measured:.4f} ohm against "
        f"Hammerstad's {expected:.4f} ({error * 100:+.4f}%); compare the "
        "hand-built gate's GATE line, which this must match closely"
    )
    assert float(np.max(np.abs(results.s(1))[quasi_static])) < 0.01, (
        "the line is reflecting, so the Through padding did not survive "
        "translation and the line is not infinite"
    )


# ---------------------------------------------------------------------------
# The other two port kinds
# ---------------------------------------------------------------------------


#: How far a lumped port's pick reaches along x. A trace running off the grid
#: axes ends on a diagonal, and this stands in for what its bounding box gains.
PICK_DEPTH = 1.0


def lumped_pick(strip, *, area, name="Picked"):
    """A lumped port fed from a pick spanning ``PICK_DEPTH`` and the trace width.

    ``area`` is the whole difference between the two shapes this stands for: a
    pad covers what its bounds say, and an outline is a diagonal across them.
    The bounds are identical, which is why the caller has to say which it is.
    """
    strip.Shape.face(name, (-50.0, -WIDTH / 2, HEIGHT), (-50.0 + PICK_DEPTH, WIDTH / 2, HEIGHT))
    picked = strip.Shape.getElement(name)
    picked.Faces = [picked] if area else []
    return lumped_port(1, strip, ground(), SourceEntity=(strip, [name]))


def lumped_port(number, source, reference, **overrides):
    properties = dict(
        Number=number,
        Excitation=True,
        Resistance=50.0,
        ReferenceImpedance=50.0,
        ReferencedTo=FIXED_IMPEDANCE,
        Length=0.0,
        ExcitationAxis="-Z",
        SourceEntity=(source, ["Face1"]),
        ReferenceEntity=(reference, ["Face1"]),
    )
    properties.update(overrides)
    return Obj("EMPortLumped", f"Lumped{number}", **properties)


class TestALumpedPort:
    def test_it_spans_the_gap_from_source_to_reference(self):
        strip, plane = trace(), ground()
        port = document.problem(model(port=lumped_port(1, strip, plane)).Objects[0]).ports[0]

        assert port.start[2] == pytest.approx(HEIGHT)
        assert port.stop[2] == pytest.approx(0.0)
        assert port.excite_sign == -1
        assert port.feed_resistance == 50.0

    def test_a_ground_plane_reference_does_not_widen_it(self):
        """The reference says *where* the far side of the gap is, not how wide
        the port is.

        On a real board the ground plane covers the whole footprint. Unioning
        the two entities would turn a lumped port between a trace end and the
        ground into a sheet resistor spanning the entire model - and it would
        solve, and the answer would be nonsense.
        """
        strip, plane = trace(), ground()
        port = document.problem(model(port=lumped_port(1, strip, plane)).Objects[0]).ports[0]

        for dim in (0, 1):
            extent = abs(port.stop[dim] - port.start[dim])
            assert extent <= WIDTH, (
                f"the port spans {extent:.4g} mm along {'xyz'[dim]}, wider than "
                f"the {WIDTH} mm trace it is fed from"
            )

    def test_it_needs_a_gap_to_drive_across(self):
        strip = trace()
        port = lumped_port(1, strip, strip)
        with pytest.raises(document.TranslationError, match="there is none"):
            document.problem(model(port=port).Objects[0])

    @pytest.mark.parametrize(
        ("area", "flattened"),
        [(True, False), (False, True)],
        ids=["a pad keeps its depth", "an outline is flattened"],
    )
    def test_what_the_pick_is_decides_whether_it_is_flattened(self, area, flattened):
        """openEMS has only axis-aligned ports, so an outline that runs off the
        axes has to be flattened onto the plane it already is - and a pad driven
        across its own face has to keep the depth that is really conductor."""
        port = lumped_pick(trace(), area=area)
        built = document.problem(model(port=port).Objects[0]).ports[0]

        assert (abs(built.stop[0] - built.start[0]) <= portbox.FLATNESS) is flattened
        if not flattened:
            assert abs(built.stop[0] - built.start[0]) == pytest.approx(PICK_DEPTH, abs=1e-12)


# A WR-42 guide: air inside, conducting walls as boundary conditions rather
# than geometry, absorbing only on the ends. Same guide the waveguide gate uses.
GUIDE_A, GUIDE_B, GUIDE_L = 10.7, 4.3, 50.0


def waveguide_model(inset=0.0, **port_overrides):
    shape = Shape((0, 0, 0), (GUIDE_A, GUIDE_B, GUIDE_L))
    shape.face("Face1", (0, 0, inset), (GUIDE_A, GUIDE_B, inset))
    guide = Obj("Part::Box", "Guide", shape)
    air = Obj(
        "EMMaterial",
        "Air",
        MaterialType="Dielectric",
        Permittivity=1.0,
        Permeability=1.0,
        Conductivity=0.0,
        LossTangent=0.0,
        MeasuredAt=0.0,
        Thickness=0.0,
    )
    properties = dict(
        Number=1,
        Excitation=True,
        ReferenceImpedance=50.0,
        ReferencedTo=FIXED_IMPEDANCE,
        Length=0.0,
        CrossSection=(guide, ["Face1"]),
        PropagationAxis="Z",
        Mode="TE10",
    )
    properties.update(port_overrides)
    port = Obj("EMPortRectWaveguide", "WG1", **properties)

    settings = mesh_settings(
        **{f"AirCells{a}{s}": 0 for a in "XYZ" for s in ("Min", "Max")},
        **{f"Padding{a}{s}": "Air" for a in "XYZ" for s in ("Min", "Max")},
    )
    solver_obj = simulation(
        BoundaryXMin="PEC",
        BoundaryXMax="PEC",
        BoundaryYMin="PEC",
        BoundaryYMax="PEC",
    )
    return Document(
        analysis(FrequencyStart=18e9, FrequencyStop=26e9),
        settings,
        guide,
        binding("AirBinding", air, guide),
        port,
        solver_obj,
    )


class TestAWaveguidePort:
    def test_it_spans_the_cross_section_and_a_few_cells_of_depth(self):
        """Not the guide's length.

        ``AddRectWaveGuidePort`` puts the excitation on the near face of the
        port box and the probes on the far one, so a box spanning the whole
        guide would measure at the opposite end.

        The two extents are asserted as the fixture's own dimensions rather than
        re-typed in metres: spelling one fact twice means editing the guide
        fails this for the wrong reason.
        """
        problem = document.problem(waveguide_model(inset=10.0).Objects[0])
        port = problem.ports[0]

        assert port.waveguide_arguments(1e-3) == (
            pytest.approx(GUIDE_A * 1e-3),
            pytest.approx(GUIDE_B * 1e-3),
            "TE10",
        )
        assert port.mode == "TE10"
        assert port.length == pytest.approx(5 * problem.grid.params["dielectric_res"])
        assert port.length < GUIDE_L / 4

    def test_the_walls_keep_the_grid_out_of_the_absorber(self):
        problem = document.problem(waveguide_model(inset=10.0).Objects[0])
        assert problem.boundary == ("PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8")
        assert problem.grid.params["pml_cells"] == [0, 0, 8]

    def test_it_asks_for_a_grid_line_at_each_of_its_planes(self):
        """openEMS does not snap a waveguide port. Off a line it discretises
        nothing, the run excites nothing, and every S-parameter comes back 0/0
        with no warning anywhere."""
        problem = document.problem(waveguide_model(inset=10.0).Objects[0])
        port = problem.ports[0]

        for position in port.required_lines()[2]:
            assert any(abs(float(line) - position) < 1e-9 for line in problem.grid.z), (
                f"no grid line at z={position}"
            )

    def test_the_absorber_is_added_outside_the_guide(self):
        """Which is what makes the guide's own end face a legal place for a port.

        ``pml_cells`` lays absorber cells *beyond* the meshed domain, not inside
        it, so a domain meshed exactly to the guide walls keeps its whole
        interior usable. Were it the other way round, the natural thing to click
        - the end face of the guide - would put the excitation inside the
        PML, and the run would report an S-matrix for a wave being deliberately
        eaten.

        The WR-42 gate still insets its ports by ten cells. Nothing here
        establishes a minimum, so nothing refuses a port at the edge; this only
        pins down which side of the domain wall the absorber is on.
        """
        from Microwave.Solvers.openems import preflight

        problem = document.problem(waveguide_model(inset=0.0).Objects[0])
        interior = preflight.absorber._absorber_bounds(problem.grid, 2)

        assert interior == pytest.approx((0.0, GUIDE_L))
        assert float(problem.grid.z[0]) < 0.0, "absorber cells belong outside"
        assert float(problem.grid.z[-1]) > GUIDE_L
        assert preflight.refusals(preflight.check(problem)) == []


# The coaxial line the translation tests below are drawn on: a bore of 3.5 mm
# with a 1 mm inner conductor, 80 mm long, which is the fixture in tests/coax.py.
COAX_INNER = 1.0
COAX_OUTER = 3.5
COAX_LENGTH = 80.0


def coaxial_model(**port_overrides):
    """One tube with an annular end face, and a coaxial port on it."""
    shape = Shape(
        (-COAX_OUTER, -COAX_OUTER, 0.0), (COAX_OUTER, COAX_OUTER, COAX_LENGTH), solid=True
    )
    shape.ring("Face1", (0.0, 0.0, 0.0), COAX_INNER, COAX_OUTER, flat=2)
    line = Obj("Part::Feature", "Line", shape)
    ptfe = Obj(
        "EMMaterial",
        "PTFE",
        MaterialType="Dielectric",
        Permittivity=2.1,
        Permeability=1.0,
        Conductivity=0.0,
        LossTangent=0.0,
        MeasuredAt=0.0,
        Thickness=0.0,
    )
    properties = dict(
        Number=1,
        Excitation=True,
        ReferenceImpedance=50.0,
        ReferencedTo=PORT_IMPEDANCE,
        Annulus=(line, ["Face1"]),
        PropagationAxis="Z",
        FeedOffset=16.0,
        MeasurementDistance=24.0,
        Length=0.0,
    )
    properties.update(port_overrides)
    # No air around the line and walls instead of an absorber: the line is 7 mm
    # across, and eight absorber cells sized for a 6 GHz wavelength would eat it
    # whole. Nothing here is about the domain.
    settings = mesh_settings(
        **{f"AirCells{a}{s}": 0 for a in "XYZ" for s in ("Min", "Max")},
        **{f"Padding{a}{s}": "Air" for a in "XYZ" for s in ("Min", "Max")},
    )
    return Document(
        analysis(FrequencyStart=1e9, FrequencyStop=6e9),
        settings,
        line,
        binding("PTFEBinding", ptfe, line),
        Obj("EMPortCoaxial", "Coax1", **properties),
        simulation(
            BoundaryXMin="PEC",
            BoundaryXMax="PEC",
            BoundaryYMin="PEC",
            BoundaryYMax="PEC",
        ),
    )


class TestACoaxialPort:
    """One pick carries the whole line: two radii, a centre and an axis."""

    def translated(self, **overrides):
        return document.problem(coaxial_model(**overrides).Objects[0]).ports[0]

    def test_the_radii_come_off_the_ring_rather_than_off_a_property(self):
        """Nothing about the two conductors is typed anywhere, so nothing about
        them can disagree with the drawing - which matters more here than for
        any other kind, the impedance being nothing but their ratio."""
        port = self.translated()

        assert port.kind == "coaxial"
        assert port.inner_radius == COAX_INNER
        assert port.outer_radius == COAX_OUTER
        assert port.bore_centre == (0.0, 0.0, 0.0)

    def test_the_two_planes_are_measured_in_from_the_ring(self):
        port = self.translated()

        assert port.feed_shift == 16.0
        assert port.measurement_shift == 40.0
        assert port.measurement_position() == 40.0

    def test_an_unset_length_ends_the_port_at_its_probes(self):
        assert self.translated().length == 40.0

    def test_a_stated_length_is_taken(self):
        assert self.translated(Length=COAX_LENGTH).length == COAX_LENGTH

    def test_a_port_pointing_out_of_the_line_is_refused(self):
        with pytest.raises(document.TranslationError, match="reach out of the structure"):
            self.translated(PropagationAxis="-Z")

    def test_a_face_that_is_not_a_ring_is_refused_by_name(self):
        """The message has to say what to pick instead: a coaxial port is the
        one kind whose selection is not a face of the obvious shape."""
        model = coaxial_model()
        model.Objects[2].Shape.face(
            "Face2", (-COAX_OUTER, -COAX_OUTER, 0.0), (COAX_OUTER, COAX_OUTER, 0.0)
        )
        model.Objects[4].Annulus = (model.Objects[2], ["Face2"])

        with pytest.raises(document.TranslationError, match="boundaries"):
            document.problem(model.Objects[0])

    def test_a_ring_of_metal_is_refused_by_saying_which_ring_to_pick(self):
        """The shield's end face is a ring too, and it is the plausible wrong
        pick. Taken, every probe and the excitation shell would sit inside the
        conductor - and the run would finish and report numbers for it."""
        model = coaxial_model()
        metal = model.Objects[3].Material
        metal.MaterialType = "PEC"
        # A conductor carrying either of these is refused earlier, on the
        # material rather than on the port, and would mask what is under test.
        metal.Permittivity = 1.0
        metal.Permeability = 1.0

        with pytest.raises(document.TranslationError, match="which is a conductor"):
            document.problem(model.Objects[0])

    def test_two_rings_on_one_port_are_refused(self):
        """The box is the union of everything named and the radii come off one
        ring, so a second one would set the outer radius from the pair and the
        inner from the first - a line nobody drew, with nothing to say so."""
        model = coaxial_model()
        line = model.Objects[2]
        line.Shape.ring("Face2", (0.0, 0.0, COAX_LENGTH), COAX_INNER, COAX_OUTER, flat=2)
        model.Objects[4].Annulus = (line, ["Face1", "Face2"])

        with pytest.raises(document.TranslationError, match="names 2 sub-elements"):
            document.problem(model.Objects[0])

    def test_probes_on_the_source_are_refused_with_the_number_to_use(self):
        with pytest.raises(document.TranslationError, match="MeasurementDistance is 0"):
            self.translated(MeasurementDistance=0.0)

    def test_it_carries_no_excitation_axis_and_no_metal(self):
        """The port lays nothing: the conductors are the user's, and the field
        is radial. Both are refused by the envelope, so what this holds is that
        the translation does not try to supply them."""
        port = self.translated()

        assert port.excitation_axis is None
        assert port.metal == ""


class TestWhatTheMarkupMustResolveTo:
    """Materials, bindings and selections have to come out unambiguous.

    Everything here is about the step between "the user drew and labelled it"
    and "the adapter has one solid with one material": overlap resolved by
    priority, a label that names two materials, a selection of several faces,
    and the two ways a binding can be incomplete. A wrong answer at this step
    describes a *different structure* and then solves it perfectly.
    """

    def test_a_dielectric_underlays_its_metal(self):
        """CSXCAD resolves overlapping primitives by priority, so a substrate
        drawn over its own ground plane would erase it on a tie."""
        problem = document.problem(model().Objects[0])
        by_label = {solid.label: solid for solid in problem.solids}

        assert by_label["Substrate"].priority < by_label["Ground"].priority
        assert by_label["Ground"].priority < problem.ports[0].priority

    def test_two_materials_with_one_label_are_refused(self):
        """Bindings name materials by label, so a duplicate makes the reference
        ambiguous - and the loser would be silently substituted."""
        doc = model()
        impostor = copper()
        impostor.Conductivity = 1.0e6
        part(doc, "GroundBinding").Material = impostor

        with pytest.raises(document.TranslationError, match="both labelled"):
            document.problem(doc.Objects[0])

    def test_the_same_material_twice_is_fine(self):
        """The trace and the ground are both copper, and that is not a clash.
        Refusing on the label alone would reject every ordinary board."""
        problem = document.problem(model().Objects[0])
        assert sum(m.name == "Copper" for m in problem.materials) == 1

    def test_missing_mesh_settings_is_refused(self):
        doc = model()
        removes(doc, "MeshSettings")
        with pytest.raises(document.TranslationError, match="holds no mesh policy"):
            document.problem(doc.Objects[0])

    def test_a_multi_face_selection_is_unioned(self):
        """``App::PropertyLinkSub`` carries a *list* of names. Taking only the
        first would silently shrink the port."""
        strip, plane = trace(), ground()
        strip.Shape.face("Face1b", (-LENGTH / 2, WIDTH / 2, HEIGHT), (-LENGTH / 2, WIDTH, HEIGHT))
        port = microstrip_port(1, strip, plane, TraceEnd=(strip, ["Face1", "Face1b"]))
        translated = document.problem(model(port=port).Objects[0]).ports[0]

        assert translated.start[1] == pytest.approx(-WIDTH / 2)
        assert translated.stop[1] == pytest.approx(WIDTH)

    def test_a_binding_with_no_material_is_refused(self):
        doc = model()
        part(doc, "DielectricBinding").Material = None
        with pytest.raises(document.TranslationError, match="no material assigned"):
            document.problem(doc.Objects[0])


class TestGeometryTheUserActuallyDrew:
    """A translated model must describe the same structure, not a similar one.

    Every case here produces a Problem that meshes, pre-flights clean and
    solves. The only thing wrong with each is that it is a different device.
    """

    def test_a_material_bound_to_a_face_stays_a_face(self):
        """Binding copper to the top face of a substrate is how a ground plane
        gets drawn. Reading the owning solid's box instead turns that sheet into
        a block filling the whole board - 1.6 mm of solid copper where the
        user drew a foil, shorting out the very dielectric it sits on.
        """
        board = substrate()
        board.Shape.face("Top", (-LENGTH / 2, -BOARD / 2, HEIGHT), (LENGTH / 2, BOARD / 2, HEIGHT))
        doc = model()
        replaces(doc, "Substrate", board)
        part(doc, "DielectricBinding").References = [(board, [])]  # FR4 to the solid
        part(doc, "GroundBinding").References = [(board, ["Top"])]  # copper to one face

        foil = next(
            s
            for s in document.problem(doc.Objects[0]).solids
            if s.material == "Copper" and "Top" in s.label
        )
        assert foil.lower[2] == pytest.approx(HEIGHT)
        assert foil.upper[2] == pytest.approx(HEIGHT), (
            "the binding named a face; the solid it belongs to is 1.6 mm thick"
        )

    def test_each_named_face_becomes_its_own_region(self):
        """Not their union. Two faces on different planes union into a box that
        is neither of them - here, the whole substrate."""
        board = substrate()
        board.Shape.face("Top", (-LENGTH / 2, -BOARD / 2, HEIGHT), (LENGTH / 2, BOARD / 2, HEIGHT))
        board.Shape.face("Bottom", (-LENGTH / 2, -BOARD / 2, 0.0), (LENGTH / 2, BOARD / 2, 0.0))
        doc = model()
        replaces(doc, "Substrate", board)
        part(doc, "DielectricBinding").References = [(board, [])]
        part(doc, "GroundBinding").References = [(board, ["Top", "Bottom"])]

        foils = [
            s
            for s in document.problem(doc.Objects[0]).solids
            if s.material == "Copper" and s.label.startswith("Substrate:")
        ]
        assert len(foils) == 2
        assert {f.lower[2] for f in foils} == {0.0, HEIGHT}
        assert all(f.lower[2] == f.upper[2] for f in foils)

    def test_a_thick_ground_plane_is_driven_to_its_surface(self):
        """Its mid-plane is inside the metal.

        A ground drawn as real copper rather than a face is 35 um thick. Taking
        the middle of that box puts the port's far end half a conductor deep in
        PEC, so the port drives across a gap longer than the real one. The
        zero-thickness ground the other fixtures use hides this exactly: its
        middle and its surface are the same number.
        """
        foil = Shape((-LENGTH / 2, -BOARD / 2, -0.035), (LENGTH / 2, BOARD / 2, 0.0))
        foil.face("Face1", (-LENGTH / 2, -BOARD / 2, -0.035), (LENGTH / 2, BOARD / 2, 0.0))
        thick = Obj("Part::Box", "Ground", foil)

        strip = trace()
        doc = model(port=microstrip_port(1, strip, thick))
        replaces(doc, "Ground", thick)
        part(doc, "GroundBinding").References = [(thick, [])]

        port = document.problem(doc.Objects[0]).ports[0]
        assert port.stop[2] == pytest.approx(0.0), (
            f"the port ends at z={port.stop[2]:.4g}, inside the copper, "
            "rather than on the surface facing the trace"
        )

    def test_a_thick_trace_is_flattened_onto_the_substrate(self):
        """Downward, not upward. MSLPort's strip is geometrically flat, and the
        closed forms this is gated against assume it sits on the dielectric."""
        strip = Shape((-LENGTH / 2, -WIDTH / 2, HEIGHT), (LENGTH / 2, WIDTH / 2, HEIGHT + 0.035))
        strip.face(
            "Face1", (-LENGTH / 2, -WIDTH / 2, HEIGHT), (-LENGTH / 2, WIDTH / 2, HEIGHT + 0.035)
        )
        thick = Obj("Part::Box", "Trace", strip)

        doc = model(port=microstrip_port(1, thick, ground()))
        replaces(doc, "Trace", thick)
        part(doc, "TraceBinding").References = [(thick, [])]

        port = document.problem(doc.Objects[0]).ports[0]
        assert port.start[2] == pytest.approx(HEIGHT)

    def test_a_lumped_port_on_a_solid_is_refused(self):
        """The comment said "select faces"; only a 1-D selection was refused.

        A whole 3 mm trace passes every extent check, and the port comes out
        spanning the entire solid - a parasitic sheet resistor across the
        board rather than a localised feed.
        """
        strip = Shape((-LENGTH / 2, -WIDTH / 2, HEIGHT), (LENGTH / 2, WIDTH / 2, HEIGHT + 0.035))
        strip.face(
            "Face1", (-LENGTH / 2, -WIDTH / 2, HEIGHT), (-LENGTH / 2, WIDTH / 2, HEIGHT + 0.035)
        )
        thick = Obj("Part::Box", "Trace", strip)

        port = lumped_port(1, thick, ground(), SourceEntity=(thick, []))
        with pytest.raises(document.TranslationError, match="That is a solid"):
            document.problem(model(port=port).Objects[0])

    def test_a_negative_length_would_reverse_the_port(self):
        """It would put the far face behind the near one, and the adapter's Port
        would read that as a port facing the other way - launching the wave
        out of the structure instead of into it.

        Not reachable from the GUI: ``App::PropertyLength`` clamps a negative to
        zero, measured on FreeCAD 1.1.1, and so do ``FeedOffset``,
        ``MinElementSize`` and ``ElementSize``. What is reachable is
        ``App::PropertyFloat`` - ``FeedResistance``, ``Resistance`` - and
        ``App::PropertyInteger``, and those keep a negative as typed. This one
        stays because ``document.problem`` may be handed a study assembled
        somewhere other than the property editor, but it is an internal
        invariant rather than the user-facing refusal its old name promised.
        """
        refuses_a_microstrip("reverses the port", Length=-5.0)

    def test_a_negative_feed_resistance_is_refused(self):
        refuses_a_microstrip("negative resistance", FeedResistance=-50.0)

    def test_a_negative_lumped_resistance_is_refused(self):
        """openEMS does not refuse it; it raises UnboundLocalError.

        ``LumpedPort`` binds its element in the ``R > 0`` and ``R == 0``
        branches only, so a negative R falls off the end of the if-chain and
        fails inside the bindings on a name that means nothing here.
        """
        doc = model(port=lumped_port(1, trace(), ground(), Resistance=-1.0))
        with pytest.raises(document.TranslationError, match="negative resistance"):
            document.problem(doc.Objects[0])


class TestOneObjectDrawnInSeveralLumps:
    """Two lumps of one object are two objects to everything downstream.

    A compound is what a boolean, a fillet and a chamfer all hand back, and what
    a user gets from drawing two pads in one operation - so this is the ordinary
    shape of a drawing and not an exotic one. What makes it its own case is that
    the lumps arrive together: nothing outside the object is paired with them,
    so if they are not measured against each other they are not measured at all.

    A dielectric throughout, because a gap is measured whatever the material is
    and the stand-in shapes here carry no surface for a conductor's own demands
    to be read off. A metal compound is a corpus specimen.
    """

    #: The finer arm of the rate below, and half the coarser one. Both are well
    #: under any cell the bulk sizing would choose here, which is what the rate
    #: needs: a gap approaching the bulk cell stops binding, and the two arms
    #: would then be scored on a grid neither of them set.
    GAP = 0.15

    #: How far the rate may sit from the ratio of the two gaps. A grid lays
    #: whole cells across a span, so the size it settles on is that span over a
    #: count and moves in steps of its own - two grids drawn to demands a factor
    #: of two apart need not scale by exactly two.
    PLACEMENT = 0.05

    def lumps(self, gap=GAP):
        """One object holding two solids ``gap`` apart along x, and the solids.

        Clear of the board in z, so nothing here shares space with the
        acceptance geometry the rest of the document is.
        """
        near = Shape((-12.0, -2.0, 5.0), (-8.0, 2.0, 9.0))
        far = Shape((-8.0 + gap, -2.0, 5.0), (-4.0 + gap, 2.0, 9.0))
        return Obj("Part::Feature", "Lumps", Compound(near, far)), near, far

    def across(self, gap):
        """The finest cell the grid puts anywhere across a gap of ``gap``.

        The lumps are drawn short of their own boxes so they reach the mesher as
        triangles, which is what a curved pad does and what makes the gap
        something only a measurement can find - a pair of boxes pins its own
        faces and the thirds rule sizes what is between them.
        """
        near = Shape((-12.0, -2.0, 5.0), (-8.0, 2.0, 9.0), fill=0.71)
        far = Shape((-8.0 + gap, -2.0, 5.0), (-4.0 + gap, 2.0, 9.0), fill=0.71)
        obj = Obj("Part::Feature", "Lumps", Compound(near, far))
        doc = model()
        _add(doc, obj)
        _add(doc, binding("LumpBinding", fr4(), obj))
        lines = document.mesh(doc.Objects[0]).grid.x
        return min(b - a for a, b in zip(lines, lines[1:]) if a < -8.0 + gap and b > -8.0)

    def test_each_lump_reaches_the_mesher_as_itself(self):
        """The whole of the fault: a piece handed the object it came out of
        measures the distance from a shape to itself, which is zero, and zero
        is two solids touching rather than a gap."""
        obj, near, far = self.lumps()
        assert [piece.shape for piece in geometry.solid_boxes(obj)] == [near, far]

    def test_a_shape_that_is_one_lump_is_still_its_own_piece(self):
        """The other half of it. Reaching into ``Solids`` unconditionally would
        pass the test above and hand a face's piece the solid behind it."""
        board = substrate()
        assert [piece.shape for piece in geometry.solid_boxes(board)] == [board.Shape]

    def test_the_grid_follows_the_gap_between_them(self):
        """And the consequence, read through the document rather than beside it.

        Scored as a rate and not as a size: halving the clearance has to halve
        the cells laid across it. A grid that never measured the gap answers the
        bulk cell to both and the rate comes back one, whatever the drawing did
        - where a test comparing one grid against one length would pass on any
        model whose bulk size happened to land under it.
        """
        wide, narrow = self.across(2.0 * self.GAP), self.across(self.GAP)
        assert wide / narrow == pytest.approx(2.0, rel=self.PLACEMENT), (
            f"halving the gap took the cells across it from {wide:.4g} to "
            f"{narrow:.4g}, so the grid there is not the gap's doing"
        )


class TestOneObjectDrawnInSeveralSheets:
    """The same object drawn as surfaces, which do not decompose the way volumes do.

    A volume is its solids. A surface is not its faces: a shell is one surface
    however many faces were fused into it, and an unclosed shell is a shape the
    engine reads as containing no point at all - so taking a shape apart by its
    faces would hand back pieces each held exactly where the whole of it has to
    be refused. The regions are the shells plus every face belonging to none,
    and two pads drawn in one operation are the second kind.

    What that rule rests on is the kernel's, so the corpus is where it is
    asserted; here is the dispatch it turns on.
    """

    def pads(self):
        """One object holding two sheets a clearance apart, and the sheets.

        Cut across a corner so neither can be laid as rectangles, because a pair
        that can pins its own grid lines on every side and nothing between them
        is left to measure. Clear of the board in z for the reason the lumps
        above are.
        """
        near = self.pad(-12.0)
        far = self.pad(-7.0)
        return Obj("Part::Feature", "Pads", Compound(near, far)), near, far

    @staticmethod
    def pad(left):
        """One sheet spanning 4 mm from ``left``, with a corner taken off it."""
        return Shape(
            (left, -2.0, 5.0),
            (left + 4.0, 2.0, 5.0),
            rings=[[(left, -2.0), (left + 4.0, -2.0), (left + 4.0, 0.0), (left, 2.0)]],
        )

    def test_each_sheet_reaches_the_mesher_as_itself(self):
        """The whole of the fault: one piece covering both sheets makes the
        clearance between them a distance from a shape to itself, which is zero,
        and zero is two conductors touching rather than a gap."""
        obj, near, far = self.pads()
        assert [piece.shape for piece in geometry.solid_boxes(obj)] == [near, far]

    def test_an_object_that_is_one_sheet_is_still_one_piece(self):
        """The other half of it, and what keeps the split from being
        unconditional: one region is the whole object, under the name it was
        drawn with. Splitting it anyway hands back the region instead, under a
        number nobody drew and a shape the object is not.
        """
        sheet = Shape((-12.0, -2.0, 5.0), (-8.0, 2.0, 5.0))
        obj = Obj("Part::Feature", "Pad", Compound(sheet))
        assert [(piece.label, piece.shape) for piece in geometry.solid_boxes(obj)] == [
            ("Pad", obj.Shape)
        ]


class TestZeroMeansOppositeThingsToTheTwoResistances:
    """One number, two quantities, and openEMS reads zero differently for each.

    ``MSLPort`` spells "no series resistor" as an infinite ``Feed_R`` and
    reserves ``Feed_R == 0`` for a metal short across the feed. ``LumpedPort``
    has no infinity: ``R == 0`` lays metal across the gap, which is a real
    configuration and the only way to ask for one.

    So a microstrip's zero must become "omit the keyword", and a lumped port's
    zero must travel as a zero. Collapsing both through ``value or None`` - one
    idiom, applied to two meanings - turns a user's 0 ohm lumped element into a
    port the envelope refuses, since a lumped port with no resistance is not a
    thing openEMS can build.
    """

    def test_a_microstrip_zero_becomes_no_resistor_at_all(self):
        assert translated_microstrip(FeedResistance=0.0).feed_resistance is None

    def test_a_microstrip_keeps_a_real_one(self):
        assert translated_microstrip(FeedResistance=50.0).feed_resistance == 50.0

    def test_a_lumped_zero_stays_a_zero(self):
        problem = document.problem(
            model(port=lumped_port(1, trace(), ground(), Resistance=0.0)).Objects[0]
        )
        assert problem.ports[0].feed_resistance == 0.0

    def test_a_lumped_resistance_is_carried_as_given(self):
        problem = document.problem(
            model(port=lumped_port(1, trace(), ground(), Resistance=25.0)).Objects[0]
        )
        assert problem.ports[0].feed_resistance == 25.0


# ---------------------------------------------------------------------------
# Real solves. These prove the translation against physics rather than shape.
# ---------------------------------------------------------------------------

WG_FREQ_MIN, WG_FREQ_MAX = 20e9, 26e9

#: lambda/30 in vacuum at the top of the band, as a fraction, because that is
#: how the document states mesh policy. Matches the hand-built WR-42 gate.
WG_RES_FRACTION = 1.0 / 30


def _wr42_document(port_depth_cells=5):
    """WR-42 as a user would actually draw it: an air box, walls as boundaries.

    The ports sit on the guide's own end faces, because those are the only
    faces of a solid box that FreeCAD lets you pick. The hand-built gate insets
    its ports ten cells; this asks whether the reachable configuration works.
    """
    guide = Shape((0, 0, 0), (GUIDE_A, GUIDE_B, GUIDE_L))
    guide.face("Near", (0, 0, 0), (GUIDE_A, GUIDE_B, 0))
    guide.face("Far", (0, 0, GUIDE_L), (GUIDE_A, GUIDE_B, GUIDE_L))
    obj = Obj("Part::Box", "Guide", guide)
    air = Obj(
        "EMMaterial",
        "Air",
        MaterialType="Dielectric",
        Permittivity=1.0,
        Permeability=1.0,
        Conductivity=0.0,
        LossTangent=0.0,
        MeasuredAt=0.0,
        Thickness=0.0,
    )

    wavelength = SPEED_OF_LIGHT / WG_FREQ_MAX * 1e3
    depth = port_depth_cells * WG_RES_FRACTION * wavelength

    def port(number, face, axis, excite):
        return Obj(
            "EMPortRectWaveguide",
            f"WG{number}",
            Number=number,
            Excitation=excite,
            FeedResistance=0.0,
            ReferenceImpedance=50.0,
            ReferencedTo=FIXED_IMPEDANCE,
            Length=depth,
            CrossSection=(obj, [face]),
            PropagationAxis=axis,
            Mode="TE10",
        )

    settings = mesh_settings(
        ElementsPerWavelength=1.0 / WG_RES_FRACTION,
        EdgeRefinement=1.0,
        **{f"AirCells{a}{s}": 0 for a in "XYZ" for s in ("Min", "Max")},
        **{f"Padding{a}{s}": "Air" for a in "XYZ" for s in ("Min", "Max")},
    )
    solver_obj = simulation(
        MaxTimesteps=12000,
        BoundaryXMin="PEC",
        BoundaryXMax="PEC",
        BoundaryYMin="PEC",
        BoundaryYMax="PEC",
    )
    return Document(
        analysis(FrequencyStart=WG_FREQ_MIN, FrequencyStop=WG_FREQ_MAX),
        settings,
        obj,
        binding("AirBinding", air, obj),
        port(1, "Near", "Z", True),
        port(2, "Far", "-Z", False),
        solver_obj,
    ), depth


@pytest.mark.slow
def test_a_document_waveguide_has_the_phase_constant_theory_says(interpreter, tmp_path):
    """The waveguide gate, driven from a document instead of by hand.

    Exact, unlike the microstrip case: a rectangular guide's phase constant is
    closed-form, with no empirical fit and no error bar. What is asserted is
    what the solver *measures* - the phase of the wave that crossed the guide
    - and not its reference impedance, which RectWGPort computes analytically
    from the mode and would therefore be comparing theory against itself.
    """
    import numpy as np

    from Microwave.Solvers.openems import preflight, read, run, write
    from tests.analytic import reference

    doc, depth = _wr42_document()
    problem = document.problem(doc.Objects[0])
    preflight.refuse_if_blocked(preflight.check(problem))

    # Probes sit on each port box's far face, so this is what the fit recovers.
    separation = (GUIDE_L - 2 * depth) * 1e-3

    run.run(write.write(problem, tmp_path), interpreter=interpreter)
    solved = read.read(tmp_path)

    beta = np.real(reference.phase_constant(solved.frequency, GUIDE_A * 1e-3, GUIDE_B * 1e-3, 1, 0))
    slope, _ = np.polyfit(beta, np.unwrap(np.angle(solved.s(2))), 1)
    recovered = -slope
    error = (recovered - separation) / separation

    assert abs(error) < 0.01, (
        f"the phase of S21 implies the measurement planes are "
        f"{recovered * 1e3:.3f} mm apart; the document puts them "
        f"{separation * 1e3:.3f} mm apart ({error * 100:+.2f}%)"
    )


@pytest.mark.slow
def test_a_document_waveguide_is_matched_and_conserves_power(interpreter, tmp_path):
    """Two exact statements about a hollow PEC guide, from the same solve.

    A lossless guide must return everything it does not pass, and a uniform one
    must barely reflect. Both are properties of the *structure*, so they fail if
    the translation put a wall or a port plane somewhere the document did not.
    """
    import numpy as np

    from Microwave.Solvers.openems import preflight, read, run, write

    problem = document.problem(_wr42_document()[0].Objects[0])
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(write.write(problem, tmp_path), interpreter=interpreter)
    solved = read.read(tmp_path)

    reflected, through = np.abs(solved.s(1)), np.abs(solved.s(2))

    assert float(np.max(reflected)) < 0.02, (
        f"|S11| reaches {float(np.max(reflected)):.4f}; a uniform guide should "
        "barely reflect, so something is discontinuous"
    )
    # Asserted alongside the transmission floor on purpose: energy conservation
    # alone is satisfied by a perfect reflector, so a translation that filled
    # the guide with PEC would pass it (S11 = 1, S21 = 0) while describing a
    # solid block. Neither assertion means much without the other.
    assert float(np.min(through)) > 0.9, (
        f"only {float(np.min(through)):.4f} of the wave gets through; the guide "
        "is obstructed, not hollow"
    )
    leakage = float(np.max(np.abs(reflected**2 + through**2 - 1)))
    assert leakage < 0.005, (
        f"power is off by {leakage:.2e}; a hollow PEC guide in vacuum is "
        "lossless, so the only slack is absorber leakage at the two ends"
    )


@pytest.mark.slow
def test_a_document_lumped_port_presents_the_resistance_it_was_given(interpreter, tmp_path):
    """A lumped resistor terminating the acceptance line.

    Deliberately mismatched. A matched load only proves the reflection is
    small, which it would also be if the port were nearly anything near 50 ohm;
    a 200 ohm load has to produce a *specific* reflection, and the formula for
    it is exact:

        |S11| = |(R - Z0) / (R + Z0)|

    Nothing in it is empirical. Z0 is not taken from a closed form either - it
    is what the microstrip port measures in the same run, so the two port kinds
    check each other and no formula is compared against itself. Both numbers come
    out of the solve on the ``GATE`` line; the same load set to 50 ohm reads
    0.0177 (-35 dB), which is the contrast the mismatch is chosen for.

    Sensitivity is the point: a port presenting 100 ohm instead of 200 would
    read 0.33 here, nowhere near the tolerance.

    It is also the standing proof that a **zero-thickness** lumped box is
    legitimate. The terminator is fed from the trace's end face, so its box is a
    plane with no extent across x - and that plane sits 39.7 um from the nearest
    grid line, because the thirds rule puts lines either side of a conductor edge
    rather than on it. openEMS snaps a lumped element's box to the mesh, so it
    lands on a line anyway: solved as drawn, moved half a cell, and given a cell
    of thickness, all three agree, because openEMS snaps a lumped element to
    the mesh.
    """
    import numpy as np

    from Microwave.Solvers.openems import preflight, read, run, write

    load = 200.0
    strip, plane = trace(), ground()
    strip.Shape.face("Far", (LENGTH / 2, -WIDTH / 2, HEIGHT), (LENGTH / 2, WIDTH / 2, HEIGHT))

    feed = microstrip_port(1, strip, plane, Length=70.0, FeedOffset=14.0, MeasurementDistance=28.0)
    terminator = lumped_port(
        2,
        strip,
        plane,
        Excitation=False,
        Resistance=load,
        ReferenceImpedance=load,
        ReferencedTo=FIXED_IMPEDANCE,
        SourceEntity=(strip, ["Far"]),
        ReferenceEntity=(plane, ["Face1"]),
    )
    # The far end is air rather than Through: the line has to actually end for
    # there to be anything for the resistor to terminate.
    doc = model(port=feed, settings=mesh_settings(PaddingXMax="Air"))
    replaces(doc, "Ground", plane)
    replaces(doc, "Trace", strip)
    part(doc, "GroundBinding").References = [(plane, [])]
    part(doc, "TraceBinding").References = [(strip, [])]
    doc.Objects.append(terminator)

    problem = document.problem(doc.Objects[0])
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(write.write(problem, tmp_path), interpreter=interpreter)
    solved = read.read(tmp_path)

    in_band = solved.band(1e9, 2e9)
    z0 = float(np.mean(solved.port(1).impedance[in_band]))
    measured = float(np.mean(np.abs(solved.s(1)[in_band])))
    predicted = abs((load - z0) / (load + z0))
    error = (measured - predicted) / predicted

    print(
        f"\nGATE lumped {load:.0f} ohm on a {z0:.2f} ohm line: |S11| = {measured:.4f}, "
        f"theory {predicted:.4f}, {error * 100:+.2f}%"
    )

    assert abs(error) < 0.05, (
        f"a {load:.0f} ohm termination on a {z0:.2f} ohm line reflected "
        f"{measured:.4f}, but transmission-line theory says {predicted:.4f} "
        f"({error * 100:+.2f}%). The lumped port is not presenting the "
        "resistance the document gave it, or is not where the document put it"
    )


class TestMaterialsResolvePerFace:
    """One solid may carry different materials on different faces."""

    def _two_faced_trace(self):
        """A trace with real thickness, so its faces have area to bind to."""
        top = HEIGHT + 0.035
        strip = Shape((-LENGTH / 2, -WIDTH / 2, HEIGHT), (LENGTH / 2, WIDTH / 2, top))
        strip.face(
            "Face1", (-LENGTH / 2, -WIDTH / 2, HEIGHT), (-LENGTH / 2, WIDTH / 2, top)
        )  # the end the port feeds
        strip.face("Top", (-LENGTH / 2, -WIDTH / 2, top), (LENGTH / 2, WIDTH / 2, top))
        return Obj("Part::Box", "Trace", strip)

    def test_the_port_takes_the_material_bound_to_its_own_face(self):
        """Keyed by object alone, the second binding overwrites the first and
        the port is laid in whichever material happened to be processed last.

        Both faces here are conductors, so nothing downstream would complain
        - the strip would simply be made of the wrong metal, with the wrong
        conductivity, and the loss would come out wrong.
        """
        strip = self._two_faced_trace()
        gold = copper()
        gold.Label = "Gold"
        gold.Conductivity = 4.1e7

        # Air padding, not Through: a region bound to the trace's *end* face
        # would otherwise sit in the band Through clips away.
        doc = model(
            port=microstrip_port(1, strip, ground()),
            settings=mesh_settings(PaddingXMin="Air", PaddingXMax="Air"),
        )
        replaces(doc, "Trace", strip)
        part(doc, "TraceBinding").References = [(strip, ["Face1"])]
        doc.Objects.insert(8, binding("GoldBinding", gold, strip))
        part(doc, "Port1").References = [(strip, ["Top"])]

        assert document.problem(doc.Objects[0]).ports[0].metal == "Copper"

    def test_a_whole_solid_binding_still_covers_a_face_selection(self):
        """The ordinary case: bind the trace, point the port at one of its
        faces, and the two must still agree."""
        assert document.problem(model().Objects[0]).ports[0].metal == "Copper"


class TestTheLumpedPortFootprintIsSymmetric:
    def test_swapping_source_and_reference_does_not_resize_the_port(self):
        """Source and reference name polarity, not size.

        Nothing stops a user picking the ground plane as the source. Deriving
        the port's cross-section from the source alone made that choice change
        the device: the port grew to the whole board and became a parasitic
        sheet resistor across the entire model, which builds and solves.
        """
        strip, plane = trace(), ground()
        forward = document.problem(model(port=lumped_port(1, strip, plane)).Objects[0]).ports[0]
        swapped = document.problem(model(port=lumped_port(1, plane, strip)).Objects[0]).ports[0]

        for dim in (0, 1):
            assert abs(forward.stop[dim] - forward.start[dim]) == pytest.approx(
                abs(swapped.stop[dim] - swapped.start[dim])
            )
        # Only the polarity flips, which is the one thing that should.
        assert forward.excite_sign == -swapped.excite_sign

    def test_entities_that_do_not_face_each_other_are_refused(self):
        """A port between two conductors that do not overlap has no gap to
        drive; the union of them is a box containing mostly empty space."""
        strip, plane = trace(), ground()
        plane.Shape.face(
            "Corner", (LENGTH / 2 - 1, BOARD / 2 - 1, 0.0), (LENGTH / 2, BOARD / 2, 0.0)
        )
        port = lumped_port(1, strip, plane, ReferenceEntity=(plane, ["Corner"]))
        with pytest.raises(document.TranslationError, match="do not overlap"):
            document.problem(model(port=port).Objects[0])


class TestTheGridLandsWherethePortAsked:
    """Both of a microstrip port's planes are set deliberately. The grid has to
    put a line where each one was asked for, and leave the cells there alone.

    ``MSLPort`` snaps rather than failing - the source goes to the nearest
    line and the probe triplet to the three nearest the measurement plane -
    so every fault in this class is silent. It costs a fraction of a cell of
    reference-plane offset, or an order of accuracy in the differencing, and
    the run completes and reports a plausible impedance either way.
    """

    def grid_and_port(self, **overrides):
        import numpy as np

        port = microstrip_port(1, trace(), ground(), **overrides)
        problem = document.problem(model(port=port).Objects[0])
        return np.asarray(problem.grid[0]), problem.ports[0]

    def nearest(self, grid, position):
        import numpy as np

        return float(grid[int(np.argmin(np.abs(grid - position)))])

    def test_a_line_lands_exactly_on_the_source(self):
        """Measured before this was forced: the source asked for x = -30 and
        got -30.167, a sixth of a millimetre away, because nothing pinned it."""
        grid, port = self.grid_and_port()
        wanted = port.start[0] + port.direction * port.feed_shift
        assert self.nearest(grid, wanted) == pytest.approx(wanted, abs=1e-9)

    def test_a_line_lands_exactly_on_the_measurement_plane(self):
        grid, port = self.grid_and_port()
        wanted = port.measurement_position()
        assert self.nearest(grid, wanted) == pytest.approx(wanted, abs=1e-9)

    def test_it_follows_the_feed_wherever_it_is_put(self):
        """A position that is not a round number, so a line landing there
        cannot be one the grading would have produced anyway."""
        grid, port = self.grid_and_port(FeedOffset=17.31)
        wanted = port.start[0] + port.direction * port.feed_shift
        assert self.nearest(grid, wanted) == pytest.approx(wanted, abs=1e-9)

    # Length 0 ends the box at the measurement plane, where the forced line
    # makes the thirds rule stand down by itself - so that case alone cannot
    # tell whether the suppression works. A stated Length puts the box's end
    # somewhere nothing else pins, which is where it is load-bearing.
    @pytest.mark.parametrize("length", [0.0, 70.0])
    def test_the_strip_the_port_lays_does_not_refine_its_own_ends(self, length):
        """The box's far face is not a conductor edge - the metal carries on
        as the user's trace - so the cells there must stay bulk.

        This is the fault that made the box's length matter at all. With the
        thirds rule applied it meshed 0.04 mm there; with only the thirds rule
        suppressed and the matching sizing constraint left in, 0.136 mm. The
        cap is 0.715.
        """
        import numpy as np

        grid, port = self.grid_and_port(Length=length)
        end = port.stop[0]
        index = int(np.argmin(np.abs(grid - end)))
        cells = [grid[index] - grid[index - 1], grid[index + 1] - grid[index]]
        floor = 0.5 * float(np.median(np.diff(grid)))
        assert min(cells) > floor, (
            f"cells at the port box's end are {cells}, refined against a "
            f"median of {np.median(np.diff(grid)):.4f}"
        )

    def test_the_box_length_past_the_probes_changes_nothing(self):
        """The question the box's length was supposed to answer, answered.

        ``MSLPort`` reads its probe triplet off the *global* grid, so the box
        bounds nothing that is measured; its length decides only how much strip
        is laid, over a trace that is already there. Both grids below must be
        identical - if they are not, ending the box at the plane costs
        something and the default is wrong.
        """
        import numpy as np

        short, _ = self.grid_and_port()
        long, _ = self.grid_and_port(Length=LENGTH)
        assert np.array_equal(short, long)


class TestPreflightWarningsAreNotUnconditional:
    """A warning that always fires teaches people to skip reading them."""

    def test_a_healthy_model_produces_no_port_warnings(self):
        from Microwave.Solvers.openems import preflight

        findings = preflight.check(document.problem(model().Objects[0]))
        warnings = [f for f in findings if f.severity == preflight.WARN]
        assert not [w for w in warnings if "near field" in w.message], (
            "the acceptance configuration measures 30 mm from its feed, which "
            "is 0.21 wavelengths; warning about it would make the check noise"
        )
        assert not [w for w in warnings if "uneven" in w.message]

    def test_the_clearance_warning_tracks_the_separation(self):
        """Fires below the limit, silent above it, on the same geometry."""
        from Microwave.Solvers.openems import preflight

        def warned(feed, separation):
            port = microstrip_port(
                1, trace(), ground(), FeedOffset=feed, MeasurementDistance=separation
            )
            findings = preflight.check(document.problem(model(port=port).Objects[0]))
            return any("near field" in f.message for f in findings)

        assert warned(13.6, 1.7), "1.7 mm apart is 0.012 wavelengths"
        assert not warned(13.6, 18.0), "18 mm apart is 0.13 wavelengths"


class TestTheGridInputsDigest:
    """A staleness key that is cheap enough to ask on every recompute.

    The mesher is a pure function of exactly these inputs, so the key is
    complete rather than an approximation - but it costs a translation and no
    meshing, which is what makes an out-of-date badge affordable *and* precise.
    """

    def test_it_is_stable_across_calls(self, study):
        assert document.grid_inputs_digest(study) == document.grid_inputs_digest(study)

    def test_it_moves_when_the_mesh_policy_moves(self, study):
        before = document.grid_inputs_digest(study)
        policy(study).ElementsPerWavelength = 30.0
        assert document.grid_inputs_digest(study) != before

    def test_it_moves_when_the_geometry_moves(self, study):
        """A wider trace puts its edges somewhere else, so the grid must move."""
        before = document.grid_inputs_digest(study)
        strip = next(o for o in study.Document.Objects if o.Name == "Trace")
        wider = Shape((-LENGTH / 2, -WIDTH, HEIGHT), (LENGTH / 2, WIDTH, HEIGHT))
        wider.face("Face1", (-LENGTH / 2, -WIDTH, HEIGHT), (-LENGTH / 2, WIDTH, HEIGHT))
        strip.Shape = wider
        assert document.grid_inputs_digest(study) != before

    def test_it_moves_when_the_frequency_moves(self, study):
        """The wavelength sets every cell size, so this is a mesh input."""
        before = document.grid_inputs_digest(study)
        study.FrequencyStop = 20e9
        assert document.grid_inputs_digest(study) != before

    def test_it_moves_when_the_padding_mode_moves(self, study):
        before = document.grid_inputs_digest(study)
        policy(study).PaddingYMin = "Through"
        assert document.grid_inputs_digest(study) != before

    @pytest.mark.parametrize(
        "name, value",
        [
            ("MaxTimesteps", 999_999),
            ("Threads", 4),
            ("EnergyDecay", -40.0),
            # It scales the timestep, which is not a grid line. The mesh report
            # does show it, but the badge is about the *drawing*, and forcing a
            # remesh to change a number the panel recomputes anyway is how a
            # badge gets ignored.
            ("TimestepFactor", 0.5),
        ],
    )
    def test_it_ignores_what_the_grid_does_not_depend_on(self, study, name, value):
        """These move the envelope and not one cell."""
        before = document.grid_inputs_digest(study)
        setattr(solver_of(study), name, value)
        assert document.grid_inputs_digest(study) == before

    def test_and_how_far_down_the_response_is_read(self, study):
        """On the study rather than the solver, and still not a grid input: it
        decides how a finished run is judged, not how one is meshed."""
        before = document.grid_inputs_digest(study)
        study.SmallestResponse = -40.0
        assert document.grid_inputs_digest(study) == before

    def test_equal_inputs_mean_an_equal_grid(self, study):
        """The claim the key rests on: the mesher is deterministic in these."""
        first = document.mesh(study)
        assert document.grid_inputs_digest(study) == document.grid_inputs_digest(study)
        second = document.mesh(study)
        for dim in range(3):
            assert list(first.lines[dim]) == list(second.lines[dim])

    def test_it_moves_when_a_refinement_region_appears(self, doc, study):
        before = document.grid_inputs_digest(study)
        block = _add(
            doc,
            Obj("Part::Box", "Pad", Shape((-4.0, -1.5, 0.0), (4.0, 1.5, HEIGHT))),
        )
        region = _add(doc, refinement("Fine", block))
        assert document.grid_inputs_digest(study) != before

        moved = document.grid_inputs_digest(study)
        region.ElementSize = 0.02
        assert document.grid_inputs_digest(study) != moved

        moved_again = document.grid_inputs_digest(study)
        block.Shape = Shape((-3.0, -1.5, 0.0), (5.0, 1.5, HEIGHT))
        assert document.grid_inputs_digest(study) != moved_again, (
            "the box moved, so the grid moved, but the key did not"
        )

        resized = document.grid_inputs_digest(study)
        region.MinElementsAcross = 8
        assert document.grid_inputs_digest(study) != resized

        region.MinElementsAcross = 0
        block.Shape = Shape((-4.0, -1.5, 0.0), (4.0, 1.5, HEIGHT))
        region.ElementSize = 0.05
        region.Enabled = False
        assert document.grid_inputs_digest(study) == before

    def test_it_ignores_a_refinement_region_being_relabelled(self, doc, study):
        """The label reaches error messages and not one cell."""
        block = _add(
            doc,
            Obj("Part::Box", "Pad", Shape((-4.0, -1.5, 0.0), (4.0, 1.5, HEIGHT))),
        )
        region = _add(doc, refinement("Fine", block))
        before = document.grid_inputs_digest(study)
        region.Label = "Renamed"
        assert document.grid_inputs_digest(study) == before

    def test_a_different_key_means_a_different_grid(self, study):
        before_key = document.grid_inputs_digest(study)
        before_grid = list(document.mesh(study).lines[2])
        policy(study).ElementsPerWavelength = 30.0
        assert document.grid_inputs_digest(study) != before_key
        assert list(document.mesh(study).lines[2]) != before_grid


class TestATriangulationHasToStillBeTheShape:
    """Two guards on the way in, both of which have to be able to fail.

    A requested tessellation fineness is a hint - FreeCAD returned the same mesh
    for two different requests - so a translation that trusted it would ship
    whatever the kernel felt like producing.
    """

    def board(self, **overrides):
        board = substrate()
        board.Shape = Shape(
            (-LENGTH / 2, -BOARD / 2, 0.0),
            (LENGTH / 2, BOARD / 2, HEIGHT),
            fill=0.71,
            **overrides,
        )
        doc = model()
        replaces(doc, "Substrate", board)
        part(doc, "DielectricBinding").References = [(board, [])]
        return doc, board

    def test_the_fineness_asked_for_is_finer_than_the_shape_is_thin(self):
        """A chord error the size of the feature is no triangulation at all.

        Asserted against the shape rather than against the constant that sets
        it, which would move with any change and so pin nothing.
        """
        doc, board = self.board()
        document.problem(doc.Objects[0])
        assert 0.0 < board.Shape.asked_deflection < HEIGHT

    def test_and_it_follows_the_shape_rather_than_being_fixed(self):
        """A thinner solid is asked for a finer triangulation, in proportion:
        one absolute fineness would over-tessellate a board and under-tessellate
        a foil."""
        doc, thick = self.board()
        document.problem(doc.Objects[0])
        board = substrate()
        board.Shape = Shape(
            (-LENGTH / 2, -BOARD / 2, 0.0), (LENGTH / 2, BOARD / 2, HEIGHT / 4), fill=0.71
        )
        thin_doc = model()
        replaces(thin_doc, "Substrate", board)
        part(thin_doc, "DielectricBinding").References = [(board, [])]
        document.problem(thin_doc.Objects[0])
        assert board.Shape.asked_deflection == pytest.approx(thick.Shape.asked_deflection / 4)

    def test_asking_more_finely_is_what_happens_before_refusing(self):
        """A first answer that misses the bound is not evidence about the shape.

        The request is a hint, so a coarse answer says nothing about what a
        finer request would give - and on a real kernel several requests an
        order apart return the identical mesh. So it is sharpened and asked
        again, and only a shape that will not improve is turned away.
        """
        doc, board = self.board(coarsens=0.5 / (geometry.DEFLECTION_OF_EXTENT * HEIGHT))
        assert document.problem(doc.Objects[0]).solids
        assert board.Shape.asked_deflection < geometry.DEFLECTION_OF_EXTENT * HEIGHT

    def test_a_triangulation_that_will_not_improve_is_refused(self):
        """Not a tolerance on a fit: it is the difference between solving what
        was drawn and solving something else that fits in the same box."""
        doc, _ = self.board(coarsens=0.5 / (geometry.DEFLECTION_OF_EXTENT * HEIGHT), stubborn=True)
        with pytest.raises(document.TranslationError, match="losing"):
            document.problem(doc.Objects[0])

    def test_and_a_loss_inside_the_bound_is_accepted(self):
        doc, _ = self.board(
            coarsens=0.5 * geometry.MAX_SHAPE_LOSS / (geometry.DEFLECTION_OF_EXTENT * HEIGHT),
            stubborn=True,
        )
        assert document.problem(doc.Objects[0]).solids

    def test_the_refusal_says_it_already_tried_asking_more_finely(self):
        doc, _ = self.board(coarsens=0.5 / (geometry.DEFLECTION_OF_EXTENT * HEIGHT), stubborn=True)
        with pytest.raises(document.TranslationError, match="sharper requests"):
            document.problem(doc.Objects[0])


class TestASolidBelowTheOriginIsCarriedAllTheSame:
    """openEMS casts its containment ray outward from a shape's *maximum*
    corner, and scaling a negative coordinate moves it further from the origin
    rather than away from the shape. For a solid lying below the origin on every
    axis the ray ends inside it and every answer is inverted - measured, and it
    is silent.

    Nothing about that is a property of the drawing, so nothing here refuses
    one. It is a property of where openEMS is asked to hold it, and the driver
    holds every structure at the origin."""

    def solid(self, upper):
        from Microwave.Solvers.openems.model import Solid

        span = [u - 1.0 for u in upper]
        unit = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]
        return Solid(
            material="copper",
            lower=tuple(span),
            upper=tuple(upper),
            label="Reflector",
            vertices=tuple(tuple(span[i] + u[i] for i in range(3)) for u in unit),
            faces=(
                (0, 3, 2),
                (0, 2, 1),
                (4, 5, 6),
                (4, 6, 7),
                (0, 1, 5),
                (0, 5, 4),
                (1, 2, 6),
                (1, 6, 5),
                (2, 3, 7),
                (2, 7, 6),
                (3, 0, 4),
                (3, 4, 7),
            ),
        )

    @pytest.mark.parametrize(
        "upper",
        [
            pytest.param((-1.0, -1.0, -1.0), id="wholly below the origin"),
            pytest.param((-1.0, -1.0, 0.5), id="one maximum above it"),
            pytest.param((50.0, 30.0, 0.0), id="a board sitting on the origin plane"),
            pytest.param((50.0, 30.0, 12.0), id="wholly above it"),
        ],
    )
    def test_the_engine_is_given_it_at_the_origin(self, upper):
        from Microwave.Solvers.openems.model import origin_offset

        solid = self.solid(upper)
        placed = solid.moved(origin_offset([solid]))
        assert min(min(placed.lower), min(min(v) for v in placed.vertices)) == pytest.approx(
            0.0, abs=0.0
        )

    def test_which_leaves_its_size_and_its_shape_alone(self):
        solid = self.solid((-1.0, -1.0, -1.0))
        placed = solid.moved(origin_offset([solid]))
        assert placed.faces == solid.faces
        for mine, theirs in zip(placed.vertices, solid.vertices):
            assert [b - a for a, b in zip(mine, placed.lower)] == pytest.approx(
                [b - a for a, b in zip(theirs, solid.lower)]
            )


class TestATriangulatedSolidIsSizedAsAMaterial:
    """It contributes no region, so everything a region asked for on its behalf
    has to be asked another way or is knowingly not asked at all."""

    def problem(self):
        board = substrate()
        board.Shape = Shape(
            (-LENGTH / 2, -BOARD / 2, 0.0), (LENGTH / 2, BOARD / 2, HEIGHT), fill=0.71
        )
        doc = model()
        replaces(doc, "Substrate", board)
        part(doc, "DielectricBinding").References = [(board, [])]
        return document.problem(doc.Objects[0])

    def test_it_contributes_no_region(self):
        """Everything a Region does is reasoning about a box that *is* the
        shape - pinning its faces, the thirds rule, asking whether another
        conductor covers it. Given a box that merely bounds one, each of those
        is wrong quietly."""
        from Microwave.Solvers.openems.write import regions

        problem = self.problem()
        kinds = {m.name: m.kind for m in problem.materials}
        bounds = ((-1e3,) * 3, (1e3,) * 3)
        shapes = regions(problem.solids, problem.ports, kinds, bounds)
        assert not [r for r in shapes if r.label == "Substrate"]

    def test_it_keeps_its_material_bulk_size(self):
        """A rotated board must be meshed as finely inside as a flat one: the
        wave's speed is a property of the material, not of the shape."""
        from Microwave.Solvers.openems.write import features

        problem = self.problem()
        found = features(problem.solids, {"FR4": 0.5})
        assert [f.source for f in found] == ["'Substrate' bulk"]
        assert found[0].cells()[0] == pytest.approx(0.5)

    def test_and_asks_nothing_when_no_bulk_size_was_given(self):
        from Microwave.Solvers.openems.write import features

        assert features(self.problem().solids, None) == []

    def test_a_coarsened_one_asks_at_the_size_it_settled_for(self):
        """The route the driver takes, which reads solids back off the file.

        A triangulated solid contributes no region, so this demand is the whole
        of what it asks the grid - and it is the geometry a coarsening is most
        likely to be aimed at, since a box carries no detail to give up.
        """
        from Microwave.Solvers.openems.write import features

        problem = self.problem()
        loose = [dataclasses.replace(s, relaxed_to=4.0) for s in problem.solids]
        assert features(problem.solids, {"FR4": 0.5})[0].cells()[0] == pytest.approx(0.5)
        assert features(loose, {"FR4": 0.5})[0].cells()[0] == pytest.approx(4.0)

    def test_a_document_holding_one_still_meshes(self):
        """The whole path, rather than its pieces: a curved solid reaches a
        finished grid without refusing and without exploding."""
        problem = self.problem()
        assert all(len(problem.grid[dim]) > 4 for dim in range(3))

    def test_the_grid_it_produces_is_a_grid_anybody_could_solve(self):
        """The bounding-box thickness is deliberately not fed to the connection
        criterion, and this is what that decision buys: a box states extents and
        no direction, so a thin solid would otherwise demand cells that fit
        inside its thickness across the whole of its length."""
        problem = self.problem()
        shape = [len(problem.grid[dim]) for dim in range(3)]
        assert math.prod(shape) < 1_000_000, shape


class TestASheetIsFoundAtItsOwnPlane:
    """A zero-thickness conductor exists only where a grid line falls exactly on
    it: openEMS applies metal at E-field sample points, and for a sheet those sit
    on a main-grid line of its normal axis. Off the line it is not modelled at
    all, and the run completes having solved a board with no trace on it.

    A sheet cut into rectangles gets that from its own region. One carried as an
    area has no region, so the plane is required of the mesher directly.
    """

    def sheet(self):
        problem = document.problem(diagonal_trace().Objects[0])
        return problem, next(s for s in problem.solids if s.label.startswith("Trace"))

    def meshed_at(self, elevation):
        """A grid built around one triangulated sheet at ``elevation``.

        Placed where nothing else in the model has a face, so that the line
        being looked for can only have come from the sheet. Asserting it on a
        trace lying on a substrate proves nothing: the substrate's own top face
        asks for a line at the same plane.
        """
        from Microwave.Solvers.openems.mesh import MeshParams
        from Microwave.Solvers.openems.model import Material, Port, Solid
        from Microwave.Solvers.openems.write import plan_mesh

        sheet = Solid(
            material="copper",
            lower=(0.0, 0.0, elevation),
            upper=(4.0, 4.0, elevation),
            label="Patch",
            vertices=(
                (0.0, 0.0, elevation),
                (4.0, 0.0, elevation),
                (4.0, 4.0, elevation),
                (0.0, 4.0, elevation),
            ),
            faces=((0, 1, 2), (0, 2, 3)),
            sheet_normal=2,
        )
        ports = [
            Port(
                number=1,
                kind="lumped",
                start=(1.0, 1.0, 0.0),
                stop=(2.0, 2.0, 1.0),
                excitation_axis=2,
                propagation_axis=0,
                feed_resistance=50.0,
            )
        ]
        lines, _, _ = plan_mesh(
            [sheet],
            ports,
            [Material(name="copper", kind="pec")],
            MeshParams(metal_res=0.5, dielectric_res=2.0),
        )
        return lines

    def test_the_grid_holds_a_line_exactly_at_the_sheet(self):
        """Exactly, not nearly: one ulp out and the conductor is not
        discretised, and openEMS says so only as an "Unused primitive" warning
        whose benign form reads identically."""
        elevation = 3.37
        lines = self.meshed_at(elevation)
        assert elevation in {float(v) for v in lines[2]}

    def test_and_the_sheet_is_what_put_it_there(self):
        """Moving the sheet moves the line, so nothing else in the model is
        supplying it."""
        assert 2.11 in {float(v) for v in self.meshed_at(2.11)[2]}
        assert 2.11 not in {float(v) for v in self.meshed_at(3.37)[2]}


class TestASheetIsLaidAtExactlyOnePlane:
    """A drawing is flat to within the kernel's tolerance rather than exactly, so
    a triangulation of one spans a hair on that axis. openEMS models a sheet at a
    single elevation, so that hair has to go somewhere deliberate.
    """

    def sheet(self, **overrides):
        trace = diagonal_trace()
        shape = next(o for o in trace.Objects if o.Label == "Trace").Shape
        for name, value in overrides.items():
            setattr(shape, name, value)
        problem = document.problem(trace.Objects[0])
        return next(s for s in problem.solids if s.label.startswith("Trace"))

    def test_a_triangulation_a_hair_off_the_plane_is_collapsed_onto_it(self):
        """Rather than carried through and refused by the envelope, which would
        blame the user for a thickness they did not draw."""
        sheet = self.sheet(wobble=geometry.FLATNESS / 10.0)
        axis = sheet.sheet_normal
        assert sheet.lower[axis] == sheet.upper[axis]
        assert {point[axis] for point in sheet.vertices} == {sheet.lower[axis]}

    def test_and_it_is_collapsed_onto_the_plane_the_shape_was_judged_flat_at(self):
        """Not onto wherever the triangulation happened to land, which would
        move the conductor by the size of the kernel's own tolerance."""
        sheet = self.sheet(wobble=geometry.FLATNESS / 10.0)
        assert sheet.lower[sheet.sheet_normal] == pytest.approx(HEIGHT, abs=0.0)


class TestASheetsAreaHasToSurviveTriangulation:
    """The same guard the solid path has, on the quantity a sheet has instead of
    a volume. Without it an outline could be triangulated to something coarser
    than was drawn and solved as that."""

    def board(self, **overrides):
        trace = diagonal_trace()
        shape = next(o for o in trace.Objects if o.Label == "Trace").Shape
        for name, value in overrides.items():
            setattr(shape, name, value)
        return trace

    def test_asking_more_finely_is_what_happens_before_refusing(self):
        doc = self.board(coarsens=0.5 / (geometry.DEFLECTION_OF_EXTENT * WIDTH))
        assert document.problem(doc.Objects[0]).solids

    def test_an_outline_that_will_not_improve_is_refused(self):
        doc = self.board(coarsens=0.5 / (geometry.DEFLECTION_OF_EXTENT * WIDTH), stubborn=True)
        with pytest.raises(document.TranslationError, match="losing"):
            document.problem(doc.Objects[0])

    def test_and_the_refusal_talks_about_area_rather_than_volume(self):
        """A sheet has no volume, so quoting one would describe a measurement
        that was never taken."""
        doc = self.board(coarsens=0.5 / (geometry.DEFLECTION_OF_EXTENT * WIDTH), stubborn=True)
        with pytest.raises(document.TranslationError) as refusal:
            document.problem(doc.Objects[0])
        assert "area" in str(refusal.value)
        assert "volume" not in str(refusal.value)


class TestAnAreaIsRefusedForBeingAnArea:
    """A surface with a boundary carries no region, and the kernel says so with
    a volume that is rounding either side of zero. Which side it lands on is
    arithmetic noise, so nothing the user is told may turn on it."""

    def surface(self, volume, unclosed=True):
        """A shape carrying no solid, flat on no axis, open unless asked.

        Being flat on no axis is what leaves it nowhere to lie: openEMS lays a
        zero-thickness conductor at one elevation on one axis, and a surface
        tilted across all three has no elevation to be laid at.
        """
        board = substrate()
        board.Shape = Shape((0.0, 0.0, 0.0), (LENGTH, BOARD, HEIGHT), unclosed=unclosed)
        board.Shape.Volume = volume
        board.Shape.Solids = []
        doc = model()
        replaces(doc, "Substrate", board)
        part(doc, "DielectricBinding").References = [(board, [])]
        return doc

    def refusal(self, volume):
        with pytest.raises(document.TranslationError) as refused:
            document.problem(self.surface(volume).Objects[0])
        return str(refused.value)

    def test_it_says_the_shape_is_flat_on_no_axis(self):
        """Rather than that a sheet has to be drawn with axis-aligned edges: an
        outline of any shape is meshed, so a refusal describing a limit the
        layer does not have sends the user to redraw something that would
        already have worked."""
        reason = self.refusal(0.0)
        assert "flat on one of the three axes" in reason
        assert "axis-aligned edges" not in reason

    def test_and_says_the_same_whichever_side_of_zero_the_volume_lands(self):
        """The kernel computes a figure across an open surface either way, and
        the two are the same shape.

        The magnitude is the rounding a figure of this shape's own size carries,
        rather than a number chosen to be small: what has to be told from a
        region is what the arithmetic leaves behind, and that scales with the
        shape.
        """
        rounding = LENGTH * BOARD * HEIGHT * sys.float_info.epsilon
        assert self.refusal(rounding) == self.refusal(-rounding)

    def test_but_an_open_surface_with_a_region_in_it_is_a_fault_in_the_drawing(self):
        """Not an area at all: something enclosing that much and still open has
        a hole in it, and saying it should have been given thickness would send
        the user to fix the wrong thing."""
        reason = self.refusal(LENGTH * BOARD * HEIGHT * 0.71)
        assert "Check geometry" in reason
        assert "flat on one of the three axes" not in reason

    def test_and_a_closed_one_carrying_no_solid_is_meshed(self):
        """The bound is against the space the shape spans rather than against
        zero, so it has to swallow the kernel's rounding and leave a real
        region alone. This is the side that would fail quietly, by refusing
        something meshable."""
        doc = self.surface(LENGTH * BOARD * HEIGHT * 0.71, unclosed=False)
        board = next(s for s in document.problem(doc.Objects[0]).solids if s.label == "Substrate")
        assert board.faces

    def test_and_a_conductor_drawn_the_same_way_is_given_thickness_instead(self):
        """The split this class is one half of: what a dielectric skin leaves
        out decides the answer, and what a metal skin leaves out does not."""
        doc = self.surface(0.0)
        replaces(doc, "DielectricBinding", binding("SkinBinding", copper(), part(doc, "Substrate")))
        solved = document.problem(doc.Objects[0])
        assert any(solid.label == "Substrate" for solid in solved.solids)


class TestAConductorDrawnAsASkinIsGivenThickness:
    """A metal surface carries no thickness and needs none - the field inside a
    conductor is zero either way - so one is supplied rather than demanded."""

    def vacuum_cell(self, doc):
        """The cell metal is meshed at in vacuum, in mm, read off the document.

        Off the objects rather than restated here, so the band or the policy can
        move in the fixture without this quietly describing the old one.
        """
        study, settings = doc.Objects[0], part(doc, "MeshSettings")
        top = float(study.FrequencyStop)
        across = float(settings.ElementsPerWavelength)
        return units.SPEED_OF_LIGHT / top * units.MM_PER_M / across / float(settings.EdgeRefinement)

    def skin(self, **overrides):
        """A conductor drawn as an open surface, flat on no axis."""
        shape = overrides.pop("shape", {})
        doc = model(**overrides)
        board = part(doc, "Substrate")
        board.Shape = Shape((0.0, 0.0, 0.0), (LENGTH, BOARD, HEIGHT), **shape)
        board.Shape.Volume = 0.0
        board.Shape.Solids = []
        replaces(doc, "DielectricBinding", binding("SkinBinding", copper(), board))
        return doc, board

    def test_it_is_offset_by_the_thickness_the_grid_stops_reading(self):
        """Not a length anybody chose. A cross-section is spent as the cell whose
        body diagonal it is, so a thickness of this asks for exactly the cell a
        conductor's edges are already sized at, and never for anything finer."""
        doc, board = self.skin()
        document.problem(doc.Objects[0])
        assert board.Shape.asked_offset == pytest.approx(
            fits_inside(self.vacuum_cell(doc)), rel=1e-12, abs=0.0
        )

    def test_and_what_reaches_the_engine_is_a_region_rather_than_an_area(self):
        """Which is the whole of the change: the same drawing used to reach the
        engine as nothing at all.

        Where that region *is* is not asserted here. A stand-in built out of
        boxes has no side to prefer and no surface to sweep, so the direction the
        offset runs and the space it covers are scored against the kernel in the
        corpus, where both are real.
        """
        doc, _ = self.skin()
        solid = next(s for s in document.problem(doc.Objects[0]).solids if s.label == "Substrate")
        assert solid.faces, "a skin has to leave as a region, not as an area"

    def test_and_the_length_it_was_given_travels_with_it(self):
        """Pre-flight is what announces the length, and it reads the envelope -
        so a thickness that stops at this module is one no run can name."""
        doc, _ = self.skin()
        solid = next(s for s in document.problem(doc.Objects[0]).solids if s.label == "Substrate")
        assert solid.thickened == pytest.approx(
            fits_inside(self.vacuum_cell(doc)), rel=1e-12, abs=0.0
        )

    def test_while_a_drawing_that_carried_its_own_thickness_reports_none(self):
        """Nothing to say, and saying it anyway would put the note on every
        model anybody draws."""
        solved = document.problem(model().Objects[0])
        assert [solid.thickened for solid in solved.solids] == [0.0] * len(solved.solids)

    def test_but_not_where_the_thickness_is_a_large_share_of_the_width(self):
        """A skin is free only while it stays thin across the surface carrying
        it. Once it does not, what would be solved is a bar, and no number coming
        back from it would say so."""
        doc, _ = self.skin(analysis=analysis(FrequencyStart=1e6, FrequencyStop=1e7))
        with pytest.raises(document.TranslationError) as refused:
            document.problem(doc.Objects[0])
        assert "measures across" in str(refused.value)
        assert "bar rather than the sheet" in str(refused.value)

    def test_nor_where_the_metal_built_is_not_a_sweep_of_the_surface(self):
        """The other way a skin stops being one, and the way a width cannot see:
        a surface curving back on itself sweeps far more or far less than its
        area times the thickness, and a flat one sweeps exactly that however
        thick it is told to be. So the solid is measured, not just the drawing.
        """
        doc, board = self.skin()
        board.Shape.swells = 1.0 + 2.0 * geometry.MOST_OF_A_SKIN
        with pytest.raises(document.TranslationError) as refused:
            document.problem(doc.Objects[0])
        assert "the radius this shape turns through" in str(refused.value)
        assert "block rather than the sheet" in str(refused.value)

    def test_and_not_where_the_kernel_will_not_offset_it(self):
        """A surface folded tighter than the thickness has nowhere to put it.
        The kernel is what knows that, so what it says is carried out."""
        doc, board = self.skin()
        board.Shape.offsets = False
        with pytest.raises(document.TranslationError) as refused:
            document.problem(doc.Objects[0])
        assert "would not offset it into a solid" in str(refused.value)
        assert "MakeOffsetShape" in str(refused.value)

    def test_and_a_solid_that_will_not_close_is_still_a_fault_in_the_drawing(self):
        """Thickness is offered to a shape carrying no solid at all. One that
        carries a solid has a thickness already, so a surface of it that will not
        knit is a drawing to fix rather than a skin to thicken."""
        doc = model()
        board = part(doc, "Substrate")
        board.Shape = Shape((0.0, 0.0, 0.0), (LENGTH, BOARD, HEIGHT), fill=0.5, unclosed=True)
        replaces(doc, "DielectricBinding", binding("SkinBinding", copper(), board))
        with pytest.raises(document.TranslationError) as refused:
            document.problem(doc.Objects[0])
        assert "Check geometry" in str(refused.value)


class TestADocumentOlderThanItsClasses:
    """A property added since a file was written, met on the route that reads it.

    FreeCAD stores the properties an object *had* and does not reconcile a
    restored one against its class, so a document written by an earlier build
    comes back without whatever has been added since. Every property here is
    reached by duck typing, so the first read is a bare ``AttributeError`` -
    which the panel shows as an internal error and a traceback, reading as a
    fault in the workbench rather than a document out of step with it.

    Old documents stay unsupported. What is asserted is that being unsupported
    arrives as a sentence naming the object and the property.
    """

    def missing(self, name):
        """A document with one property taken back off whichever object has it."""
        doc = model()
        held = [obj for obj in doc.Objects if hasattr(obj, name)]
        assert len(held) == 1, f"{name} is on {len(held)} objects, so this is ambiguous"
        delattr(held[0], name)
        return doc, held[0]

    def test_a_study_without_a_property_names_it(self):
        doc, _ = self.missing("SmallestResponse")
        with pytest.raises(document.TranslationError) as refused:
            document.problem(doc.Objects[0])
        assert "SmallestResponse" in str(refused.value)
        assert "Analysis" in str(refused.value)

    def test_and_says_what_to_do_about_it(self):
        doc, _ = self.missing("SmallestResponse")
        with pytest.raises(document.TranslationError) as refused:
            document.problem(doc.Objects[0])
        message = str(refused.value)
        assert "re-create" in message.lower()
        assert "earlier build" in message

    def test_it_does_not_claim_to_know_whose_fault_it_is(self):
        """The adapter cannot tell a document missing a property from itself
        misspelling one - they are the same event seen from inside. So the
        message covers both, and a run that is really our bug still tells the
        user to report it."""
        doc, _ = self.missing("SmallestResponse")
        with pytest.raises(document.TranslationError) as refused:
            document.problem(doc.Objects[0])
        assert "fault in the workbench" in str(refused.value)

    #: Properties whose only refusal is this guard, one per object kind it
    #: reaches. ``ElementsPerWavelength`` is deliberately not among them:
    #: ``policy._check_mesh_policy`` names that one itself, so a test using it
    #: passes with the guard taken out and proves nothing.
    UNGUARDED_ELSEWHERE = ("MaxGrowthRatio", "MinElementsAcross", "NumFrequencyPoints", "PMLCells")

    @pytest.mark.parametrize("name", UNGUARDED_ELSEWHERE)
    def test_the_mesh_route_names_it_too(self, name):
        """Run is where this was met, because the properties added so far are
        read there. The guard is on the shared read rather than on Run, so the
        next property to go on a mesh or a solver object is caught by the same
        sentence - each of these reaches the grid and nothing else refuses it.
        """
        doc, _ = self.missing(name)
        with pytest.raises(document.TranslationError) as refused:
            document.mesh(doc.Objects[0])
        assert name in str(refused.value)

    def test_an_internal_attribute_is_still_a_traceback(self, monkeypatch):
        """The discriminator, end to end, and the whole reason rewriting any of
        these is safe. An ``AttributeError`` this adapter raised against itself
        would blame the user for our fault and hide ours, so it goes on being
        the crash it is - through the same route a real document takes.
        """

        def broken(_analysis):
            raise AttributeError("internal", name="no_such_thing", obj=object())

        monkeypatch.setattr(document, "_smallest_response", broken)
        with pytest.raises(AttributeError):
            document.problem(model().Objects[0])

    def test_a_property_the_object_really_has_is_not_blamed(self):
        """A CapWords miss on an object that carries it is our bug somewhere
        else, and stays one."""
        error = AttributeError("...", name="SmallestResponse", obj=object())
        assert properties.stale_document(error, [model().Objects[0]]) is None

    def test_an_object_that_is_not_ours_is_never_blamed(self):
        """A plain CAD solid has no proxy of ours, so it lacks every property
        name there is. Without the kind check the first one in the document
        would be named for whatever the adapter failed to read anywhere."""
        error = AttributeError("...", name="SmallestResponse", obj=object())
        board = Obj("", "Board")
        board.Proxy = None
        assert properties.stale_document(error, [board]) is None
