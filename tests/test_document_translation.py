# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the adapter reads out of a FreeCAD document.

The fakes here are deliberately not FreeCAD stubs. ``document.py`` imports no
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

import math

import numpy as np
import pytest

from Microwave.Objects.ports import FIXED_IMPEDANCE, PORT_IMPEDANCE
from Microwave.Solvers.openems import document
from Microwave.Solvers.openems.model import THROUGH

# The acceptance line: a 3 mm trace on 1.6 mm FR4, ground underneath.
WIDTH = 3.0
HEIGHT = 1.6
LENGTH = 100.0
BOARD = 30.0
EPS_R = 4.4


# ---------------------------------------------------------------------------
# Fakes: the whole contract document.py has with FreeCAD
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


class Shape:
    """A box-shaped shape. ``fill`` below 1 makes it something else.

    ``rings`` gives it *corners*: closed rings of ``(u, v)`` in the plane of a
    flat shape, from which ``Area`` and ``Edges`` are both derived, so the two
    cannot disagree the way ``fill`` and a hand-written area could. A shape that
    has to be cut up needs corners, because a fill fraction says how much of the
    box is covered and never says where.

    ``area`` overrides what the rings add up to, and exists for the one shape
    they cannot describe between them: a compound of overlapping faces, where
    the kernel reports each face's own area and the outline encloses the overlap
    only once.
    """

    def __init__(self, lower, upper, fill=1.0, rings=None, area=None):
        self.BoundBox = BoundBox(lower, upper)
        self._faces = {}
        self.Edges = []
        extents = [b - a for a, b in zip(lower, upper)]
        flat = [d for d in range(3) if extents[d] <= document.FLATNESS]
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
        elif flat:
            self.Area = math.prod(e for d, e in enumerate(extents) if d not in flat) * fill
            self.Volume = 0.0
        else:
            self.Volume = math.prod(extents) * fill
            self.Area = 2 * sum(extents[a] * extents[b] for a, b in ((0, 1), (0, 2), (1, 2)))

    def face(self, name, lower, upper):
        """Register a named sub-shape. A face is a Shape too: the translation
        validates whatever a binding names, so it needs area and volume."""
        self._faces[name] = Shape(lower, upper)
        return self

    def getElement(self, name):
        return self._faces[name]


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
        wavelength = document.SPEED_OF_LIGHT / 10e9 / math.sqrt(EPS_R) * 1e3

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
    def test_a_rotated_solid_is_not_a_box(self):
        """Verified against the shape's own volume, so the refusal does not
        depend on recognising a type. A rotated box has a bounding box like any
        other shape, and openEMS would staircase it without comment."""
        board = substrate()
        board.Shape = Shape(
            (-LENGTH / 2, -BOARD / 2, 0.0), (LENGTH / 2, BOARD / 2, HEIGHT), fill=0.71
        )
        doc = model()
        replaces(doc, "Substrate", board)
        part(doc, "DielectricBinding").References = [(board, [])]

        # A solid is judged by volume, and the refusal says which measurement
        # it took: an area here would be a different check reporting.
        with pytest.raises(document.TranslationError, match="its volume is"):
            document.problem(doc.Objects[0])

    def test_a_cylinder_is_not_a_box(self):
        """pi/4 of its bounding box, and nothing about its type says so."""
        board = substrate()
        board.Shape = Shape(
            (-LENGTH / 2, -BOARD / 2, 0.0),
            (LENGTH / 2, BOARD / 2, HEIGHT),
            fill=math.pi / 4,
        )
        doc = model()
        replaces(doc, "Substrate", board)
        part(doc, "DielectricBinding").References = [(board, [])]

        with pytest.raises(document.TranslationError, match="78.5%"):
            document.problem(doc.Objects[0])

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

    def test_a_flat_shape_with_one_diagonal_edge_is_still_refused(self):
        """The cut is exact or it does not happen. One edge off the axes is a
        staircase openEMS would build without saying so, and a best fit here is
        the silent approximation the whole check exists to prevent."""
        with pytest.raises(document.TranslationError) as refusal:
            document.problem(diagonal_trace().Objects[0])
        said = str(refusal.value)
        assert "not axis-aligned" in said
        # Flat in one axis, so nothing about it has a volume. Quoting one would
        # describe a measurement the check never took.
        assert "volume" not in said

    def test_a_shape_a_percent_short_of_its_box_is_still_refused(self):
        """The tolerance is for float arithmetic, not for a close enough fit.
        A shape that misses by a per cent is staircased like any other."""
        board = substrate()
        board.Shape = Shape(
            (-LENGTH / 2, -BOARD / 2, 0.0), (LENGTH / 2, BOARD / 2, HEIGHT), fill=0.99
        )
        doc = model()
        replaces(doc, "Substrate", board)
        part(doc, "DielectricBinding").References = [(board, [])]

        with pytest.raises(document.TranslationError, match="its volume is"):
            document.problem(doc.Objects[0])

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
        assert document._PORT_IMPEDANCE == PORT_IMPEDANCE

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
        original = document.Material

        def explode(*args, **kwargs):
            raise RuntimeError("an adapter bug, not a property")

        document.Material = explode
        try:
            with pytest.raises(RuntimeError, match="an adapter bug"):
                document.problem(doc.Objects[0])
        finally:
            document.Material = original

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
            2 * math.pi * problem.frequency.center * document.VACUUM_PERMITTIVITY * EPS_R * 0.02
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
        wavelength = document.SPEED_OF_LIGHT / problem.frequency.stop / math.sqrt(EPS_R) * 1e3
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
        wavelength_mm = document.SPEED_OF_LIGHT / 10e9 * 1e3
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

    wavelength = document.SPEED_OF_LIGHT / WG_FREQ_MAX * 1e3
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
