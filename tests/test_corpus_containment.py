# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Does the engine hold the shape the user drew, everywhere they drew it?

Everything the translation does - triangulating, orienting, choosing a plane for
a sheet, ordering a polygon's coordinate lists - exists to make one thing true:
that openEMS finds material where the CAD kernel says there is material, and
finds none where it says there is none. That is a question with a right answer
at every point in space, and both sides can be asked it.

So this asks both. The kernel's verdict travels in the artifact; the engine is
built here, through the adapter's own emission path rather than a copy of it, and
asked the same points. A disagreement is scored by which way it goes:

* **The engine says air where the drawing says metal.** A conductor with a hole
  in it. Nothing is printed, the run completes, and the S-matrix is about a
  different device.
* **The engine says metal where the drawing says air.** A conductor that grew.
  It shorts what it reaches, and again in silence.

Neither is reported by anything the engine does, which is why they are worth a
test of their own: no other check in this suite would notice either one.

The points that decide it are **on the boundary**, and they are the ones a
scatter through the volume never finds. A zero-thickness sheet is hit by no
random point at all, so a containment check made only of those passes on a sheet
that was never emitted.
"""

from __future__ import annotations

import pytest

from Microwave.Solvers.openems.model import Solid, origin_offset
from tests import corpus, triangulated
from tests.conftest import corpus_record as _record

# The engine's own bindings, without which none of this can be asked at all.
pytest.importorskip("CSXCAD", reason="the openEMS bindings are not on this interpreter")

pytestmark = pytest.mark.slow


def _band(record) -> float:
    """How near the boundary a point has to be for a disagreement to mean nothing.

    A polygonised surface lies off the surface it approximates, so within that
    distance the kernel and the engine disagree honestly and the comparison
    settles nothing. The width is the departure the probe *measured* between the
    two surfaces, not a property of the triangles: a long edge across a flat
    face departs from it by nothing at all, and a short one across a tight curve
    departs by more than its neighbours.
    """
    return max(float(record.get("departure", 0.0)), 1e-9)


def _solids(pieces):
    """The pieces as envelope solids, in the coordinates they were drawn in.

    What the predicate is asked about, and what the engine is built from once
    it has been placed.
    """
    return [
        Solid(
            material="corpus",
            lower=tuple(piece["lower"]),
            upper=tuple(piece["upper"]),
            label=piece["label"],
            vertices=tuple(tuple(v) for v in piece["vertices"]),
            faces=tuple(tuple(f) for f in piece["faces"]),
            sheet_normal=piece["sheet_normal"],
        )
        for piece in pieces
    ]


def _built(pieces):
    """The pieces as openEMS holds them, through the adapter's own emission.

    ``add_solid`` rather than a copy of the choice it makes, so that a primitive
    the adapter would pick differently is not silently picked correctly here.
    And at the origin, which is where the driver puts every structure - a shape
    drawn below it is read inside out by the engine, so building one here where
    it was drawn would ask the engine a question the adapter never asks it.

    Returns the offset with the property, because a point put to it has to be
    asked in the coordinates it was built in.
    """
    from CSXCAD import ContinuousStructure

    from Microwave.Solvers.openems.driver import add_solid

    csx = ContinuousStructure()
    prop = csx.AddMetal("corpus")
    solids = _solids(pieces)
    offset = origin_offset(solids)
    for solid in solids:
        add_solid(prop, solid.moved(offset))
    # A polygon reads a bounding box that its constructor leaves zeroed, so
    # until this runs it contains nothing. A polyhedron updates itself and a box
    # needs none of it.
    for number in range(prop.GetQtyPrimitives()):
        prop.GetPrimitive(number).Update()
    return prop, offset


def _inside(built, point) -> bool:
    prop, offset = built
    moved = [p + d for p, d in zip(point, offset)]
    return any(
        prop.GetPrimitive(number).IsInside(moved) for number in range(prop.GetQtyPrimitives())
    )


@pytest.mark.parametrize(
    "specimen",
    [s for s in corpus.specimens() if s.expect == "meshed"],
    ids=lambda s: s.name,
)
class TestTheEngineHoldsWhatWasDrawn:
    def scored(self, record):
        """Samples far enough from the boundary for the two answers to have to
        agree, paired with what each side says."""
        built = _built(record["pieces"])
        band = _band(record)
        for sample in record["samples"]:
            if sample["inside"] is None or sample["distance"] <= band:
                continue
            yield sample, sample["inside"], _inside(built, sample["point"])

    def test_no_part_of_the_conductor_is_missing(self, specimen, artifacts):
        """The fatal direction. A conductor the engine cannot find does not fail
        loudly - it changes the answer and says nothing."""
        record = _record(artifacts, specimen.name)
        if record["measured"]["volume"] == 0.0:
            pytest.skip("a sheet has no interior, and is checked on its plane instead")
        missing = [
            sample["point"] for sample, drawn, built in self.scored(record) if drawn and not built
        ]
        assert not missing, (
            f"{specimen.name!r} ({specimen.subject}): the drawing has material at "
            f"{len(missing)} sampled points and the engine has none, the first at "
            f"{missing[0]}. Whatever is solved there is not what was drawn"
        )

    def test_the_conductor_did_not_grow(self, specimen, artifacts):
        """The other direction, and just as quiet. Metal where the drawing has
        air shorts whatever it reaches."""
        record = _record(artifacts, specimen.name)
        if record["measured"]["volume"] == 0.0:
            pytest.skip("a sheet has no interior, and is checked on its plane instead")
        grown = [
            sample["point"] for sample, drawn, built in self.scored(record) if built and not drawn
        ]
        assert not grown, (
            f"{specimen.name!r} ({specimen.subject}): the engine has material at "
            f"{len(grown)} sampled points the drawing leaves empty, the first at "
            f"{grown[0]}"
        )

    def test_something_was_actually_emitted(self, specimen, artifacts):
        """A structure that contains no point at all is the way an unclosed
        surface fails: the object is simply absent, and openEMS prints a warning
        whose benign form reads identically."""
        record = _record(artifacts, specimen.name)
        built = _built(record["pieces"])
        emitted = built[0].GetQtyPrimitives()
        assert emitted > 0
        held = [sample for sample in record["samples"] if _inside(built, sample["point"])]
        assert held, (
            f"{specimen.name!r} ({specimen.subject}) was emitted as {emitted} "
            "primitives that contain none of the points the shape was sampled at, "
            "so nothing of it is in the simulation"
        )


class TestTheOneShapeThePlacementIsFor:
    """Containment is decided by casting a ray from a point built by scaling the
    bounding box's maximum corner, which is only outward while some part of the
    shape is above the origin. For a solid lying wholly at or below it there is
    no outward, and every answer comes back inverted - the object absent where it
    is, and present where it is not.

    The class above already asks this specimen the same questions as every
    other, and gets the same answers, because the driver hands the engine no
    negative coordinate. What is asserted here is that the specimen is still the
    case that would prove it: a shape that reaches the polyhedron path and is
    drawn wholly below the origin, so the ray would be cast inward if it were
    built where it was drawn."""

    def test_the_specimen_still_reaches_the_ray_that_needs_it(self, artifacts):
        record = _record(artifacts, "negative_octant")
        assert record["status"] == "meshed", "the geometry layer has no quarrel with it"
        assert max(max(piece["upper"]) for piece in record["pieces"]) <= 0.0
        assert any(piece["faces"] for piece in record["pieces"]), (
            "this shape has to reach the polyhedron path, or the ray is never cast"
        )

    def test_and_the_engine_answers_backwards_without_it(self):
        """The defect itself, so the placement is measured against something
        rather than believed.

        Drawn rather than taken from the corpus, because reproducing it takes a
        particular shape. The endpoint is the maximum corner scaled by one plus
        a random fraction, so it lands somewhere between that corner and twice
        it: for a box spanning ``-2a`` to ``-a`` that interval is the box, and
        the endpoint is inside it whatever the fraction comes out as. A specimen
        of some other proportion is inverted only for some of them, which is a
        test that passes on a broken adapter about half the time.
        """
        from CSXCAD import ContinuousStructure

        from Microwave.Solvers.openems.driver import add_solid

        vertices, faces = triangulated.bar((1.0, 1.0, 1.0), centre=(-1.5, -1.5, -1.5))
        lower, upper = triangulated.extent(vertices)
        drawn = Solid(
            material="corpus",
            lower=lower,
            upper=upper,
            label="inside out",
            vertices=tuple(tuple(v) for v in vertices),
            faces=tuple(tuple(f) for f in faces),
        )
        middle = tuple(0.5 * (a + b) for a, b in zip(lower, upper))

        def held(solid, point):
            csx = ContinuousStructure()
            prop = csx.AddMetal(solid.label)
            add_solid(prop, solid)
            for number in range(prop.GetQtyPrimitives()):
                prop.GetPrimitive(number).Update()
            return _inside((prop, (0.0, 0.0, 0.0)), point)

        offset = origin_offset([drawn])
        assert not held(drawn, middle), (
            "the engine has to be wrong about this shape where it is drawn, or "
            "the placement is guarding nothing"
        )
        assert held(drawn.moved(offset), [p + d for p, d in zip(middle, offset)])


class TestASheetIsFoundOnItsOwnPlane:
    """A zero-thickness conductor is modelled where a grid line falls on it, so
    where its plane is *is* where the conductor is. Nothing in a volume scatter
    reaches that plane, which is why these are separate."""

    def sheets(self, artifacts):
        for specimen in corpus.specimens():
            record = artifacts.get(specimen.name)
            if record is None or record["status"] != "meshed":
                continue
            if any(piece["sheet_normal"] is not None for piece in record["pieces"]):
                yield specimen.name, record

    def test_there_are_sheets_in_the_corpus_at_all(self, artifacts):
        """Or the class below asserts nothing, quietly."""
        assert list(self.sheets(artifacts))

    def test_a_point_on_the_sheet_is_in_the_engine(self, artifacts):
        for name, record in self.sheets(artifacts):
            built = _built(record["pieces"])
            on_it = [s for s in record["samples"] if s["on_surface"]]
            assert on_it, f"{name!r} was sampled nowhere on its own surface"
            missing = [s["point"] for s in on_it if not _inside(built, s["point"])]
            assert not missing, (
                f"{name!r} is a sheet the engine does not hold at {len(missing)} of "
                f"{len(on_it)} points on it, the first at {missing[0]}. A sheet that "
                "contains nothing is absent from the run without anything being said"
            )

    def test_the_sheet_sits_at_the_plane_it_was_drawn_on(self, artifacts):
        """Only two coordinates and an elevation are sent, so a sheet put at the
        wrong elevation is solved somewhere the user did not draw it."""
        for name, record in self.sheets(artifacts):
            drawn = record["measured"]["bound_box"]
            for piece in record["pieces"]:
                if piece["sheet_normal"] is None:
                    continue
                axis = piece["sheet_normal"]
                assert piece["lower"][axis] == pytest.approx(drawn[axis], rel=0.0, abs=1e-9), (
                    f"{name!r} is drawn flat at {drawn[axis]:g} and "
                    f"emitted at {piece['lower'][axis]:g}"
                )


@pytest.mark.parametrize(
    "specimen",
    [s for s in corpus.specimens() if s.expect == "meshed"],
    ids=lambda s: s.name,
)
class TestThePredicateAgreesWithTheKernel:
    """The adapter's own containment answer, against the CAD kernel's.

    Everything that asks whether a grid still holds the drawing - the
    connectivity check a run makes, the permittivity a test integrates - reads
    :func:`~Microwave.Solvers.openems.containment.contains` and nothing else, so
    a predicate that is wrong about a real shape is a check that agrees with
    whatever it is checking. Arithmetic settles it on shapes a formula can
    describe; only the kernel can settle it on a fillet, a loft or a STEP file.

    The same samples and the same band the engine is judged on, so a
    disagreement here is between the predicate and the drawing rather than
    between two ways of choosing points.
    """

    def scored(self, record):
        import numpy as np

        from Microwave.Solvers.openems.containment import contains

        band = _band(record)
        samples = [
            sample
            for sample in record["samples"]
            if sample["inside"] is not None and sample["distance"] > band
        ]
        if not samples:
            return []
        points = np.array([sample["point"] for sample in samples], dtype=float)
        held = np.zeros(len(points), dtype=bool)
        for solid in _solids(record["pieces"]):
            held |= contains(solid, points)
        return list(zip(samples, held))

    def test_it_finds_the_material_the_drawing_has(self, specimen, artifacts):
        record = _record(artifacts, specimen.name)
        if record["measured"]["volume"] == 0.0:
            pytest.skip("a sheet has no interior, and is answered on its plane instead")
        missing = [
            sample["point"] for sample, held in self.scored(record) if sample["inside"] and not held
        ]
        assert not missing, (
            f"{specimen.name!r} ({specimen.subject}): the drawing has material at "
            f"{len(missing)} sampled points and the predicate finds none, the first "
            f"at {missing[0]}"
        )

    def test_it_finds_none_where_the_drawing_has_none(self, specimen, artifacts):
        record = _record(artifacts, specimen.name)
        if record["measured"]["volume"] == 0.0:
            pytest.skip("a sheet has no interior, and is answered on its plane instead")
        grown = [
            sample["point"] for sample, held in self.scored(record) if held and not sample["inside"]
        ]
        assert not grown, (
            f"{specimen.name!r} ({specimen.subject}): the predicate finds material at "
            f"{len(grown)} sampled points the drawing leaves empty, the first at "
            f"{grown[0]}"
        )
