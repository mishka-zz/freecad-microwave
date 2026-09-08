# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Properties of a finished grid that hold without any reference to compare to.

Every other mesh test asks whether one grid is right. These ask whether the
mesher is the same mesher whichever way the model is presented, which is a
question a single grid cannot answer and which no closed form is needed for:
draw the structure again on other axes, mirrored, or somewhere else in space,
and the grid must come back turned the same way.

A rigid motion is not the only presentation. A solid arrives as the boxes it was
cut into, and the same metal cut another way must mesh the same. That
dependence is harder to see: the cut leaves every coordinate where it was and
shows up only as a grid nobody can compare against anything.
:class:`TestOneSolidCutTwoWays` holds the mesher to it.

That catches a class of defect the per-case tests cannot see. A rule that reads
one axis where it meant another, a constant that only suits the sizes the
example happens to use, a tolerance that binds near the origin and not at 100 mm
- each of these produces a perfectly plausible grid for the case it was written
against, and a different one for the same structure drawn a millimetre to the
left.

The orbit is a catalogue of structures crossed with a group of isometries, and
it is only worth what the catalogue covers - so :class:`TestTheCatalogueItself`
asserts, of each scene, the thing that scene is in the catalogue for. Those
guards are not decoration: a refinement box that stopped refining or a
declaration that stopped being read leaves a scene meshing perfectly well and
asking nothing, and the orbit around it stays green throughout.

What the orbit cannot see is worth stating too. It compares the mesher against
itself, so a rule that is wrong the same way whichever way the model is turned
passes every test here. Reading a region's clipped extent where its drawn one
was meant is that kind of fault, and it belongs to the tests that compare a grid
against arithmetic.

Nothing here imports openEMS, CSXCAD or FreeCAD.
"""

from __future__ import annotations

import itertools
from dataclasses import MISSING, dataclass, fields, replace

import numpy as np
import pytest

import Microwave.Solvers.openems.mesh as mesh
from Microwave.portbox import FLATNESS
from Microwave.Solvers.openems.grid import MeshLines
from Microwave.Solvers.openems.mesh import generate_mesh_lines
from Microwave.Solvers.openems.metal import (
    Conductor,
    _bands,
    _edge_pair,
    _edge_size,
    _face_depth,
    _grouped_into_conductors,
    _met_by_metal,
    _one_pair_at_each_face,
    conductor_faces,
    conductor_pieces,
    conductor_run,
    edge_lines,
    width_axes,
)
from Microwave.Solvers.openems.regions import (
    DIMENSIONS,
    EDGE_LINE_INSIDE,
    MaterialClass,
    MeshParams,
    Region,
    SizingRegion,
)
from Microwave.Solvers.openems.sizing import Feature
from Microwave.Solvers.openems.sizing_field import _symmetrize
from tests.mesh_fixtures import assert_graded_within

#: How far a *translated* grid's cells may sit from the original's, relative to
#: the cell. A thousandth of a cell, which is below anything an FDTD run can
#: notice - and it is a declared budget rather than a derived bound. The
#: integration grid each gap is sampled on is sized by truncating a length
#: ratio, so a translation that tips one gap's sample count by a single sample
#: moves every line in that gap slightly. What is *exact* under a translation is
#: the cell count, which is what a solve costs, and that is asserted as exact.
PLACEMENT = 1e-3

#: How far a mirrored or relabelled grid may sit from the original's, in units
#: of the last bit of the coordinates being compared. A mirrored grid is built
#: by the same arithmetic on negated coordinates, which is not the negation of
#: the same arithmetic - placement walks each gap from the other end - so the
#: agreement is rounding rather than exactness. Stated in ulp rather than in
#: millimetres so that it means the same thing for a structure drawn at the
#: origin and for one drawn a metre away.
ROUNDING = 64


@dataclass(frozen=True)
class Scene:
    """A whole meshing problem, in the arguments ``generate_mesh_lines`` takes."""

    regions: tuple[Region, ...] = ()
    domain: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-10.0, -10.0, -5.0),
        (10.0, 10.0, 5.0),
    )
    params: MeshParams = MeshParams(metal_res=0.2, dielectric_res=1.0)
    forced: tuple[tuple[float, ...], ...] = ((), (), ())
    sizing: tuple[SizingRegion, ...] = ()
    features: tuple[Feature, ...] = ()

    def mesh(self) -> MeshLines:
        return generate_mesh_lines(
            self.regions, self.domain, self.params, self.forced, self.sizing, self.features
        )


@dataclass(frozen=True)
class Move:
    """A rigid motion of the whole problem: relabel the axes, mirror some of
    them, then translate.

    ``order[d]`` is the axis the moved axis ``d`` was drawn on, which is how a
    per-axis policy - ``max_ratio``, ``pml_cells`` - has to travel with the
    geometry: turning a board on its side must turn its grading with it, or the
    comparison tests the policy rather than the mesher.
    """

    name: str
    order: tuple[int, int, int] = (0, 1, 2)
    sign: tuple[float, float, float] = (1.0, 1.0, 1.0)
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def in_place(self) -> bool:
        """True where the move is a relabelling or a mirror and not a translation.

        The two are compared differently: without a translation the arithmetic
        is the same arithmetic on rearranged inputs and the grids agree to
        rounding, while a translation changes the numbers the integration
        samples fall on. See :data:`ROUNDING` and :data:`PLACEMENT`.
        """
        return self.shift == (0.0, 0.0, 0.0)

    def value(self, values, dim: int) -> float:
        """One coordinate of a moved point, taken from the axis it came from."""
        return self.sign[dim] * values[self.order[dim]] + self.shift[dim]

    def point(self, point) -> tuple[float, float, float]:
        return tuple(self.value(point, dim) for dim in range(DIMENSIONS))

    def box(self, lower, upper):
        """A moved box, as corners. A mirror swaps which corner is the lower one."""
        near, far = self.point(lower), self.point(upper)
        return (
            tuple(min(a, b) for a, b in zip(near, far)),
            tuple(max(a, b) for a, b in zip(near, far)),
        )

    def relabel(self, values):
        """A per-axis quantity that is not a position: a ratio, a count, an extent."""
        return tuple(values[self.order[dim]] for dim in range(DIMENSIONS))

    def axis(self, dim: int) -> int:
        """Where the axis drawn as ``dim`` ends up."""
        return self.order.index(dim)

    def scene(self, scene: Scene) -> Scene:
        lower, upper = self.box(*scene.domain)
        return Scene(
            regions=tuple(self._region(region) for region in scene.regions),
            domain=(lower, upper),
            params=replace(
                scene.params,
                max_ratio=self.relabel(scene.params.max_ratio),
                pml_cells=self.relabel(scene.params.pml_cells),
            ),
            forced=tuple(
                tuple(sorted(self.value((v, v, v), dim) for v in scene.forced[self.order[dim]]))
                for dim in range(DIMENSIONS)
            ),
            sizing=tuple(self._sizing(region) for region in scene.sizing),
            features=tuple(self._feature(feature) for feature in scene.features),
        )

    def _region(self, region: Region) -> Region:
        lower, upper = self.box(region.lower, region.upper)
        return replace(
            region,
            lower=lower,
            upper=upper,
            continuous=frozenset(self.axis(dim) for dim in region.continuous),
            drawn=None if region.drawn is None else self.relabel(region.drawn),
        )

    def _sizing(self, region: SizingRegion) -> SizingRegion:
        lower, upper = self.box(region.lower, region.upper)
        return replace(region, lower=lower, upper=upper)

    def _feature(self, feature: Feature) -> Feature:
        lower, upper = self.box(feature.lower, feature.upper)
        normal = None
        if feature.normal is not None:
            # A direction, so it is relabelled and mirrored but never translated.
            normal = tuple(
                self.sign[dim] * feature.normal[self.order[dim]] for dim in range(DIMENSIONS)
            )
        return replace(feature, lower=lower, upper=upper, normal=normal)

    def lines(self, lines: MeshLines, dim: int) -> np.ndarray:
        """Where the grid ``lines`` ought to be after the move."""
        return np.sort(self.sign[dim] * lines[self.order[dim]] + self.shift[dim])

    def pins(self, lines: MeshLines, dim: int) -> list[float]:
        """Where the anchors ought to be after the move."""
        return sorted(
            self.value((pin.position,) * DIMENSIONS, dim)
            for pin in lines.fixed[self.order[dim]]
            if pin.required
        )


def _stackup() -> Scene:
    """Microstrip: a dielectric, a ground sheet under it and a trace on top."""
    return Scene(
        regions=(
            Region((-8, -8, 0), (8, 8, 1.6), MaterialClass.DIELECTRIC, "sub"),
            Region((-8, -8, 0), (8, 8, 0), MaterialClass.METAL, "gnd"),
            Region((-1.5, -8, 1.6), (1.5, 8, 1.6), MaterialClass.METAL, "trace"),
        )
    )


def _block() -> Scene:
    """A solid conductor, centred, so the symmetry fold fires on all three axes."""
    return Scene(regions=(Region((-2, -3, -1), (2, 3, 1), MaterialClass.METAL, "block"),))


def _off_centre() -> Scene:
    """The same kind of structure with nothing centred and nothing symmetric."""
    return Scene(
        regions=(
            Region((1, -3, 0.4), (5, 2, 2.0), MaterialClass.DIELECTRIC, "slab"),
            Region((1, -3, 0.4), (5, 2, 0.4), MaterialClass.METAL, "foil"),
        )
    )


def _per_axis_policy() -> Scene:
    """A different grading ratio and a different absorber on each axis.

    One ratio for all three cannot tell an axis that reads its own policy from
    one that reads the first entry and applies it everywhere - but only where
    every axis actually grades, so the pad is here to make all three ramp from
    the metal size up to the bulk one. An axis that comes out uniform obeys
    every ratio there is.

    One axis has no absorber at all, which is the case per-axis absorbers exist
    for: a waveguide is PEC on its side walls and absorbing only at its ends.
    """
    return Scene(
        regions=(
            Region((-8, -8, 0), (8, 8, 1.6), MaterialClass.DIELECTRIC, "sub"),
            Region((-1, -1, 1.6), (1, 1, 1.6), MaterialClass.METAL, "pad"),
        ),
        params=MeshParams(
            metal_res=0.2,
            dielectric_res=1.0,
            max_ratio=(1.3, 1.15, 1.45),
            min_lines=3,
            pml_cells=(8, 6, 0),
        ),
    )


def _refined() -> Scene:
    """A refinement box, which contributes constraints and pins nothing."""
    return Scene(
        regions=(Region((-8, -8, 0), (8, 8, 1.6), MaterialClass.DIELECTRIC, "sub"),),
        sizing=(SizingRegion((-2, -2, 0), (2, 2, 1.6), 0.15, label="box"),),
    )


def _coupled() -> Scene:
    """Two traces with a gap between them, and the gap declared as a feature.

    The second feature is outside the domain in one axis only, which is the
    shape of the question :func:`~.mesh._features_inside` asks: a demand is
    projected onto each axis separately, so one that misses in a single axis
    would otherwise refine slabs on the other two, somewhere its geometry is
    not. Being outside in one axis alone is what tells an axis-by-axis test
    from one that looks at the first axis and stops.
    """
    return Scene(
        regions=(
            Region((-8, -8, 0), (8, 8, 1.6), MaterialClass.DIELECTRIC, "sub"),
            Region((-4, -8, 1.6), (-1, 8, 1.6), MaterialClass.METAL, "left"),
            Region((1, -8, 1.6), (4, 8, 1.6), MaterialClass.METAL, "right"),
        ),
        features=(
            Feature(0.4, (1.0, 0.0, 0.0), (-1, 0, 1.6), (1, 0, 1.6), "gap"),
            Feature(0.05, None, (-1, 0, 20.0), (1, 0, 20.0), "elsewhere"),
        ),
    )


def _port() -> Scene:
    """A strip fed through the domain wall, the way a transmission-line port is.

    Three paths meet here and none is reachable from a plain drawing. The feed's
    end along the propagation axis is marked continuous, so it is where the port
    hands over rather than a conductor edge, and is not refined as one. The trace
    and the pad are one metal drawn in two pieces, so the seam between them is
    not a boundary and asks for no line. And the feed reaches the domain wall,
    where an edge has no outside to put its line in.
    """
    return Scene(
        regions=(
            Region((-10, -10, 0), (10, 10, 1.6), MaterialClass.DIELECTRIC, "sub"),
            Region(
                (-10, -1.5, 1.6),
                (0, 1.5, 1.6),
                MaterialClass.METAL,
                "feed",
                material_name="copper",
                continuous=frozenset({0}),
            ),
            Region(
                (0, -1.5, 1.6), (6, 1.5, 1.6), MaterialClass.METAL, "trace", material_name="copper"
            ),
            Region(
                (6, -1.5, 1.6), (9, 1.5, 1.6), MaterialClass.METAL, "pad", material_name="copper"
            ),
        )
    )


def _forced() -> Scene:
    """Port planes: positions that must be lines though no region asks for them.

    Off any round number, so that the grid the domain would have given anyway
    does not already have a line there - a forced position that coincides with
    one tests the coincidence.
    """
    return Scene(
        regions=(Region((-8, -8, 0), (8, 8, 1.6), MaterialClass.DIELECTRIC, "sub"),),
        forced=((-6.3, 5.7), (), ()),
    )


def _clipped() -> Scene:
    """A board reaching far past the domain, of which a sliver survives.

    Sized by what was drawn rather than by what is left, which is the one place
    a region carries a per-axis extent of its own. The sliver is narrow enough
    that its own extent would ask for cells, and the board is wide enough that
    the drawn extent asks for none.
    """
    return Scene(
        regions=(
            Region(
                (-10, -10, 0),
                (-8, 10, 1.6),
                MaterialClass.DIELECTRIC,
                "board",
                drawn=(100.0, 40.0, 1.6),
            ),
        )
    )


def _capped() -> Scene:
    """A vacuum ceiling above the bulk size, and a region asking for its own."""
    return Scene(
        regions=(Region((-8, -8, 0), (8, 8, 1.6), MaterialClass.DIELECTRIC, "sub", size=0.4),),
        params=MeshParams(metal_res=0.2, dielectric_res=1.0, cap=2.0),
    )


def _bare() -> Scene:
    """Nothing drawn at all: the grid the domain and the policy give on their own."""
    return Scene()


CATALOGUE = {
    "bare": _bare,
    "stackup": _stackup,
    "block": _block,
    "off centre": _off_centre,
    "per axis policy": _per_axis_policy,
    "refined": _refined,
    "coupled": _coupled,
    "port": _port,
    "forced": _forced,
    "clipped": _clipped,
    "capped": _capped,
}


def _moves() -> list[Move]:
    moves = [
        Move(f"relabelled {order}", order=order)
        for order in itertools.permutations(range(3))
        if order != (0, 1, 2)
    ]
    for dim in range(DIMENSIONS):
        sign = [1.0, 1.0, 1.0]
        sign[dim] = -1.0
        moves.append(Move(f"mirrored in {'xyz'[dim]}", sign=tuple(sign)))
    # A right angle is a relabelling and a mirror together, which is worth
    # stating on its own: either one alone can be right while the composition
    # is not.
    moves.append(Move("turned a right angle about z", order=(1, 0, 2), sign=(-1.0, 1.0, 1.0)))
    moves.append(Move("turned a right angle about x", order=(0, 2, 1), sign=(1.0, -1.0, 1.0)))
    for shift in [
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        (1000.0, 1000.0, 1000.0),
        (1 / 3, -7 / 11, 0.1),
        (-13.7, 2.2, 9.9),
    ]:
        moves.append(Move(f"moved to {shift}", shift=shift))
    return moves


MOVES = _moves()

_MESHED: dict[tuple[str, str], MeshLines] = {}


def meshed(scene_name: str, move: Move) -> MeshLines:
    """The grid for one catalogue entry under one move, built once.

    Every class below walks the whole orbit, and the same few hundred grids
    would otherwise be rebuilt for each of them.
    """
    key = (scene_name, move.name)
    if key not in _MESHED:
        _MESHED[key] = move.scene(CATALOGUE[scene_name]()).mesh()
    return _MESHED[key]


IDENTITY = Move("as drawn")


def ulp(*arrays) -> float:
    """The last bit of the largest coordinate among ``arrays``."""
    return float(np.spacing(max(float(np.max(np.abs(array))) for array in arrays)))


def asymmetry(axis: np.ndarray) -> float:
    """How far an axis is from being its own mirror image about its centre."""
    centre = (axis[0] + axis[-1]) / 2.0
    return float(np.max(np.abs(axis - (2.0 * centre - axis[::-1]))))


@pytest.fixture(params=sorted(CATALOGUE), ids=lambda name: name.replace(" ", "-"))
def scene_name(request) -> str:
    return request.param


def _ids(moves):
    return [move.name.replace(" ", "-") for move in moves]


@pytest.fixture(params=MOVES, ids=_ids(MOVES))
def move(request) -> Move:
    return request.param


IN_PLACE = [move for move in MOVES if move.in_place]
TRANSLATIONS = [move for move in MOVES if not move.in_place]


@pytest.fixture(params=IN_PLACE, ids=_ids(IN_PLACE))
def in_place_move(request) -> Move:
    return request.param


@pytest.fixture(params=TRANSLATIONS, ids=_ids(TRANSLATIONS))
def translation(request) -> Move:
    return request.param


class TestTheModelCanBeTurnedAndMeshedAgain:
    """The grid must follow the model through an isometry.

    An axis-aligned grid commutes with the motions that map the axes onto each
    other: relabel the axes and the lines are relabelled, mirror the model and
    the lines mirror, move it and the lines move with it. Anything the mesher
    does that depends on *which* axis it is on, or on where the model sits
    rather than on its shape, shows up here and only here.
    """

    def test_the_cell_count_follows_the_model(self, scene_name, move):
        """The count is exact under every motion, translations included.

        It is the quantity a solve is paid for, so it is the one the invariant
        has to hold exactly - a mesher that quietly costs three percent more
        because a board was drawn away from the origin is a defect, however
        close the lines are.
        """
        reference, moved = meshed(scene_name, IDENTITY), meshed(scene_name, move)
        assert moved.shape == move.relabel(reference.shape)

    def test_relabelling_the_axes_changes_nothing_at_all(self, scene_name):
        """Not nearly: the same arithmetic on the same numbers.

        Each axis is meshed by one function that is told the axis only so it can
        name it in a refusal, so a relabelled problem is the identical
        computation. An approximate comparison here would accept a rule that
        read the wrong entry of a per-axis policy and happened to land close.
        """
        reference = meshed(scene_name, IDENTITY)
        for order in itertools.permutations(range(DIMENSIONS)):
            relabelled = Move(f"relabelled {order}", order=order)
            moved = meshed(scene_name, relabelled)
            for dim in range(DIMENSIONS):
                assert np.array_equal(moved[dim], relabelled.lines(reference, dim)), (
                    f"axis {dim} of the {scene_name} scene differs when it is drawn as "
                    f"axis {relabelled.order[dim]}"
                )

    def test_a_mirrored_model_gives_a_mirrored_grid(self, scene_name, in_place_move):
        reference, moved = meshed(scene_name, IDENTITY), meshed(scene_name, in_place_move)
        for dim in range(DIMENSIONS):
            want = in_place_move.lines(reference, dim)
            assert np.max(np.abs(moved[dim] - want)) <= ROUNDING * ulp(want, moved[dim])

    def test_moving_the_model_does_not_resize_its_cells(self, scene_name, translation):
        """Offset independence, which the sizing field claims and nothing checked.

        Every length the field is built from is a distance, so nothing in it
        knows where the origin is. Cell *sizes* are therefore the invariant to
        state; the positions inherit it, and stating it on the positions instead
        would compare a number that grows with the translation.
        """
        reference, moved = meshed(scene_name, IDENTITY), meshed(scene_name, translation)
        for dim in range(DIMENSIONS):
            want = np.diff(translation.lines(reference, dim))
            assert np.max(np.abs(np.diff(moved[dim]) - want) / want) <= PLACEMENT

    def test_the_anchors_move_with_the_model(self, scene_name, move):
        """An anchor is a position, so it is compared as one.

        Exactly where the model was not translated. Under a translation the
        thirds rule adds its offset to a coordinate that has already been moved
        rather than the other way about, which is the same position and not
        always the same float, so the claim there is rounding.
        """
        reference, moved = meshed(scene_name, IDENTITY), meshed(scene_name, move)
        for dim in range(DIMENSIONS):
            want = move.pins(reference, dim)
            got = sorted(pin.position for pin in moved.fixed[dim] if pin.required)
            assert len(got) == len(want), (
                f"axis {dim} of the {scene_name} scene anchors "
                f"{len(got)} positions after being {move.name}, not {len(want)}"
            )
            if move.in_place:
                assert got == want
            elif want:
                assert np.max(np.abs(np.array(got) - np.array(want))) <= ROUNDING * ulp(
                    np.array(want)
                )

    def test_a_symmetric_model_is_folded_wherever_it_is_drawn(self, move, monkeypatch):
        """The fold has to fire on a symmetric structure however it is placed.

        Asserted on the decision rather than on the grid, because the grid
        cannot answer it: placement lands close enough on its own that an axis
        which was never folded still mirrors to within a few bits, and a
        structure drawn at the origin folds to no visible difference at all.

        What that hides is the fold quietly ceasing to run away from the origin.
        Deciding whether a structure is symmetric means comparing a position
        against its mirror image, and about a centre the grid cannot hold
        exactly, those two are never the same float - so the comparison needs a
        tolerance, and a tolerance that stops scaling stops finding anything.
        """
        folded = []
        original = _symmetrize
        monkeypatch.setattr(
            mesh,
            "_symmetrize",
            lambda interior, anchors: (folded.append(None), original(interior, anchors))[1],
        )
        move.scene(CATALOGUE["block"]()).mesh()
        assert len(folded) == DIMENSIONS, (
            f"a structure symmetric in every axis was folded on {len(folded)} of them "
            f"when it was {move.name}"
        )


class TestEveryGridInTheOrbitIsWellFormed:
    """The two promises whose loss is silent, asked of every grid in the orbit.

    The mesher's own validation catches a grid that is out of order, infinite or
    below the cell floor, and does it before the grid is returned - so asking
    again here would only re-run it. These two it cannot catch. Grading is
    checked against one ratio for all three axes everywhere else, and an anchor
    is checked to a tolerance everywhere else, and a tolerance is precisely what
    an anchor does not have.
    """

    def test_no_two_neighbouring_cells_differ_by_more_than_the_ratio(self, scene_name, move):
        """Per axis against that axis' own ratio.

        The grading limit is one number per axis and the tests that check it
        pass one number for all three, so an axis reading somebody else's
        ratio has never been contradicted.
        """
        scene = move.scene(CATALOGUE[scene_name]())
        assert_graded_within(meshed(scene_name, move), scene.params.max_ratio)

    def test_every_anchor_is_a_grid_line_exactly(self, scene_name, move):
        """A zero-thickness sheet is discretised only where a line equals its
        position exactly, and one ulp of drift removes the conductor from the
        model without removing it from the report."""
        lines = meshed(scene_name, move)
        for dim in range(DIMENSIONS):
            available = set(lines[dim].tolist())
            for pin in lines.fixed[dim]:
                if pin.required:
                    assert pin.position in available, (
                        f"{pin.source} at {pin.position!r} is anchored on axis {dim} but "
                        f"the nearest line is "
                        f"{lines[dim][np.argmin(np.abs(lines[dim] - pin.position))]!r}"
                    )


class TestTheCatalogueItself:
    """The orbit is worth what the catalogue covers, so say what it covers.

    Without this, a scene quietly falling out of the interesting path - a
    refinement box that stopped refining, a structure that stopped being
    symmetric - would leave the whole file passing while testing less.
    """

    def test_each_move_is_an_isometry(self, move):
        """Axes relabelled, some of them mirrored, the whole thing translated."""
        assert sorted(move.order) == list(range(DIMENSIONS))
        assert set(move.sign) <= {1.0, -1.0}

    def test_both_kinds_of_move_are_present(self):
        """Emptying either list would leave every test that reads it green while
        it tested nothing, and the two are the file's whole subject."""
        assert IN_PLACE and TRANSLATIONS
        assert len(IN_PLACE) + len(TRANSLATIONS) == len(MOVES)

    def test_one_scene_is_symmetric_in_every_axis(self):
        """The fold rewrites every line on an axis, so an orbit that never
        reaches it leaves the most invasive step in the mesher untested."""
        lines = meshed("block", IDENTITY)
        for dim in range(DIMENSIONS):
            assert asymmetry(lines[dim]) == 0.0

    def test_one_scene_is_symmetric_in_no_axis(self):
        lines = meshed("off centre", IDENTITY)
        for dim in range(DIMENSIONS):
            assert asymmetry(lines[dim]) > 1e-9

    def test_the_refinement_box_refines(self):
        """A sizing region that had stopped biting would still pass every
        invariant above, having become one more plain dielectric."""
        with_box = meshed("refined", IDENTITY)
        without = Scene(regions=CATALOGUE["refined"]().regions).mesh()
        assert with_box.smallest_cell() < without.smallest_cell()

    def test_the_forced_lines_are_somewhere_the_grid_would_not_have_gone(self):
        """A forced position that the domain would have put a line on anyway
        proves the coincidence rather than the forcing."""
        scene = CATALOGUE["forced"]()
        lines = meshed("forced", IDENTITY)
        for position in scene.forced[0]:
            assert position in set(lines.x.tolist())
        assert not np.array_equal(lines.x, replace(scene, forced=((), (), ())).mesh().x)

    def test_the_policy_scene_grades_on_every_axis(self):
        """Each axis against its own ratio, which is only a claim about that
        axis where the axis grades at all - a uniform one obeys every ratio
        there is, and the tightest of the three is the one that would go
        unnoticed."""
        scene = CATALOGUE["per axis policy"]()
        lines = meshed("per axis policy", IDENTITY)
        assert len(set(scene.params.max_ratio)) == DIMENSIONS
        assert len(set(scene.params.pml_cells)) == DIMENSIONS
        for dim in range(DIMENSIONS):
            spacings = np.diff(lines[dim])
            observed = np.max(
                np.maximum(spacings[1:] / spacings[:-1], spacings[:-1] / spacings[1:])
            )
            limit = scene.params.max_ratio[dim]
            assert observed > 1.0 + (limit - 1.0) / 2.0, (
                f"axis {dim} grades at {observed}, nowhere near its limit of {limit}; "
                "the ratio it was given is not being tested"
            )

    @pytest.mark.parametrize("declaration", ["continuous", "material_name"])
    def test_the_port_scene_needs_everything_it_declares(self, declaration):
        """A continuous face and a shared material name each change where lines
        go, so a scene that had stopped carrying one would mesh perfectly well
        and quietly stop asking the question."""
        blank = {"continuous": frozenset(), "material_name": ""}[declaration]
        scene = CATALOGUE["port"]()
        plain = replace(
            scene,
            regions=tuple(replace(region, **{declaration: blank}) for region in scene.regions),
        )
        assert not np.array_equal(plain.mesh().x, meshed("port", IDENTITY).x)

    def test_the_port_scene_reaches_the_domain_wall(self):
        """A conductor edge at the wall has no outside to put its line in, so it
        is not refined at all - a path only a conductor touching the wall
        reaches, and one a scene can lose by being drawn a millimetre shorter."""
        scene = CATALOGUE["port"]()
        feed = next(region for region in scene.regions if region.label == "feed")
        assert feed.lower[0] == scene.domain[0][0]

    def test_the_clipped_board_is_sized_by_what_was_drawn(self):
        """The drawn extent must be the one that decides, which it only is where
        the surviving sliver would have decided otherwise."""
        scene = CATALOGUE["clipped"]()
        as_drawn = replace(
            scene, regions=tuple(replace(region, drawn=None) for region in scene.regions)
        )
        assert as_drawn.mesh().shape != meshed("clipped", IDENTITY).shape

    def test_the_stray_feature_is_outside_the_domain_in_one_axis_only(self):
        """Outside in every axis is a demand any test of the three would drop."""
        scene = CATALOGUE["coupled"]()
        lower, upper = scene.domain
        stray = scene.features[-1]
        outside = [
            dim
            for dim in range(DIMENSIONS)
            if stray.upper[dim] < lower[dim] or stray.lower[dim] > upper[dim]
        ]
        assert len(outside) == 1

    def test_the_capped_scene_has_a_ceiling_above_its_bulk(self):
        params = CATALOGUE["capped"]().params
        assert params.ceiling > params.dielectric_res


#: Outer span of the hollow box the tilings below partition, per axis, in mm.
SHELL = (10.0, 10.0, 4.0)
#: How thick its wall is drawn, in mm. Every tiling's boxes are built from these
#: two, so a tiling that stopped holding the same metal is a fault in the tiling.
WALL = 0.5


def _metal(lower, upper, label: str) -> Region:
    return Region(
        lower=lower, upper=upper, material=MaterialClass.METAL, label=label, material_name="Copper"
    )


def _standing_on_the_floor() -> list[Region]:
    """A floor spanning the footprint, with four walls standing on it."""
    x, y, z = SHELL
    return [
        _metal((0.0, 0.0, 0.0), (x, y, WALL), "floor"),
        _metal((0.0, 0.0, WALL), (WALL, y, z), "x low wall"),
        _metal((x - WALL, 0.0, WALL), (x, y, z), "x high wall"),
        _metal((WALL, 0.0, WALL), (x - WALL, WALL, z), "y low wall"),
        _metal((WALL, y - WALL, WALL), (x - WALL, y, z), "y high wall"),
    ]


def _between_two_slabs() -> list[Region]:
    """The same metal, cut on x first: two full-height slabs with the rest
    between them."""
    x, y, z = SHELL
    return [
        _metal((0.0, 0.0, 0.0), (WALL, y, z), "x low slab"),
        _metal((x - WALL, 0.0, 0.0), (x, y, z), "x high slab"),
        _metal((WALL, 0.0, 0.0), (x - WALL, y, WALL), "floor"),
        _metal((WALL, 0.0, WALL), (x - WALL, WALL, z), "y low wall"),
        _metal((WALL, y - WALL, WALL), (x - WALL, y, z), "y high wall"),
    ]


TILINGS = {"standing on the floor": _standing_on_the_floor, "between two slabs": _between_two_slabs}

#: Domain and policy the tilings are meshed in. Clear of the shell on every side
#: so that no face is excused the edge treatment for sitting at a wall.
CUT_DOMAIN = ((-5.0, -5.0, -5.0), (15.0, 15.0, 9.0))
CUT_PARAMS = MeshParams(metal_res=0.4, dielectric_res=2.0)


def _occupies(regions, point) -> bool:
    return any(
        all(region.lower[d] <= point[d] <= region.upper[d] for d in range(DIMENSIONS))
        for region in regions
    )


class TestOneSolidCutTwoWays:
    """Two partitions of one hollow box, which must mesh the same.

    A solid bounded by planes square to the grid reaches the mesher as the boxes
    it was cut into, and the mesher reads each box on its own - so the cut is a
    second thing the answer must not depend on, and unlike a rigid motion it
    leaves the coordinates alone and is invisible in the result.

    A hollow box is the specimen because each of its four side planes has metal
    of two depths against it: the wall standing on that plane, and the floor
    running the whole footprint away from it. Which of the two a box against the
    plane reports is decided by where the sweep cut, and both are true of the
    metal.

    What a face is worth is read off the piece of metal's own table of faces -
    :func:`conductor_faces` - rather than off the run of whichever box
    happened to lie against the plane, so the answer belongs to the metal and
    the cut reaches it nowhere.
    """

    def test_the_two_tilings_hold_the_same_metal(self):
        """A sanity check on the specimen, not a proof of equal regions: the
        boxes are disjoint, the volumes agree, and a lattice finer than the wall
        finds the same metal. Without it a tiling drifting off the shell would
        leave every comparison below green and about two structures."""
        tilings = [builder() for builder in TILINGS.values()]
        assert len({tuple(sorted((r.lower, r.upper) for r in one)) for one in tilings}) == len(
            tilings
        )
        for regions in tilings:
            for one, other in itertools.combinations(regions, 2):
                assert any(
                    one.lower[d] >= other.upper[d] - FLATNESS
                    or other.lower[d] >= one.upper[d] - FLATNESS
                    for d in range(DIMENSIONS)
                ), f"{one.label} and {other.label} overlap, so the volumes below prove nothing"
        volume = [
            sum(np.prod([region.extent(d) for d in range(DIMENSIONS)]) for region in regions)
            for regions in tilings
        ]
        assert volume[0] == pytest.approx(volume[1], rel=1e-12)
        # Half the wall, so no lattice line can straddle a wall and miss it.
        step = WALL / 2.0
        lattice = [np.arange(-step, SHELL[dim] + 2 * step, step) for dim in range(DIMENSIONS)]
        for point in itertools.product(*lattice):
            assert _occupies(tilings[0], point) == _occupies(tilings[1], point), point

    @pytest.mark.parametrize("dim", range(DIMENSIONS))
    def test_the_two_tilings_pin_the_same_lines(self, dim: int):
        """The anchors, which is where the cut reaches the grid: a face of the
        metal is pinned or it is not, and no grading covers the difference."""
        anchored = []
        for builder in TILINGS.values():
            mandatory, _ = mesh._fixed_positions(
                _grouped_into_conductors(builder()),
                dim,
                CUT_DOMAIN[0][dim],
                CUT_DOMAIN[1][dim],
                CUT_PARAMS,
            )
            anchored.append(sorted({round(position, 9) for position, _ in mandatory}))
        assert anchored[0] == anchored[1]

    @pytest.mark.parametrize("dim", range(DIMENSIONS))
    def test_the_two_tilings_give_the_same_grid(self, dim: int):
        """Cells rather than positions, and to :data:`PLACEMENT` rather than to
        rounding: the two cuts hand the sizing field different bends, so the
        integral between two anchors is summed over different pieces and the
        lines in a gap move. That is the same budget a translation is held to
        and for the same reason - a gap's count tipping by one. The count is
        what a solve costs and is asserted exactly."""
        grids = [
            generate_mesh_lines(builder(), CUT_DOMAIN, CUT_PARAMS)[dim]
            for builder in TILINGS.values()
        ]
        assert len(grids[0]) == len(grids[1])
        want = np.diff(grids[0])
        assert np.max(np.abs(np.diff(grids[1]) - want) / want) <= PLACEMENT

    def test_a_face_of_the_metal_is_resolved_at_the_finest_depth_against_it(self):
        """The plane at x = 0 has metal of two depths behind it - the wall
        standing on it, and the floor running the whole footprint away from it -
        and the wall is what has to be held. Resolving the floor's depth there
        would put the pair a wall's whole width apart.

        Both regions ask about that plane, and both ask the same thing: the
        depths belong to the piece of metal, so which box is asked decides
        nothing."""
        regions = _standing_on_the_floor()
        lines = generate_mesh_lines(regions, CUT_DOMAIN, CUT_PARAMS)
        grouped = _grouped_into_conductors(regions)
        wall, floor = (
            next(region for region in grouped.regions if region.label == label)
            for label in ("x low wall", "floor")
        )
        assert _face_depth(wall, 0, False, CUT_PARAMS) == pytest.approx(WALL, abs=0.0)
        fine = _edge_size(wall, 0, False, CUT_PARAMS)
        assert _edge_size(floor, 0, False, CUT_PARAMS) == fine
        # What the floor would ask for on its own run away from that plane,
        # which is what a face read off the box against it comes to. The same
        # box grouped by itself, so nothing else is there for it to be part of.
        alone = _grouped_into_conductors(
            [region for region in regions if region.label == "floor"]
        ).regions[0]
        coarse = _edge_size(alone, 0, False, CUT_PARAMS)
        assert fine < coarse
        for cell, present in ((fine, True), (coarse, False)):
            pair = edge_lines(0.0, -1.0, cell, CUT_PARAMS.edge_line_inside)
            for position in pair:
                assert (
                    bool(np.any(np.isclose(lines.x, position, rtol=0.0, atol=1e-9))) is present
                ), (position, cell)

    def test_two_blocks_sharing_a_plane_keep_an_edge_each(self):
        """A face is a plane and a side. Two separate blocks - one ending at the
        plane from below, one starting from it a few millimetres along - have an
        edge at that coordinate looking opposite ways, and the void one of them
        faces is the side the other's metal is on. Two edges, and each is asked
        for at its own depth."""
        below = _metal((0.0, 0.0, -2.0), (4.0, 4.0, 0.0), "below")
        above = _metal((5.0, 0.0, 0.0), (9.0, 4.0, 0.6), "above")
        domain = ((-6.0, -6.0, -6.0), (15.0, 10.0, 6.0))
        lines = generate_mesh_lines([below, above], domain, CUT_PARAMS)
        grouped = _grouped_into_conductors([below, above])
        sizes = set()
        for region, outward, at_high in zip(grouped.regions, (1.0, -1.0), (True, False)):
            cell = _edge_size(region, 2, at_high, CUT_PARAMS)
            sizes.add(cell)
            for position in edge_lines(0.0, outward, cell, CUT_PARAMS.edge_line_inside):
                assert np.any(np.isclose(lines.z, position, rtol=0.0, atol=1e-9)), (
                    f"{region.label} lost its edge at z = 0"
                )
        assert len(sizes) == 2, "both edges ask the same cell, so neither can be dropped"

    @pytest.mark.parametrize("dim", range(DIMENSIONS))
    def test_every_box_against_one_plane_asks_the_same_cell(self, dim: int):
        """What lets one pair stand for a whole face: a depth is read off the
        piece's table, so two boxes of one piece against one plane cannot
        disagree about what that plane is worth."""
        for builder in TILINGS.values():
            asked: dict[tuple[float, bool], set[float]] = {}
            for region in _grouped_into_conductors(builder()).regions:
                for at_high in (False, True):
                    if _face_depth(region, dim, at_high, CUT_PARAMS) is None:
                        continue
                    plane = region.upper[dim] if at_high else region.lower[dim]
                    asked.setdefault((round(plane, 9), at_high), set()).add(
                        _edge_size(region, dim, at_high, CUT_PARAMS)
                    )
            assert asked, "no face asked for anything, so this proves nothing"
            for face, cells in asked.items():
                assert len(cells) == 1, (face, cells)

    @pytest.mark.parametrize("sliver", (0.05, 0.1, 0.2))
    def test_a_thin_piece_against_an_edge_does_not_cost_it_the_treatment(self, sliver: float):
        """Whether a face gets the thirds rule at all is decided on the extent
        of the metal, never on the box's own. A cut leaving a piece thinner than
        the cell against a real edge would otherwise pin a plain line there -
        which is a line *on* the conductor face, the one placement the rule
        exists to avoid - while the same strip drawn whole is resolved."""
        strip = ((0.0, 0.0, 0.0), (4.0, 1.0, 0.05))
        domain = ((-5.0, -5.0, -5.0), (9.0, 6.0, 5.0))
        whole = [_metal(*strip, "strip")]
        cut = [
            _metal(strip[0], (sliver, strip[1][1], strip[1][2]), "end"),
            _metal((sliver, 0.0, 0.0), strip[1], "rest"),
        ]
        assert sliver < _edge_size(
            _grouped_into_conductors(cut).regions[0], 0, False, CUT_PARAMS
        ), "the piece is not thinner than the cell, so nothing here is at stake"
        anchored = []
        for regions in (whole, cut):
            mandatory, _ = mesh._fixed_positions(
                _grouped_into_conductors(regions), 0, domain[0][0], domain[1][0], CUT_PARAMS
            )
            anchored.append(sorted({round(position, 9) for position, _ in mandatory}))
        assert anchored[0] == anchored[1]
        # And the pair is really there, so the two do not agree by both losing it.
        cell = _edge_size(_grouped_into_conductors(whole).regions[0], 0, False, CUT_PARAMS)
        for position in edge_lines(0.0, -1.0, cell, CUT_PARAMS.edge_line_inside):
            assert round(position, 9) in anchored[0]

    def test_a_conductor_thinner_than_a_cell_still_gets_plain_faces(self):
        """The other side of the rule above, and what the extent is read for: a
        piece of metal that really is thinner than the cell has its two pairs
        crossing, so both faces are pinned plainly instead. Reading the metal
        rather than the box does not weaken that - a conductor drawn whole is
        its own metal."""
        foil = [_metal((0.0, 0.0, 0.0), (0.2, 4.0, 4.0), "foil")]
        domain = ((-5.0, -5.0, -5.0), (9.0, 9.0, 9.0))
        mandatory, _ = mesh._fixed_positions(
            _grouped_into_conductors(foil), 0, domain[0][0], domain[1][0], CUT_PARAMS
        )
        assert sorted({round(position, 9) for position, _ in mandatory}) == [
            domain[0][0],
            0.0,
            0.2,
            domain[1][0],
        ]


class TestWhatGroupingHandsOn:
    """:func:`_grouped_into_conductors` produces the one value the grid is
    built from, so what it carries forward is the whole of what the drawing
    said about a box plus what it worked out about the metal."""

    def test_a_conductor_carries_the_region_it_was_made_from_whole(self):
        """Grouping copies a region field by field, so a field added to
        :class:`regions.Region` with a default and not copied is lost without a
        word: the box meshes at that default instead of at what was drawn. A
        field with no default raises instead, and the checker names it. Asked of
        the fields rather than of a list written here, so the day the first
        happens this fails rather than the grid moving."""
        drawn = Region(
            lower=(0.0, 0.0, 0.0),
            upper=(4.0, 2.0, 0.5),
            material=MaterialClass.METAL,
            label="drawn",
            material_name="Copper",
            size=0.3,
            continuous=frozenset({1}),
            drawn=(9.0, 2.0, 0.5),
            relaxed_to=0.7,
        )
        settled = [
            (
                field.name,
                field.default if field.default_factory is MISSING else field.default_factory(),
            )
            for field in fields(Region)
            if not (field.default is MISSING and field.default_factory is MISSING)
        ]
        assert all(getattr(drawn, name) != default for name, default in settled), (
            "a field left at its default reads the same carried or dropped"
        )
        carried = _grouped_into_conductors([drawn]).regions[0]
        for field in fields(Region):
            assert getattr(carried, field.name) == getattr(drawn, field.name), field.name

    def test_a_region_that_is_not_metal_is_handed_on_as_it_arrived(self):
        """Only metal has a piece to belong to, so a dielectric comes back the
        object that went in rather than a copy of it.

        Drawn with metal beside it, because a list holding no metal at all is
        handed straight back and would prove this of a route nothing takes."""
        board = Region(
            lower=(-8.0, -8.0, 0.0),
            upper=(8.0, 8.0, 1.6),
            material=MaterialClass.DIELECTRIC,
            label="Substrate",
        )
        ground = _metal((-8.0, -8.0, 0.0), (8.0, 8.0, 0.0), "GroundPlane")
        handed = _grouped_into_conductors([board, ground]).regions
        assert handed[0] is board
        assert isinstance(handed[1], Conductor)


class TestWhichPairsSurviveAtOneFace:
    """:func:`_one_pair_at_each_face` on its own, which is where the rule
    that collapses a face's pairs is stated.

    A pure function of a list, so the awkward arrangements - two sides at one
    coordinate, two conductors at one coordinate, coordinates a hair apart - are
    written down rather than searched for in a structure that produces them.

    Every pair here asks the same cell, because that is what the mesher hands
    it: a depth is read off the piece of metal, so which box asked cannot
    change the answer. What survives is therefore never decided by a size, and
    a specimen that varied one would be testing a distinction the caller cannot
    make.
    """

    @staticmethod
    def _pair(edge: float, cell: float, region: Conductor, outward: float = -1.0):
        inside, outside = edge_lines(edge, outward, cell, CUT_PARAMS.edge_line_inside)
        return (edge, inside, outside, region)

    @staticmethod
    def _piece(lower, upper, label: str, piece: int = 0) -> Conductor:
        """One box of a conductor, as ``_grouped_into_conductors`` hands it on.

        Grouped on its own, so its run, its faces and its extent are the box's
        own and nothing is written here that the grouping would not produce.
        The piece is then set, because whether two boxes are one conductor is
        what these specimens vary and drawing them touching would settle it
        somewhere else."""
        alone = _grouped_into_conductors([_metal(lower, upper, label)]).regions[0]
        return replace(alone, piece=piece)

    def test_the_pairs_come_back_in_the_order_they_arrived(self):
        """The anchors are built in this order and :func:`mesh._snap` keeps the
        first at a coincident position, so the order decides which box a face is
        named after in a refusal and in the report."""
        one = self._piece((0.0, 0.0, 0.0), (4.0, 4.0, 1.0), "one", piece=0)
        two = self._piece((9.0, 0.0, 0.0), (13.0, 4.0, 1.0), "two", piece=1)
        given = [self._pair(9.0, 0.2, two), self._pair(0.0, 0.2, one)]
        assert [candidate[3].label for candidate in _one_pair_at_each_face(given)] == [
            "two",
            "one",
        ]

    def test_two_sides_of_one_plane_survive_whatever_order_they_arrive_in(self):
        """Interleaved, so a rule that only looks at the pair it kept last sees a
        face of the other side between two of this one and starts again."""
        piece = self._piece((0.0, 0.0, -2.0), (4.0, 4.0, 2.0), "step")
        given = [
            self._pair(0.0, 0.2, piece, outward=-1.0),
            self._pair(0.0, 0.2, piece, outward=1.0),
            self._pair(0.0, 0.2, piece, outward=-1.0),
        ]
        kept = _one_pair_at_each_face(given)
        assert sorted(outside > inside for _, inside, outside, _ in kept) == [False, True]

    def test_a_conductor_the_metal_does_not_reach_keeps_its_own_pair(self):
        """A face is answered by lines the same metal asked for. Otherwise the
        cell at one conductor is decided by an unrelated object elsewhere in the
        model, and moves when that object is drawn a nanometre further off."""
        near = self._piece((0.0, 0.0, 0.0), (20.0, 10.0, 2.0), "ground", piece=0)
        far = self._piece((26.0, 0.0, 0.0), (28.0, 2.0, 0.6), "pad", piece=1)
        given = [self._pair(0.0, 0.15, near), self._pair(0.0, 0.15, far)]
        assert len(_one_pair_at_each_face(given)) == 2

    def test_faces_the_kernel_can_tell_apart_stay_two_faces(self):
        """The merge is for one plane read off two boxes, which agree to the
        kernel's own tolerance and no worse. Two features a fiftieth of a
        millimetre apart are two features and each keeps its edge."""
        piece = self._piece((0.0, 0.0, 0.0), (4.0, 4.0, 1.0), "piece")
        given = [self._pair(0.0, 0.2, piece), self._pair(0.02, 0.2, piece)]
        assert len(_one_pair_at_each_face(given)) == 2

    def test_one_plane_read_off_two_boxes_is_one_face(self):
        """The other side of it: a coordinate that differs only in the last bits
        of the kernel's arithmetic is not two faces."""
        piece = self._piece((0.0, 0.0, 0.0), (4.0, 4.0, 1.0), "piece")
        given = [self._pair(0.0, 0.2, piece), self._pair(1e-9, 0.2, piece)]
        assert len(_one_pair_at_each_face(given)) == 1

    def test_a_run_the_piece_reaches_only_through_another_is_still_one_face(self):
        """A piece of metal arrives as several runs and two of them need not meet
        each other; a third meeting both is what makes them one. Being one piece
        is walked rather than tested, so all three are one face here however far
        apart the arms sit."""
        low = self._piece((0.0, 0.0, 0.0), (4.0, 4.0, 4.0), "low arm")
        middle = self._piece((0.0, 0.0, 0.0), (12.0, 4.0, 1.0), "spine")
        high = self._piece((8.0, 0.0, 0.0), (12.0, 4.0, 4.0), "high arm")
        given = [
            self._pair(0.0, 0.2, low),
            self._pair(0.0, 0.2, high),
            self._pair(0.0, 0.2, middle),
        ]
        assert len(_one_pair_at_each_face(given)) == 1

    @pytest.mark.parametrize("dim", range(DIMENSIONS))
    def test_a_block_drawn_in_pieces_meshes_as_the_block(self, dim: int):
        """The case a single neighbour cannot answer: the plate's continuation
        upward is two pads and neither covers it, so a run read off one box
        reports a third of the metal that is there. Drawn whole, drawn as a plate
        with two pads on it, and drawn as two half plates under one top."""
        block = [_metal((0.0, 0.0, 0.0), (10.0, 10.0, 3.0), "whole")]
        plate_and_pads = [
            _metal((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), "plate"),
            _metal((0.0, 0.0, 1.0), (5.0, 10.0, 3.0), "pad a"),
            _metal((5.0, 0.0, 1.0), (10.0, 10.0, 3.0), "pad b"),
        ]
        halves_and_top = [
            _metal((0.0, 0.0, 0.0), (5.0, 10.0, 1.0), "half a"),
            _metal((5.0, 0.0, 0.0), (10.0, 10.0, 1.0), "half b"),
            _metal((0.0, 0.0, 1.0), (10.0, 10.0, 3.0), "top"),
        ]
        grids = [
            generate_mesh_lines(drawing, CUT_DOMAIN, CUT_PARAMS)[dim]
            for drawing in (block, plate_and_pads, halves_and_top)
        ]
        for other in grids[1:]:
            assert len(other) == len(grids[0])
            assert np.max(np.abs(np.diff(other) - np.diff(grids[0])) / np.diff(grids[0])) <= (
                PLACEMENT
            )

    def test_and_no_single_neighbour_covers_that_plate(self):
        """Without this the case above is three drawings of a block that any rule
        would agree about."""
        boxes = [
            ((0.0, 0.0, 0.0), (10.0, 10.0, 1.0)),
            ((0.0, 0.0, 1.0), (5.0, 10.0, 3.0)),
            ((5.0, 0.0, 1.0), (10.0, 10.0, 3.0)),
        ]
        covering = [
            box
            for box in boxes[1:]
            if all(box[0][a] <= FLATNESS and box[1][a] >= 10.0 - FLATNESS for a in (0, 1))
        ]
        assert not covering
        assert conductor_run(boxes[0][0], boxes[0][1], boxes, 2) == (0.0, 3.0)


class TestWhatARunIsMeasuredOver:
    """:func:`conductor_run` on its own: the cross-section it divides, and
    the two ways dividing it wrongly hides a conductor rather than failing."""

    def test_a_conductor_with_no_thickness_still_has_a_run(self):
        """A sheet's cross-section is a plane, so it has no bands unless a span
        of no width counts as one. Without that the minimum is taken over
        nothing, every sheet reports itself unbounded, and every one of them
        quietly loses the edge treatment while the run completes."""
        sheet = ((0.0, 0.0, 0.0), (4.0, 6.0, 0.0))
        assert conductor_run(sheet[0], sheet[1], [sheet], 0) == (0.0, 4.0)
        assert conductor_run(sheet[0], sheet[1], [sheet], 2) == (0.0, 0.0)

    def test_two_sheets_butted_in_their_own_plane_are_one_run(self):
        near = ((0.0, 0.0, 0.0), (4.0, 6.0, 0.0))
        far = ((4.0, 0.0, 0.0), (9.0, 6.0, 0.0))
        assert conductor_run(near[0], near[1], [near, far], 0) == (0.0, 9.0)

    def test_metal_merely_alongside_does_not_carry_a_run_on(self):
        """What carries a run on is covering a band's middle, not touching the
        cross-section. A neighbour lying against this box's side is beside it,
        and a rule that counted it would grow a run along a face two conductors
        happen to share."""
        strip = ((0.0, 0.0, 0.0), (4.0, 2.0, 1.0))
        alongside = ((0.0, 2.0, 0.0), (9.0, 4.0, 1.0))
        assert conductor_run(strip[0], strip[1], [strip, alongside], 0) == (0.0, 4.0)

    def test_a_neighbour_a_hair_out_of_line_still_carries_the_run(self):
        """Two faces the kernel cannot separate are one face. They do divide the
        cross-section into a band narrower than that, and what keeps the run
        whole across it is that covering a middle is asked to within FLATNESS -
        so both neighbours cover the sliver between them. Without that, every
        run would stop at its own box: finer everywhere, and looking like the
        fault this rule removes rather than like a bug."""
        plate = ((0.0, 0.0, 0.0), (4.0, 4.0, 1.0))
        near = ((0.0, 0.0, 1.0), (2.0, 4.0, 3.0))
        far = ((2.0 + FLATNESS / 2.0, 0.0, 1.0), (4.0, 4.0, 3.0))
        assert conductor_run(plate[0], plate[1], [plate, near, far], 2) == (0.0, 3.0)

    def test_a_face_met_by_two_metals_is_pinned_for_the_one_that_differs(self):
        """Once the metal past a face may be several regions, which of them is
        named stops being arbitrary: the caller pins a plain line where the
        material differs and nothing where it does not, and a property boundary
        with no line on it moves by up to a cell. So the odd one out is the one
        to name, whichever order the regions arrive in."""
        plate = _metal((0.0, 0.0, 0.0), (4.0, 4.0, 1.0), "plate")
        same = replace(_metal((0.0, 0.0, 1.0), (2.0, 4.0, 3.0), "same metal"))
        other = replace(
            _metal((2.0, 0.0, 1.0), (4.0, 4.0, 3.0), "other metal"), material_name="PEC"
        )
        for order in ([plate, same, other], [plate, other, same]):
            grouped = _grouped_into_conductors(order)
            met = _met_by_metal(grouped.regions[0], grouped, 2, True)
            assert met is not None and met.material_name == "PEC"

    def test_and_a_face_met_by_one_metal_throughout_names_that_one(self):
        plate = _metal((0.0, 0.0, 0.0), (4.0, 4.0, 1.0), "plate")
        halves = [
            _metal((0.0, 0.0, 1.0), (2.0, 4.0, 3.0), "half a"),
            _metal((2.0, 0.0, 1.0), (4.0, 4.0, 3.0), "half b"),
        ]
        grouped = _grouped_into_conductors([plate, *halves])
        met = _met_by_metal(grouped.regions[0], grouped, 2, True)
        assert met is not None and met.material_name == plate.material_name

    def test_a_face_outside_the_cross_section_is_not_a_band_boundary(self):
        """A neighbour may reach into the cross-section and go on well past it,
        and where it ends outside says nothing about what covers the inside. A
        band boundary there puts a middle outside the cross-section, and the run
        is then read off metal this box has no face against."""
        assert _bands(0.0, 4.0, [-3.0, 2.0, 9.0]) == [(0.0, 2.0), (2.0, 4.0)]
        assert _bands(0.0, 4.0, [-3.0, 9.0]) == [(0.0, 4.0)]

    def test_a_band_boundary_sits_at_both_faces_of_a_neighbour(self):
        """A run is read at one point per band, so a band must not straddle a
        place where coverage changes. Taking boundaries from a neighbour's near
        face alone leaves the far one inside a band: here a single pad covers
        half the plate, and a band spanning both halves reads the covered half
        and reports the plate three times as deep as it is."""
        plate = ((0.0, 0.0, 0.0), (4.0, 4.0, 1.0))
        pad = ((0.0, 0.0, 1.0), (2.0, 4.0, 3.0))
        assert conductor_run(plate[0], plate[1], [plate, pad], 2) == (0.0, 1.0)

    def test_metal_ending_at_a_face_is_not_past_it(self):
        """A region flush with the face lies on this side and covers nothing
        beyond, so it cannot be what the metal past the face is made of - and
        naming it would pin a line on a seam that is copper on both sides."""
        plate = _metal((0.0, 0.0, 0.0), (4.0, 4.0, 1.0), "plate")
        above = _metal((0.0, 0.0, 1.0), (4.0, 4.0, 3.0), "above")
        under = replace(_metal((0.0, 0.0, -1.0), (2.0, 4.0, 1.0), "under"), material_name="PEC")
        grouped = _grouped_into_conductors([plate, above, under])
        met = _met_by_metal(grouped.regions[0], grouped, 2, True)
        assert met is not None and met.material_name == plate.material_name

    def test_metal_alongside_is_not_metal_past_the_face(self):
        """A conductor sharing only a plane with this one lies over none of its
        face, so what it is made of says nothing about that face. Otherwise a
        copper seam gets a line pinned on it because somebody drew a PEC block
        beside it, and the grid moves with a distant object."""
        lower = _metal((0.0, 0.0, 0.0), (4.0, 4.0, 1.0), "lower half")
        upper = _metal((0.0, 0.0, 1.0), (4.0, 4.0, 2.0), "upper half")
        beside = replace(_metal((4.0, 0.0, 0.0), (8.0, 4.0, 2.0), "beside"), material_name="PEC")
        for regions in ([lower, upper], [lower, upper, beside]):
            grouped = _grouped_into_conductors(regions)
            met = _met_by_metal(grouped.regions[0], grouped, 2, True)
            assert met is not None and met.material_name == lower.material_name

    def test_a_chain_that_closes_through_a_joined_pair_is_one_piece(self):
        """Connectedness is not pairwise, so it is walked rather than tested:
        two boxes that touch nothing of each other are one piece once a third
        touches both, and joining anything but the two chains' own
        representatives splits what was already joined."""
        boxes = [
            ((2.0, 2.0, 3.0), (3.0, 3.0, 4.0)),
            ((2.0, 2.0, 2.0), (4.0, 4.0, 4.0)),
            ((3.0, 3.0, 1.0), (4.0, 4.0, 2.0)),
            ((3.0, 1.0, 3.0), (4.0, 2.0, 4.0)),
        ]
        assert len(set(conductor_pieces(boxes))) == 1

    def test_a_conductor_touching_nothing_is_its_own_piece(self):
        boxes = [((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), ((9.0, 9.0, 9.0), (10.0, 10.0, 10.0))]
        assert len(set(conductor_pieces(boxes))) == 2

    def test_a_decomposed_conductor_pins_no_seam(self):
        """One strip cut into three boxes, and no line at either cut.

        Where the metal ends is read off what grouping filled in, so the run
        spans the whole strip and neither cut is a face of anything. The
        absorber's own pass asks the same question of the same regions and gets
        the same answer, which is what makes a decomposed conductor mesh as the
        shape it was cut from."""
        strip = [
            _metal((0.0, 0.0, 0.0), (3.0, 1.0, 0.2), "a"),
            _metal((3.0, 0.0, 0.0), (6.0, 1.0, 0.2), "b"),
            _metal((6.0, 0.0, 0.0), (9.0, 1.0, 0.2), "c"),
        ]
        grouped = _grouped_into_conductors(strip)
        pinned, _ = mesh._fixed_positions(grouped, 0, -5.0, 14.0, CUT_PARAMS)
        assert not [
            source for _, source in pinned if "edge at 3" in source or "edge at 6" in source
        ]


class TestWhatFacesAPieceOfMetalHas:
    """:func:`conductor_faces` on its own: which planes a piece of metal
    ends at along an axis, and how deep it is behind each.

    A run says how far the metal reaches everywhere across a box, which is a
    width; a face is open over only part of that cross-section, and how deep the
    metal is behind the open part is a different number and the one a thirds
    pair is sized from. Asking it of the piece is what makes it independent of
    the cut.
    """

    @staticmethod
    def _faces(boxes, dim):
        return {
            (face.plane, face.at_high): face.depths for face in conductor_faces(boxes, boxes, dim)
        }

    def test_one_box_has_its_own_two_faces(self):
        box = ((0.0, 0.0, 0.0), (4.0, 6.0, 1.0))
        assert self._faces([box], 0) == {(0.0, False): (4.0,), (4.0, True): (4.0,)}

    def test_a_plane_two_boxes_are_cut_at_is_not_a_band_of_its_own(self):
        """A decomposed solid puts several boxes' faces at one coordinate, so the
        cuts a cross-section is divided at arrive repeated. Divided at one twice,
        the band left between the two is the plane itself - and a band stands for
        the coverage at its middle, so the metal on *both* sides answers there
        and the run comes back deeper than any column the metal has.

        :func:`conductor_run` never saw it, taking the shallowest band and
        never the deepest; reading each band's ends is what makes it visible."""
        low = ((0.0, 0.0, 0.0), (1.0, 1.0, 2.0))
        near = ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        far = ((2.0, 0.0, 1.0), (3.0, 1.0, 2.0))
        boxes = [low, near, far]
        assert _bands(0.0, 2.0, [end[2] for box in boxes for end in box]) == [
            (0.0, 1.0),
            (1.0, 2.0),
        ]
        # Below z = 1 the metal runs x 0..2; above it, 0..1 and 2..3. Nothing is
        # three deep, and the plane at z = 1 is where a run of three came from.
        assert self._faces(boxes, 0)[(0.0, False)] == (1.0, 2.0)
        assert self._faces(boxes, 0)[(3.0, True)] == (1.0,)

    def test_a_seam_is_not_a_face(self):
        """Two conductors butted face to face are one piece of metal, so the
        plane where they meet is a boundary of nothing and the run behind either
        end is the whole of it."""
        near = ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        far = ((2.0, 0.0, 0.0), (5.0, 1.0, 1.0))
        assert self._faces([near, far], 0) == {(0.0, False): (5.0,), (5.0, True): (5.0,)}

    def test_a_plane_a_face_of_the_metal_in_several_places_carries_each_depth(self):
        """A comb's teeth all end on their own planes and all start on one. The
        plane they start on is a face nowhere - the spine is behind it - but the
        depths behind each tooth's own end differ, and a single number for the
        plane would be one tooth's answer given to the others."""
        spine = ((0.0, 0.0, 0.0), (6.0, 1.0, 1.0))
        teeth = [((2.0 * k, 1.0, 0.0), (2.0 * k + 1.0, 2.0 + k, 1.0)) for k in range(3)]
        assert self._faces([spine, *teeth], 1) == {
            # The spine alone between two teeth, and each tooth over it.
            (0.0, False): (1.0, 2.0, 3.0, 4.0),
            # The spine's own top, where no tooth stands on it.
            (1.0, True): (1.0,),
            **{(2.0 + k, True): (2.0 + k,) for k in range(3)},
        }

    def test_the_depth_behind_a_face_is_not_the_run_across_the_box(self):
        """The two numbers side by side, on the shape that separates them. The
        plate's continuation upward is two pads and neither covers its whole
        cross-section, so the run stops at the plate - while the metal behind
        the plane at the bottom is the block's full height."""
        plate = ((0.0, 0.0, 0.0), (10.0, 10.0, 1.0))
        pads = [((0.0, 0.0, 1.0), (5.0, 10.0, 3.0)), ((5.0, 0.0, 1.0), (10.0, 10.0, 3.0))]
        boxes = [plate, *pads]
        assert self._faces(boxes, 2)[(0.0, False)] == (3.0,)
        assert conductor_run(plate[0], plate[1], boxes, 2) == (0.0, 3.0)

    def test_a_depth_no_width_can_be_read_off_does_not_decide_a_plane(self):
        """One piece of metal can end on one plane in two places, and one of them
        be a fin thinner than a cell. The fin is getting plain lines of its own -
        there is no share of it to hold - so letting its depth decide the plane
        would size the arm's pair from metal that is not asking for one, and the
        cell would come out a fraction of anything the policy named."""
        arm = _metal((0.0, 0.0, 0.0), (10.0, 2.0, 1.0), "arm")
        fin = _metal((0.0, 0.0, 1.0), (0.05, 2.0, 5.0), "fin")
        assert self._faces([(arm.lower, arm.upper), (fin.lower, fin.upper)], 0)[(0.0, False)] == (
            0.05,
            10.0,
        )
        # Both boxes have a face on that plane and both are answered by the arm's
        # depth, which is what makes one pair able to stand for the plane: a box
        # with nothing to hold asks for what the plane asks for rather than for
        # something of its own.
        for region in _grouped_into_conductors([arm, fin]).regions:
            assert _face_depth(region, 0, False, CUT_PARAMS) == 10.0
            assert _edge_size(region, 0, False, CUT_PARAMS) == CUT_PARAMS.metal_res

    def test_a_face_is_answered_by_its_own_piece_of_metal(self):
        """Two conductors ending on one plane far apart. A table built over both
        would hand the pad's depth to the ground plane, and the cell at a ground
        plane would then move when an unrelated object was drawn a little
        shorter."""
        plane = _metal((0.0, 0.0, 0.0), (20.0, 10.0, 1.0), "ground")
        pad = _metal((0.0, 20.0, 0.0), (2.0, 22.0, 1.0), "pad")
        ground, pad_only = _grouped_into_conductors([plane, pad]).regions
        assert ground.piece != pad_only.piece
        assert _face_depth(ground, 0, False, CUT_PARAMS) == 20.0
        assert _face_depth(pad_only, 0, False, CUT_PARAMS) == 2.0

    def test_a_seam_band_cannot_talk_a_face_out_of_its_treatment(self):
        """Whether an axis is a width is a question about the piece of metal, and
        the box's *run* is not that: a run is the intersection over the box's
        cross-section, so a box whose face is part seam reports a run narrower
        than every column of the metal - and narrower than a cell, on which the
        face is declined and pinned plainly. Which is a line on a conductor face
        with a width of metal behind it, and the cut decides whether it happens.

        A Z of three steps, cut two ways, every step just under a cell - which
        is the regime the question lives in."""
        step = 0.35
        shape = {
            "on the middle": [
                ((0.0, 1 * step, 0.0), (step, 2 * step, 2 * step)),
                ((0.0, 2 * step, 1 * step), (step, 3 * step, 2 * step)),
                ((0.0, 0.0, 0.0), (step, 1 * step, 1 * step)),
            ],
            "on the slab": [
                ((0.0, 0.0, 0.0), (step, 2 * step, 1 * step)),
                ((0.0, 1 * step, 1 * step), (step, 2 * step, 2 * step)),
                ((0.0, 2 * step, 1 * step), (step, 3 * step, 2 * step)),
            ],
        }
        anchored = []
        for name, boxes in shape.items():
            regions = [
                _metal(lower, upper, f"{name} {n}") for n, (lower, upper) in enumerate(boxes)
            ]
            grouped = _grouped_into_conductors(regions)
            assert any(
                width_axes(region.run[0], region.run[1], CUT_PARAMS.metal_res)
                != width_axes(region.piece_box[0], region.piece_box[1], CUT_PARAMS.metal_res)
                for region in grouped.regions
            ), f"{name}: no box's run disagrees with the piece, so nothing here is at stake"
            mandatory, _ = mesh._fixed_positions(grouped, 1, -5.0, 5.0, CUT_PARAMS)
            anchored.append(sorted(round(position, 9) for position, _ in mandatory))
        assert anchored[0] == anchored[1]
        # And both resolve the two faces rather than agreeing by pinning them.
        assert 2 * step not in anchored[0] and step not in anchored[0]

    @pytest.mark.parametrize("dim", range(DIMENSIONS))
    def test_the_two_tilings_have_the_same_faces(self, dim: int):
        """The property the whole of it is for, asserted on the table rather than
        on the grid it ends up in."""
        tables = [
            self._faces([(region.lower, region.upper) for region in builder()], dim)
            for builder in TILINGS.values()
        ]
        assert tables[0] == tables[1]


class TestTheTwoInnerLinesCannotCross:
    """The one thing a region's *own* extent still decides.

    Each thirds pair puts its inner line a share of a cell inside the metal, so
    two of them meet when the region is thinner than those two shares together.
    Where the share is small the arithmetic cannot reach it - a face is only an
    edge on an axis the conductor is wider than a cell of, and the share is a
    fraction of that cell - so the guard is reachable only where the share is
    most of the cell, which is what makes it a policy rather than a constant.
    """

    BLOCK = [_metal((0.0, 0.0, 0.0), (0.5, 4.0, 4.0), "block")]
    DOMAIN = (-5.0, 6.0)

    def _anchors(self, inside: float):
        params = replace(CUT_PARAMS, edge_line_inside=inside)
        mandatory, _ = mesh._fixed_positions(
            _grouped_into_conductors(self.BLOCK), 0, *self.DOMAIN, params
        )
        return sorted(round(position, 6) for position, _ in mandatory)

    def test_at_the_default_share_both_faces_are_resolved(self):
        assert self._anchors(EDGE_LINE_INSIDE) == [-5.0, -0.025, 0.0125, 0.4875, 0.525, 6.0]

    def test_at_a_share_that_would_cross_both_faces_are_pinned_plainly(self):
        """Nothing is refused and no pair is placed on top of another: the faces
        are held where they are, which is what a grid can still say about them."""
        assert self._anchors(0.96) == [-5.0, 0.0, 0.5, 6.0]


class TestWhatOneFaceIsAsked:
    """:func:`metal._edge_pair` is the one question the pinning and the edge
    demand both put about a conductor face, so what it may be asked about is
    wider than what either caller goes on to use."""

    #: A plate thinner than the cell the policy asks for at metal, so z is not a
    #: width the grid resolves and the thirds rule does not apply across it.
    FOIL = ((-4.0, -4.0, 0.0), (4.0, 4.0, 0.1))

    def _pair(self, dim: int, at_high: bool):
        grouped = _grouped_into_conductors([_metal(*self.FOIL, "Foil")])
        region = grouped.regions[0]
        return region, _edge_pair(region, grouped, dim, at_high, CUT_PARAMS, -50.0, 50.0)

    def test_a_face_on_an_axis_that_is_not_a_width_is_still_given_a_verdict(self):
        """The verdict is the metal\'s own and is reached without a depth. The
        pinning asks for it first and drops the face afterwards on the depth, so
        a helper that answered only where there was a width would leave the
        thirds rule deciding on a value nobody computed."""
        region, face = self._pair(2, False)
        assert _face_depth(region, 2, False, CUT_PARAMS) is None, (
            "the foil is thicker than a cell here, so z is a width and this is "
            "not the case the test is about"
        )
        assert face.resolve is True

    def test_the_inner_line_of_the_pair_is_the_one_in_the_metal(self):
        """Both callers read the two by name, and swapping them would put the
        conductor face a whole cell from where it was drawn."""
        for dim in range(DIMENSIONS):
            for at_high in (False, True):
                region, face = self._pair(dim, at_high)
                assert min(face.inside, face.outside) < face.position, (dim, at_high)
                assert face.position < max(face.inside, face.outside), (dim, at_high)
                assert (face.inside < face.position) is at_high, (dim, at_high)


class TestWhatAnAxisAnchorsOn:
    """:func:`mesh._fixed_positions` on the two things its answer turns on that
    no grid can show: what the thirds rule was judged against, and which of two
    coincident anchors names the line."""

    def test_one_pair_is_not_crowded_by_another(self):
        """The pairs are judged against what the axis pinned before any of them
        was laid. Judged against a running list instead, a face keeps its pair
        or gives it up according to the order the regions arrived in, and the
        conductor drawn second is meshed worse than the one drawn first.

        The gap is narrower than the cell at these faces, so each conductor's
        outer line stands inside the other's pair."""
        pair = [
            _metal((-3.1, -2.0, -2.0), (-0.1, 2.0, 2.0), "left"),
            _metal((0.1, -2.0, -2.0), (3.1, 2.0, 2.0), "right"),
        ]
        mandatory, _ = mesh._fixed_positions(
            _grouped_into_conductors(pair), 0, -8.0, 8.0, CUT_PARAMS
        )
        placed = {source: position for position, source in mandatory}
        assert (
            placed["'right' edge at 0.1, outside"]
            < placed["'left' edge at -0.1, outside"]
            < placed["'right' edge at 0.1, inside"]
        ) and (
            placed["'left' edge at -0.1, inside"]
            < placed["'right' edge at 0.1, outside"]
            < placed["'left' edge at -0.1, outside"]
        ), "the two pairs do not reach each other, so nothing here is at stake"
        assert sorted(source for source in placed if "edge at -0.1" in source) == [
            "'left' edge at -0.1, inside",
            "'left' edge at -0.1, outside",
        ]
        assert sorted(source for source in placed if "edge at 0.1" in source) == [
            "'right' edge at 0.1, inside",
            "'right' edge at 0.1, outside",
        ]

    def test_a_face_drawn_on_the_domain_wall_is_named_for_the_wall(self):
        """Two anchors at one coordinate are one line, and :func:`mesh._snap`
        keeps the first the axis offered. The walls are offered first, so a
        refusal and a report name the wall rather than whatever was drawn
        against it."""
        pad = [_metal((0.0, -2.0, -2.0), (0.1, 2.0, 2.0), "pad")]
        mandatory, preferred = mesh._fixed_positions(
            _grouped_into_conductors(pad), 0, -5.0, 0.1, CUT_PARAMS
        )
        assert (0.1, "'pad' face at the domain wall") in mandatory, (
            "the pad's face is not pinned on the wall, so no two anchors coincide"
        )
        named = {
            line.position: line.source
            for line in mesh._snap(mandatory, preferred, CUT_PARAMS.floor, 0)
        }
        assert named[0.1] == "domain upper bound"

    def test_a_plane_read_off_two_boxes_is_answered_once(self):
        """The pairs are collapsed to one at each face on the way through. Left
        uncollapsed, a solid cut into boxes asks at the same plane once per box,
        and two boxes whose faces differ inside the kernel's own tolerance ask
        for two pairs a nanometre apart."""
        cut = [
            _metal((0.0, -2.0, -2.0), (3.0, 0.0, 2.0), "a"),
            _metal((1e-9, 0.0, -2.0), (3.0, 2.0, 2.0), "b"),
        ]
        mandatory, _ = mesh._fixed_positions(
            _grouped_into_conductors(cut), 0, -8.0, 8.0, CUT_PARAMS
        )
        assert sorted(source for _, source in mandatory if "edge at" in source) == [
            "'a' edge at 0, inside",
            "'a' edge at 0, outside",
            "'a' edge at 3, inside",
            "'a' edge at 3, outside",
        ]

    def test_a_pair_that_yielded_owns_no_span(self):
        """A span exists to keep everything else out of the edge cell, and a
        pair that gave way laid no edge cell to keep anything out of. Held
        anyway, the interface below is dropped for crowding a pair that is not
        there."""
        regions = [
            _metal((0.0, -2.0, -2.0), (3.0, 2.0, 2.0), "strip"),
            Region(
                lower=(-0.08, -2.0, -2.0),
                upper=(3.0, 2.0, 2.0),
                material=MaterialClass.DIELECTRIC,
                label="slab",
            ),
        ]
        mandatory, preferred = mesh._fixed_positions(
            _grouped_into_conductors(regions), 0, -8.0, 8.0, CUT_PARAMS, forced=(-0.05,)
        )
        assert (0.0, "'strip' edge at 0, crowded") in mandatory, (
            "the strip's edge kept its pair, so no span is at stake"
        )
        assert (-0.08, "'slab' lower face") in preferred

    def test_a_face_the_metal_carries_on_past_is_an_anchor(self):
        """A continuous face gets no thirds pair and is still where the metal
        ends. Offered as a preference instead, it is dropped wherever it crowds
        anything, and openEMS then builds the strip between two lines that are
        not the ones it was drawn between."""
        feed = [_metal((0.0, -2.0, -2.0), (3.0, 2.0, 2.0), "feed")]
        feed[0] = replace(feed[0], continuous=frozenset({0}))
        mandatory, preferred = mesh._fixed_positions(
            _grouped_into_conductors(feed), 0, -8.0, 8.0, CUT_PARAMS
        )
        assert [source for _, source in mandatory if "feed" in source] == [
            "'feed' lower face, continuous",
            "'feed' upper face, continuous",
        ]
        assert preferred == []
