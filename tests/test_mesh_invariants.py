# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Properties of a finished grid that hold without any reference to compare to.

Every other mesh test asks whether one grid is right. These ask whether the
mesher is the same mesher whichever way the model is turned, which is a
question a single grid cannot answer and which no closed form is needed for:
draw the structure again on other axes, mirrored, or somewhere else in space,
and the grid must come back turned the same way.

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
from dataclasses import dataclass, replace

import numpy as np
import pytest

import Microwave.Solvers.openems.mesh as mesh
from Microwave.Solvers.openems.mesh import (
    DIMENSIONS,
    MaterialClass,
    MeshLines,
    MeshParams,
    Region,
    SizingRegion,
    generate_mesh_lines,
)
from Microwave.Solvers.openems.sizing import Feature
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
        original = mesh._symmetrize
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
