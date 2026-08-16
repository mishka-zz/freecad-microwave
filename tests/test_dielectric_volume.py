# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How much dielectric the grid holds, against how much was drawn.

A conductor fails loudly once anything looks: it comes apart, and the pieces can
be counted. A dielectric never does. An interface half a cell from where it was
drawn produces a grid that is well formed, graded, symmetric and entirely
plausible, and moves the phase velocity through that region by half a cell's
worth for the whole run. Nothing in the suite would notice, which is why this
exists.

It takes two measures of the permittivity over the domain, and the reason is
worth stating because one of them is the obvious choice and it is not enough on
its own.

**How much**, the integral of the permittivity above air's. It answers a slab
emitted larger or smaller than it was drawn, and a face that staircased away
more than it gained. It barely answers where anything is: a rigid displacement
moves no volume, and all the integral sees when a slab moves is its own
staircase re-rolling under it.

**Where**, the first moment of that same excess. It is the drawn shape's own
centroid, and a displacement moves it by the displacement while growing the slab
about its centre does not move it at all.

Neither is a refinement of the other, and
:class:`TestEachMeasureAnswersItsOwnFault` is what says so - each has to respond
more to the fault it is for than to the fault the other is for, which is a
comparison and needs no tolerance to make.

Both are held against arithmetic on the drawing - a volume and a centroid, each
from the divergence theorem over the same triangles the engine is given - so
nothing here is compared against a reference simulation or against the
rasterisation being judged.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from Microwave.Solvers.openems.containment import contains
from Microwave.Solvers.openems.mesh import (
    MaterialClass,
    MeshParams,
    Region,
    generate_mesh_lines,
)
from Microwave.Solvers.openems.model import Solid
from tests.triangulated import bar, centre_of, extent, volume

#: The domain everything here is drawn inside, in mm.
DOMAIN = ((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0))

#: Relative permittivity of the one dielectric. FR4's, so the contrast is the
#: one a real board has rather than an arbitrarily large number that would make
#: any error look small beside it.
EPSILON = 4.4

#: How far the integrated permittivity may sit from the drawing's, relative to
#: the drawing's. A declared budget: what a staircase costs depends on how much
#: boundary the shape has and how fine the grid is, and neither is fixed here.
#: ``test_the_budget_would_reject_a_slab_of_the_wrong_size`` is what says it
#: is tight enough to catch a slab emitted the wrong size.
BUDGET = 0.005

#: How far the permittivity's centre may sit from the drawing's, in cells. A
#: staircase moves it by the imbalance between what it gained and what it lost,
#: which is a small fraction of one cell. ``test_a_displacement_is_caught`` is
#: what says this is tight enough to reject a slab in the wrong place rather
#: than accept it.
OFF_CENTRE = 0.2

#: Cell sizes the sweeps run at, in mm. All of them resolve the slab's thinnest
#: dimension several times over, which is the regime a model is meshed in;
#: coarser than the feature the staircase is the whole answer rather than a
#: correction to it, and there is nothing left for either measure to be about.
RESOLUTIONS = [1.0, 0.5, 0.4, 0.25]


#: Where the slab sits. Off the origin on every axis and off any round number,
#: because a slab centred in a domain that is centred on the origin has a
#: staircase symmetric about its own centre - which puts its centroid exactly
#: where it was drawn whatever the grid, and leaves the measure of *where*
#: nothing to measure.
PLACED = (1.3, -0.7, 0.9)


def turned_slab(displaced: float = 0.0) -> Solid:
    """A dielectric bar across the grid, optionally emitted off where it was drawn.

    Turned, because an axis-aligned box has its faces pinned by the mesher and
    is then held exactly - which is worth asserting and is not the case this
    file is about. A bar at an angle to every axis has no face a line can land
    on, so the whole of its boundary is staircased.
    """
    centre = tuple(value + displaced for value in PLACED)
    vertices, faces = bar((12.0, 5.0, 4.0), math.pi / 6.0, centre)
    lower, upper = extent(vertices)
    return Solid(
        material="FR4",
        lower=lower,
        upper=upper,
        vertices=tuple(vertices),
        faces=tuple(faces),
        label="Slab",
    )


def resized(scale: float) -> Solid:
    """The same slab emitted larger than it was drawn, about its own centre.

    Scaled about the centre so that this is a change of *size* and nothing else:
    scaled about anything else it would move as well, and a measure responding
    to it would not say which of the two it had noticed.
    """
    slab = turned_slab()
    return replace(
        slab,
        vertices=tuple(
            tuple(PLACED[dim] + (value - PLACED[dim]) * scale for dim, value in enumerate(point))
            for point in slab.vertices
        ),
    )


def axis_aligned_slab() -> Solid:
    return Solid(material="FR4", lower=(-6.0, -4.0, -2.0), upper=(6.0, 4.0, 2.0), label="Board")


def grid_for(solid: Solid, size: float) -> list[np.ndarray]:
    """The grid the mesher actually produces for this drawing at this resolution."""
    region = Region(
        lower=tuple(solid.lower),
        upper=tuple(solid.upper),
        material=MaterialClass.DIELECTRIC,
        label=solid.label,
    )
    lines = generate_mesh_lines(
        [region],
        DOMAIN,
        MeshParams(metal_res=size, dielectric_res=size, max_ratio=(1.4,) * 3, pml_cells=0),
    )
    return [lines[dim] for dim in range(3)]


def rasterised(lines: list[np.ndarray], solid: Solid):
    """Every cell's volume, its centre, and whether the solid holds it.

    Each cell takes the material at its own centre, which is how a rectilinear
    grid decides: a cell is one material, and the centre is the point that says
    which.
    """
    centres = [0.5 * (axis[:-1] + axis[1:]) for axis in lines]
    widths = [np.diff(axis) for axis in lines]
    cells = np.einsum("i,j,k->ijk", *widths).ravel()
    points = np.stack([part.ravel() for part in np.meshgrid(*centres, indexing="ij")], axis=-1)
    return cells, points, contains(solid, points)


def held(lines: list[np.ndarray], solid: Solid) -> float:
    """The permittivity the grid holds above air's, integrated over the domain.

    The excess and not the permittivity itself, for the same reason the centre
    below is weighted by it: air is most of the domain and is drawn by nobody,
    so leaving it in makes every departure a small fraction of the padding
    somebody happened to choose. Taken this way the number is about the slab.
    """
    cells, _, inside = rasterised(lines, solid)
    return float(np.sum(cells * np.where(inside, EPSILON - 1.0, 0.0)))


def held_centre(lines: list[np.ndarray], solid: Solid) -> np.ndarray:
    """Where the grid puts the permittivity above air's.

    Weighting by the *excess* rather than by the permittivity itself is what
    keeps the answer from being the domain's own centre with the slab as a
    small correction to it: air contributes nothing, so this is the centroid of
    the material the grid believes is there.
    """
    cells, points, inside = rasterised(lines, solid)
    weight = cells * np.where(inside, EPSILON - 1.0, 0.0)
    return (weight[:, None] * points).sum(axis=0) / weight.sum()


def drawn(solid: Solid) -> float:
    """The same integral off the drawing: the slab's own volume times its contrast."""
    inside = (
        volume(solid.vertices, solid.faces)
        if solid.is_mesh
        else float(np.prod(np.asarray(solid.upper) - np.asarray(solid.lower)))
    )
    return inside * (EPSILON - 1.0)


def departure(size: float, solid: Solid, emitted: Solid | None = None) -> float:
    """How far the grid's answer sits from the drawing's, relative.

    ``emitted`` is what the grid is asked about where that differs from what was
    drawn, which is the only way to state a misplaced interface: the reference
    stays with the drawing while the rasterisation follows the shape the engine
    was actually handed.
    """
    lines = grid_for(solid, size)
    want = drawn(solid)
    return abs(held(lines, emitted or solid) - want) / want


def off_centre(size: float, solid: Solid, emitted: Solid | None = None) -> float:
    """How far the grid puts the slab's centre from where it was drawn, in cells."""
    lines = grid_for(solid, size)
    want = np.asarray(centre_of(solid.vertices, solid.faces))
    return float(np.max(np.abs(held_centre(lines, emitted or solid) - want)) / size)


class TestABoxIsHeldExactly:
    """A dielectric whose faces the mesher put lines on has no staircase at all,
    and anything less than agreement to rounding would be a defect elsewhere."""

    def test_the_grid_holds_what_was_drawn(self):
        solid = axis_aligned_slab()
        lines = grid_for(solid, 0.5)
        assert held(lines, solid) == pytest.approx(drawn(solid), rel=1e-12)

    def test_its_faces_really_are_on_lines(self):
        """Without this the case above says only that a coarse box happened to
        land well, and it would stop being the exact case without failing."""
        solid = axis_aligned_slab()
        lines = grid_for(solid, 0.5)
        for dim, axis in enumerate(lines):
            for face in (solid.lower[dim], solid.upper[dim]):
                assert np.min(np.abs(axis - face)) < 1e-9


class TestAStaircasedInterfaceStillHoldsTheDrawing:
    """What a rectilinear grid does to a boundary at an angle to every axis.

    Asserted across a range of resolutions rather than as a trend between two,
    and that is not a weaker claim but a truer one. What a staircase costs is
    the difference between what it gained and what it lost, so it is already
    small, and refining does not walk it down smoothly - it re-rolls it. A test
    demanding that the finer of two grids be nearer would be asserting the sign
    of that noise.
    """

    @pytest.mark.parametrize("size", RESOLUTIONS)
    def test_the_grid_holds_what_was_drawn(self, size):
        assert departure(size, turned_slab()) <= BUDGET

    @pytest.mark.parametrize("size", RESOLUTIONS)
    def test_the_grid_puts_it_where_it_was_drawn(self, size):
        assert off_centre(size, turned_slab()) <= OFF_CENTRE

    def test_the_budget_would_reject_a_slab_of_the_wrong_size(self):
        """The control the integral needs, and the fault it is the right measure
        for: a solid emitted a little larger than it was drawn.

        Without it the budget is an upper bound with nothing pushing up against
        it, and every case above would pass on a measure that always answered
        zero.
        """
        assert departure(0.4, turned_slab()) <= BUDGET
        assert departure(0.4, turned_slab(), emitted=resized(1.01)) > BUDGET


class TestEachMeasureAnswersItsOwnFault:
    """Why there are two, stated as the comparison rather than as a remark.

    A later reader looking at a volume and a centroid could reasonably take one
    for a refinement of the other and delete it. These say what each is for, and
    they say it without a tolerance: each measure has to respond more to the
    fault it exists for than to the one the other exists for.
    """

    def test_the_integral_answers_size_rather_than_place(self):
        """A rigid displacement moves no volume. What the integral does see when
        a slab moves is its staircase re-rolling under it, and that is smaller
        than what one per cent of the slab's own size costs - though a half cell
        is by far the larger error of the two."""
        size = 0.4
        displaced = departure(size, turned_slab(), emitted=turned_slab(displaced=size / 2.0))
        resize = departure(size, turned_slab(), emitted=resized(1.01))
        assert resize > displaced

    def test_the_centre_answers_place_rather_than_size(self):
        """A slab grown about its own centre has not moved, so the measure of
        where it is must barely notice - and the displacement must move it."""
        size = 0.4
        displaced = off_centre(size, turned_slab(), emitted=turned_slab(displaced=size / 2.0))
        resize = off_centre(size, turned_slab(), emitted=resized(1.01))
        assert displaced > resize


class TestWhereTheGridPutsIt:
    """The half the integral cannot answer, and why it needs answering apart."""

    @pytest.mark.parametrize("cells", [0.5, 1.0, 2.0])
    def test_a_displacement_is_caught(self, cells):
        """The control that makes the tolerance mean something.

        The drawing stays where it is and the shape the grid is asked about
        moves, which is exactly the defect: a solid emitted off where the user
        put it. A tolerance that accepted this would accept the fault it is
        there to catch.
        """
        size = 0.4
        assert off_centre(size, turned_slab(), emitted=turned_slab(displaced=size * cells)) > (
            OFF_CENTRE
        )

    @pytest.mark.parametrize("cells", [0.5, 1.0, 2.0])
    def test_it_reads_how_far_and_not_merely_that_it_moved(self, cells):
        """To within a cell, which is as well as a cell-centre rasterisation can
        answer: a cell either holds the material or it does not, so the measure
        moves in whole cells however smoothly the shape does. A user's next move
        depends on the distance, and half a cell and ten want different repairs.
        """
        size = 0.4
        moved = off_centre(size, turned_slab(), emitted=turned_slab(displaced=size * cells))
        assert moved == pytest.approx(cells, abs=1.0)
