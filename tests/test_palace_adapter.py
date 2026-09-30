# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The Palace adapter: what it declares, and the configuration it writes.

Every test here runs where no solver is installed. The configuration is a value,
so what it says can be read back without Palace and without a mesh.
"""

import ast
import json
import math
import os
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import Microwave
from Microwave import drawn, units
from Microwave.Gmsh import coverage
from Microwave.Gmsh.vocabulary import (
    BOTH,
    FRONTIER,
    INTERIOR,
    AtRim,
    Demand,
    Edges,
    Label,
    Mesh,
    Near,
    Part,
    Piece,
    Reached,
    Refused,
    Uncut,
    Unmeshed,
)
from Microwave.Materials.model import KINDS
from Microwave.Solvers import gmsh_mesh_report
from Microwave.Solvers.cancellation import Cancellation, Cancelled
from Microwave.Solvers.errors import TranslationError
from Microwave.Solvers.mesh_regions import check_relaxations_were_used, relaxations
from Microwave.Solvers.palace import balance, pipeline, reduced
from Microwave.Solvers.palace import config as config_module
from Microwave.Solvers.palace import policy as palace_policy
from Microwave.Solvers.palace import run as run_module
from Microwave.Solvers.palace.attributes import (
    check,
    configured,
    divided_by,
    meeting,
    voltage_across,
)
from Microwave.Solvers.palace.capabilities import (
    HZ_PER_GHZ,
    METRES_PER_UNIT,
    capabilities,
)
from Microwave.Solvers.palace.config import (
    CONVERGENCE_MEMORY,
    ESTIMATOR_ITERATIONS,
    FEWEST_SOLVES,
    RANK_LOSS_TANGENT,
    Adaptive,
    Driven,
    Filling,
    Material,
    SurfaceConductivity,
    Sweep,
    WavePort,
)
from Microwave.Solvers.palace.document import CONDUCTS, OVERFLOW, PASSES, THICK, WALL, problem
from Microwave.Solvers.palace.modes import (
    MODES,
    SolvedShort,
    judged,
    propagates,
    reflected,
)
from Microwave.Solvers.palace.policy import PROFILE, band, demand, one_metal, sweeping
from Microwave.Solvers.palace.policy import said as unlaid_said
from Microwave.Solvers.palace.problem import Coarsening
from Microwave.Solvers.palace.read import (
    FLUX_TABLE,
    IMPEDANCE_TABLE,
    MODES_TABLE,
    TABLE,
    ResultsError,
    Scattering,
    modes,
    scattering,
)
from Microwave.Solvers.palace.run import (
    BINARY_ENV_VAR,
    FLOOR,
    SolverFailed,
    SolverNotFound,
    SolverUnsupported,
    complaints,
    find_launcher,
    find_solver,
    remarks,
    solve,
    supported,
    validate,
    version,
)
from Microwave.Solvers.palace.write import (
    CHECKED_NAME,
    CONFIG_NAME,
    MESH_NAME,
    MODES_NAME,
    OUTPUT_NAME,
    configure,
    configure_modes,
    draw,
)
from Microwave.Solvers.properties import WANTED, kind, label
from tests import import_sweep
from tests.conftest import needed, only_declared
from tests.processes import gone_within, killed_after_first_line
from tests.test_acceptance_palace_septum import POWER_TOLERANCE as SEPTUM_POWER
from tests.test_acceptance_palace_waveguide import POWER_TOLERANCE as WAVEGUIDE_POWER

#: What a Palace this adapter runs says when asked its version, in the form
#: palace/main.cpp prints it.
STATED = "Palace version: v0.18.1"

#: A vacuum wavelength at the top of the band, so a size derived from it is the
#: whole wavelength divided by the count asked for.
VACUUM = 1.0

#: The middle of a WR-42 guide 20 mm long drawn from the origin, which stands on
#: the model's side of both of its end faces.
INSIDE = (10.0, 5.35, 2.15)

WR42 = dict(
    mesh="wr42.msh",
    output="postpro/wr42",
    materials=(Material(attributes=(1,)),),
    perfect_conductor=(2,),
    ports=(
        WavePort(index=1, attributes=(3,), behind=INSIDE, excited=True),
        WavePort(index=2, attributes=(4,), behind=INSIDE, excited=True),
    ),
    sweep=Sweep(20e9, 26e9, 13),
    order=3,
)


def driven(**changed):
    """The guide, with whatever this test wanted different about it."""
    return Driven(**{**WR42, **changed})


class TestWhatTheAdapterDeclares:
    def test_it_says_what_it_is_and_needs_no_solver_to_say_it(self):
        """The declaration decides which solvers could answer a model, so it has
        to be readable on a machine where none of them is installed.
        """
        declared = capabilities()
        assert declared.solver == "Palace"
        assert declared.supports_output("s_parameters")
        assert declared.supports_port("rect_waveguide")
        assert declared.supports_port("lumped")

    def test_it_claims_no_port_it_has_not_been_driven_through(self):
        """A name here that nothing has run turns a refusal into a failure."""
        declared = capabilities()
        assert not declared.supports_port("microstrip")
        assert not declared.supports_port("coaxial")

    def test_it_claims_an_open_domain_and_names_the_port_it_is_not_driven_through(self):
        """The workbench reserves the air round an open study and Palace absorbs
        on its skin. A wave port beside that surface is a capability not offered
        yet, and the declaration says so where a user reads why it is refused.
        """
        declared = capabilities()
        assert {"open", "enclosed"} <= declared.domains
        assert "wave port" in declared.notes["wave_port_beside_open"]

    def test_it_names_every_kind_a_catalog_holds(self):
        """A catalog is solver-neutral, so a kind it can hold that this adapter
        does not declare is a material the user picked and cannot run here."""
        for held in KINDS:
            assert capabilities().supports_material(held)

    def test_it_names_a_perfect_conductor_it_writes_as_a_condition(self):
        assert capabilities().supports_material("pec")


class TestTheUnitsTheConfigurationIsWrittenIn:
    def test_the_band_is_written_in_gigahertz(self):
        """Twenty gigahertz is twenty thousand million Hertz, and the number
        written is the one Palace reads. Stated rather than derived from the
        conversion the code uses, which would agree with itself whatever it held.
        """
        written = driven().to_dict()["Solver"]["Driven"]["Samples"][0]
        assert written["MinFreq"] == 20.0
        assert written["MaxFreq"] == 26.0

    def test_the_mesh_carries_what_one_of_its_units_is_in_metres(self):
        """The mesh is written in the units the shapes were drawn in, and Palace
        is told the scale rather than handed rescaled coordinates. This workbench
        draws in millimetres, so one unit is a thousandth of a metre.
        """
        assert driven().to_dict()["Model"]["L0"] == 0.001

    def test_the_two_conversions_say_what_they_are(self):
        """Both are read by the configuration and by nothing else, so a test
        comparing the file against them cannot see either move.
        """
        assert HZ_PER_GHZ == 1e9
        assert METRES_PER_UNIT == 0.001

    def test_the_band_is_asked_for_as_a_count_and_not_as_a_step(self):
        """A step derived from a count is a division that has to come out right
        at the far end of the band.
        """
        written = driven(sweep=Sweep(20e9, 26e9, 7)).to_dict()["Solver"]["Driven"]["Samples"][0]
        assert written["NSample"] == 7
        assert "FreqStep" not in written


class TestWhichPortsAreDriven:
    def test_every_driven_port_gets_an_excitation_of_its_own(self):
        """Palace forms a scattering matrix only where each excitation drives
        exactly one port, and writes no table at all otherwise.
        """
        assert driven().excitations == (1, 2)

    def test_an_excitation_carries_the_number_of_the_port_it_drives(self):
        """Past one excitation Palace requires the index to be the port's own
        number and stops before the first solve where it is not, so a count over
        the driven ports builds a configuration it refuses.
        """
        ports = (
            WavePort(index=3, attributes=(3,), behind=INSIDE, excited=True),
            WavePort(index=7, attributes=(4,), behind=INSIDE, excited=True),
        )
        written = driven(ports=ports).to_dict()["Boundaries"]["WavePort"]
        assert [(one["Index"], one["Excitation"]) for one in written] == [(3, 3), (7, 7)]

    def test_a_port_that_is_not_driven_carries_no_excitation(self):
        ports = (
            WavePort(index=1, attributes=(3,), behind=INSIDE, excited=True),
            WavePort(index=2, attributes=(4,), behind=INSIDE),
        )
        written = driven(ports=ports).to_dict()["Boundaries"]["WavePort"]
        assert "Excitation" in written[0]
        assert "Excitation" not in written[1]

    def test_a_run_with_nothing_driving_it_is_refused(self):
        ports = (
            WavePort(index=1, attributes=(3,), behind=INSIDE),
            WavePort(index=2, attributes=(4,), behind=INSIDE),
        )
        with pytest.raises(ValueError, match="no port is driven"):
            driven(ports=ports)

    def test_two_ports_carrying_one_number_are_refused(self):
        """The number names the port in every table the run writes, and the
        refusal names the number that is repeated rather than every number.
        """
        ports = (
            WavePort(index=1, attributes=(3,), behind=INSIDE, excited=True),
            WavePort(index=1, attributes=(4,), behind=INSIDE, excited=True),
            WavePort(index=2, attributes=(5,), behind=INSIDE, excited=True),
        )
        with pytest.raises(ValueError, match=r"carries the number \[1\]"):
            driven(ports=ports)


class TestWhatAPortStatesAboutItself:
    def test_the_reference_plane_is_written_as_a_distance_to_de_embed(self):
        ports = (
            WavePort(index=1, attributes=(3,), behind=INSIDE, offset=5.0, excited=True),
            WavePort(index=2, attributes=(4,), behind=INSIDE, excited=True),
        )
        written = driven(ports=ports).to_dict()["Boundaries"]["WavePort"]
        assert written[0]["Offset"] == 5.0
        assert written[1]["Offset"] == 0.0

    def test_a_reference_plane_inside_the_model_is_refused(self):
        with pytest.raises(ValueError, match="stands inside the model"):
            WavePort(index=1, attributes=(3,), behind=INSIDE, offset=-5.0)

    def test_a_port_standing_on_nothing_is_refused(self):
        with pytest.raises(ValueError, match="at least one attribute"):
            WavePort(index=1, attributes=(), behind=INSIDE)

    def test_a_mode_is_an_ordinal_and_is_ranked_from_one(self):
        with pytest.raises(ValueError, match="a mode is ranked"):
            WavePort(index=1, attributes=(3,), behind=INSIDE, mode=0)


class TestWhatFillsTheDomain:
    def test_a_material_carrying_both_models_of_loss_is_refused(self):
        """Palace stops on the pair, and a loss tangent and a conductivity are
        two ways of saying one thing.
        """
        with pytest.raises(ValueError, match="one or the other"):
            Filling(loss_tangent=0.02, conductivity=5.8e7)

    def test_either_model_of_loss_alone_is_written(self):
        lossy = Material(
            attributes=(1,), filling=Filling(permittivity=4.3, loss_tangent=0.02)
        ).to_dict()
        assert lossy["LossTan"] == 0.02
        assert "Conductivity" not in lossy
        conducting = Material(attributes=(1,), filling=Filling(conductivity=5.8e7)).to_dict()
        assert conducting["Conductivity"] == 5.8e7
        assert "LossTan" not in conducting

    def test_a_lossless_material_says_neither(self):
        """A zero written out is a value Palace would configure; leaving it out
        is the absence of the condition.
        """
        written = Material(attributes=(1,)).to_dict()
        assert "LossTan" not in written
        assert "Conductivity" not in written

    def test_a_material_filling_no_attribute_is_refused(self):
        """A material names the pieces of the mesh it fills, and one that names
        none would be written and configure nothing.
        """
        with pytest.raises(ValueError, match="at least one attribute"):
            Material(attributes=())

    def test_a_mesh_with_no_perfect_conductor_in_it_is_refused(self):
        """An exterior attribute nobody names becomes a perfect magnetic
        conductor with a warning, so a wall left out is not left out of the
        problem.
        """
        with pytest.raises(ValueError, match="perfect magnetic conductor"):
            driven(perfect_conductor=())

    def test_a_mesh_no_material_fills_is_refused(self):
        """A domain attribute no material names is removed from the mesh with
        one line of print, so an empty list is a problem that quietly shrinks.
        """
        with pytest.raises(ValueError, match="Palace deletes what none names"):
            driven(materials=())


class TestRanksThatCouldDisagree:
    """Where metal of finite conductivity meets a wave port and no material on
    the port's face is lossy, each of Palace's ranks decides alone whether the
    port's mode solve makes a collective assembly. A loss tangent on every
    material decides it for all."""

    LOSSY_METAL = (SurfaceConductivity(attributes=(5,), conductivity=1.57e7),)

    def test_every_lossless_material_carries_the_loss_tangent(self):
        run = driven(
            conducting=self.LOSSY_METAL,
            materials=(Material(attributes=(1,)), Material((6,), Filling(permittivity=2.2))),
        )
        assert run.ranks_can_disagree
        written = run.to_dict()["Domains"]["Materials"]
        assert [material["LossTan"] for material in written] == [RANK_LOSS_TANGENT] * 2

    def test_the_loss_tangent_is_a_loss_below_a_doubles_precision(self):
        """Palace counts a material lossy where its tangent is above zero."""
        assert 0.0 < RANK_LOSS_TANGENT < sys.float_info.epsilon

    @pytest.mark.parametrize(
        "filling", [Filling(loss_tangent=0.02), Filling(conductivity=5.8)], ids=["tangent", "sigma"]
    )
    def test_a_lossy_material_keeps_its_loss_beside_a_lossless_one(self, filling):
        """The port's mode solve holds only the materials on the port's face, so
        a lossy material elsewhere leaves the ranks free to disagree there."""
        run = driven(
            conducting=self.LOSSY_METAL,
            materials=(Material(attributes=(1,)), Material((6,), filling)),
        )
        air, lossy = run.to_dict()["Domains"]["Materials"]
        assert air["LossTan"] == RANK_LOSS_TANGENT
        assert lossy == run.materials[1].to_dict()

    def test_a_run_without_lossy_metal_is_written_as_it_is(self):
        run = driven()
        assert not run.ranks_can_disagree
        assert "LossTan" not in run.to_dict()["Domains"]["Materials"][0]


class TestTheFileThatIsWritten:
    def test_the_json_says_what_the_value_says(self):
        config = driven()
        assert json.loads(config.to_json()) == config.to_dict()

    def test_it_is_a_driven_run_and_says_where_its_tables_go(self):
        written = driven().to_dict()["Problem"]
        assert written["Type"] == "Driven"
        assert written["Output"] == "postpro/wr42"

    def test_the_walls_are_a_condition_rather_than_a_body(self):
        """No metal is drawn: what carries a perfect conductor is a face."""
        assert driven().to_dict()["Boundaries"]["PEC"]["Attributes"] == [2]

    def test_the_solver_order_is_written_and_is_palaces_own(self):
        """Away from what the workbench ships, so a written-in figure fails."""
        assert driven(order=4).to_dict()["Solver"]["Order"] == 4

    def test_an_order_of_nothing_is_refused(self):
        with pytest.raises(ValueError, match="an order is positive"):
            driven(order=0)

    def test_it_insists_on_being_told_the_order(self):
        """No default. What this workbench ships is on the solver object the
        user edits, and a second copy here is the one every caller that said
        nothing would silently get."""
        with pytest.raises(TypeError):
            Driven(**{name: value for name, value in WR42.items() if name != "order"})


class TestABandOfOneFrequency:
    def test_it_is_asked_for_as_a_frequency_and_not_as_a_block(self):
        """Palace spaces a linear block by dividing the width by one less than
        the count, so a block of one sample divides by zero and the run solves
        at a frequency that is not a number and writes a table of them.
        """
        written = driven(sweep=Sweep(20e9, 26e9, 1)).to_dict()["Solver"]["Driven"]["Samples"][0]
        assert written == {"Type": "Point", "Freq": [20.0]}

    def test_a_band_of_no_width_is_one_frequency_too(self):
        """Palace drops repeated samples after expanding a block, so a start and
        a stop that are equal come back as one row however many were asked for.
        """
        written = driven(sweep=Sweep(20e9, 20e9, 13)).to_dict()["Solver"]["Driven"]["Samples"][0]
        assert written == {"Type": "Point", "Freq": [20.0]}

    def test_what_the_run_writes_is_what_the_reader_holds_it_to(self):
        assert Sweep(20e9, 26e9, 1).samples == 1
        assert Sweep(20e9, 20e9, 13).samples == 1
        assert Sweep(20e9, 26e9, 13).samples == 13


class TestTheBandItself:
    def test_a_band_that_runs_backwards_is_refused(self):
        with pytest.raises(ValueError, match="ends at"):
            Sweep(26e9, 20e9, 13)

    def test_a_band_reaching_zero_is_refused(self):
        """A frequency-domain method answers one point at a time, and it has
        nothing to answer at nothing.
        """
        with pytest.raises(ValueError, match="nothing to answer at zero"):
            Sweep(0.0, 26e9, 13)

    def test_a_band_asked_for_at_no_points_is_refused(self):
        with pytest.raises(ValueError, match="asked for at 0 points"):
            Sweep(20e9, 26e9, 0)


def obj(kind, name, **properties):
    """A document object as the adapter meets one: an attribute bag with a proxy."""
    return SimpleNamespace(Proxy=type(kind, (), {})(), Label=name, Name=name, **properties)


def palace_solver(name="Palace", **overrides):
    """A Palace solver as the adapter meets one, at its declared sweep."""
    properties = dict(Order=2, Sweep="Adaptive", SweepTolerance=1e-3, SweepSolves=20)
    properties.update(overrides)
    return obj("EMSolverPalace", name, **properties)


#: How far across the end face of a WR-42 guide reaches, corner to corner.
WR42_FACE = math.hypot(10.7, 4.3)


class Drawn:
    """A shape as the stage owning a directory meets one: it writes itself out.

    It also says where its mass centres, how far across it reaches and how far
    it spans along each axis, which is what a port's face is read for. Each is that of the end
    face of a WR-42 guide drawn from the origin along x, whatever the shape
    stands for.
    """

    #: How many shapes have been made, so that each stands in a box of its own.
    made = 0

    def __init__(self, name, centre=(0.0, 5.35, 2.15), reach=WR42_FACE, spans=(0.0, 10.7, 4.3)):
        self.name = name
        self.CenterOfMass = SimpleNamespace(x=centre[0], y=centre[1], z=centre[2])
        # Two regions a test binds are two bodies apart, as drawings that do
        # not nest are, unless the test places one inside the other.
        Drawn.made += 1
        apart = 1000.0 * Drawn.made
        self.BoundBox = SimpleNamespace(
            DiagonalLength=reach,
            XLength=spans[0],
            YLength=spans[1],
            ZLength=spans[2],
            XMin=apart,
            XMax=apart + 1.0,
            YMin=0.0,
            YMax=1.0,
            ZMin=0.0,
            ZMax=1.0,
        )

    def exportBrep(self, path):
        Path(path).write_text(self.name, encoding="utf-8")

    def fuse(self, others):
        """The faces glued: an edge two of them hold as one object stays one."""
        return SimpleNamespace(Faces=[face for one in (self, *others) for face in one.Faces])


def square_to(axis):
    """How far a face square to ``axis``, as a port states one, spans along each axis."""
    spans = [10.7, 10.7, 4.3]
    spans["XYZ".index(axis.lstrip("-"))] = 0.0
    return tuple(spans)


class Shape(Drawn):
    """A solid as the adapter reads one: a listing of faces, addressed by position."""

    def __init__(self, faces=6, name="pipe"):
        super().__init__(name)
        self.Faces = [Drawn(f"{name}.face{index}") for index in range(1, faces + 1)]
        self.Solids = [self]

    def getElement(self, name):
        return self.Faces[int(name[len("Face") :]) - 1]


class Sheet(Drawn):
    """A sheet as the adapter reads one: faces and no solid."""

    def __init__(self, name="septum", faces=1):
        super().__init__(name)
        self.Faces = [Drawn(f"{name}.face{index}") for index in range(1, faces + 1)]
        self.Solids = []


def mesh_settings(**overrides):
    """A mesh policy as the document carries one, reached by duck typing."""
    properties = dict(
        MinElementSize=0.0,
        CurveTolerance=0.0,
        MinElementsAcross=9,
        Clearance=0.0,
        Medium=None,
        # The domain ends on the drawing. A face saying Air reserves free space
        # beyond it, which the open studies of test_palace_open.py are about.
        **{f"Padding{axis}{side}": "Ends" for axis in "XYZ" for side in ("Min", "Max")},
    )
    properties.update(overrides)
    return obj("EMMeshPolicy", "MeshSettings", **only_declared("mesh", "EMMeshPolicy", properties))


def gmsh_mesh(**overrides):
    """The tetrahedral mesh's own settings, reached the same way."""
    properties = dict(
        ElementsPerWavelength=20.0,
        EdgeRefinement=8.0,
        ElementsPerTurn=6,
        MaxGrowthRatio=1.6,
    )
    properties.update(overrides)
    return obj("EMGmshMesh", "GmshMesh", **only_declared("mesh", "EMGmshMesh", properties))


def study(**overrides):
    properties = dict(
        FrequencyStart=20e9, FrequencyStop=26e9, NumFrequencyPoints=13, SmallestResponse=0.0
    )
    properties.update(overrides)
    return obj("EMAnalysis", "Study", **properties)


class TestTheBandTheStudyStates:
    def test_it_is_read_as_the_band_and_the_points_across_it(self):
        asked = band(study())
        assert (asked.start, asked.stop, asked.points) == (20e9, 26e9, 13)

    def test_a_band_that_describes_no_sweep_is_refused_by_the_study_it_came_from(self):
        with pytest.raises(TranslationError, match="'Study'"):
            band(study(FrequencyStop=10e9))

    def test_a_property_carrying_a_unit_is_read_through_it(self):
        """A FreeCAD quantity is not a float, and every property here is reached
        by duck typing.
        """
        quantity = SimpleNamespace(Value=26e9)
        assert band(study(FrequencyStop=quantity)).stop == 26e9


def members_of(found, wanted):
    (member,) = [member for member in found.Group if kind(member) == wanted]
    return member


class TestHowTheBandIsSwept:
    """Adaptive where the points outnumber the full solves the sweep may take,
    and every point solved in full otherwise."""

    def test_many_points_are_swept_adaptively_to_the_solvers_tolerance(self):
        found, _ = guide(NumFrequencyPoints=21)
        members_of(found, "EMSolverPalace").SweepTolerance = 1e-4
        members_of(found, "EMSolverPalace").SweepSolves = 12
        assert problem(found).sweep.adaptive == Adaptive(tolerance=1e-4, solves=12)

    def test_no_more_points_than_solves_are_each_solved_in_full(self):
        found, _ = guide(NumFrequencyPoints=20)
        assert problem(found).sweep.adaptive is None

    def test_a_discrete_sweep_solves_every_point(self):
        found, _ = guide(NumFrequencyPoints=501)
        members_of(found, "EMSolverPalace").Sweep = "Discrete"
        assert problem(found).sweep.adaptive is None

    @pytest.mark.parametrize(
        ("name", "value", "said"),
        [
            ("SweepTolerance", 0.0, "relative error is above zero"),
            ("SweepTolerance", 1.0, "relative error is above zero"),
            ("SweepSolves", FEWEST_SOLVES - 1, f"no sooner than {FEWEST_SOLVES}"),
        ],
    )
    def test_a_setting_no_sweep_can_meet_is_refused_by_the_solver(self, name, value, said):
        found, _ = guide(NumFrequencyPoints=501)
        setattr(members_of(found, "EMSolverPalace"), name, value)
        with pytest.raises(TranslationError, match=f"'Palace': .*{said}"):
            problem(found)

    def test_a_discrete_sweep_is_not_refused_for_a_tolerance_it_does_not_use(self):
        found, _ = guide(NumFrequencyPoints=501)
        members_of(found, "EMSolverPalace").Sweep = "Discrete"
        members_of(found, "EMSolverPalace").SweepTolerance = 0.0
        assert problem(found).sweep.adaptive is None

    def test_the_fewest_solves_is_the_least_a_sweep_can_converge_in(self):
        Adaptive(tolerance=1e-3, solves=FEWEST_SOLVES)

    def test_an_adaptive_sweep_of_no_more_points_than_solves_is_refused(self):
        with pytest.raises(ValueError, match="as many as solving each point may take"):
            Sweep(20e9, 26e9, 20, adaptive=Adaptive(1e-3, 20))

    def test_a_band_of_one_frequency_is_never_adaptive(self):
        """Palace writes one row for it however many points were asked for."""
        with pytest.raises(ValueError, match="asked for at 1 points"):
            Sweep(20e9, 20e9, 501, adaptive=Adaptive(1e-3, 20))

    def test_an_adaptive_sweep_is_written_with_its_tolerance_its_cap_and_its_memory(self):
        adaptive = Adaptive(tolerance=1e-4, solves=12)
        written = driven(sweep=Sweep(20e9, 26e9, 101, adaptive=adaptive)).to_dict()
        assert written["Solver"]["Driven"] == {
            "Samples": [Sweep(20e9, 26e9, 101).to_dict()],
            "AdaptiveTol": 1e-4,
            "AdaptiveMaxSamples": 12,
            "AdaptiveConvergenceMemory": CONVERGENCE_MEMORY,
        }

    def test_a_discrete_sweep_writes_the_samples_alone(self):
        written = driven(sweep=Sweep(20e9, 26e9, 101)).to_dict()
        assert written["Solver"]["Driven"] == {"Samples": [Sweep(20e9, 26e9, 101).to_dict()]}

    def test_what_is_said_names_the_points_the_tolerance_and_the_cap(self):
        said = sweeping(Sweep(20e9, 26e9, 101, adaptive=Adaptive(tolerance=1e-4, solves=12)))
        assert "101 points" in said
        assert "field error of 0.0001" in said
        assert "12 full solves" in said

    def test_what_is_said_of_a_discrete_sweep_names_the_points(self):
        assert "101 points" in sweeping(Sweep(20e9, 26e9, 101))

    def test_what_is_said_of_one_frequency_counts_no_points(self):
        assert sweeping(Sweep(20e9, 20e9, 501)) == "The one frequency is solved in full"

    def test_the_choices_are_the_objects_own(self):
        """Stated twice, because the object imports FreeCAD and the adapter does not."""
        from Microwave.Objects.solver import SWEEPS

        assert SWEEPS == palace_policy.SWEEPS
        assert palace_policy.DISCRETE in SWEEPS


class TestTheElementSizeTheMesherIsAskedFor:
    def test_it_is_a_length_in_millimetres(self):
        """A vacuum wavelength at thirty gigahertz is about ten millimetres, so
        twenty elements across it is about half of one. What this pins is the
        unit: the same arithmetic in metres is out by a thousand.
        """
        asked = demand(mesh_settings(), gmsh_mesh(), stop=30e9, slowest=VACUUM)
        assert 0.4 < asked.coarsest < 0.6

    def test_a_slower_material_asks_for_a_smaller_element(self):
        """A wave slows by the square root of the product of permittivity and
        permeability, so meshing to the vacuum wavelength under-resolves a
        material by exactly that factor.
        """
        vacuum = demand(mesh_settings(), gmsh_mesh(), stop=30e9, slowest=VACUUM)
        slower = demand(mesh_settings(), gmsh_mesh(), stop=30e9, slowest=4.0)
        assert slower.coarsest == pytest.approx(vacuum.coarsest / math.sqrt(4.0))

    def test_asking_for_twice_as_many_across_a_wavelength_halves_it(self):
        coarse = demand(mesh_settings(), gmsh_mesh(), stop=30e9, slowest=VACUUM)
        fine = demand(
            mesh_settings(), gmsh_mesh(ElementsPerWavelength=40.0), stop=30e9, slowest=VACUUM
        )
        assert fine.coarsest == pytest.approx(coarse.coarsest / 2.0)

    def test_the_top_of_the_band_is_where_the_size_is_taken(self):
        high = demand(mesh_settings(), gmsh_mesh(), stop=60e9, slowest=VACUUM)
        low = demand(mesh_settings(), gmsh_mesh(), stop=30e9, slowest=VACUUM)
        assert high.coarsest == pytest.approx(low.coarsest / 2.0)

    def test_no_floor_stated_derives_one_from_the_slowest_region(self):
        """The size from curvature grows without bound at a cone's apex, and a
        floor ends it, so a policy stating none still has one."""
        for slowest in (VACUUM, 4.0):
            asked = demand(mesh_settings(), gmsh_mesh(), stop=30e9, slowest=slowest)
            assert asked.finest == pytest.approx(asked.coarsest / palace_policy.FLOOR, rel=1e-12)

    def test_the_derived_floor_follows_the_slowest_region_rather_than_the_medium(self):
        asked = demand(
            mesh_settings(),
            gmsh_mesh(),
            stop=30e9,
            slowest=4.0,
            within=(("Substrate", 4.0),),
            reserving=True,
        )
        bulk = asked.coarsest / math.sqrt(4.0)
        assert asked.finest == pytest.approx(bulk / palace_policy.FLOOR, rel=1e-12)

    def test_the_count_round_a_turn_is_carried_through_as_it_was_stated(self):
        for per_turn in (0, 1, 6, 12):
            asked = demand(
                mesh_settings(), gmsh_mesh(ElementsPerTurn=per_turn), stop=30e9, slowest=VACUUM
            )
            assert asked.per_turn == per_turn

    def test_the_count_round_a_turn_reaches_a_demand_that_lays_places(self):
        asked = demand(
            mesh_settings(),
            gmsh_mesh(ElementsPerTurn=9),
            stop=30e9,
            slowest=4.0,
            within=(("Substrate", 4.0),),
            reserving=True,
        )
        assert asked.places and asked.per_turn == 9

    @pytest.mark.parametrize("per_turn", [-1, -6])
    def test_a_count_round_a_turn_below_nothing_is_refused(self, per_turn):
        with pytest.raises(TranslationError, match="ElementsPerTurn"):
            demand(mesh_settings(), gmsh_mesh(ElementsPerTurn=per_turn), stop=30e9, slowest=VACUUM)

    def test_a_floor_is_carried_through_as_it_was_stated(self):
        asked = demand(mesh_settings(MinElementSize=0.01), gmsh_mesh(), stop=30e9, slowest=VACUUM)
        assert asked.finest == 0.01

    def test_a_count_of_nothing_across_a_wavelength_is_refused(self):
        with pytest.raises(TranslationError, match="across a wavelength"):
            demand(mesh_settings(), gmsh_mesh(ElementsPerWavelength=0.0), stop=30e9, slowest=VACUUM)

    def test_a_floor_below_nothing_is_refused(self):
        with pytest.raises(TranslationError, match="a length or nothing at all"):
            demand(mesh_settings(MinElementSize=-1.0), gmsh_mesh(), stop=30e9, slowest=VACUUM)

    def test_a_floor_above_the_ceiling_is_refused_rather_than_taken(self):
        """Gmsh takes it silently and meshes to the ceiling, so every element
        would be the coarsest one and nothing would say so.
        """
        with pytest.raises(TranslationError, match="above the ceiling"):
            demand(mesh_settings(MinElementSize=5.0), gmsh_mesh(), stop=30e9, slowest=VACUUM)


class TestWhatThisBackendCannotBeAskedFor:
    def test_air_outside_the_drawing_is_not_refused_by_the_demand(self):
        """The adapter reserves the air, so the demand has nothing to refuse
        about it. Asked for no size per region, it asks for the size a closed
        study asks for."""
        settings = mesh_settings(PaddingXMin="Air", PaddingZMax="Air")
        asked = demand(settings, gmsh_mesh(), stop=30e9, slowest=VACUUM)
        assert asked == demand(mesh_settings(), gmsh_mesh(), stop=30e9, slowest=VACUUM)

    def test_whether_a_study_is_open_is_read_off_the_policy_alone(self):
        """No Yee grid is in the study, and none is asked: what lies beyond the
        structure is a fact about the problem. A study losing the other
        pipeline's object does not change which problem this one solves, and
        here the guide's wave ports are refused as a study open to free space.
        """
        opened = with_members(settings=mesh_settings(PaddingYMax="Air"))
        with pytest.raises(TranslationError, match="PaddingYMax"):
            problem(opened)

    def test_a_face_that_ends_on_the_drawing_asks_for_nothing(self):
        """Which is what this backend builds anyway, so nothing is missing."""
        assert demand(mesh_settings(), gmsh_mesh(), stop=30e9, slowest=VACUUM).coarsest > 0

    def test_a_face_the_structure_runs_out_through_is_refused_where_no_port_bounds_it(self):
        """That value says a wave leaves and is not reflected. Where the model
        ends the boundary carries a perfect wall, so the wave is reflected whole
        - which is the one thing the value exists to prevent.
        """
        settings = mesh_settings(PaddingZMax="Through")
        with pytest.raises(TranslationError, match="PaddingZMax") as raised:
            demand(settings, gmsh_mesh(), stop=30e9, slowest=VACUUM)
        assert "runs out through the absorber" in str(raised.value)
        assert "perfect wall" in str(raised.value)

    def test_and_honoured_where_a_ports_plane_bounds_that_face(self):
        """The mesher leaves out what stands behind a port's plane, so the model
        ends on the port and what a wave meets there is the port's own
        condition. ``bounded`` is read off the study's own ports.
        """
        settings = mesh_settings(PaddingZMax="Through")
        asked = demand(settings, gmsh_mesh(), stop=30e9, slowest=VACUUM, bounded=("ZMax", "XMin"))
        assert asked.coarsest > 0

    def test_which_face_a_port_bounds_is_read_off_the_way_it_faces(self):
        """End to end, because ``bounded`` is a list the translation derives and
        a test passing one in would exercise the parameter rather than the
        derivation. The guide's ports both face along x, so the lower x face is
        bounded and the upper one is a wall.
        """
        through = with_members(settings=mesh_settings(PaddingXMin="Through"))
        assert problem(through).demand.coarsest > 0

        walled = with_members(settings=mesh_settings(PaddingXMax="Through"))
        with pytest.raises(TranslationError, match="PaddingXMax"):
            problem(walled)

    def test_holding_a_curved_surface_to_a_distance_is_refused(self):
        """The surface is meshed rather than sent as facets, so a distance from
        the drawing reaches nothing.
        """
        with pytest.raises(TranslationError, match="CurveTolerance"):
            demand(mesh_settings(CurveTolerance=0.001), gmsh_mesh(), stop=30e9, slowest=VACUUM)


class TestTheProfileTheMesherIsGiven:
    def test_it_fills_a_volume(self):
        """The field lives in three dimensions, and a surface method's profile
        would ask for the skin of the same body.
        """
        assert PROFILE.top == 3

    def test_the_elements_carry_the_shape_of_a_curved_boundary(self):
        """A straight-sided element meets a curve along the chord between its
        corners, and no number of them removes that.
        """
        assert PROFILE.element_order > 1
        assert PROFILE.curved

    def test_it_is_written_in_the_revision_the_backend_reads(self):
        assert PROFILE.written == "msh22"

    def test_what_it_fills_is_one_region(self):
        """A driven run's field enters through its ports, and a part no face
        joins to the rest is closed off by the wall where it ends. Palace
        solves that part as a cavity of its own and says nothing, so a guide
        drawn as two bodies a hair apart reflects everything at both ports.
        """
        assert PROFILE.connected


def dielectric(**overrides):
    properties = dict(
        MaterialType="Dielectric",
        Permittivity=1.0,
        Permeability=1.0,
        LossTangent=0.0,
        Conductivity=0.0,
        Thickness=0.035,
        MeasuredAt=0.0,
        SourceDigest="",
    )
    properties.update(overrides)
    return obj("EMMaterial", "Air", **properties)


def waveguide_port(number, body, face, **overrides):
    properties = dict(
        Number=number,
        Excitation=True,
        ReferencedTo="Port impedance",
        ReferenceImpedance=50.0,
        Mode="TE10",
        ReferenceDepth=0.0,
        PropagationAxis="X",
        CrossSection=(body, [face]),
    )
    properties.update(overrides)
    return obj("EMPortRectWaveguide", f"Port{number}", **properties)


def septum(name="Septum"):
    """A sheet drawn inside the guide, as a plane is."""
    return obj("Part::Plane", name, Shape=Sheet(name=name.lower()))


def metal_binding(sheet, material=None, sub=()):
    """A binding of a perfect conductor to one drawn shape."""
    return obj(
        "EMMaterialBinding",
        f"{sheet.Name}Metal",
        Material=material if material is not None else dielectric(MaterialType="PEC"),
        References=[(sheet, list(sub))],
    )


def guide(**overrides):
    """A study over one hollow body, with a port on a face at each end."""
    body = obj("Part::Feature", "Pipe", Shape=Shape())
    material = overrides.pop("material", dielectric())
    binding = obj("EMMaterialBinding", "AirFill", Material=material, References=[(body, [])])
    members = overrides.pop(
        "members",
        [
            palace_solver(),
            mesh_settings(),
            gmsh_mesh(),
            binding,
            waveguide_port(1, body, "Face1"),
            waveguide_port(2, body, "Face2"),
        ],
    )
    return study(Group=members, **overrides), body


def sheet_metal(**overrides):
    """A conducting sheet as the bundled catalog states its brass."""
    properties = dict(
        MaterialType="ConductingSheet",
        Permittivity=1.0,
        Permeability=1.0,
        LossTangent=0.0,
        Conductivity=1.57e7,
        Thickness=0.5,
        MeasuredAt=0.0,
        SourceDigest="",
    )
    properties.update(overrides)
    return obj("EMMaterial", "Brass", **properties)


def lossy_guide(material=None, **overrides):
    """The guide, with a sheet of finite conductivity drawn inside it."""
    found, _ = guide(**overrides)
    found.Group = [*found.Group, metal_binding(septum(), material=material or sheet_metal())]
    return found


def frequency_at(depths, metal):
    """Where ``metal`` is ``depths`` skin depths thick, in Hz."""
    metres = metal.Thickness / units.MM_PER_M
    return (depths / metres) ** 2 / (
        math.pi * units.VACUUM_PERMEABILITY * metal.Permeability * metal.Conductivity
    )


def lossy_band(start, stop, metal=None):
    """A guide with a lossy sheet in it, over a band of its own."""
    metal = metal or sheet_metal()
    return problem(lossy_guide(material=metal, FrequencyStart=start, FrequencyStop=stop)).lossy[0]


#: The wave impedance of free space, in ohms.
FREE_SPACE = units.VACUUM_PERMEABILITY * units.SPEED_OF_LIGHT


def film(passes, conductivity=1.0e5, beside=FREE_SPACE):
    """A conducting sheet as thick as lets ``passes`` of the field through, as a
    film does, beside a medium of impedance ``beside``."""
    conductance = 2.0 * (1.0 / passes - 1.0) / beside
    return sheet_metal(
        Conductivity=conductivity, Thickness=conductance * units.MM_PER_M / conductivity
    )


class TestAResistiveFilm:
    """Palace takes each face of a sheet inside the region as a wall of its own,
    so a sheet passes nothing; a film of sheet resistance near the wave
    impedance passes a large part of the field, and is refused."""

    def test_a_film_is_refused_naming_the_binding_the_material_and_its_resistance(self):
        """A kilohm a square across a guide, about twice TE10's impedance there."""
        metal = sheet_metal(Conductivity=1.0e5, Thickness=1.0e-5)
        with pytest.raises(TranslationError) as refused:
            problem(lossy_guide(material=metal))
        said = str(refused.value)
        assert "'SeptumMetal' binds 'Brass'" in said
        assert "1000 ohms a square" in said

    def test_the_bar_is_the_share_of_the_field_let_through(self):
        """Either side of it, in free space."""
        problem(lossy_guide(material=film(PASSES * (1.0 - 1e-6))))
        with pytest.raises(TranslationError, match="lets through up to"):
            problem(lossy_guide(material=film(PASSES * (1.0 + 1e-6))))

    def test_a_denser_medium_beside_the_sheet_lets_more_through_it(self):
        """A permittivity of four halves the wave impedance, so a sheet just
        under the bar in free space is over it there."""
        metal = film(PASSES * (1.0 - 1e-6))
        found = lossy_guide(material=metal)
        for member in found.Group:
            if kind(member) == "EMMaterialBinding" and member.Name == "AirFill":
                member.Material = dielectric(Permittivity=4.0)
        with pytest.raises(TranslationError, match="of the model of 188.4 ohms"):
            problem(found)

    @pytest.mark.parametrize("first", [True, False])
    def test_the_densest_medium_in_the_model_is_the_one_a_sheet_is_judged_beside(self, first):
        """Air round a block of permittivity four: the sheet may stand against
        either, and the block lets more through it, whichever is bound first."""
        found = lossy_guide(material=film(PASSES * (1.0 - 1e-6)))
        block = obj(
            "EMMaterialBinding",
            "BlockFill",
            Material=dielectric(Permittivity=4.0),
            References=[(obj("Part::Feature", "Block", Shape=Shape()), [])],
        )
        found.Group = [block, *found.Group] if first else [*found.Group, block]
        with pytest.raises(TranslationError, match="of the model of 188.4 ohms"):
            problem(found)

    def test_a_conducting_medium_is_judged_at_the_bottom_of_the_band(self):
        """Its conductivity lowers the impedance most where the frequency is
        lowest, so the sheet is judged where it lets most through."""
        found = lossy_guide(material=film(PASSES * (1.0 - 1e-6)))
        for member in found.Group:
            if kind(member) == "EMMaterialBinding" and member.Name == "AirFill":
                member.Material = dielectric(Conductivity=0.5)
        start = 2.0 * math.pi * 20e9 * units.VACUUM_PERMITTIVITY
        lowest = FREE_SPACE / abs(complex(1.0, 0.5 / start)) ** 0.5
        with pytest.raises(TranslationError, match=f"of the model of {lowest:.4g} ohms"):
            problem(found)

    def test_a_loss_tangent_makes_the_permittivity_complex_in_proportion_to_it(self):
        """The imaginary part is the permittivity times the loss tangent, and the
        impedance is taken over the magnitude."""
        found = lossy_guide(material=film(0.5))
        for member in found.Group:
            if kind(member) == "EMMaterialBinding" and member.Name == "AirFill":
                member.Material = dielectric(Permittivity=4.0, LossTangent=0.5)
        lowest = FREE_SPACE / abs(complex(4.0, 4.0 * 0.5)) ** 0.5
        with pytest.raises(TranslationError, match=f"of the model of {lowest:.4g} ohms"):
            problem(found)

    def test_a_permeability_raises_the_impedance_and_lets_less_through(self):
        """A permeability of four doubles the impedance, so a sheet just over the
        bar in free space is under it there."""
        found = lossy_guide(material=film(PASSES * (1.0 + 1e-6)))
        for member in found.Group:
            if kind(member) == "EMMaterialBinding" and member.Name == "AirFill":
                member.Material = dielectric(Permeability=4.0)
        assert problem(found).lossy

    def test_a_model_with_no_sheet_is_not_asked_for_an_impedance(self):
        """The regions are read for their impedance only to judge a sheet by."""
        found, _ = guide()
        found.Group = [
            *found.Group,
            obj(
                "EMMaterialBinding",
                "BlockFill",
                Material=dielectric(Permittivity=0.0),
                References=[(obj("Part::Feature", "Block", Shape=Shape()), [])],
            ),
        ]
        try:
            problem(found)
        except TranslationError as refused:
            assert "fills its region" not in str(refused)

    def test_a_region_that_carries_no_wave_is_refused_before_a_sheet_is_judged(self):
        found = lossy_guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding" and member.Name == "AirFill":
                member.Material = dielectric(Permittivity=0.0)
        with pytest.raises(TranslationError, match="'Air': a relative permittivity of 0"):
            problem(found)

    def test_the_phase_of_a_lossy_mediums_impedance_lets_more_through(self):
        """A sheet meets a wave whose impedance is turned from real, and the sum
        in the share is then shorter than its magnitudes added: a sheet a
        millionth under the bar by the magnitude alone is over it."""
        medium = FREE_SPACE / abs(complex(4.0, -4.0)) ** 0.5
        found = lossy_guide(material=film(PASSES * (1.0 - 1e-6), beside=medium))
        for member in found.Group:
            if kind(member) == "EMMaterialBinding" and member.Name == "AirFill":
                member.Material = dielectric(Permittivity=4.0, LossTangent=1.0)
        with pytest.raises(TranslationError, match="lets through up to"):
            problem(found)

    def test_a_sheet_is_asked_to_conduct_far_past_what_it_displaces(self):
        """Either side of the bar at the top of the band, on a sheet thick
        enough to let nothing through by its sheet resistance."""
        displaced = 2.0 * math.pi * 26e9 * units.VACUUM_PERMITTIVITY
        edge = CONDUCTS * displaced
        problem(lossy_guide(material=sheet_metal(Conductivity=edge * (1 + 1e-9), Thickness=1e3)))
        with pytest.raises(TranslationError, match="times the displacement current"):
            problem(
                lossy_guide(material=sheet_metal(Conductivity=edge * (1 - 1e-9), Thickness=1e3))
            )

    def test_a_thick_lossy_dielectric_resonating_across_its_thickness_is_refused(self):
        """A sheet carrying less conduction than displacement current, thick
        and of low permeability, lets through twice its share at the
        thickness where it resonates."""
        metal = sheet_metal(
            Conductivity=0.05018702368187339, Permeability=5e-9, Thickness=105992.64
        )
        with pytest.raises(TranslationError, match="times the displacement current"):
            problem(lossy_guide(material=metal))

    def test_a_band_starting_below_a_doubles_range_is_answered_rather_than_divided_by(self):
        """The frequency times the permittivity of free space is zero there, and
        a conductivity over it is taken in turn rather than over the product."""
        found = lossy_guide(FrequencyStart=1e-320)
        for member in found.Group:
            if kind(member) == "EMMaterialBinding" and member.Name == "AirFill":
                member.Material = dielectric(Conductivity=1e-3)
        with pytest.raises(TranslationError):
            problem(found)

    @pytest.mark.parametrize("thickness", [5e-324, 1e-300])
    def test_a_thickness_too_small_to_be_metal_is_refused(self, thickness):
        """Below a double's range inside Palace the first is read as none, and
        the thick metal's impedance is applied; the second makes the wall a
        magnetic one. Both are films of a conductance far under any metal's."""
        with pytest.raises(TranslationError, match="lets through up to"):
            problem(lossy_guide(material=sheet_metal(Thickness=thickness)))

    def test_a_conductance_past_a_doubles_range_lets_nothing_through(self):
        """The share is two over something infinite, which is nothing, rather
        than a quotient that is not a number."""
        metal = sheet_metal(Conductivity=1e308, Thickness=1000.0)
        (sheet,) = problem(lossy_guide(material=metal)).lossy
        assert sheet.conductivity == 1e308


def sealed_guide():
    """The guide with a sheet of perfect conductor drawn inside it."""
    found, _ = guide()
    found.Group = [*found.Group, metal_binding(septum())]
    return problem(found)


def rod_guide():
    """The guide with a body of perfect conductor standing inside it."""
    found, _ = guide()
    rod = obj("Part::Feature", "Rod", Shape=Shape(name="rod"))
    found.Group = [*found.Group, metal_binding(rod)]
    return problem(found)


class TestABodyOfPerfectConductor:
    """A body bound to a perfect conductor leaves the region, and the faces it
    leaves behind carry the condition."""

    def test_it_is_a_conductor_drawn_as_a_body(self):
        (rod,) = rod_guide().conductors
        assert (rod.label, rod.solid) == ("RodMetal", True)

    def test_it_reaches_the_mesher_filled_and_leaving(self):
        described = rod_guide()
        assert ("RodMetal", 3) in [(name, dimension) for name, dimension, _ in described.labelled]
        handed = described.pieces({label: ("shape.brep",) for label, _, _ in described.labelled})
        (rod,) = [piece for piece in handed if piece.label == "RodMetal"]
        assert rod.leaves and rod.divides

    def test_it_stands_above_the_region_it_is_drawn_in(self):
        """The mesher gives a piece two labels were drawn over to the higher
        priority, and only the body leaving takes the piece out."""
        described = rod_guide()
        handed = described.pieces({label: ("shape.brep",) for label, _, _ in described.labelled})
        priority = {piece.label: piece.priority for piece in handed}
        assert priority["RodMetal"] > priority[described.regions[0].label]

    def test_a_sheet_neither_leaves_nor_stands_above_anything(self):
        described = sealed_guide()
        handed = described.pieces({label: ("shape.brep",) for label, _, _ in described.labelled})
        (sheet,) = [piece for piece in handed if piece.label == described.conductors[0].label]
        assert (sheet.leaves, sheet.priority) == (False, 0)

    def test_its_faces_carry_the_perfect_conductor(self):
        described = rod_guide()
        mesh = meshed(described)
        run = configured(described, mesh, "results")
        assert mesh.labels["RodMetal"].tag in run.perfect_conductor


def metal_bodies(monkeypatch, *names, shared=(), sheets=()):
    """The guide with a body of perfect conductor per name in ``names`` and a
    sheet per name in ``sheets``, each under a binding of its own, in that
    order. The bodies each pair in ``shared`` names meet, and no others do."""
    pairs = [set(pair) for pair in shared]
    monkeypatch.setattr(
        drawn, "meets", lambda one, other, boxes=None: {one.name, other.name} in pairs
    )
    found, _ = guide()
    bodies = [obj("Part::Feature", name, Shape=Shape(name=name)) for name in names]
    found.Group = [
        *found.Group,
        *(metal_binding(body) for body in bodies),
        *(metal_binding(septum(name)) for name in sheets),
    ]
    return found


def covered_by(found, binding):
    """The body a binding covers, found by the binding's label."""
    (bound,) = [one for one in found.Group if one.Label == binding]
    return bound.References[0][0]


class TestBodiesOfPerfectConductorSharingAVolume:
    """Every body of metal reaches the mesher at one priority, and the mesher
    refuses two labels at one priority over one piece. Two perfect conductors
    carry one condition, so bindings whose bodies meet are drawn under one
    label."""

    def test_they_are_one_conductor_under_the_first_binding(self, monkeypatch):
        described = problem(metal_bodies(monkeypatch, "Block", "Via", shared=[("Block", "Via")]))
        (metal,) = described.conductors
        assert (metal.label, metal.joined, metal.solid) == ("BlockMetal", ("ViaMetal",), True)
        assert [shape.name for shape in metal.shapes] == ["Block", "Via"]
        labels = [name for name, _, _ in described.labelled]
        assert labels.count("BlockMetal") == 1 and "ViaMetal" not in labels

    def test_bindings_join_through_each_other(self, monkeypatch):
        """The pad meets the barrel and the barrel the plane, and the pad and the
        plane share nothing: one metal all the same."""
        found = metal_bodies(
            monkeypatch,
            "Pad",
            "Plane",
            "Barrel",
            shared=[("Barrel", "Plane"), ("Pad", "Barrel")],
        )
        (metal,) = problem(found).conductors
        assert (metal.label, metal.joined) == ("PadMetal", ("PlaneMetal", "BarrelMetal"))

    def test_a_group_found_first_joins_whole(self, monkeypatch):
        """The pad and the barrel are one metal before the via is reached, and the
        via then joins them to the plane."""
        found = metal_bodies(
            monkeypatch,
            "Plane",
            "Pad",
            "Barrel",
            "Via",
            shared=[("Pad", "Barrel"), ("Plane", "Via"), ("Pad", "Via")],
        )
        (metal,) = problem(found).conductors
        assert (metal.label, metal.joined) == (
            "PlaneMetal",
            ("PadMetal", "BarrelMetal", "ViaMetal"),
        )

    def test_bodies_that_do_not_meet_stay_apart(self, monkeypatch):
        described = problem(metal_bodies(monkeypatch, "Block", "Via", "Post", shared=[]))
        assert [(one.label, one.joined) for one in described.conductors] == [
            ("BlockMetal", ()),
            ("ViaMetal", ()),
            ("PostMetal", ()),
        ]

    def test_only_the_bindings_that_meet_join(self, monkeypatch):
        found = metal_bodies(monkeypatch, "Block", "Post", "Via", shared=[("Block", "Via")])
        assert [(one.label, one.joined) for one in problem(found).conductors] == [
            ("BlockMetal", ("ViaMetal",)),
            ("PostMetal", ()),
        ]

    def test_a_pair_the_kernel_cannot_measure_is_left_apart(self, monkeypatch):
        """Nothing then says the bodies meet, and the mesher refuses a piece the
        two share, naming both."""
        found = metal_bodies(monkeypatch, "Block", "Via")

        def unmeasured(one, other, boxes=None):
            raise drawn.Unmeasured("BRep_API: command not done")

        monkeypatch.setattr(drawn, "meets", unmeasured)
        assert [one.joined for one in problem(found).conductors] == [(), ()]

    def test_every_body_of_a_binding_joins_with_it(self, monkeypatch):
        """The plane's binding covers two bodies, and the via meets the second."""
        pairs = [{"Hole", "Via"}]
        monkeypatch.setattr(
            drawn, "meets", lambda one, other, boxes=None: {one.name, other.name} in pairs
        )
        found, _ = guide()
        plane, hole, via = (Shape(name=name) for name in ("Plane", "Hole", "Via"))
        planes = obj(
            "EMMaterialBinding",
            "PlaneMetal",
            Material=dielectric(MaterialType="PEC"),
            References=[
                (obj("Part::Feature", "Plane", Shape=plane), [""]),
                (obj("Part::Feature", "Hole", Shape=hole), [""]),
            ],
        )
        found.Group = [
            *found.Group,
            planes,
            metal_binding(obj("Part::Feature", "Via", Shape=via)),
        ]
        (metal,) = problem(found).conductors
        assert (metal.label, metal.joined) == ("PlaneMetal", ("ViaMetal",))
        assert [shape.name for shape in metal.shapes] == ["Plane", "Hole", "Via"]

    def test_a_distance_the_kernel_fails_to_measure_leaves_the_pair_apart(self):
        """Measured through :func:`Microwave.drawn.meets`, over bodies whose
        boxes meet."""
        found, _ = guide()
        block, via = Shape(name="Block"), Shape(name="Via")
        via.BoundBox = block.BoundBox

        def failing(other):
            raise RuntimeError("BRep_API: command not done")

        block.distToShape = via.distToShape = failing
        found.Group = [
            *found.Group,
            *(metal_binding(obj("Part::Feature", one.name, Shape=one)) for one in (block, via)),
        ]
        assert [one.joined for one in problem(found).conductors] == [(), ()]

    def test_a_sheet_is_left_to_itself(self, monkeypatch):
        """A sheet on a body's face is the sheet's, which the mesher settles."""
        found = metal_bodies(monkeypatch, "Block", "Via", sheets=["Septum"])
        monkeypatch.setattr(drawn, "meets", lambda *_, **__: True)
        assert [(one.label, one.solid, one.joined) for one in problem(found).conductors] == [
            ("BlockMetal", True, ("ViaMetal",)),
            ("SeptumMetal", False, ()),
        ]

    def test_a_coarsening_of_one_joined_binding_names_the_label_it_is_part_of(self, monkeypatch):
        """A rim settles for a coarser size only where every face its label's
        bindings name is coarsened, and the via's binding is drawn under the
        block's label."""
        found = metal_bodies(monkeypatch, "Block", "Via", shared=[("Block", "Via")])
        via = covered_by(found, "ViaMetal")
        found.Group = [
            *found.Group,
            refinement([(via, [""])], Mode="Coarsen", ElementSize=2.0),
        ]
        (said,) = [one for one in problem(found).unlaid if isinstance(one, Coarsening)]
        assert (said.part, said.bodies, said.joined) == (("BlockMetal",), (), (("ViaMetal",),))
        (line,) = [
            line
            for line in unlaid_said(problem(found).unlaid, answered(problem(found)))
            if line.startswith("'Ring'")
        ]
        assert ("names some faces of 'BlockMetal' (drawn with 'ViaMetal') and not all") in line

    def test_a_coarsening_names_what_each_label_holds(self, monkeypatch):
        described = problem(metal_bodies(monkeypatch, "Block"))
        record = Coarsening(
            subject="Ring",
            size=2.0,
            part=("BlockMetal", "PostMetal"),
            joined=(("ViaMetal",), ()),
        )
        (line,) = unlaid_said((record,), answered(described))
        assert (
            "names some faces of 'BlockMetal' (drawn with 'ViaMetal') and 'PostMetal' and not all"
        ) in line

    def test_a_label_joined_to_another_is_still_a_name_taken(self, monkeypatch):
        found = metal_bodies(monkeypatch, "Block", "Via", shared=[("Block", "Via")])
        port = [one for one in found.Group if one.Label == "Port1"][0]
        port.Label = "ViaMetal"
        with pytest.raises(TranslationError, match="two objects are called 'ViaMetal'"):
            problem(found)

    def test_the_run_says_which_bindings_it_holds_as_one_metal(self, monkeypatch):
        found = metal_bodies(
            monkeypatch, "Block", "Via", "Post", shared=[("Block", "Via"), ("Post", "Via")]
        )
        assert one_metal(problem(found).conductors) == [
            "'BlockMetal', 'ViaMetal' and 'PostMetal' bind perfect conductors whose bodies "
            "meet, directly or through one another, so the mesh holds them as one metal "
            "under 'BlockMetal'"
        ]


class TestWhichLabelsNoFieldCrosses:
    """Palace duplicates the boundary elements of every attribute a condition
    names but a lumped port's, so those faces stand between two problems."""

    def test_a_sheet_of_perfect_conductor_divides(self):
        described = sealed_guide()
        assert described.dividing == {described.conductors[0].label}

    def test_a_sheet_of_finite_conductivity_divides(self):
        described = problem(lossy_guide())
        assert described.dividing == {described.lossy[0].label}

    def test_a_port_and_the_wall_do_not(self):
        """A wave port's face and the wall are where the model ends, so they
        join nothing to begin with and need say nothing."""
        described = problem(guide()[0])
        assert described.dividing == frozenset()
        assert described.wall not in described.dividing

    def test_what_divides_reaches_the_mesher_on_its_own_pieces(self):
        described = sealed_guide()
        metal = described.conductors[0].label
        handed = described.pieces({label: ("shape.brep",) for label, _, _ in described.labelled})
        assert {piece.label for piece in handed if piece.divides} == {metal}


class TestAPartOfTheRegionNoPortStandsOn:
    """The mesher says which parts a condition leaves standing apart, and this
    layer is the one that knows a part with no port on it is driven by nothing."""

    def named(self, *labels):
        """A part of the region standing on these labels."""
        return Part(labels=tuple(labels), place="(10, 7, 3) to (18, 13, 7)")

    def whole(self, described):
        """The part every port and the wall stand on."""
        return Part(
            labels=(
                described.regions[0].label,
                described.conductors[0].label,
                described.wall,
                *(feed.label for feed in described.feeds),
            ),
            place="(0, 0, 0) to (30, 20, 10)",
        )

    def test_a_part_the_metal_closes_off_is_refused_by_the_metal_and_the_place(self):
        described = sealed_guide()
        metal = described.conductors[0].label
        sealed = self.named(described.regions[0].label, metal)
        with pytest.raises(TranslationError) as refused:
            check(described, meshed(described, parted=(self.whole(described), sealed)))
        assert "the part of the region at (10, 7, 3) to (18, 13, 7)" in str(refused.value)
        assert f"closed off by {metal!r}" in str(refused.value)

    def test_a_part_closed_by_a_label_this_problem_does_not_state_names_its_own(self):
        """A mesh made elsewhere reaches ``configured``, and its parts may stand
        on labels no condition of this problem names."""
        described = sealed_guide()
        stranger = self.named(described.regions[0].label, "ShieldBinding")
        with pytest.raises(TranslationError) as refused:
            check(described, meshed(described, parted=(self.whole(described), stranger)))
        assert "closed off by 'ShieldBinding'" in str(refused.value)
        assert described.regions[0].label not in str(refused.value), (
            "what fills the part does not close it off"
        )

    def test_a_part_a_port_stands_on_is_taken(self):
        described = sealed_guide()
        check(described, meshed(described, parted=(self.whole(described),)))

    def test_a_mesh_naming_no_parts_is_taken(self):
        """Which is every mesh made before a label said anything divides it."""
        described = sealed_guide()
        check(described, meshed(described))

    def test_the_parts_the_mesher_makes_are_the_ones_this_refusal_reads(self, tmp_path):
        """The two halves of the guard driven end to end.

        Everything above hands ``check`` parts written by hand, and the mesher's
        own tests never see a Palace problem. What neither can catch is the
        mesher putting a label on a part under a name or a rule the refusal does
        not read - in particular that a dividing label lands on both sides of
        its own cut, which is the whole of what keeps the message from naming
        nothing.
        """
        gmsh_mesh = needed("gmsh", "Gmsh is not installed")
        assert gmsh_mesh is not None
        from Microwave.Gmsh.mesh import mesh as build

        described = sealed_guide()
        air, metal = described.regions[0].label, described.conductors[0].label
        drawn = pathlib.Path(__file__).parent / "_drawings"
        handed = [
            Piece(air, 3, str(drawn / "skin_sealing_a_cavity.0.brep")),
            Piece(metal, 2, str(drawn / "skin_sealing_a_cavity.1.brep"), divides=True),
        ]
        got = build(
            handed,
            Demand(coarsest=8.0, finest=2.0),
            replace(PROFILE, element_order=1, curved=False, connected=False),
            str(tmp_path),
            "sealed",
            remainder=described.wall,
        )
        assert len(got.parted) == 2, "the skin cuts the air into the space it seals and the rest"
        with pytest.raises(TranslationError) as refused:
            check(described, replace(meshed(described), parted=got.parted))
        assert f"closed off by {metal!r}" in str(refused.value), (
            "the mesher put the dividing label on the sealed part under the name the "
            "problem states it by"
        )


class TestASheetOfFiniteConductivity:
    def test_it_carries_the_metal_its_material_states(self):
        described = problem(lossy_guide())
        (sheet,) = described.lossy
        assert sheet.label == "SeptumMetal"
        assert sheet.conductivity == 1.57e7
        assert sheet.permeability == 1.0
        assert described.conductors == ()

    def test_it_reaches_the_mesher_as_a_surface_under_its_binding(self):
        described = problem(lossy_guide())
        assert ("SeptumMetal", 2) in [
            (name, dimension) for name, dimension, _ in described.labelled
        ]

    def test_a_sheet_thick_against_its_skin_depth_over_the_band_is_given_no_thickness(self):
        """Past that, the thickness changes nothing Palace computes, and written
        it is the number Palace's factor overflows on."""
        assert problem(lossy_guide()).lossy[0].thickness == 0.0

    @pytest.mark.parametrize("permeability", [1.0, 4.0])
    def test_the_thickness_is_left_out_from_where_the_bottom_of_the_band_is_thick(
        self, permeability
    ):
        metal = sheet_metal(Permeability=permeability)
        start = frequency_at(THICK * 1.01, metal)
        assert lossy_band(start, 2 * start, metal).thickness == 0.0

    @pytest.mark.parametrize("permeability", [1.0, 4.0])
    def test_the_thickness_is_kept_where_the_bottom_of_the_band_is_thin(self, permeability):
        metal = sheet_metal(Permeability=permeability)
        start = frequency_at(THICK * 0.99, metal)
        assert lossy_band(start, 2 * start, metal).thickness == 0.5

    def test_a_conductivity_past_a_doubles_range_is_thick_rather_than_a_crash(self):
        metal = sheet_metal(Conductivity=sys.float_info.max)
        assert problem(lossy_guide(material=metal)).lossy[0].thickness == 0.0

    def test_a_band_wide_enough_to_overflow_the_thickness_is_refused_before_a_mesh(self):
        """The overflow is taken at twice the thickness, which is what Palace
        uses where the sheet ends the model, and that is not known before the
        mesh exists."""
        start = frequency_at(THICK / 2, sheet_metal())
        with pytest.raises(TranslationError, match="Brass.*not a number"):
            lossy_band(start, frequency_at(OVERFLOW / 2 * 1.01, sheet_metal()))

    def test_a_band_just_short_of_the_overflow_is_taken_with_its_thickness(self):
        start = frequency_at(THICK / 2, sheet_metal())
        stop = frequency_at(OVERFLOW / 2 * 0.99, sheet_metal())
        assert lossy_band(start, stop).thickness == 0.5

    @pytest.mark.parametrize(
        "changed, said",
        [
            (dict(Conductivity=0.0), "Conductivity is 0 S/m, and a conducting sheet conducts"),
            (dict(Conductivity=math.nan), "Conductivity is nan S/m"),
            (dict(Conductivity=math.inf), "Conductivity is inf S/m"),
            (dict(Thickness=0.0), "Thickness is 0 mm, and a conducting sheet states"),
            (dict(Permeability=0.0), "Permeability is 0, and a relative permeability"),
            (dict(Permittivity=4.3), "carries Permittivity 4.3, which reaches nothing"),
            (dict(LossTangent=0.025), "carries LossTangent 0.025, which reaches nothing"),
        ],
    )
    def test_what_a_sheet_cannot_carry_is_refused_naming_the_material(self, changed, said):
        with pytest.raises(TranslationError, match=f"'Brass'.*{said}"):
            problem(lossy_guide(material=sheet_metal(**changed)))

    @pytest.mark.parametrize(
        "changed, said",
        [
            (dict(Permittivity=4.3), "Permittivity 4.3"),
            (dict(Permeability=4.0), "Permeability 4"),
            (dict(Permeability=0.0), "Permeability 0"),
            (dict(LossTangent=0.025), "LossTangent 0.025"),
        ],
    )
    def test_what_reaches_nothing_on_a_perfect_conductor_is_refused(self, changed, said):
        metal = dielectric(MaterialType="PEC", **changed)
        with pytest.raises(TranslationError, match=f"PEC material and carries {said}"):
            problem(lossy_guide(material=metal))

    def test_a_perfect_conductor_is_not_refused_for_its_thickness(self):
        """Every material carries a thickness by default, so one on a perfect
        conductor says nothing the user typed."""
        metal = dielectric(MaterialType="PEC", Thickness=0.035)
        assert len(problem(lossy_guide(material=metal)).conductors) == 1

    @pytest.mark.parametrize(
        "kind, name, stated, reason, reader",
        [
            ("PEC", "Permittivity", 4.3, "a condition on a face", "Dielectric"),
            ("PEC", "LossTangent", 0.02, "a condition on a face", "Dielectric"),
            ("PEC", "Permeability", 2.0, "no skin depth and no loss", "ConductingSheet"),
            ("PEC", "Conductivity", 5.8e7, "no skin depth and no loss", "ConductingSheet"),
            ("ConductingSheet", "Permittivity", 4.3, "a condition on a face", "Dielectric"),
            ("ConductingSheet", "LossTangent", 0.02, "a condition on a face", "Dielectric"),
        ],
    )
    def test_each_value_metal_does_not_read_is_refused_saying_why(
        self, kind, name, stated, reason, reader
    ):
        base = sheet_metal() if kind == "ConductingSheet" else dielectric(MaterialType="PEC")
        setattr(base, name, stated)
        with pytest.raises(TranslationError) as refused:
            problem(lossy_guide(material=base))
        assert f"{name} {stated:g}, which reaches nothing" in str(refused.value)
        assert reason in str(refused.value) and f"make this a {reader}" in str(refused.value)

    def test_a_perfect_conductor_carrying_a_conductivity_is_refused(self):
        """A perfect conductor has no loss to take one from, and a sheet switched
        to PEC in the property editor keeps the conductivity it had, which the
        run would drop without a word."""
        metal = dielectric(MaterialType="PEC", Conductivity=5.8e7)
        with pytest.raises(TranslationError, match="Conductivity 5.8e.07, which reaches nothing"):
            problem(lossy_guide(material=metal))
        with pytest.raises(TranslationError, match="make this a ConductingSheet"):
            problem(lossy_guide(material=metal))

    @pytest.mark.parametrize("name", ["LossTangent", "Conductivity"])
    @pytest.mark.parametrize("stated", [-0.02, math.nan, math.inf])
    def test_a_dielectric_loss_that_is_not_a_loss_is_refused(self, name, stated):
        """Palace would take a negative one as a material that supplies energy."""
        with pytest.raises(TranslationError, match="not a loss|not a number a loss can be"):
            problem(guide(material=dielectric(**{name: stated}))[0])

    @pytest.mark.parametrize("name", ["Permittivity", "Permeability"])
    @pytest.mark.parametrize("stated", [0.0, -2.0, math.nan, math.inf])
    def test_a_filling_that_carries_no_wave_is_refused_naming_it(self, name, stated):
        """Refused by name before anything divides by it."""
        with pytest.raises(TranslationError, match=f"'Air': a relative {name.lower()}"):
            problem(guide(material=dielectric(**{name: stated}))[0])

    def test_a_body_of_finite_conductivity_is_refused(self):
        """This adapter writes such a metal as a surface impedance on a sheet."""
        found, _ = guide()
        rod = obj("Part::Feature", "Rod", Shape=Shape(name="rod"))
        found.Group = [*found.Group, metal_binding(rod, material=sheet_metal())]
        with pytest.raises(
            TranslationError,
            match="'Rod', which is a body, and this solver models such a metal as a sheet",
        ):
            problem(found)

    def test_a_sheet_sharing_a_name_with_a_port_is_refused(self):
        found = lossy_guide()
        found.Group[-1].Label = "Port1"
        with pytest.raises(TranslationError, match="two objects are called 'Port1'"):
            problem(found)


class TestTheConductivityConditionWritten:
    def test_a_thick_sheet_is_written_with_no_thickness(self):
        written = SurfaceConductivity(attributes=(5,), conductivity=1.57e7).to_dict()
        assert written == {
            "Attributes": [5],
            "Conductivity": 1.57e7,
            "Permeability": 1.0,
            "External": False,
        }

    def test_a_thickness_is_written_in_the_units_the_mesh_is_drawn_in(self):
        """Palace's schema states it in mesh length units, which ``L0`` makes
        millimetres here."""
        written = SurfaceConductivity(
            attributes=(5,), conductivity=5.8e7, thickness=0.018, external=True
        ).to_dict()
        assert written["Thickness"] == 0.018
        assert written["External"] is True
        assert driven().to_dict()["Model"]["L0"] == 1.0 / units.MM_PER_M

    @pytest.mark.parametrize(
        "changed, said",
        [
            (dict(attributes=()), "at least one attribute"),
            (dict(conductivity=0.0), "a metal conducts"),
            (dict(conductivity=math.inf), "a metal conducts"),
            (dict(permeability=0.0), "relative permeability is positive"),
            (dict(thickness=-0.5), "zero or positive"),
            (dict(thickness=math.nan), "zero or positive"),
            (dict(thickness=math.inf), "zero or positive"),
        ],
    )
    def test_a_condition_palace_would_stop_on_cannot_be_built(self, changed, said):
        with pytest.raises(ValueError, match=said):
            SurfaceConductivity(**{"attributes": (5,), "conductivity": 1.57e7, **changed})

    def test_a_model_whose_metal_is_all_lossy_writes_no_perfect_conductor(self):
        metal = SurfaceConductivity(attributes=(2,), conductivity=1.57e7, external=True)
        written = driven(perfect_conductor=(), conducting=(metal,)).to_dict()["Boundaries"]
        assert "PEC" not in written
        assert written["Conductivity"] == [metal.to_dict()]

    def test_a_model_with_no_metal_at_all_cannot_be_built(self):
        with pytest.raises(ValueError, match="no attribute carries metal"):
            driven(perfect_conductor=())

    def test_a_model_with_no_lossy_metal_writes_no_conductivity(self):
        assert "Conductivity" not in driven().to_dict()["Boundaries"]


class TestHowTheLossIsModelled:
    """What a result says Palace was given, one record to a material."""

    @pytest.mark.parametrize(
        "where, faces",
        [
            ({}, ["boundary"]),
            ({"inside": ("SeptumMetal",)}, ["inside"]),
            ({"both": ("SeptumMetal",)}, ["boundary", "inside"]),
        ],
    )
    def test_a_sheet_carries_its_metal_on_the_faces_the_region_meets(self, where, faces):
        described = problem(lossy_guide())
        (record,) = pipeline.modelled(described, meshed(described, **where))
        assert record["sheet"] == "surface impedance"
        assert record["faces"] == faces

    def test_two_sheets_of_one_metal_are_one_record_holding_both_sides(self):
        described = problem(lossy_guide())
        (sheet,) = described.lossy
        second = replace(sheet, label="OtherMetal")
        described = replace(described, lossy=(sheet, second))
        mesh = meshed(described, inside=("OtherMetal",))
        (record,) = pipeline.modelled(described, mesh)
        assert record["material"] == "Brass"
        assert record["faces"] == ["boundary", "inside"]

    def test_two_materials_sharing_a_label_are_two_records(self):
        """FreeCAD can be set to let two objects share a label, and a record
        keyed by the label alone would state one of the two."""
        described = problem(guide(material=dielectric(LossTangent=0.001))[0])
        (region,) = described.regions
        other = replace(
            region, label="OtherFill", filling=replace(region.filling, loss_tangent=0.003)
        )
        described = replace(described, regions=(region, other))
        assert [
            record["loss_tangent"] for record in pipeline.modelled(described, meshed(described))
        ] == [
            0.001,
            0.003,
        ]

    def test_a_record_names_the_material_and_not_what_binds_it(self):
        """The other backend names a material by its own label, and a comparison
        of the two is by that."""
        described = problem(guide(material=dielectric(Conductivity=0.5))[0])
        assert pipeline.modelled(described, meshed(described)) == (
            {"material": "Air", "held": "conductivity", "conductivity": 0.5},
        )

    def test_what_every_backend_models_alike_is_not_recorded(self):
        described = problem(guide()[0])
        assert pipeline.modelled(described, meshed(described)) == ()


class TestTheSideASheetStandsOn:
    @pytest.mark.parametrize("sits, external", [(FRONTIER, True), (INTERIOR, False)])
    def test_a_sheet_where_the_model_ends_is_written_external(self, sits, external):
        described = problem(lossy_guide())
        mesh = meshed(described, inside=("SeptumMetal",) if sits == INTERIOR else ())
        (metal,) = configured(described, mesh, "out").conducting
        assert metal.external is external
        assert metal.attributes == (mesh.labels["SeptumMetal"].tag,)

    def test_a_thin_sheet_on_both_sides_is_refused_naming_it(self):
        start = frequency_at(THICK / 2, sheet_metal())
        described = problem(lossy_guide(FrequencyStart=start, FrequencyStop=2 * start))
        with pytest.raises(TranslationError, match="'SeptumMetal' stands both inside"):
            configured(described, meshed(described, both=("SeptumMetal",)), "out")

    def test_the_metal_reaches_the_condition_as_its_material_states_it(self):
        start = frequency_at(THICK / 2, sheet_metal(Permeability=4.0))
        described = problem(
            lossy_guide(
                material=sheet_metal(Permeability=4.0),
                FrequencyStart=start,
                FrequencyStop=2 * start,
            )
        )
        (metal,) = configured(described, meshed(described), "out").conducting
        assert (metal.conductivity, metal.permeability, metal.thickness) == (1.57e7, 4.0, 0.5)

    def test_a_thick_sheet_may_stand_on_both_sides(self):
        """With no thickness written the side changes nothing Palace computes."""
        described = problem(lossy_guide())
        (metal,) = configured(described, meshed(described, both=("SeptumMetal",)), "out").conducting
        assert metal.thickness == 0.0


class TestWhatTheStudyHasToHold:
    def test_a_study_over_one_guide_describes_a_run(self):
        described = problem(guide()[0])
        assert [feed.number for feed in described.feeds] == [1, 2]
        assert len(described.regions) == 1
        assert described.order == 2

    def test_anything_but_a_study_is_refused_by_name(self):
        with pytest.raises(TranslationError, match="is not a study"):
            problem(palace_solver())

    def test_the_refusal_names_the_backend_it_is_about(self):
        """A study may hold the other backend's solver and none of this one's,
        and a message saying it holds no solver is then false with one plainly
        in the tree. The user is looking at the object the sentence denies."""
        found, _ = guide()
        found.Group = [
            member if kind(member) != "EMSolverPalace" else obj("EMSolverOpenEMS", "openEMS")
            for member in found.Group
        ]
        with pytest.raises(TranslationError) as refusal:
            problem(found)
        assert "holds no Palace solver" in str(refusal.value)

    def test_a_study_with_no_solver_of_this_kind_is_refused(self):
        found, _ = guide()
        found.Group = [member for member in found.Group if kind(member) != "EMSolverPalace"]
        with pytest.raises(TranslationError, match="holds no Palace solver"):
            problem(found)

    def test_a_study_holding_two_of_them_says_nothing_about_which_runs(self):
        found, _ = guide()
        found.Group = [*found.Group, palace_solver("Palace2")]
        with pytest.raises(TranslationError, match="more than one Palace solver"):
            problem(found)

    def test_a_study_with_no_mesh_policy_is_refused(self):
        found, _ = guide()
        found.Group = [member for member in found.Group if kind(member) != "EMMeshPolicy"]
        with pytest.raises(TranslationError, match="holds no mesh policy"):
            problem(found)

    def test_a_study_binding_no_material_has_no_region_for_a_field(self):
        found, _ = guide()
        found.Group = [member for member in found.Group if kind(member) != "EMMaterialBinding"]
        with pytest.raises(TranslationError, match="binds no material"):
            problem(found)

    def test_a_study_with_no_port_has_nothing_driving_it(self):
        found, _ = guide()
        found.Group = [member for member in found.Group if not kind(member).startswith("EMPort")]
        with pytest.raises(TranslationError, match="holds no port"):
            problem(found)

    def test_a_port_in_a_nested_group_is_still_the_studys(self):
        """Sorting ports into a subgroup is housekeeping, not a way to drop them."""
        found, body = guide()
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        rest = [member for member in found.Group if not kind(member).startswith("EMPort")]
        found.Group = [*rest, SimpleNamespace(Proxy=None, Label="Ports", Name="G", Group=ports)]
        assert [feed.number for feed in problem(found).feeds] == [1, 2]


class TestHowTheDrawingBecomesLabels:
    def test_a_guide_loaded_with_a_slab_carries_both_fillings(self):
        """Two bound bodies that touch is what a loaded guide is drawn as.

        What is held here is the translation: each binding becomes a region of
        its own with its own filling, and the wall is one attribute whatever the
        drawing was cut into. Which faces that attribute ends up carrying is the
        mesher's answer and is held against a real mesh in
        ``tests/test_gmsh_labels.py``, the mesh here being a fake that hands
        back a label per name.
        """
        found, body = guide()
        slab = obj(
            "EMMaterialBinding",
            "Slab",
            Material=dielectric(Permittivity=2.2),
            References=[(obj("Part::Feature", "Slab", Shape=Shape()), [])],
        )
        found.Group = [*found.Group, slab]
        described = problem(found)
        run = configured(described, meshed(described), "results")
        assert sorted(material.filling.permittivity for material in run.materials) == [1.0, 2.2]
        assert run.perfect_conductor == (meshed(described).labels[described.wall].tag,)

    def test_the_wall_is_a_name_and_no_shape_is_drawn_for_it(self):
        """A wall is what is left of the boundary once everything else has
        claimed its own, and the mesher is the layer that knows what is left.
        """
        described = problem(guide()[0])
        assert described.wall == "wall"
        assert "wall" not in {row[0] for row in described.labelled}

    def test_a_port_holds_the_face_its_link_names(self):
        described, body = guide()
        described = problem(described)
        assert described.feeds[0].shapes == (body.Shape.Faces[0],)
        assert described.feeds[1].shapes == (body.Shape.Faces[1],)

    def test_the_region_holds_the_body_the_binding_covers(self):
        described, body = guide()
        assert problem(described).regions[0].shapes == (body.Shape,)

    def test_no_label_is_handed_over_above_another(self):
        """The wall is not drawn, so it contests nothing, and a region is a
        volume where a port is a face, so the two cannot meet over a piece.
        """
        described = problem(guide()[0])
        files = {
            row[0]: tuple(f"{row[0]}-{n}.brep" for n in range(len(row[2])))
            for row in described.labelled
        }
        assert {piece.priority for piece in described.pieces(files)} == {0}

    def test_every_label_says_which_dimension_it_is_declared_at(self):
        """A file arrives with its whole hierarchy, so a solid's own faces come
        back beside it and the declaration is what separates them.
        """
        described = problem(guide()[0])
        at = {row[0]: row[1] for row in described.labelled}
        assert at["AirFill"] == 3
        assert at["Port1"] == 2

    def test_the_mesher_is_asked_for_one_piece_per_shape(self):
        described = problem(guide()[0])
        files = {
            row[0]: tuple(f"{row[0]}-{n}.brep" for n in range(len(row[2])))
            for row in described.labelled
        }
        pieces = described.pieces(files)
        assert len(pieces) == sum(len(row[2]) for row in described.labelled)
        assert {piece.label for piece in pieces} == {"AirFill", "Port1", "Port2"}

    def test_a_label_written_to_fewer_files_than_it_has_shapes_is_refused(self):
        described = problem(guide()[0])
        files = {row[0]: () for row in described.labelled}
        with pytest.raises(ValueError, match="files were written"):
            described.pieces(files)

    def test_a_sheet_of_metal_is_a_label_at_the_dimension_of_a_face(self):
        """It is the faces it was drawn as, wherever they stand, and is handed
        over with no way into the model: nothing stands behind it to leave out."""
        found, _ = guide()
        sheet = septum()
        found.Group = [*found.Group, metal_binding(sheet)]
        described = problem(found)
        assert [(c.label, c.shapes) for c in described.conductors] == [
            ("SeptumMetal", (sheet.Shape,))
        ]
        assert ("SeptumMetal", 2, (sheet.Shape,)) in described.labelled
        files = {
            row[0]: tuple(f"{row[0]}-{n}.brep" for n in range(len(row[2])))
            for row in described.labelled
        }
        (piece,) = [p for p in described.pieces(files) if p.label == "SeptumMetal"]
        assert (piece.dimension, piece.inward, piece.priority) == (2, None, 0)

    def test_a_sheet_of_metal_is_no_region_and_fills_nothing(self):
        found, _ = guide()
        found.Group = [*found.Group, metal_binding(septum())]
        described = problem(found)
        assert [region.label for region in described.regions] == ["AirFill"]

    def test_a_sheet_of_metal_carrying_the_name_of_another_object_is_refused(self):
        found, body = guide()
        sheet = metal_binding(septum())
        sheet.Label = "Port1"
        found.Group = [*found.Group, sheet]
        with pytest.raises(TranslationError, match="two objects are called 'Port1'"):
            problem(found)

    def test_a_sheet_of_metal_carrying_the_walls_name_is_refused(self):
        found, _ = guide()
        sheet = metal_binding(septum())
        sheet.Label = "wall"
        found.Group = [*found.Group, sheet]
        with pytest.raises(TranslationError, match="the name this solver gives"):
            problem(found)


class TestWhereAPortEndsTheModel:
    def test_a_port_on_a_plane_drawn_across_the_region_stands_on_that_plane(self):
        """What the other backend needs, its guide running on past the plane into
        an absorber. Here the mesher leaves out what stands behind the plane, so
        the face handed over is the plane's own.
        """
        found, _ = guide()
        plane = obj("Part::Plane", "Reference", Shape=Shape(faces=1, name="plane"))
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        ports[0].CrossSection = (plane, ["Face1"])
        (first, _) = problem(found).feeds
        assert first.shapes == (plane.Shape.Faces[0],)

    @pytest.mark.parametrize(
        ("axis", "inward"),
        [
            ("X", (1.0, 0.0, 0.0)),
            ("-X", (-1.0, 0.0, 0.0)),
            ("Y", (0.0, 1.0, 0.0)),
            ("-Y", (0.0, -1.0, 0.0)),
            ("Z", (0.0, 0.0, 1.0)),
            ("-Z", (0.0, 0.0, -1.0)),
        ],
    )
    def test_the_propagation_axis_is_the_way_into_the_model(self, axis, inward):
        found, body = guide()
        body.Shape.Faces[0] = Drawn("pipe.face1", spans=square_to(axis))
        found.Group = [
            member for member in found.Group if not kind(member).startswith("EMPort")
        ] + [waveguide_port(1, body, "Face1", PropagationAxis=axis)]
        (feed,) = problem(found).feeds
        assert feed.inward == inward

    @pytest.mark.parametrize(
        ("spans", "axis", "refused"),
        [
            ((0.3594, 5.35, 4.3), "X", True),
            ((5.35, 9.27, 4.3), "-X", True),
            ((0.0, 10.7, 4.3), "Y", True),
            ((2e-6, 10.7, 4.3), "X", True),
            ((5e-7, 10.7, 4.3), "X", False),
            ((10.7, 0.0, 4.3), "-Y", False),
        ],
        ids=[
            "curved",
            "turned",
            "square to another axis",
            "just past flat",
            "flat to within the tolerance",
            "square to its own",
        ],
    )
    def test_a_face_not_flat_and_square_to_its_axis_is_refused(self, spans, axis, refused):
        """What the face spans along the way the wave travels, against the
        workbench's flatness."""
        found, body = guide()
        body.Shape.Faces[0] = Drawn("pipe.face1", spans=spans)
        (first,) = [member for member in found.Group if getattr(member, "Number", 0) == 1]
        first.PropagationAxis = axis
        name = axis.lstrip("-")
        spanned = spans["XYZ".index(name)]
        if refused:
            with pytest.raises(TranslationError) as stopped:
                problem(found)
            assert str(stopped.value).startswith(
                f"'Port1' stands on a face that spans {spanned:.4g} mm along {name}"
            )
        else:
            problem(found)

    def test_an_axis_that_names_no_direction_is_refused(self):
        found, body = guide()
        found.Group = [
            member for member in found.Group if not kind(member).startswith("EMPort")
        ] + [waveguide_port(1, body, "Face1", PropagationAxis="Up")]
        with pytest.raises(TranslationError, match="PropagationAxis is 'Up'"):
            problem(found)

    def test_each_port_face_reaches_the_mesher_with_its_way_in_and_nothing_else_does(self):
        described = problem(guide()[0])
        files = {
            name: tuple(f"/run/{name}-{index}.brep" for index in range(len(shapes)))
            for name, _, shapes in described.labelled
        }
        handed = {piece.label: piece.inward for piece in described.pieces(files)}
        assert handed == {
            "AirFill": None,
            "Port1": (1.0, 0.0, 0.0),
            "Port2": (1.0, 0.0, 0.0),
        }


class TestWhatTheTranslationRefuses:
    def test_a_port_standing_on_no_face_at_all_is_refused(self):
        found, _ = guide()
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        ports[0].CrossSection = None
        with pytest.raises(TranslationError, match="CrossSection is unset"):
            problem(found)

    def test_a_port_standing_on_two_faces_is_refused(self):
        """A wave port is solved as one cross-section."""
        found, body = guide()
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        ports[0].CrossSection = (body, ["Face1", "Face5"])
        with pytest.raises(TranslationError, match="Select a single face"):
            problem(found)

    def test_a_port_standing_on_an_edge_is_refused(self):
        found, body = guide()
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        ports[0].CrossSection = (body, ["Edge3"])
        with pytest.raises(TranslationError, match="which is not a face"):
            problem(found)

    def test_a_port_reported_against_a_number_is_refused(self):
        """The impedance Palace states for a wave port is not the one the other
        backend renormalises a guide from, so a fixed reference would give one
        guide two reflections."""
        found, _ = guide()
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        ports[0].ReferencedTo = "Fixed impedance"
        with pytest.raises(TranslationError, match="two numbers on one guide's reflection"):
            problem(found)

    def test_a_mode_this_adapter_does_not_carry_is_refused(self):
        """The solver ranks modes by wave number and names none of them."""
        found, _ = guide()
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        ports[0].Mode = "TE20"
        with pytest.raises(TranslationError, match="ranks a port's modes"):
            problem(found)

    def test_a_reference_plane_inside_the_guide_is_the_ports_offset(self, uniform_asked):
        """Palace de-embeds each wave port by its ``Offset``, in the mesh's own
        length unit, which is the millimetre the drawing is in."""
        found, _ = guide()
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        ports[0].ReferenceDepth = 5.0
        described = problem(found)
        offsets = {feed.number: feed.offset for feed in described.feeds}
        assert offsets == {int(ports[0].Number): 5.0, int(ports[1].Number): 0.0}
        assert [(port, reach) for port, _, _, reach, _ in uniform_asked] == [(ports[0], 5.0)]

    def test_the_guide_is_asked_to_run_on_the_way_the_port_faces(self, uniform_asked):
        found, _ = guide()
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        ports[1].PropagationAxis = "-X"
        for port in ports:
            port.ReferenceDepth = 2.0
        problem(found)
        asked = {port.Name: (axis, direction) for port, axis, direction, _, _ in uniform_asked}
        assert asked == {
            port.Name: (0, 1 if str(port.PropagationAxis) == "X" else -1) for port in ports
        }

    def test_a_reference_plane_behind_the_face_is_refused(self):
        """FreeCAD's length property holds no negative value, and a script can
        still write one."""
        found, _ = guide()
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        ports[0].ReferenceDepth = -1.0
        with pytest.raises(TranslationError, match="ReferenceDepth is -1 mm"):
            problem(found)

    def test_a_port_an_earlier_build_saved_is_refused_by_what_departs(self):
        """What departs from a class is the classes' to say; this adapter asks
        each object it reads before reading any."""
        found, _ = guide()
        port = next(member for member in found.Group if kind(member).startswith("EMPort"))
        port.Proxy.departures = lambda _obj: SimpleNamespace(
            stops=True, phrases=("has no ReferenceDepth",)
        )
        del port.ReferenceDepth
        with pytest.raises(TranslationError, match=f"'{port.Label}' has no ReferenceDepth"):
            problem(found)

    def test_a_material_is_asked_through_the_binding_that_links_it(self):
        found, _ = guide()
        binding = next(member for member in found.Group if kind(member) == "EMMaterialBinding")
        binding.Proxy.linked = lambda _obj: (binding.Material,)
        binding.Material.Proxy.departures = lambda _obj: SimpleNamespace(
            stops=True, phrases=("has no DispersionFrequency",)
        )
        with pytest.raises(TranslationError, match="has no DispersionFrequency"):
            problem(found)

    @pytest.mark.parametrize(
        "wanted, lacking",
        [("EMSolverPalace", "Order"), ("EMMeshPolicy", "Medium"), ("EMGmshMesh", "MaxGrowthRatio")],
    )
    def test_each_member_read_is_asked(self, wanted, lacking):
        found, _ = guide()
        member = next(member for member in found.Group if kind(member) == wanted)
        said = SimpleNamespace(stops=True, phrases=(f"has no {lacking}",))
        member.Proxy.departures = lambda _obj: said
        with pytest.raises(TranslationError, match=f"'{member.Label}' has no {lacking}"):
            problem(found)

    def test_a_member_this_backend_does_not_read_is_not_asked(self):
        """The other backend's recipe is read by no Palace run, so what an
        earlier build left on it does not stop one."""
        found, _ = guide()
        grid = obj("EMYeeGrid", "YeeGrid")
        grid.Proxy.departures = lambda _obj: SimpleNamespace(
            stops=True, phrases=("has no ElementsPerWavelength",)
        )
        found.Group = [*found.Group, grid]
        problem(found)

    def test_two_ports_carrying_one_number_are_refused(self):
        found, body = guide()
        third = waveguide_port(1, body, "Face3")
        third.Label = third.Name = "PortThree"
        found.Group = [*found.Group, third]
        with pytest.raises(TranslationError, match="indexes every table"):
            problem(found)

    def test_a_port_kind_this_adapter_cannot_drive_is_refused(self):
        found, body = guide()
        microstrip = obj("EMPortMicrostrip", "Strip", Number=3, Excitation=True)
        found.Group = [*found.Group, microstrip]
        with pytest.raises(TranslationError, match="'Strip' is a microstrip port.*lumped"):
            problem(found)

    def test_no_port_set_to_excite_leaves_the_run_nothing_to_answer(self):
        found, _ = guide()
        for member in found.Group:
            if kind(member).startswith("EMPort"):
                member.Excitation = False
        with pytest.raises(TranslationError, match="no port is set to excite"):
            problem(found)

    def test_metal_bound_to_bodies_and_sheets_at_once_is_refused(self):
        """A body leaves the region and a sheet stands in it, so one label
        cannot be both."""
        found, _ = guide()
        rod = obj("Part::Feature", "Rod", Shape=Shape(name="rod"))
        binding = metal_binding(rod)
        binding.References = [*binding.References, (septum(), [])]
        found.Group = [*found.Group, binding]
        with pytest.raises(TranslationError, match="binds metal to bodies and to sheets at once"):
            problem(found)

    def test_a_body_bound_whole_twice_is_refused(self):
        """A body of metal stands above every region, so the metal would take
        the body from the dielectric with nothing said."""
        found, body = guide()
        found.Group = [*found.Group, metal_binding(body)]
        with pytest.raises(TranslationError) as refused:
            problem(found)
        assert str(refused.value) == (
            "'Pipe' is bound whole by 'AirFill' and by 'PipeMetal', and one shape is made of "
            "one material. Bind it once"
        )

    def test_a_face_of_a_bound_body_is_not_the_body(self):
        """A sheet bound on a face of a region is a condition on that face."""
        found, body = guide()
        found.Group = [*found.Group, metal_binding(body, sub=("Face3",))]
        assert problem(found).conductors[0].solid is False

    def test_metal_bound_to_what_holds_no_face_is_refused(self):
        found, _ = guide()
        wire = obj("Part::Feature", "Wire", Shape=Sheet(faces=0))
        found.Group = [*found.Group, metal_binding(wire)]
        with pytest.raises(TranslationError, match="'Wire', which holds no face"):
            problem(found)

    def test_metal_bound_to_a_face_of_a_body_is_that_face(self):
        """A face picked off the region itself is a sheet like any other."""
        found, body = guide()
        found.Group = [*found.Group, metal_binding(body, sub=["Face3", "Face4"])]
        (conductor,) = problem(found).conductors
        assert conductor.shapes == (body.Shape.Faces[2], body.Shape.Faces[3])

    def test_metal_bound_to_an_edge_is_refused(self):
        found, body = guide()
        found.Group = [*found.Group, metal_binding(body, sub=["Face3", "Edge1"])]
        with pytest.raises(TranslationError, match="covers Edge1 of 'Pipe', which is not a face"):
            problem(found)

    def test_metal_bound_to_nothing_is_refused(self):
        found, _ = guide()
        binding = metal_binding(septum())
        binding.References = []
        found.Group = [*found.Group, binding]
        with pytest.raises(TranslationError, match="the sheet or the body the metal is drawn as"):
            problem(found)

    def test_a_dispersive_material_is_refused_by_the_material_it_came_from(self):
        """Palace states a material as one number for the whole band, and a
        table of them has no single one to be."""
        found, _ = guide(material=dielectric(MaterialType="FrequencyDependentDielectric"))
        with pytest.raises(
            TranslationError, match="FrequencyDependentDielectric material, and this solver"
        ):
            problem(found)

    def test_a_sheet_of_finite_conductivity_with_no_dielectric_is_judged_against_vacuum(
        self,
    ):
        """With no region to judge the sheet against, the vacuum the study would
        reserve is what it is judged against, and nothing fails on the way to
        the refusal of a study with no room for a field."""
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.Material = dielectric(MaterialType="PEC")
        found.Group = [*found.Group, metal_binding(septum(), material=sheet_metal())]
        with pytest.raises(TranslationError, match="binds no dielectric to anything"):
            problem(found)

    def test_a_study_binding_nothing_but_metal_has_no_region_for_a_field(self):
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.Material = dielectric(MaterialType="PEC")
                member.References = [(septum(), [])]
        with pytest.raises(TranslationError, match="binds no dielectric to anything"):
            problem(found)

    def test_a_material_carrying_both_models_of_loss_names_the_object_it_came_from(self):
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.Material = dielectric(LossTangent=0.02, Conductivity=5.8e7)
        with pytest.raises(TranslationError, match="'Air'"):
            problem(found)

    def test_a_binding_naming_no_material_is_refused(self):
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.Material = None
        with pytest.raises(TranslationError, match="names no material"):
            problem(found)

    @pytest.mark.parametrize("linked", ["study", "body", *(label(one) for one in guide()[0].Group)])
    def test_a_binding_linking_what_is_not_a_material_is_refused_naming_both(self, linked):
        """The user can set the link to an object of any kind."""
        found, body = guide()
        members = {label(one): one for one in found.Group}
        target = {"study": found, "body": body, **members}[linked]
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.Material = target
        said = f"'AirFill': Material links {label(target)!r}, which is not a material"
        with pytest.raises(TranslationError, match=said):
            problem(found)

    def test_a_binding_covering_part_of_a_surface_is_refused(self):
        """A material fills a body, and a face has no volume to fill."""
        found, body = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.References = [(body, ["Face1"])]
        with pytest.raises(TranslationError, match="part of its surface"):
            problem(found)

    def test_a_binding_covering_nothing_is_refused(self):
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.References = []
        with pytest.raises(TranslationError, match="covers nothing"):
            problem(found)

    def test_a_solver_order_of_nothing_is_refused(self):
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMSolverPalace":
                member.Order = 0
        with pytest.raises(TranslationError, match="polynomial order"):
            problem(found)

    def test_two_objects_sharing_a_name_cannot_be_told_apart_in_the_mesh(self):
        found, body = guide()
        ports = [member for member in found.Group if kind(member).startswith("EMPort")]
        ports[1].Label = ports[0].Label
        with pytest.raises(TranslationError, match="Rename one of them"):
            problem(found)

    def test_a_binding_and_a_port_sharing_a_name_are_refused(self):
        """A label is what a shape is written to disk under and what the mesher
        groups by, so the two would arrive as one and the region's body would be
        in no group at all.
        """
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.Label = "Port1"
        with pytest.raises(TranslationError, match="Rename one of them"):
            problem(found)

    def test_a_body_carrying_the_name_this_solver_gives_the_boundary_is_refused(self):
        """The wall is one group like any other, and a region drawn under that
        name would reach it as a face rather than as a body.
        """
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.Label = "wall"
        with pytest.raises(TranslationError, match="Rename it"):
            problem(found)


class TestWhatTheStudyDecidesAboutTheMesh:
    def test_the_slowest_material_in_the_model_sizes_the_element(self):
        plain = problem(guide()[0]).demand
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.Material = dielectric(Permittivity=4.0)
        assert problem(found).demand.coarsest == pytest.approx(plain.coarsest / 2.0)

    def test_a_magnetic_material_sizes_it_too(self):
        """A wave slows by the root of the product, so leaving permeability out
        meshes a ferrite as though it were not magnetic.
        """
        plain = problem(guide()[0]).demand
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.Material = dielectric(Permeability=4.0)
        assert problem(found).demand.coarsest == pytest.approx(plain.coarsest / 2.0)

    def test_the_profile_is_the_adapters_own_and_not_read_off_the_study(self):
        assert problem(guide()[0]).profile is PROFILE

    def test_the_band_reaches_the_sweep(self):
        described = problem(guide()[0])
        assert (described.sweep.start, described.sweep.stop) == (20e9, 26e9)


def meshed(described, inside=(), both=(), absent=(), parted=()):
    """The mesh as it comes back, with each label in a group of its own.

    Where a label lies is the mesher's answer off the fragmented model, so a
    test that wants a face inside the model says so here rather than by drawing
    one - which is the whole reason the question is asked after meshing.

    Each face is bounded by one closed curve of its own, and the wall's face
    holds every port's curve besides, so a port stands on one guide with the
    wall round it. :func:`cross_section` draws a port's face otherwise. Every
    face stands on the first region, which is the one a guide has, and every
    port's face is measured as a WR-42 end, flat along its own axis.
    """
    where = {}
    region = tuple(one.label for one in described.regions[:1])
    ends = {feed.label: feed.inward for feed in described.feeds if feed.inward is not None}
    rows = [*described.labelled, (described.wall, 2, ())]
    # A body of metal is handed over filled and comes back as its faces.
    leaving = {conductor.label for conductor in described.conductors if conductor.solid}
    for tag, (name, declared, _shapes) in enumerate(rows, start=1):
        if name in absent:
            continue
        dimension = declared - 1 if name in leaving else declared
        if dimension == 3:
            sits = None
        elif name in inside:
            sits = INTERIOR
        elif name in both:
            sits = BOTH
        else:
            sits = FRONTIER
        where[name] = Label(
            dimension=dimension,
            tag=tag,
            entities=(tag,),
            sits=sits,
            beside=region if dimension == 2 else (),
        )
        if name in ends:
            normal = next(axis for axis, part in enumerate(ends[name]) if part)
            sides = iter((10.7, 4.3))
            upper = tuple(0.0 if axis == normal else next(sides) for axis in range(3))
            where[name] = replace(where[name], lower=(0.0, 0.0, 0.0), upper=upper, size=10.7 * 4.3)
    faces = [label.tag for label in where.values() if label.dimension == 2]
    on_face = {face: ((1, OWN_CURVE + face),) for face in faces}
    if described.wall in where:
        wall = where[described.wall].tag
        on_face[wall] = tuple(
            sorted(
                {
                    *on_face[wall],
                    *((1, OWN_CURVE + where[feed.label].tag) for feed in described.feeds),
                }
            )
        )
    return Mesh(
        path="/run/model.msh",
        labels=where,
        worst_quality={},
        edges=Edges(shortest=0.0, longest=0.0),
        settled=(),
        version="",
        algorithm={},
        rims={
            2: on_face,
            1: {OWN_CURVE + face: ((0, OWN_POINT + face),) for face in faces},
        },
        parted=parted,
    )


#: Where the curves and points :func:`meshed` gives each face are numbered
#: from, clear of every number :func:`cross_section` is handed.
OWN_CURVE = 900
OWN_POINT = 9000


def cross_section(described, port, faces, ends, held, left=None, **kwargs):
    """The mesh, with ``port`` standing on ``faces`` - each face and the curves
    on it - and each curve in ``ends`` ending on the points given.

    ``held`` names the labels whose faces hold which of those curves, the wall
    among them, which is what says the metal meets the port there. ``left``
    names the points the fragmenting left standing on a face of the port.
    """
    mesh = meshed(described, **kwargs)
    labels = dict(mesh.labels)
    labels[port] = replace(labels[port], entities=tuple(faces))
    on_face = dict(mesh.rims[2])
    for face, curves in faces.items():
        on_face[face] = tuple(
            sorted({*((1, c) for c in curves), *((0, p) for p in (left or {}).get(face, ()))})
        )
    for name, curves in held.items():
        first = labels[name].entities[0]
        on_face[first] = tuple(sorted({*on_face[first], *((1, c) for c in curves)}))
    on_curve = {**mesh.rims[1], **{c: tuple((0, p) for p in on) for c, on in ends.items()}}
    return replace(mesh, labels=labels, rims={2: on_face, 1: on_curve})


#: A port's face with a curve across it: the corners run round from the
#: bottom left, and the curve from the middle of the left side to the middle of
#: the right divides it into the part below and the part above.
BELOW, ABOVE = 11, 12
ACROSS = 103
HALVES = {BELOW: (101, 102, ACROSS, 104), ABOVE: (ACROSS, 105, 106, 107)}
HALVES_END = {
    101: (1001, 1002),
    102: (1002, 1006),
    ACROSS: (1005, 1006),
    104: (1001, 1005),
    105: (1003, 1006),
    106: (1003, 1004),
    107: (1004, 1005),
}
HALVES_ROUND = (101, 102, 104, 105, 106, 107)

#: One face with its four sides round it, and a curve standing in it that
#: touches none of them: a strip clear of the walls.
WHOLE = 11
ROUND = (101, 102, 103, 104)
ROUND_END = {101: (1001, 1002), 102: (1002, 1003), 103: (1003, 1004), 104: (1001, 1004)}
STRIP = 105


def with_a_sheet(material=None):
    """The guide with a sheet of metal drawn in it, as the problem describes it."""
    found, _ = guide()
    found.Group = [*found.Group, metal_binding(septum(), material=material)]
    return problem(found)


class TestTheCrossSectionAPortStandsOn:
    """A port's mode is solved on its face alone, with the metal meeting it as
    the walls, so what the metal makes of the face is what the port carries."""

    def test_a_sheet_dividing_a_ports_face_is_what_divides_it(self):
        """Dividing is not refused here: whether more than one part carries a
        wave is the mode run's answer, and this names the sheet for it."""
        described = with_a_sheet()
        mesh = cross_section(
            described,
            "Port1",
            HALVES,
            HALVES_END,
            {described.wall: HALVES_ROUND, "SeptumMetal": (ACROSS,)},
        )
        assert configured(described, mesh, "results").ports
        assert divided_by("Port1", described, mesh) == "'SeptumMetal'"

    def test_a_sheet_of_finite_conductivity_divides_it_as_well(self):
        described = with_a_sheet(material=sheet_metal())
        mesh = cross_section(
            described,
            "Port1",
            HALVES,
            HALVES_END,
            {described.wall: HALVES_ROUND, "SeptumMetal": (ACROSS,)},
        )
        assert divided_by("Port1", described, mesh) == "'SeptumMetal'"

    def test_what_meets_the_face_is_every_label_holding_a_curve_on_it(self):
        """The wall round it and a sheet across it; not a sheet elsewhere."""
        found, _ = guide()
        found.Group = [*found.Group, metal_binding(septum()), metal_binding(septum("Iris"))]
        described = problem(found)
        mesh = cross_section(
            described,
            "Port1",
            HALVES,
            HALVES_END,
            {described.wall: HALVES_ROUND, "SeptumMetal": (ACROSS,)},
        )
        assert meeting("Port1", described, mesh) == frozenset(
            mesh.labels[name].tag for name in (described.wall, "SeptumMetal")
        )

    @staticmethod
    def two_ports_on_halves(described, held):
        """Port1 on the part below the curve across the face and Port2 on the
        part above, the curve held by the labels given besides the wall."""
        mesh = cross_section(
            described,
            "Port1",
            {BELOW: HALVES[BELOW]},
            HALVES_END,
            {described.wall: HALVES_ROUND, **held},
        )
        labels = dict(mesh.labels)
        labels["Port2"] = replace(labels["Port2"], entities=(ABOVE,))
        on_face = {**mesh.rims[2], ABOVE: tuple((1, c) for c in HALVES[ABOVE])}
        return replace(mesh, labels=labels, rims={**mesh.rims, 2: on_face})

    def test_two_ports_meeting_along_a_curve_no_metal_holds_are_refused(self):
        described = problem(guide()[0])
        mesh = self.two_ports_on_halves(described, {})
        with pytest.raises(
            TranslationError,
            match="'Port1' and 'Port2' stand on faces that meet along a curve no metal holds",
        ):
            configured(described, mesh, "results")

    @pytest.mark.parametrize("material", [None, sheet_metal()], ids=["perfect", "finite"])
    def test_two_ports_either_side_of_metal_are_solved(self, material):
        """A sheet along the curve is the wall each port's mode is solved with."""
        described = with_a_sheet(material=material)
        mesh = self.two_ports_on_halves(described, {"SeptumMetal": (ACROSS,)})
        assert configured(described, mesh, "results").ports

    def test_one_port_whose_face_is_cut_in_pieces_meets_nothing(self):
        """A dielectric's interface cuts a face into pieces joined along a curve
        no metal holds, and they are one port's."""
        described = problem(guide()[0])
        mesh = cross_section(described, "Port1", HALVES, HALVES_END, {described.wall: HALVES_ROUND})
        assert configured(described, mesh, "results").ports

    def test_a_port_is_asked_about_every_piece_of_its_face(self):
        """Port2 meets the second piece of Port1's face, along a side the wall
        leaves; the wall holds Port2's other side."""
        described = problem(guide()[0])
        open_side, far_side = 105, 108
        mesh = cross_section(
            described,
            "Port1",
            HALVES,
            HALVES_END,
            {described.wall: (*(c for c in HALVES_ROUND if c != open_side), far_side)},
        )
        labels = dict(mesh.labels)
        labels["Port2"] = replace(labels["Port2"], entities=(13,))
        on_face = {**mesh.rims[2], 13: ((1, open_side), (1, far_side))}
        on_curve = {**mesh.rims[1], far_side: ((0, 1003), (0, 1006))}
        mesh = replace(mesh, labels=labels, rims={2: on_face, 1: on_curve})
        with pytest.raises(TranslationError, match="'Port1' and 'Port2' stand on faces"):
            configured(described, mesh, "results")

    def test_metal_between_faces_of_one_part_is_not_named_as_dividing_it(self):
        """A strip lying along a dielectric's interface stands between two faces
        the interface joins, and divides nothing; the sheet between that part
        and the next is what divides the face."""
        found, _ = guide()
        found.Group = [*found.Group, metal_binding(septum()), metal_binding(septum("Iris"))]
        described = problem(found)
        mesh = cross_section(
            described,
            "Port1",
            {11: (211, 201, 202), 12: (212, 201, 202, 203), 13: (213, 203)},
            {211: (1, 2), 212: (2, 3), 201: (2, 5), 202: (1, 4), 203: (3, 4), 213: (3, 4)},
            {described.wall: (211, 212, 213), "IrisMetal": (202,), "SeptumMetal": (203,)},
        )
        assert divided_by("Port1", described, mesh) == "'SeptumMetal'"

    def test_a_curve_no_metal_holds_joins_the_two_sides(self):
        """A dielectric's interface running across the face is such a curve,
        and the port is one guide filled with two materials."""
        described = with_a_sheet()
        mesh = cross_section(described, "Port1", HALVES, HALVES_END, {described.wall: HALVES_ROUND})
        assert divided_by("Port1", described, mesh) == ""

    def test_faces_no_curve_joins_are_parts_the_region_divides(self):
        """A region notched through to the port's face leaves it in two faces
        that share nothing, and no sheet is to blame."""
        described = problem(guide()[0])
        apart = {BELOW: (101, 102, 103, 104), ABOVE: (105, 106, 107, 108)}
        ends = {
            **ROUND_END,
            105: (1005, 1006),
            106: (1006, 1007),
            107: (1007, 1008),
            108: (1005, 1008),
        }
        mesh = cross_section(
            described, "Port1", apart, ends, {described.wall: tuple(range(101, 109))}
        )
        assert divided_by("Port1", described, mesh) == "the region's own boundary"

    def test_metal_standing_apart_from_the_wall_is_refused_naming_it(self):
        described = with_a_sheet()
        mesh = cross_section(
            described,
            "Port1",
            {WHOLE: (*ROUND, STRIP)},
            {**ROUND_END, STRIP: (1005, 1006)},
            {described.wall: ROUND, "SeptumMetal": (STRIP,)},
        )
        with pytest.raises(TranslationError) as refused:
            configured(described, mesh, "results")
        said = str(refused.value)
        assert "2 pieces" in said
        assert "'SeptumMetal'" in said
        assert "the region's own boundary" in said

    def test_a_hole_in_the_region_is_a_second_conductor_too(self):
        """A post drawn as a hole in the body the field fills is bounded by the
        wall, so the second conductor is the region's own boundary."""
        described = problem(guide()[0])
        hole = 105
        mesh = cross_section(
            described,
            "Port1",
            {WHOLE: (*ROUND, hole)},
            {**ROUND_END, hole: (1005,)},
            {described.wall: (*ROUND, hole)},
        )
        with pytest.raises(TranslationError, match="2 pieces"):
            configured(described, mesh, "results")

    def test_metal_joined_to_the_wall_that_divides_nothing_is_a_ridged_guide(self):
        """A fin off the floor ends on a point of the floor's curve, and the
        port is one guide with one conductor round it."""
        described = with_a_sheet()
        floor, fin = (101, 108), 105
        mesh = cross_section(
            described,
            "Port1",
            {WHOLE: (*floor, 102, 103, 104, fin)},
            {
                **ROUND_END,
                101: (1001, 1007),
                108: (1007, 1002),
                fin: (1007, 1008),
            },
            {described.wall: (*floor, 102, 103, 104), "SeptumMetal": (fin,)},
        )
        assert configured(described, mesh, "results").ports

    def test_each_piece_of_metal_is_named_by_everything_it_is_made_of(self):
        """A fin joined to the wall is one piece with it, and a strip standing
        apart is another, so the refusal names the fin with the wall."""
        found, _ = guide()
        found.Group = [*found.Group, metal_binding(septum()), metal_binding(septum("Iris"))]
        described = problem(found)
        floor, fin = (101, 108), 106
        mesh = cross_section(
            described,
            "Port1",
            {WHOLE: (*floor, 102, 103, 104, fin, STRIP)},
            {
                **ROUND_END,
                101: (1001, 1007),
                108: (1007, 1002),
                fin: (1007, 1008),
                STRIP: (1005, 1006),
            },
            {
                described.wall: (*floor, 102, 103, 104),
                "SeptumMetal": (fin,),
                "IrisMetal": (STRIP,),
            },
        )
        with pytest.raises(TranslationError) as refused:
            configured(described, mesh, "results")
        said = str(refused.value)
        assert "'SeptumMetal' and the region's own boundary" in said
        assert "; 'IrisMetal' -" in said

    def test_metal_apart_from_the_wall_on_any_part_of_the_face_is_refused(self):
        """A dielectric's interface joins the face's two sides into one guide, and
        a strip standing clear of the walls in the upper side is a second
        conductor in it."""
        described = with_a_sheet()
        mesh = cross_section(
            described,
            "Port1",
            {BELOW: HALVES[BELOW], ABOVE: (*HALVES[ABOVE], 108)},
            {**HALVES_END, 108: (1007, 1008)},
            {described.wall: HALVES_ROUND, "SeptumMetal": (108,)},
        )
        with pytest.raises(TranslationError, match="2 pieces"):
            configured(described, mesh, "results")

    def test_every_part_of_a_divided_face_is_asked_for_a_second_conductor(self):
        """A strip clear of the walls in the upper half, over a sheet dividing
        the face: a port on each part would leave the strip in the upper one,
        so the conductor is what is said."""
        found, _ = guide()
        found.Group = [*found.Group, metal_binding(septum()), metal_binding(septum("Iris"))]
        described = problem(found)
        mesh = cross_section(
            described,
            "Port1",
            {BELOW: HALVES[BELOW], ABOVE: (*HALVES[ABOVE], 108)},
            {**HALVES_END, 108: (1007, 1008)},
            {described.wall: HALVES_ROUND, "SeptumMetal": (ACROSS,), "IrisMetal": (108,)},
        )
        with pytest.raises(TranslationError, match="pieces that do not touch there"):
            configured(described, mesh, "results")

    def test_a_tube_crossing_the_face_is_refused_as_the_second_conductor_it_is(self):
        """A tube meeting the face divides it into the ring outside and the disc
        inside, and holds the ring between two conductors. A port on each part
        would leave the ring as it is, so the conductor is what is said."""
        described = with_a_sheet()
        ring, disc, circle = 11, 12, 105
        mesh = cross_section(
            described,
            "Port1",
            {ring: (*ROUND, circle), disc: (circle,)},
            {**ROUND_END, circle: (1005,)},
            {described.wall: ROUND, "SeptumMetal": (circle,)},
        )
        with pytest.raises(TranslationError, match="pieces that do not touch there"):
            configured(described, mesh, "results")

    def test_a_perfect_conductor_touching_the_face_at_a_point_is_a_second_conductor(self):
        """The corner of a sheet standing on the floor touches the face at one
        point, which the solve fixes as it fixes a perfect conductor's curves."""
        described = with_a_sheet()
        corner = 1009
        mesh = cross_section(
            described,
            "Port1",
            {WHOLE: ROUND},
            {**ROUND_END, 110: (corner, 1010), 111: (1010, 1011), 112: (1011, corner)},
            {described.wall: ROUND, "SeptumMetal": (110, 111, 112)},
            left={WHOLE: (corner,)},
        )
        with pytest.raises(TranslationError) as refused:
            configured(described, mesh, "results")
        said = str(refused.value)
        assert "2 pieces" in said and "'SeptumMetal'" in said

    def test_a_finite_conductor_touching_the_face_at_a_point_is_not(self):
        """Its condition is an integral along a curve, and a point adds nothing
        to it, so the solve sees no second conductor there."""
        described = with_a_sheet(material=sheet_metal())
        corner = 1009
        mesh = cross_section(
            described,
            "Port1",
            {WHOLE: ROUND},
            {**ROUND_END, 110: (corner, 1010), 111: (1010, 1011), 112: (1011, corner)},
            {described.wall: ROUND, "SeptumMetal": (110, 111, 112)},
            left={WHOLE: (corner,)},
        )
        assert configured(described, mesh, "results").ports

    def test_a_perfect_conductor_touching_a_curve_no_metal_holds_is_counted_there(self):
        """A corner resting on a dielectric's interface across the face is a
        point of that curve rather than one left standing on the face."""
        described = with_a_sheet()
        corner = 1009
        mesh = cross_section(
            described,
            "Port1",
            {BELOW: (101, 102, 103, 113, 104), ABOVE: (103, 113, 105, 106, 107)},
            {
                **HALVES_END,
                103: (1005, corner),
                113: (corner, 1006),
                110: (corner, 1010),
                111: (1010, 1011),
                112: (1011, corner),
            },
            {described.wall: HALVES_ROUND, "SeptumMetal": (110, 111, 112)},
        )
        with pytest.raises(TranslationError, match="2 pieces"):
            configured(described, mesh, "results")

    def test_a_circle_touching_the_wall_at_its_one_point_is_joined_to_it(self):
        """A closed curve is bounded by one point, and where that point is on
        the wall the metal round it is one conductor with the wall."""
        described = with_a_sheet()
        tube = 105
        mesh = cross_section(
            described,
            "Port1",
            {WHOLE: (*ROUND, tube)},
            {**ROUND_END, tube: (1001,)},
            {described.wall: ROUND, "SeptumMetal": (tube,)},
        )
        assert configured(described, mesh, "results").ports

    def test_a_mesh_that_does_not_say_what_is_on_a_ports_face_is_not_read_as_whole(self):
        """A face missing from the map would otherwise count as one guide with
        nothing on it, which is the answer the check exists to doubt."""
        described = problem(guide()[0])
        mesh = meshed(described)
        port = mesh.labels["Port1"].tag
        bare = {face: on for face, on in mesh.rims[2].items() if face != port}
        with pytest.raises(KeyError, match=f"^{port}$"):
            configured(described, replace(mesh, rims={**mesh.rims, 2: bare}), "results")


class TestTheFilesTheMesherIsAskedToRead:
    def test_every_shape_reaches_a_file_of_its_own(self, tmp_path):
        """One file per shape is what makes the label map exact: the entities an
        import returns are that file's, so nothing is matched by position.
        """
        described = problem(guide()[0])
        pieces, _ = draw(described, tmp_path)
        drawn = sum(len(row[2]) for row in described.labelled)
        assert len(pieces) == drawn
        assert len({piece.file for piece in pieces}) == drawn

    def test_a_piece_names_the_file_the_shape_wrote_itself_into(self, tmp_path):
        described = problem(guide()[0])
        for piece in draw(described, tmp_path)[0]:
            assert Path(piece.file).is_file()

    def test_each_piece_keeps_the_dimension_the_label_states(self, tmp_path):
        described = problem(guide()[0])
        stated = {row[0]: row[1] for row in described.labelled}
        for piece in draw(described, tmp_path)[0]:
            assert piece.dimension == stated[piece.label]

    def test_a_label_that_is_no_file_name_still_writes_inside_the_directory(self, tmp_path):
        """A label is whatever the user typed, and a separator in one would write
        outside the directory that was named.
        """
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.Label = "../Air Fill"
        pieces, _ = draw(problem(found), tmp_path)
        for piece in pieces:
            assert Path(piece.file).parent == tmp_path

    def test_the_directory_is_made_where_there_is_none(self, tmp_path):
        described = problem(guide()[0])
        pieces, _ = draw(described, tmp_path / "run" / "shapes")
        assert Path(pieces[0].file).is_file()

    def test_two_labels_alike_as_file_names_still_reach_different_files(self, tmp_path):
        """A file name keeps what it can of a label and no more, so what makes
        it unique is the ordinal in front of it.
        """
        found, body = guide()
        second = obj("Part::Feature", "Stub", Shape=Shape(name="stub"))
        found.Group.append(
            obj(
                "EMMaterialBinding",
                "Air/Fill",
                Material=dielectric(),
                References=[(second, [])],
            )
        )
        for member in found.Group:
            if kind(member) == "EMMaterialBinding" and member.Label == "AirFill":
                member.Label = "Air Fill"
        pieces, _ = draw(problem(found), tmp_path)
        assert len({piece.file for piece in pieces}) == len(pieces)


class TestWhatIsWrittenBesideTheMesh:
    def test_the_configuration_goes_under_the_name_the_run_is_given(self, tmp_path):
        where = tmp_path / "run" / "wr42"
        path = configure(driven(), where)
        assert path.name == CONFIG_NAME
        assert path.parent == where

    def test_what_lands_on_disk_is_what_the_value_says(self, tmp_path):
        run = driven()
        assert json.loads(configure(run, tmp_path).read_text()) == run.to_dict()


class TestWhatTheMeshSaysAboutTheDrawing:
    def test_each_label_is_written_as_the_attribute_its_group_carries(self):
        """Palace names a region and a condition by a mesh attribute, and an
        attribute is the physical group the mesher put the label into.
        """
        described = problem(guide()[0])
        mesh = meshed(described)
        run = configured(described, mesh, "results")
        assert [material.attributes for material in run.materials] == [
            (mesh.labels["AirFill"].tag,)
        ]
        assert run.perfect_conductor == (mesh.labels["wall"].tag,)
        assert [port.attributes for port in run.ports] == [
            (mesh.labels["Port1"].tag,),
            (mesh.labels["Port2"].tag,),
        ]

    def test_the_run_names_the_mesh_the_mesher_wrote_and_where_its_tables_go(self):
        described = problem(guide()[0])
        run = configured(described, meshed(described), "/run/results")
        assert run.mesh == "/run/model.msh"
        assert run.output == "/run/results"

    def test_what_fills_a_region_reaches_the_attribute_it_became(self):
        found, _ = guide(material=dielectric(Permittivity=2.2, LossTangent=0.009))
        described = problem(found)
        (material,) = configured(described, meshed(described), "results").materials
        assert material.filling.permittivity == 2.2
        assert material.filling.loss_tangent == 0.009

    def test_the_band_the_order_and_which_ports_are_driven_are_carried_through(self):
        found, _ = guide()
        for member in found.Group:
            if kind(member) == "EMSolverPalace":
                member.Order = 3
        described = problem(found)
        run = configured(described, meshed(described), "results")
        assert run.sweep is described.sweep
        assert run.order == 3
        assert sorted(run.excitations) == [1, 2]

    def test_a_feed_carries_its_mode_and_its_reference_plane_into_the_port(self):
        """The document states neither yet, so both are asserted against the
        problem this stage takes rather than against a drawing.
        """
        described = problem(guide()[0])
        moved = replace(
            described,
            feeds=(replace(described.feeds[0], mode=2, offset=1.5), *described.feeds[1:]),
        )
        run = configured(moved, meshed(moved), "results")
        assert (run.ports[0].mode, run.ports[0].offset) == (2, 1.5)

    def test_the_wall_is_asked_for_by_name_and_nothing_is_drawn_for_it(self):
        """It is the mesher's answer to what the rest of the labels left, so the
        adapter reads the group under the name it asked for and hands over no
        face of its own. Nothing here can put it inside the model.
        """
        described = problem(guide()[0])
        mesh = meshed(described)
        assert described.wall not in {row[0] for row in described.labelled}
        assert configured(described, mesh, "results").perfect_conductor == (
            mesh.labels[described.wall].tag,
        )

    def test_a_port_standing_on_a_face_inside_the_model_is_refused(self):
        described = problem(guide()[0])
        with pytest.raises(TranslationError, match="Point it at a face on the outside"):
            configured(described, meshed(described, inside=("Port2",)), "results")

    def test_a_port_partly_inside_the_model_is_refused_too(self):
        """The fragmenting cuts a face into pieces, so a port drawn across the
        interface between two bodies keeps some of the boundary and some of the
        inside. Held apart from the case above because the guard reads one word
        and a label lying partly outside is the way the sound drawing arrives.
        """
        described = problem(guide()[0])
        with pytest.raises(TranslationError, match="Point it at a face on the outside"):
            configured(described, meshed(described, both=("Port2",)), "results")

    def test_every_sheet_of_metal_carries_the_perfect_conductor_beside_the_wall(self):
        """Inside the model as readily as on its boundary: a sheet the region
        runs round on both sides is a wall the field meets from both."""
        found, _ = guide()
        found.Group = [*found.Group, metal_binding(septum()), metal_binding(septum("Iris"))]
        described = problem(found)
        mesh = meshed(described, inside=("SeptumMetal",), both=("IrisMetal",))
        assert sorted(configured(described, mesh, "results").perfect_conductor) == sorted(
            mesh.labels[name].tag for name in ("wall", "SeptumMetal", "IrisMetal")
        )

    def test_a_boundary_drawn_all_as_metal_leaves_no_wall_and_is_still_a_run(self):
        """Where every face of the boundary is a port or a sheet somebody drew,
        the mesher makes no group for the wall, and the metal carries the
        condition alone."""
        found, _ = guide()
        found.Group = [*found.Group, metal_binding(septum())]
        described = problem(found)
        mesh = meshed(described, absent=(described.wall,))
        assert configured(described, mesh, "results").perfect_conductor == (
            mesh.labels["SeptumMetal"].tag,
        )

    def test_a_mesh_made_without_a_sheet_of_metal_is_not_written_without_it(self):
        """Only the wall may be missing from the mesh: a sheet somebody drew
        left out of the condition is an open guide solved as a septum."""
        found, _ = guide()
        found.Group = [*found.Group, metal_binding(septum())]
        described = problem(found)
        with pytest.raises(KeyError, match="SeptumMetal"):
            configured(described, meshed(described, absent=("SeptumMetal",)), "results")

    def test_a_port_on_the_frontier_is_what_the_run_is_built_from(self):
        described = problem(guide()[0])
        mesh = meshed(described)
        assert all(mesh.labels[feed.label].sits == FRONTIER for feed in described.feeds)
        assert configured(described, mesh, "results").ports


def stands_in(tmp_path, says="", status=0, name="palace"):
    """A Palace that prints what this test wants read and exits like that.

    Every fault below is a clean exit and a full set of tables, so what is being
    exercised is the reading of the log rather than the status.
    """
    script = tmp_path / name
    # A quoted here-document prints the text as it stands, quotes and all.
    printed = f"cat <<'SAID'\n{says}\nSAID\n" if says else ""
    script.write_text(f'#!/bin/sh\necho "$@"\n{printed}exit {status}\n', encoding="utf-8")
    script.chmod(0o755)
    return script


#: What Palace prints ahead of a warning, with the colour it writes whether or
#: not anything is reading it as a terminal.
WARNED = "\033[38;2;255;255;000m--> Warning!\033[0m"

#: A frequency's solve that stopped short of the tolerance, and then the solve
#: behind the error estimate stopping short, as a Palace 0.18.0 driven run
#: printed them.
STOPPED_SHORT = f"""\
GMRES solver did NOT converge in 100 iterations (avg. reduction factor: 8.067e-01)

{WARNED}
Linear solver did not converge, norm(Ax-b)/norm(b) = 4.713e-10 (norm(b) = 3.466e+02)!

 Sol. ||E|| = 4.639185e+01 (||RHS|| = 2.205210e+01)
 Field energy E (9.899e-11 J) + H (9.899e-11 J) = 1.980e-10 J
 S[1][2] = -5.722e-01+8.201e-01i, |S[1][2]| = -3.454e-07, arg(S[1][2]) = +1.249e+02
 S[2][2] = +8.879e-07+1.776e-05i, |S[2][2]| = -9.500e+01, arg(S[2][2]) = +8.714e+01
 Updating solution error estimates
PCG solver did NOT converge in 10000 iterations (avg. reduction factor: 9.989e-01)

{WARNED}
Linear solver did not converge, norm(Ax-b)/norm(b) = 1.367e-05 (norm(b) = 5.101e-01)!


Completed 0 iterations of adaptive mesh refinement (AMR):
"""


def estimated(cap):
    """The estimate's solve stopping short at ``cap`` iterations, in the words
    of ``palace/linalg/iterative.cpp`` (``CgSolver::Mult``) and
    ``palace/linalg/ksp.cpp`` (``BaseKspSolver::Mult``)."""
    return (
        f" Updating solution error estimates\n"
        f"PCG solver did NOT converge in {cap} iterations (avg. reduction factor: 9.989e-01)\n"
        f"\n{WARNED}\n"
        "Linear solver did not converge, norm(Ax-b)/norm(b) = 1.367e-05 (norm(b) = 5.101e-01)!\n"
        "\n"
    )


def warned(*lines, ended=True):
    """A Palace warning as ``palace/utils/communication.hpp`` prints it: a blank
    line follows only a message whose format string ends in a newline."""
    return f"\n{WARNED}\n" + "\n".join(lines) + ("\n\n" if ended else "\n")


def mfem_warned(message, function, where):
    """An MFEM warning as ``mfem/general/error.hpp`` composes it."""
    return f"MFEM Warning: {message}\n ... in function: {function}\n ... in file: {where}\n\n"


class TestFindingTheSolver:
    def test_a_solver_that_is_there_is_the_one_used(self, tmp_path):
        assert find_solver(explicit=stands_in(tmp_path)) == stands_in(tmp_path)

    @pytest.mark.parametrize("written", ["palace", "./palace", "bin/../palace"])
    def test_a_relative_path_comes_back_naming_the_same_file_from_anywhere(
        self, tmp_path, monkeypatch, written
    ):
        """A run starts Palace in the run's own directory, where a relative path
        names another file or none, and a bare name is looked for on PATH."""
        stands_in(tmp_path)
        (tmp_path / "bin").mkdir()
        monkeypatch.chdir(tmp_path)
        found = find_solver(explicit=written)
        assert found.is_absolute()
        assert found.resolve() == (tmp_path / "palace").resolve()

    def test_a_relative_environment_setting_comes_back_absolute(self, tmp_path, monkeypatch):
        stands_in(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(BINARY_ENV_VAR, "palace")
        assert find_solver() == tmp_path / "palace"

    def test_the_environment_is_consulted_after_the_setting(self, tmp_path, monkeypatch):
        monkeypatch.setenv(BINARY_ENV_VAR, str(stands_in(tmp_path)))
        assert find_solver() == stands_in(tmp_path)

    def test_a_path_that_is_not_a_program_is_not_a_solver(self, tmp_path, monkeypatch):
        """A name that exists and cannot be run fails at the start of a run, on
        a message about a file rather than about a missing solver.
        """
        text = tmp_path / "palace"
        text.write_text("not a program", encoding="utf-8")
        monkeypatch.setenv(BINARY_ENV_VAR, str(text))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(SolverNotFound, match="Tried"):
            find_solver()

    def test_what_was_tried_is_named(self, tmp_path, monkeypatch):
        monkeypatch.setenv(BINARY_ENV_VAR, str(tmp_path / "from-the-environment"))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(SolverNotFound) as raised:
            find_solver()
        assert "from-the-environment" in str(raised.value)
        assert BINARY_ENV_VAR in str(raised.value)

    def test_a_setting_that_names_no_program_is_refused_rather_than_searched_past(
        self, tmp_path, monkeypatch
    ):
        """A Palace found elsewhere answers in place of the one the user chose,
        and says nothing about it."""
        monkeypatch.setenv(BINARY_ENV_VAR, str(stands_in(tmp_path)))
        with pytest.raises(SolverNotFound, match="asked-for"):
            find_solver(explicit=tmp_path / "asked-for")


class TestFindingTheMPILauncher:
    """Palace's launcher looks for ``mpirun`` on ``PATH`` only once a run
    starts, after the version has been read without it."""

    def program(self, where):
        where.mkdir(parents=True, exist_ok=True)
        path = where / "mpirun"
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_a_named_launcher_is_the_one_used(self, tmp_path):
        named = self.program(tmp_path / "mpi")
        assert find_launcher(named) == named

    def test_a_relative_name_comes_back_absolute(self, tmp_path, monkeypatch):
        """A run starts in the run's own directory, where a relative path names
        another file or none."""
        self.program(tmp_path / "mpi")
        monkeypatch.chdir(tmp_path)
        assert find_launcher("mpi/mpirun") == tmp_path / "mpi" / "mpirun"

    def test_a_blank_setting_finds_mpirun_on_the_path(self, tmp_path, monkeypatch):
        found = self.program(tmp_path / "bin")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        assert find_launcher() == found

    def test_a_path_without_mpirun_is_refused_naming_it(self, tmp_path, monkeypatch):
        """What a FreeCAD started from Finder or the Dock is given."""
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(SolverNotFound) as raised:
            find_launcher()
        assert str(tmp_path / "empty") in str(raised.value)

    def test_with_no_path_at_all_the_path_searched_is_the_one_named(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PATH")
        monkeypatch.setattr(run_module.os, "defpath", str(tmp_path / "default"))
        with pytest.raises(SolverNotFound) as raised:
            find_launcher()
        assert str(tmp_path / "default") in str(raised.value)

    def test_a_setting_that_names_no_program_is_refused_rather_than_searched_past(
        self, tmp_path, monkeypatch
    ):
        self.program(tmp_path / "bin")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        with pytest.raises(SolverNotFound, match="asked-for"):
            find_launcher(tmp_path / "asked-for")

    @pytest.mark.parametrize("folder", ["open mpi", "mpi*", "mpi?", "mpi[4]"])
    @pytest.mark.parametrize("named", [True, False], ids=["named", "searched"])
    def test_a_launcher_whose_path_the_shell_would_split_or_match_is_refused(
        self, tmp_path, monkeypatch, named, folder
    ):
        """Palace's launcher expands the path it is given unquoted."""
        found = self.program(tmp_path / folder)
        monkeypatch.setenv("PATH", str(tmp_path / folder))
        with pytest.raises(SolverNotFound, match="pattern"):
            find_launcher(found if named else None)


class TestWhichPalaceIsRun:
    """What ``palace --version`` says, and which answers this adapter runs on.

    A release is read off the tag ``git describe`` found at the build, which is
    all the program states about itself.
    """

    @pytest.mark.parametrize(
        "line",
        [
            "Palace version: v0.18.1",
            "Palace version: v0.18.1-dirty",
            "Palace version: v0.18.1-1-gdc0e0a9ef",
            "Palace version: v0.19.2",
            "Palace version: v1.0.0",
        ],
    )
    def test_a_release_at_or_past_the_floor_is_run(self, tmp_path, line):
        assert supported(stands_in(tmp_path, says=line)) == line

    @pytest.mark.parametrize(
        "line",
        [
            "Palace version: v0.18.0-1-gee35dbd6b",
            "Palace version: v0.17.0-dirty",
            "Palace version: v0.17.0-270-ged3959024",
            "Palace version: v0.9.99",
        ],
    )
    def test_an_older_release_is_refused_naming_what_it_said_and_the_floor(self, tmp_path, line):
        with pytest.raises(SolverUnsupported) as refused:
            supported(stands_in(tmp_path, says=line))
        assert line in str(refused.value)
        assert ".".join(str(part) for part in FLOOR) in str(refused.value)

    def test_the_floor_compares_by_number_rather_than_by_text(self, tmp_path):
        """As text, 0.9 sorts after 0.18."""
        with pytest.raises(SolverUnsupported):
            supported(stands_in(tmp_path, says="Palace version: v0.9.0"))
        assert supported(stands_in(tmp_path, says="Palace version: v0.100.0"))

    @pytest.mark.parametrize("line", ["Palace version: UNKNOWN", "Palace version: ee35dbd6b"])
    def test_a_build_that_names_no_release_is_refused(self, tmp_path, line):
        """Where git found no tag Palace states a bare commit, and where it
        found no git, UNKNOWN - and whether that build carries what the floor
        is for cannot then be told."""
        with pytest.raises(SolverUnsupported, match="names no release"):
            supported(stands_in(tmp_path, says=line))

    def test_the_launchers_own_line_ahead_of_the_answer_is_passed_over(self, tmp_path):
        """The launcher names the program it runs, and the argument it passes
        on is the word the answer is looked for by."""
        said = f">> {tmp_path}/palace-arm64.bin --version\n\n{STATED}\nSchema version: 1-7-0"
        assert supported(stands_in(tmp_path, says=said)) == STATED

    def test_it_is_asked_without_mpi(self, tmp_path):
        """``--serial`` runs the program without ``mpirun``, and the launcher
        then writes no node file where a batch scheduler's variables are set."""
        stand_in = tmp_path / "palace"
        stand_in.write_text(f'#!/bin/sh\necho "$@" > {tmp_path / "asked"}\necho "{STATED}"\n')
        stand_in.chmod(0o755)
        version(stand_in)
        assert (tmp_path / "asked").read_text().split() == ["--serial", "--version"]

    def test_an_answer_with_no_version_in_it_is_refused_quoting_it(self, tmp_path):
        with pytest.raises(SolverUnsupported, match="Could not locate"):
            version(stands_in(tmp_path, says="Error: Could not locate Palace executable"))

    def test_a_program_that_cannot_be_started_is_refused(self, tmp_path):
        with pytest.raises(SolverUnsupported, match="could not be started"):
            version(tmp_path / "nothing-here")

    def test_a_launcher_that_does_not_answer_is_stopped_with_everything_it_started(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("Microwave.Solvers.palace.run.VERSION_TIMEOUT", 3.0)
        monkeypatch.setattr("Microwave.Solvers.palace.run.GRACE", 0.5)
        child = tmp_path / "child"
        stand_in = tmp_path / "palace"
        stand_in.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {child}\nwait\n")
        stand_in.chmod(0o755)
        began = time.monotonic()
        with pytest.raises(SolverUnsupported, match="had not answered"):
            version(stand_in)
        # Well short of the child's own minute: a stop that missed it waits it out.
        assert time.monotonic() - began < 10.0
        pid = int(child.read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        pytest.fail(f"the launcher's child {pid} outlived the question")

    def test_an_answer_whose_launcher_left_a_child_on_the_pipe_is_taken(
        self, tmp_path, monkeypatch
    ):
        """The answer ends where the pipe does, and a child that inherited it
        holds it open after the launcher has gone. The child is stopped and
        what was said before it is read."""
        monkeypatch.setattr("Microwave.Solvers.palace.run.VERSION_TIMEOUT", 3.0)
        monkeypatch.setattr("Microwave.Solvers.palace.run.GRACE", 0.5)
        child = tmp_path / "child"
        stand_in = tmp_path / "palace"
        stand_in.write_text(f'#!/bin/sh\necho "{STATED}"\nsleep 60 &\necho $! > {child}\n')
        stand_in.chmod(0o755)
        began = time.monotonic()
        assert version(stand_in) == STATED
        assert time.monotonic() - began < 10.0
        pid = int(child.read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        pytest.fail(f"the launcher's child {pid} outlived the answer")

    def test_an_answer_no_encoding_can_spell_is_read_past(self, tmp_path):
        stand_in = tmp_path / "palace"
        stand_in.write_text(f"#!/bin/sh\nprintf '>> \\377 %s\\n' \"$*\"\necho '{STATED}'\n")
        stand_in.chmod(0o755)
        assert version(stand_in) == STATED

    def test_a_child_that_left_the_session_is_not_waited_on(self, tmp_path, monkeypatch):
        """Nothing here can signal it, and it holds the pipe the answer ends on."""
        monkeypatch.setattr("Microwave.Solvers.palace.run.VERSION_TIMEOUT", 3.0)
        monkeypatch.setattr("Microwave.Solvers.palace.run.GRACE", 0.5)
        stand_in = tmp_path / "palace"
        stand_in.write_text(
            f"#!/bin/sh\n{sys.executable} -c 'import os, time; os.setsid(); time.sleep(20)' &\n"
        )
        stand_in.chmod(0o755)
        began = time.monotonic()
        with pytest.raises(SolverUnsupported, match="had not answered"):
            version(stand_in)
        assert time.monotonic() - began < 10.0

    def test_a_launcher_that_ignores_the_stop_is_killed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("Microwave.Solvers.palace.run.VERSION_TIMEOUT", 3.0)
        monkeypatch.setattr("Microwave.Solvers.palace.run.GRACE", 0.5)
        stand_in = tmp_path / "palace"
        stand_in.write_text("#!/bin/sh\ntrap '' TERM\nsleep 60\n")
        stand_in.chmod(0o755)
        began = time.monotonic()
        with pytest.raises(SolverUnsupported, match="had not answered"):
            version(stand_in)
        assert time.monotonic() - began < 10.0


#: What Palace 0.18.0 printed once an adaptive sweep of a WR-90 filter had
#: converged, copied from the log whole.
CONVERGED = """\
Adaptive sampling converged with 9 frequency samples:
 n = 18, error = 3.918e-05, tol = 1.000e-02, memory = 2/2
 Sampled frequencies (GHz): 8.000e+00, 1.200e+01, 1.084e+01, 9.834e+00,
                            1.057e+01, 1.030e+01, 8.648e+00, 1.045e+01,
                            1.157e+01
 Sample errors: inf, inf, 7.616e-01, 7.337e-01, 8.652e-01,
                1.002e-01, 3.686e-02, 1.038e-04, 3.918e-05
 Total offline phase elapsed time: 1.86e+03 s
"""

#: What it printed where a sweep of a resonator in WR-42 took the four full
#: solves it was allowed with a tolerance of 1e-12, copied from the log whole.
REACHED = """\
Adaptive sampling reached maximum 4 frequency samples:
 n = 8, error = 3.471e-01, tol = 1.000e-12, memory = 0/2
 Sampled frequencies (GHz): 2.000e+01, 2.600e+01, 2.189e+01, 2.423e+01
 Sample errors: inf, inf, 1.089e+00, 3.471e-01

Adding excitation index 2 (2/2):
"""


#: The same where the sweep converged on the last solve it was allowed. The words
#: ahead of the count say only whether it reached the cap
#: (``palace/drivers/drivensolver.cpp``, ``SweepAdaptive``).
CONVERGED_AT_THE_CAP = CONVERGED.replace("converged with", "reached maximum")


def adaptive_guide(driven_ports=(1, 2)):
    """The guide, swept adaptively, with the ports ``driven_ports`` driven."""
    ports = tuple(replace(port, excited=port.index in driven_ports) for port in WR42["ports"])
    return driven(ports=ports, sweep=Sweep(8e9, 12e9, 41, adaptive=Adaptive(1e-2, 20)))


class TestHowAnAdaptiveSweepSampled:
    def test_the_summary_is_read_as_palace_printed_it(self):
        (one,) = run_module.sampled(f"It 1/41\n{CONVERGED}Done\n")
        assert one.converged
        assert (one.solves, one.error, one.tolerance) == (9, 3.918e-05, 1e-2)
        assert len(one.frequencies) == one.solves
        assert (min(one.frequencies), max(one.frequencies)) == (8e9, 12e9)

    def test_a_sweep_that_took_every_solve_is_read_as_not_converged(self):
        (one,) = run_module.sampled(REACHED)
        assert not one.converged

    def test_a_sweep_that_converged_on_its_last_solve_is_read_as_converged(self):
        (one,) = run_module.sampled(CONVERGED_AT_THE_CAP)
        assert one.converged

    def test_each_excitation_is_read_in_the_order_it_was_run(self):
        read = run_module.sampled(CONVERGED + REACHED)
        assert [one.converged for one in read] == [True, False]

    def test_a_discrete_sweep_says_nothing_of_sampling(self):
        assert run_module.sampled("It 1/41\nDone\n") == ()

    def test_each_driven_port_is_said_with_its_tolerance_and_where_it_was_solved(self):
        said = pipeline._swept(adaptive_guide((1,)), CONVERGED, {1: "WGPort1", 2: "WGPort2"})
        assert len(said) == 1
        assert said[0].startswith("'WGPort1': ")
        assert "within 0.01" in said[0]
        assert "after 9 full solves" in said[0]
        assert "8, 8.648, 9.834, 10.3, 10.45, 10.57, 10.84, 11.57, 12 GHz" in said[0]

    def test_a_port_whose_sweep_took_every_solve_fails_the_run_by_name(self):
        with pytest.raises(run_module.SolverFailed) as failed:
            pipeline._swept(
                adaptive_guide((1, 2)), CONVERGED + REACHED, {1: "WGPort1", 2: "WGPort2"}
            )
        message = str(failed.value)
        assert message.startswith("'WGPort2': ")
        assert f"all 4 full solves it is allowed before {CONVERGENCE_MEMORY} in a row" in message
        assert "tolerance 1e-12" in message
        assert "the last being 0.35 off it" in message
        assert "Raise SweepSolves or SweepTolerance" in message
        assert "set its Sweep to Discrete" in message

    def test_a_port_the_log_says_nothing_of_fails_the_run(self):
        """The model's error there is not known, and the table cannot say."""
        with pytest.raises(run_module.SolverFailed, match="1 of the 2 driven ports"):
            pipeline._swept(adaptive_guide((1, 2)), CONVERGED, {1: "WGPort1", 2: "WGPort2"})

    def test_a_discrete_sweep_is_judged_on_nothing_here(self):
        assert pipeline._swept(driven(), "", {}) == []


class TestWhereTheReducedModelIsChecked:
    """The points solved again are the ones where a model is likeliest to be
    wrong in a way that matters, and never one it was built on."""

    BAND = np.linspace(8e9, 12e9, 41)

    def answer(self, dip_at=None):
        magnitudes = np.full((self.BAND.size, 2, 1), 0.5)
        if dip_at is not None:
            magnitudes[int(np.argmin(np.abs(self.BAND - dip_at))), 0, 0] = 1e-4
        return scattered(magnitudes, np.zeros_like(magnitudes), frequency=self.BAND)

    def test_the_lowest_response_and_the_widest_gap_are_chosen(self):
        chosen = reduced.where(self.answer(dip_at=10.3e9), [(8e9, 10e9, 12e9)])
        assert chosen == pytest.approx((9e9, 10.3e9), rel=0, abs=1.0)

    def test_two_points_are_checked_where_the_lowest_is_also_the_widest(self):
        chosen = reduced.where(self.answer(dip_at=9e9), [(8e9, 10e9, 12e9)])
        assert len(chosen) == len(set(chosen)) == 2
        assert 9e9 in chosen

    def test_a_point_solved_in_full_is_never_chosen(self):
        chosen = reduced.where(self.answer(dip_at=10e9), [(8e9, 10e9, 12e9)])
        assert all(abs(point - 10e9) > reduced.SOLVED_NEAR * 0.1e9 for point in chosen)

    def test_a_point_palace_stated_to_four_figures_counts_as_solved(self):
        """10.84 GHz as Palace prints it is anywhere up to 5 MHz off the
        frequency it solved, which is more than a tenth of this band's
        spacing."""
        band = 10.8449e9 + 1e6 * np.arange(-20, 21)
        magnitudes = np.full((band.size, 2, 1), 0.5)
        magnitudes[20, 0, 0] = 1e-4
        answer = scattered(magnitudes, np.zeros_like(magnitudes), frequency=band)
        chosen = reduced.where(answer, [(band[0], 10.84e9, band[-1])])
        assert 10.8449e9 not in chosen

    def test_a_point_is_open_while_any_driven_port_left_it_unsolved(self):
        chosen = reduced.where(self.answer(), [(8e9, 12e9), tuple(self.BAND)])
        assert chosen == pytest.approx((8.1e9, 10e9), rel=0, abs=1.0)

    def test_a_band_solved_at_every_point_is_not_checked(self):
        assert reduced.where(self.answer(), [tuple(self.BAND)]) == ()

    def test_two_points_are_one_block_that_lands_on_both(self):
        written = reduced.sweep((9e9, 10.3e9)).to_dict()
        assert (written["MinFreq"], written["MaxFreq"], written["NSample"]) == (9.0, 10.3, 2)

    def test_one_point_is_a_band_of_one_frequency(self):
        assert reduced.sweep((9e9,)).to_dict() == {"Type": "Point", "Freq": [9.0]}


class TestWhatTheCheckOfTheReducedModelSays:
    def model(self):
        return scattered(
            [[[0.5], [0.01]], [[0.5], [0.02]]],
            np.zeros((2, 2, 1)),
            frequency=(9e9, 10e9),
        )

    def full(self, off):
        return scattered([[[0.5], [0.02 + off]]], np.zeros((1, 2, 1)), frequency=(10e9,))

    def test_the_largest_difference_is_found_with_its_place_and_its_term(self):
        checked = reduced.compared(self.model(), self.full(3e-4), 1.0)
        assert checked.gap == pytest.approx(3e-4, rel=1e-9)
        assert (checked.at, checked.term) == (10e9, (2, 1))

    def test_the_bar_is_a_share_of_the_smallest_response_the_study_reads(self):
        checked = reduced.compared(self.model(), self.full(3e-4), 0.01)
        assert checked.bar == pytest.approx(WANTED * 0.01, rel=1e-12)
        assert not checked.within
        assert reduced.compared(self.model(), self.full(3e-4), 1.0).within

    def test_a_model_off_by_more_than_the_bar_is_told_what_to_change(self):
        beyond = reduced.said(reduced.compared(self.model(), self.full(3e-4), 0.01))
        within = reduced.said(reduced.compared(self.model(), self.full(3e-4), 1.0))
        assert "SweepTolerance" in beyond and "SweepTolerance" not in within
        assert "-40 dB" in beyond


class TestWhatALogMeansForTheAnswer:
    def test_a_run_that_said_nothing_wrong_complains_about_nothing(self):
        assert complaints("Palace\nDone\n") == []

    def test_a_warning_is_a_failure_and_carries_what_it_said(self):
        """A condition on an attribute the mesh does not carry is dropped, and
        this is the whole of what is said about it, the attributes included
        (``palace/models/spaceoperator.cpp``, as a Palace run printed it)."""
        log = (
            warned(
                "Unknown PEC boundary attributes!",
                "Solver will just ignore them!",
                "Boundary attribute list: 9",
            )
            + "Done\n"
        )
        assert complaints(log) == [
            "Palace warned: Unknown PEC boundary attributes! Solver will just ignore them! "
            "Boundary attribute list: 9"
        ]
        assert remarks(log) == []

    def test_a_frequency_solved_short_of_the_tolerance_fails_and_the_estimate_does_not(self):
        """Palace words both the same, and the line ahead of each names the solve."""
        (failed,) = complaints(STOPPED_SHORT)
        assert "4.713e-10" in failed
        (stated,) = remarks(STOPPED_SHORT)
        assert "1.367e-05" in stated
        assert "The answer stands, because the solve that stopped short" in stated

    def test_the_projection_of_a_lumped_ports_excitation_stopped_short_fails(self):
        """It is conjugate gradient too, and stops at a cap of its own."""
        assert complaints(estimated(200))
        assert remarks(estimated(200)) == []

    def test_the_estimate_is_known_by_the_cap_every_run_writes_for_it(self):
        for written in (
            driven().to_dict(),
            driven().port_modes(1, 26e9, 3, "modes/1", frozenset()),
        ):
            cap = written["Solver"]["Linear"]["EstimatorMaxIts"]
            assert cap == ESTIMATOR_ITERATIONS
            assert complaints(estimated(cap)) == []
            assert remarks(estimated(cap))

    def test_a_warning_after_which_the_answer_stands_is_counted_and_said_once(self):
        """The estimate is made at every frequency."""
        said = estimated(ESTIMATOR_ITERATIONS) + " Sol. ||E||\n" + estimated(ESTIMATOR_ITERATIONS)
        (stated,) = remarks(said)
        assert stated.startswith("Palace warned 2 times, first: Linear solver did not converge")

    def test_the_sweep_interpolation_meeting_a_rank_deficient_matrix_fails(self):
        """It moves where the sweep solves in full next, and a sweep whose samples
        fall where the model already agrees can stop with the model off between
        them (``palace/models/romoperator.cpp``, ``ComputeMRI``)."""
        dropped = [
            warned(
                "Minimal rational interpolation encountered a rank-deficient matrix: "
                f"σ[{m}] = 1.000e-13 (σ[0] = 1.000e+00). This can indicate that the "
                "adaptive interpolation is near the accuracy limit of the HDM solves; "
                "if adaptive convergence is poor, try tightening "
                'config["Solver"]["Linear"]["Tol"] or using a looser '
                'config["Solver"]["Driven"]["AdaptiveTol"].'
            )
            for m in (3, 2)
        ]
        said = "".join(dropped)
        assert len(complaints(said)) == 2
        assert remarks(said) == []

    def test_the_boundary_terms_assembled_in_full_leave_the_answer_standing(self):
        """``palace/models/romoperator.cpp``, ``SolvePROM``."""
        said = warned(
            "Factored online A2 (farfield ABC, surface conductivity, rational "
            "impedance, Floquet Robin) disagrees with the full operator "
            "(rel. err 1.000e-06)!",
            "Reverting to the per-frequency assembled A2 for the remaining sweep. "
            "This indicates an ω-dependent boundary condition not covered by the "
            "factored path.",
        )
        assert complaints(said) == []
        assert remarks(said)

    def test_an_output_folder_not_emptied_fails_the_run(self):
        """The run empties it first, so a table an earlier solve left may be read
        as this one's (``palace/utils/outputdir.hpp``)."""
        said = warned(
            "Output folder is not empty; program will overwrite content! (out)", ended=False
        )
        assert complaints(said)

    def test_an_mfem_warning_fails_the_run_once_however_many_ranks_printed_it(self):
        """More than one rank can print it, and the function it names is a
        signature a user cannot act on."""
        once = mfem_warned(
            'Ignoring "AddToPROM" for non-adaptive simulation!',
            "palace::config::DrivenSolverData::DrivenSolverData(const json &)",
            "palace/utils/configfile.cpp:1240",
        )
        assert complaints(once + once) == [
            'Palace warned: Ignoring "AddToPROM" for non-adaptive simulation!'
        ]

    def test_split_attributes_leave_the_answer_standing_where_no_impedance_is_written(self):
        """``palace/utils/geodata.cpp``, ``AddInterfaceBdrElements``."""
        said = mfem_warned(
            "Found boundary attribute with internal and external boundary elements: 7. "
            "Impedance boundary conditions for these attributes will give erroneous "
            "results, consider separating into different attributes!",
            "int palace::mesh::AddInterfaceBdrElements(IoData &, std::unique_ptr<mfem::Mesh> &, "
            "std::unordered_map<int, int> &, MPI_Comm)",
            "palace/utils/geodata.cpp:3038",
        )
        assert complaints(said) == []
        assert remarks(said)
        written = Path(config_module.__file__).read_text()
        assert '"Impedance"' not in written and '"RationalImpedance"' not in written

    def test_the_colour_palace_writes_does_not_hide_one(self):
        """It styles the marker whether or not anything is reading it as a
        terminal, so a match on the plain words alone would never fire.
        """
        assert complaints(f"{WARNED}\nsomething\n")

    def test_a_region_taken_out_of_the_mesh_is_a_failure_too(self):
        """It is an ordinary line of print rather than a warning, and the region
        it names is simply not in the model that was solved.
        """
        said = complaints("Removed 812 unmarked domain elements from the mesh\n")
        assert len(said) == 1
        assert "812" in said[0]

    def test_a_warning_ends_where_its_message_does(self):
        said = complaints(f"{WARNED}\nfirst line\nsecond line\n\nordinary progress\n")
        assert "second line" in said[0]
        assert "ordinary progress" not in said[0]


#: What SuperLU_DIST prints where MC64 finds a structurally singular matrix,
#: derived from its format strings: ``SRC/double/dldperm_dist.c``
#: (``dldperm_dist``) and ``PrintInt32`` in ``SRC/prec-independent/util.c``,
#: which prints the name right-aligned in ten columns and each entry in ten.
SINGULAR = (
    "MC64 detects singularity .. The last 2 permutations:\n"
    "      perm:\n"
    "\t[ 0- 9]        17        18\n"
)


class TestWhatTheLibrariesSayForTheAnswer:
    """Lines the libraries Palace solves with print themselves, marked by
    neither Palace nor MFEM."""

    def test_a_structurally_singular_matrix_fails_the_run_in_superlus_words(self):
        (failed,) = complaints("Palace\n" + SINGULAR + "Done\n")
        assert failed.startswith(
            "SuperLU said: MC64 detects singularity .. The last 2 permutations: It found"
        )
        assert remarks("Palace\n" + SINGULAR) == []

    def test_its_permutation_dump_is_not_carried(self):
        (failed,) = complaints(SINGULAR)
        assert "perm" not in failed.split("permutations:", 1)[1]

    @pytest.mark.parametrize(
        "line",
        [
            # ``hypre_Memcpy`` and ``hypre_Memset`` in ``src/utilities/memory.c``.
            "hypre_Memcpy warning: copy 64 bytes from 0x0 to 0x600000c3c000 !",
            "hypre_Memset warning: set values for 64 bytes at 0x0 !",
            # A count past two gigabytes, as ``hypre_printf`` in the installed
            # library prints it: as a 32-bit integer.
            "hypre_Memcpy warning: copy -2147483640 bytes from 0x0 to 0x100c44260 !",
            "hypre_Memset warning: set values for -1294967296 bytes at 0x0 !",
        ],
    )
    def test_hypre_skipping_what_it_was_asked_to_do_fails_the_run(self, line):
        (failed,) = complaints(line + "\n")
        assert failed.startswith("hypre said: " + line)

    def test_a_line_printed_on_every_process_is_said_once_with_its_count(self):
        line = "hypre_Memset warning: set values for 64 bytes at 0x0 !"
        (failed,) = complaints(f"{line}\n{line}\n")
        assert failed.startswith(f"hypre said 2 times, first: {line}")

    def test_the_words_are_found_where_another_process_ran_into_the_line(self):
        """Processes share one stream and nothing prefixes their lines."""
        (failed,) = complaints(
            "Sol. ||E|| = 1.0MC64 detects singularity .. The last 1 permutations:\n"
        )
        assert failed.startswith("SuperLU said: MC64 detects singularity")


#: The options PETSc held on a Palace run started the workbench's way, as
#: ``-options_view`` had ``PetscFinalize`` print them.
OWN_OPTIONS = """\
#PETSc Option Table entries:
-options_view # (source: command line)
-skip_petscrc # (source: command line)
#End of PETSc Option Table entries
"""

#: A line of the same list for an option Palace sets itself where its
#: eigensolver prints, derived from ``SlepcEPSSolverBase`` in
#: ``palace/linalg/slepc.cpp`` and ``PetscOptionsView`` in
#: ``src/sys/objects/options.c``.
PALACE_OPTION = "-eps_monitor # (source: code)\n"

#: The same with ``$PETSC_OPTIONS`` set by a file ``$BASH_ENV`` names, which the
#: launcher's shell reads before it runs: one option PETSc read and one it did
#: not.
ENVIRONMENT_OPTIONS = """\
#PETSc Option Table entries:
-ems_probe_unused # (source: environment)
-eps_tol 1e-6 # (source: environment)
-options_view # (source: command line)
-skip_petscrc # (source: command line)
#End of PETSc Option Table entries
"""

#: The same with a ``.petscrc`` in the run's folder and ``-skip_petscrc`` left
#: off.
FILE_OPTIONS = """\
#PETSc Option Table entries:
-eps_view # (source: file)
-options_view # (source: command line)
#End of PETSc Option Table entries
"""


class TestWhatPetscHeld:
    """PETSc reads options from its environment and files as well as from the
    command line, and Palace hands the port mode solve whatever it finds."""

    def test_the_options_a_run_gives_petsc_leave_the_answer_alone(self):
        assert complaints("Palace\n" + OWN_OPTIONS) == []
        assert remarks(OWN_OPTIONS) == []

    def test_an_option_palace_sets_itself_leaves_the_answer_alone(self):
        assert complaints(PALACE_OPTION) == []

    def test_an_option_from_the_environment_fails_the_run_naming_each(self):
        (failed,) = complaints(ENVIRONMENT_OPTIONS)
        assert failed.startswith(
            "PETSc held options this workbench does not write: -ems_probe_unused "
            "from its environment, -eps_tol 1e-6 from its environment. "
        )
        assert "-options_view" not in failed

    def test_an_option_from_a_file_fails_the_run(self):
        (failed,) = complaints(FILE_OPTIONS)
        assert failed.startswith(
            "PETSc held options this workbench does not write: -eps_view from a file. "
        )

    def test_the_list_is_read_where_another_process_ran_into_it(self):
        """Processes share one stream and nothing prefixes their lines."""
        (failed,) = complaints("Sol. ||E|| = 1.0e-05-eps_tol 1e-6 # (source: environment)\n")
        assert ": -eps_tol 1e-6 from its environment." in failed


class TestRunningPalace:
    def test_what_palace_printed_comes_back(self, tmp_path):
        said = solve(tmp_path / "palace.json", 2, solver=stands_in(tmp_path, says="Done"))
        assert "Done" in said

    def test_what_no_encoding_can_spell_is_carried_rather_than_fatal(self, tmp_path):
        """The launcher echoes the path it runs, and a path is whatever the user
        named a folder."""
        stand_in = tmp_path / "palace"
        stand_in.write_text("#!/bin/sh\nprintf '>> \\377 %s\\n' \"$*\"\necho Done\n")
        stand_in.chmod(0o755)
        assert "Done" in solve(tmp_path / "palace.json", 1, solver=stand_in)

    def test_the_run_is_given_the_process_count_and_the_configuration(self, tmp_path):
        """The stand-in echoes its arguments, so what comes back is the command
        line rather than a claim about it. Palace reads the configuration from
        its first argument, so PETSc's options follow it.
        """
        said = solve(tmp_path / "palace.json", 4, solver=stands_in(tmp_path))
        assert said.splitlines()[0] == "-np 4 palace.json -skip_petscrc -options_view"

    def test_a_relative_solver_is_the_one_run_from_the_runs_own_directory(
        self, tmp_path, monkeypatch
    ):
        where = tmp_path / "run"
        where.mkdir()
        stands_in(tmp_path, says="Done")
        monkeypatch.chdir(tmp_path)
        assert "Done" in solve(where / "palace.json", 1, solver="palace")

    def test_the_mpi_launcher_found_is_named_to_palace(self, tmp_path):
        said = solve(
            tmp_path / "palace.json", 4, solver=stands_in(tmp_path), launcher="/bin/mpirun"
        )
        assert said.splitlines()[0] == (
            "-np 4 --launcher /bin/mpirun palace.json -skip_petscrc -options_view"
        )

    def test_a_dry_run_asks_palace_to_read_and_not_to_solve(self, tmp_path):
        """A dry run reads the configuration from its last argument, and runs
        without mpirun."""
        said = validate(tmp_path / "palace.json", solver=stands_in(tmp_path))
        assert said.splitlines()[0] == "--serial --dry-run palace.json"

    @pytest.mark.parametrize("run", [solve, validate], ids=["solve", "dry run"])
    def test_a_folder_whose_name_holds_a_space_reaches_palace_as_one_argument(self, tmp_path, run):
        """Palace's launcher splits its arguments at spaces, and a folder is
        whatever the user named it."""
        where = tmp_path / "a b"
        where.mkdir()
        stand_in = tmp_path / "palace"
        stand_in.write_text('#!/bin/sh\nfor a in "$@"; do echo "[$a]"; done\n')
        stand_in.chmod(0o755)
        args = [1] if run is solve else []
        said = run(where / "palace.json", *args, solver=stand_in)
        assert "[palace.json]" in said.splitlines()

    def test_petsc_is_given_none_of_the_options_its_environment_holds(self, tmp_path, monkeypatch):
        """PETSc reads these as well as its command line, and Palace hands the
        port mode solve whatever it finds. The launcher still finds ``mpirun``
        on the path."""
        petsc = ["PETSC_OPTIONS", "PETSC_OPTIONS_YAML"]
        for name in petsc:
            monkeypatch.setenv(name, "-eps_view")
        stand_in = tmp_path / "palace"
        stand_in.write_text(
            "#!/bin/sh\n"
            + "".join(f'echo "{name}=${{{name}-unset}}"\n' for name in [*petsc, "PATH"])
        )
        stand_in.chmod(0o755)
        said = solve(tmp_path / "palace.json", 1, solver=stand_in).splitlines()
        for name in petsc:
            assert f"{name}=unset" in said
        assert f"PATH={os.environ['PATH']}" in said

    def test_every_line_reaches_a_caller_that_asked_for_them(self, tmp_path):
        """A run takes minutes, so a panel showing progress reads them as they
        arrive rather than at the end.
        """
        seen = []
        solve(
            tmp_path / "palace.json",
            1,
            solver=stands_in(tmp_path, says="a\nb"),
            on_output=seen.append,
        )
        assert "a" in seen and "b" in seen

    def test_a_run_that_ended_badly_is_a_failure(self, tmp_path):
        with pytest.raises(SolverFailed, match="exited with code"):
            solve(tmp_path / "palace.json", 1, solver=stands_in(tmp_path, status=3))

    def test_a_run_that_stopped_saying_nothing_recognised_goes_with_its_last_lines(self, tmp_path):
        """The lines that say something, as many as twelve."""
        says = "\n\n".join(f"line {number}" for number in range(40))
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=stands_in(tmp_path, says=says, status=3))
        assert raised.value.said == tuple(f"line {number}" for number in range(28, 40))
        assert "line 0" in raised.value.log

    def test_a_warning_keeps_the_log_apart_from_the_complaint(self, tmp_path):
        solver = stands_in(tmp_path, says=f"solving\n{WARNED}\nUnknown boundary attribute 9!\n")
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=solver)
        assert "solving" in raised.value.log
        assert not any("solving" in one for one in raised.value.said)

    def test_a_clean_exit_that_warned_is_a_failure_too(self, tmp_path):
        """This is the case the exit status cannot see: a full set of tables
        about a model with a different wall in it.
        """
        solver = stands_in(tmp_path, says=f"{WARNED}\nUnknown boundary attribute 9!\n")
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=solver)
        assert any("attribute 9" in one for one in raised.value.said)

    def test_a_solver_that_is_not_there_is_named(self, tmp_path):
        with pytest.raises(SolverFailed, match="could not be started"):
            solve(tmp_path / "palace.json", 1, solver=tmp_path / "no-such-palace")

    def test_a_warning_carries_what_palace_said_rather_than_a_verdict(self, tmp_path):
        """A dropped condition and a linear solve that ran out of iterations are
        both announced this way and are not the same fault.
        """
        solver = stands_in(
            tmp_path, says=f"{WARNED}\nLinear solver did not converge, norm(Ax-b)/norm(b)\n"
        )
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=solver)
        assert any("did not converge" in one for one in raised.value.said)

    #: What a check that failed on each of two ranks prints, as Palace 0.18.0
    #: printed it for a lumped port given no resistance.
    VERIFIED = (
        "\n"
        "Verification failed: (has_circ || has_surf) is false:\n"
        " --> Lumped port boundary has no R/L/C or Rs/Ls/Cs defined, needs at least one!\n"
        " ... in function: palace::LumpedPortData::LumpedPortData()\n"
        " ... in file: palace/models/lumpedportoperator.cpp:32\n"
        "\n"
    )

    def test_a_check_that_stopped_the_run_is_said_once_without_its_signature(self, tmp_path):
        """Each rank that fails a check prints it, so two ranks print it twice."""
        solver = stands_in(tmp_path, says=f"solving\n{self.VERIFIED}{self.VERIFIED}", status=1)
        with pytest.raises(SolverFailed, match="exited with code 1") as raised:
            solve(tmp_path / "palace.json", 1, solver=solver)
        assert raised.value.said == (
            "Palace stopped: Verification failed: (has_circ || has_surf) is false: "
            "--> Lumped port boundary has no R/L/C or Rs/Ls/Cs defined, needs at least one! "
            "... in file: palace/models/lumpedportoperator.cpp:32",
        )

    def test_a_warning_before_an_abort_is_said_ahead_of_it(self, tmp_path):
        """A schema names the key it refused in a warning, and the abort after it
        says only that validation failed."""
        says = (
            f"{WARNED}\n"
            "At [\"Boundaries\"]: validation failed for additional property 'PCE': "
            "instance invalid as per false-schema\n"
            "\n"
            "MFEM abort: Configuration file validation failed!\n"
            " ... in file: palace/utils/iodata.cpp:212\n"
        )
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=stands_in(tmp_path, says=says, status=1))
        warned, stopped = raised.value.said
        assert "'PCE'" in warned
        assert stopped.startswith("Palace stopped: MFEM abort: Configuration file validation")

    #: The frame Open MPI puts about each message of its own.
    RULE = "-" * 74

    def test_a_rank_ended_by_a_signal_is_said_rather_than_the_frames_about_it(self, tmp_path):
        """A killed rank states no cause of its own, and mpirun says which."""
        signalled = (
            "mpirun noticed that process rank 1 with PID 0 on node fourteen exited on "
            "signal 9 (Killed: 9)."
        )
        says = (
            f"{self.RULE}\nPrimary job  terminated normally, but 1 process returned\n"
            f"{self.RULE}\n{self.RULE}\n{signalled}\n{self.RULE}"
        )
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=stands_in(tmp_path, says=says, status=137))
        assert raised.value.said == (f"Palace stopped: {signalled}",)

    def test_an_exception_nothing_caught_is_said_with_its_own_text(self, tmp_path):
        uncaught = (
            "terminating due to uncaught exception of type std::runtime_error: "
            "what() of the exception"
        )
        says = f"libc++abi: {uncaught}\n{self.RULE}"
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=stands_in(tmp_path, says=says, status=134))
        assert raised.value.said == (f"Palace stopped: {uncaught}",)

    def test_what_open_mpi_refused_is_taken_out_of_its_frame(self, tmp_path):
        """The first paragraph says what went wrong, and the rest is advice. Each
        rank that refuses frames its own, and what follows a frame is not in it."""
        framed = (
            f"{self.RULE}\n"
            "There are not enough slots available in the system to satisfy the 64\n"
            "slots that were requested by the application:\n"
            "\n"
            "  palace-arm64.bin\n"
            "\n"
            "Either request fewer slots for your application, or make more slots\n"
            "available for use.\n"
            f"{self.RULE}\n"
        )
        says = (
            f"{framed}{framed}"
            "[fourteen.local:51474] 1 more process has sent help message help-mpi-api.txt\n"
        )
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=stands_in(tmp_path, says=says, status=1))
        assert raised.value.said == (
            "Open MPI stopped the run: There are not enough slots available in the system "
            "to satisfy the 64 slots that were requested by the application:",
        )

    def test_what_petsc_stopped_on_is_said_in_its_words(self, tmp_path):
        """PETSc brackets its message between its own heading and a pointer to its
        troubleshooting page, and Open MPI's frame about the abort follows."""
        says = (
            "[0]PETSC ERROR: --------------------- Error Message ---------------------\n"
            "[0]PETSC ERROR: Unknown EPS type given: bogus\n"
            "[0]PETSC ERROR: See https://petsc.org/release/faq/ for trouble shooting.\n"
            "[0]PETSC ERROR: #1 EPSSetType() at src/eps/interface/epsbasic.c:172\n"
            f"{self.RULE}\n"
            "MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD\n"
            "with errorcode 86.\n"
            f"{self.RULE}"
        )
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=stands_in(tmp_path, says=says, status=86))
        assert raised.value.said == ("Palace stopped: Unknown EPS type given: bogus",)

    def test_a_table_palace_drew_is_not_taken_for_what_open_mpi_refused(self, tmp_path):
        """Palace draws its timing tables between rules of its own, shorter than
        the frame Open MPI puts about a message."""
        says = (
            "Elapsed Time Report (s)           Min.        Max.        Avg.\n"
            f"{'=' * 62}\n"
            "Initialization                   0.003       0.003       0.003\n"
            f"{'-' * 62}\n"
            "Total                            3.731       3.731       3.731"
        )
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=stands_in(tmp_path, says=says, status=1))
        assert not any("Open MPI" in line for line in raised.value.said)

    def test_a_launcher_ended_by_a_signal_says_which(self, tmp_path):
        solver = tmp_path / "palace"
        solver.write_text("#!/bin/sh\necho solving\nkill -TERM $$\n", encoding="utf-8")
        solver.chmod(0o755)
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=solver)
        assert raised.value.said[0] == "Palace's launcher was ended by signal 15 (SIGTERM)"

    def test_a_status_a_signal_gives_says_which_where_nothing_else_does(self, tmp_path):
        """The launcher is a shell script, and a shell reports a child a signal
        ended as 128 and the signal's number."""
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=stands_in(tmp_path, says="x", status=137))
        assert (
            raised.value.said[0] == "status 137 is how a process ended by signal 9 (SIGKILL) exits"
        )

    def test_a_status_says_nothing_of_a_signal_where_the_signal_is_already_said(self, tmp_path):
        signalled = (
            "mpirun noticed that process rank 1 with PID 0 on node fourteen exited on "
            "signal 9 (Killed: 9)."
        )
        with pytest.raises(SolverFailed) as raised:
            solve(
                tmp_path / "palace.json", 1, solver=stands_in(tmp_path, says=signalled, status=137)
            )
        assert raised.value.said == (f"Palace stopped: {signalled}",)

    def test_a_run_on_no_processes_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="at least one process"):
            solve(tmp_path / "palace.json", 0, solver=stands_in(tmp_path))


class TestARunCanBeStopped:
    """The panel stops a run from another thread, and the installed launcher
    is a shell script that starts ``mpirun`` as a child of its own rather than
    becoming it. So the stand-in here is shaped the same way: a launcher whose
    child does the work and holds the pipe. Signalling the launcher alone ends
    the launcher and leaves the reader blocked on a pipe the child still
    holds."""

    def launcher(self, tmp_path):
        script = tmp_path / "palace-launcher"
        script.write_text(
            "#!/bin/sh\n"
            f"{sys.executable} -c "
            "'import os, time; print(\"pid=%d\" % os.getpid(), flush=True); time.sleep(60)'\n"
            "echo the launcher outlived its child\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return script

    def test_a_stop_reaches_the_process_the_launcher_started(self, tmp_path):
        cancel = Cancellation()
        started = []

        def heard(line):
            if line.startswith("pid="):
                started.append(int(line.split("=")[1]))
                cancel.cancel()

        began = time.monotonic()
        with pytest.raises(Cancelled):
            solve(
                tmp_path / "palace.json",
                1,
                solver=self.launcher(tmp_path),
                on_output=heard,
                cancel=cancel,
            )
        assert time.monotonic() - began < 30.0
        assert started and not _alive(started[0])

    def test_a_stop_reaches_a_run_that_has_stopped_printing(self, tmp_path):
        """Its output ends while it goes on running, and the run then waits on
        it rather than reading."""
        script = tmp_path / "palace-launcher"
        script.write_text("#!/bin/sh\nexec >/dev/null 2>&1\nexec sleep 60\n", encoding="utf-8")
        script.chmod(0o755)
        cancel = Cancellation()
        threading.Timer(1.0, cancel.cancel).start()
        began = time.monotonic()
        with pytest.raises(Cancelled):
            solve(tmp_path / "palace.json", 1, solver=script, cancel=cancel)
        assert time.monotonic() - began < 30.0

    def test_a_run_ends_with_the_process_that_started_it(self, tmp_path):
        """FreeCAD can end without stopping anything, and the child here prints
        nothing more, so no write to a closed pipe ends it either."""
        line = killed_after_first_line(
            "from Microwave.Solvers.palace.run import solve\n"
            f"solve({str(tmp_path / 'palace.json')!r}, 1, "
            f"solver={str(self.launcher(tmp_path))!r}, "
            "on_output=lambda line: print(line, flush=True))\n"
        )
        assert line.startswith("pid=")
        gone_within(int(line.split("=")[1]), 10.0)

    def test_what_a_finished_run_left_running_is_ended(self, tmp_path):
        """The launcher's group outlives the launcher while anything is left
        in it."""
        left = tmp_path / "left.pid"
        script = tmp_path / "palace-launcher"
        script.write_text(
            f"#!/bin/sh\nsleep 60 </dev/null >/dev/null 2>&1 &\necho $! > {left}\necho Done\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        assert "Done" in solve(tmp_path / "palace.json", 1, solver=script)
        gone_within(int(left.read_text()), 10.0)

    def test_what_a_failed_run_left_running_is_ended_while_its_failure_is_held(self, tmp_path):
        """A caller reporting the failure holds it, and with it the frame the
        run was started from."""
        left = tmp_path / "left.pid"
        script = tmp_path / "palace-launcher"
        script.write_text(
            f"#!/bin/sh\nsleep 60 </dev/null >/dev/null 2>&1 &\necho $! > {left}\nexit 3\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        with pytest.raises(SolverFailed) as raised:
            solve(tmp_path / "palace.json", 1, solver=script)
        gone_within(int(left.read_text()), 10.0)
        assert raised.value.__traceback__ is not None

    def test_palace_is_given_no_input(self, tmp_path):
        """mpirun hands its input to the first rank, and a rank that read it
        would wait on the workbench for ever."""
        script = tmp_path / "palace-launcher"
        script.write_text(
            '#!/bin/sh\nif read line; then echo "read $line"; else echo "no input"; fi\n',
            encoding="utf-8",
        )
        script.chmod(0o755)
        cancel = Cancellation()
        said = []

        def run():
            said.append(solve(tmp_path / "palace.json", 1, solver=script, cancel=cancel))

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(10.0)
        if worker.is_alive():
            cancel.cancel()
            worker.join(10.0)
            pytest.fail("Palace was left waiting on its input")
        assert "no input" in said[0]

    def slow_to_go(self, tmp_path, on_term):
        """A launcher whose child holds the output, as mpirun does, and does
        ``on_term`` when signalled. The child's source is quoted for the shell
        in single quotes, so ``on_term`` holds none."""
        script = tmp_path / "palace-launcher"
        script.write_text(
            "#!/bin/sh\n"
            f"{sys.executable} -c "
            "'import os, signal, sys, time\n"
            f"signal.signal(signal.SIGTERM, {on_term})\n"
            'print("pid=%d" % os.getpid(), flush=True)\n'
            "time.sleep(60)'\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return script

    def callback_raising(self, started):
        def heard(line):
            if line.startswith("pid="):
                started.append(int(line.split("=")[1]))
                raise RuntimeError("the caller gave up")

        return heard

    def test_a_caller_that_gave_up_gets_control_back_once_the_run_has_gone(self, tmp_path):
        """mpirun holds the output, and ends the ranks before it ends itself,
        so it is given the time that takes."""
        started = []
        went = tmp_path / "went"
        launcher = self.slow_to_go(
            tmp_path,
            f'lambda *_: (time.sleep(3), open({json.dumps(str(went))}, "w").close(), sys.exit(0))',
        )
        with pytest.raises(RuntimeError):
            solve(
                tmp_path / "palace.json",
                1,
                solver=launcher,
                on_output=self.callback_raising(started),
            )
        gone_within(started[0], 1.0)
        assert went.exists()

    def test_a_run_being_stopped_still_ends_with_the_process_that_started_it(self, tmp_path):
        """The child here goes on the second SIGTERM only: the stop sends the
        first, and the starter dies while the child is still going."""
        count = tmp_path / "count"
        launcher = self.slow_to_go(
            tmp_path,
            f"lambda *_: sys.exit(0) if os.path.exists({json.dumps(str(count))}) "
            f'else open({json.dumps(str(count))}, "w").close()',
        )
        starter = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys\n"
                f"sys.path.insert(0, {str(Path(Microwave.__file__).parents[1])!r})\n"
                "from Microwave.Solvers.cancellation import Cancellation\n"
                "from Microwave.Solvers.palace.run import solve\n"
                "cancel = Cancellation()\n"
                "def heard(line):\n"
                "    print(line, flush=True)\n"
                "    cancel.cancel()\n"
                f"solve({str(tmp_path / 'palace.json')!r}, 1, solver={str(launcher)!r}, "
                "on_output=heard, cancel=cancel)\n",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert starter.stdout is not None
        line = starter.stdout.readline()
        assert line.startswith("pid=")
        child = int(line.split("=")[1])
        deadline = time.monotonic() + 10.0
        while not count.exists():
            assert time.monotonic() < deadline, "the stop never reached the child"
            time.sleep(0.05)
        starter.kill()
        starter.wait()
        starter.stdout.close()
        gone_within(child, 5.0)

    def test_what_will_not_go_on_a_signal_is_killed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_module, "SETTLE", 0.5)
        started = []
        launcher = self.slow_to_go(tmp_path, "signal.SIG_IGN")
        began = time.monotonic()
        with pytest.raises(RuntimeError):
            solve(
                tmp_path / "palace.json",
                1,
                solver=launcher,
                on_output=self.callback_raising(started),
            )
        assert time.monotonic() - began < 30.0
        gone_within(started[0], 1.0)

    def test_a_tether_started_without_a_session_signals_nobody(self, tmp_path):
        """Its signal names the group its shell leads, and such a shell leads
        none. The caller here leads a group of its own, as FreeCAD may."""
        caller = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import subprocess, sys, time\n"
                f"sys.path.insert(0, {str(Path(Microwave.__file__).parents[1])!r})\n"
                "from Microwave.Solvers.cancellation import tethered\n"
                "child = subprocess.Popen(tethered(['true']), stdin=subprocess.PIPE)\n"
                "child.wait()\n"
                "child.stdin.close()\n"
                "time.sleep(2)\n"
                "print('still here', flush=True)\n",
            ],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        said, _ = caller.communicate(timeout=30)
        assert caller.returncode == 0
        assert "still here" in said

    def test_a_request_that_arrives_first_starts_no_process_at_all(self, tmp_path):
        """The binary does not exist, so starting it would be a failure to
        start. Getting ``Cancelled`` is the proof."""
        cancel = Cancellation()
        cancel.cancel()
        with pytest.raises(Cancelled):
            solve(tmp_path / "palace.json", 1, solver=tmp_path / "no-such-palace", cancel=cancel)

    def test_a_stopped_run_is_not_reported_as_a_failure(self):
        """A panel reports ``SolverFailed`` in red. Nothing went wrong here; the
        user asked."""
        assert not issubclass(Cancelled, SolverFailed)


class TestWhatAStopSignals:
    def test_a_child_already_reaped_is_left_alone(self, monkeypatch):
        """Once a child is waited for, its number is free for the system to
        give to somebody else, and signalling that group signals a stranger.
        A run that finished is signalled on the way out as a matter of course."""
        from Microwave.Solvers import cancellation

        signalled = []
        monkeypatch.setattr(cancellation.os, "killpg", lambda *pair: signalled.append(pair))
        process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        process.wait()
        cancellation.terminate(process, group=True)
        assert signalled == []


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


#: What a port's mode run finds on WR-42's face at 26 GHz, in the order Palace
#: listed them: TE10, which propagates, then TE01 and TE20, which die - the
#: slower of them last.
ONE_MODE = (complex(459.06, 0.0), complex(0.0, -486.67), complex(0.0, 218.82))


def mode_table(directory, constants):
    """A table of propagation constants as Palace writes one, padded as it pads."""
    head = [
        "m",
        "Re{kn} (1/m)",
        "Im{kn} (1/m)",
        "Re{n_eff}",
        "Im{n_eff}",
        "Error (Bkwd.)",
        "Error (Abs.)",
    ]
    lines = [",".join(f"{cell:>27}" for cell in head)]
    for number, constant in enumerate(constants, start=1):
        cells = [number, constant.real, constant.imag, 0.0, 0.0, 1e-12, 1e-12]
        lines.append(",".join(f"{cell:+.12e}" for cell in cells))
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / MODES_TABLE
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def wrote(directory, out, driven, points=13, start=20.0, step=0.5, rows=None, shift=0.0):
    """A scattering table as Palace writes one.

    Column order is Palace's: every port measured for one excitation, then the
    next excitation. An excitation is named by the port it drives, which is what
    Palace requires of a run carrying more than one. Each entry carries a
    magnitude and a phase of its own, so a row read as a column comes back as a
    different number rather than the same. ``shift`` lowers every magnitude by
    that many decibels.
    """
    head = [f"{TABLE_FREQUENCY:>16}"]
    for port in driven:
        for measured in out:
            head += [f"|S[{measured}][{port}]| (dB)", f"arg(S[{measured}][{port}]) (deg.)"]
    lines = [",".join(head)]
    for sample in range(points if rows is None else rows):
        cells = [f"{start + step * sample:.8e}"]
        for port in driven:
            for measured in out:
                cells += [
                    f"{-figure(measured, port) - shift:.6e}",
                    f"{figure(measured, port):.6e}",
                ]
        lines.append(",".join(cells))
    path = Path(directory) / TABLE
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


TABLE_FREQUENCY = "f (GHz)"


def powers(directory, out, driven, points=13, start=20.0, step=0.5, rows=None, taken=None):
    """A table of the power through each port's face, as Palace writes one.

    Column order is Palace's: every port for one excitation, then the next, and
    the excitation named in brackets only where the run has more than one. Each
    face carries what :func:`wrote`'s matrix says leaves there, less the watt
    that came in at the driven port, so the account closes - and on top of it
    what ``taken`` gives for the port and the excitation, where it gives any.
    """
    single = len(driven) == 1
    head = [f"{TABLE_FREQUENCY:>16}"]
    for port in driven:
        for measured in out:
            head.append(f"Φ_pow[{measured}]{'' if single else f'[{port}]'} (W)")
    lines = [",".join(head)]
    for sample in range(points if rows is None else rows):
        cells = [f"{start + step * sample:.8e}"]
        for port in driven:
            for measured in out:
                left = abs(expected(measured, port)) ** 2 - (1.0 if measured == port else 0.0)
                left += (taken or {}).get((measured, port), 0.0)
                cells.append(f"{left:+.12e}")
        lines.append(",".join(cells))
    path = Path(directory) / FLUX_TABLE
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def impedances(directory, stated, points=13, start=20.0, step=0.5, given=None):
    """A table of the ports' impedance, as Palace writes one: each port given a
    line, its power-voltage impedance, then the impedance each excitation saw
    there, which is not the port's own. ``given`` sets a port's figure."""
    head = [f"{TABLE_FREQUENCY:>16}"]
    for port in stated:
        head += [f"Re{{Z_PV[{port}]}} (Ohm)", f"Im{{Z_PV[{port}]}} (Ohm)"]
    for port in stated:
        head += [f"Re{{Z[{port}][{port}]}} (Ohm)", f"Im{{Z[{port}][{port}]}} (Ohm)"]
    lines = [",".join(head)]
    for sample in range(points):
        cells = [f"{start + step * sample:.8e}"]
        cells += [f"{(given or {}).get(port, 400.0 + port + sample):+.12e},+0.0" for port in stated]
        cells += [f"{-1.0:+.12e},+0.0" for _ in stated]
        lines.append(",".join(cells))
    path = Path(directory) / IMPEDANCE_TABLE
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestReadingTheImpedance:
    """What a port given a line across its face states, read back."""

    LINE = ((0.0, 5.35, 0.0), (0.0, 5.35, 4.3))

    def run(self, *lines):
        return driven(
            ports=tuple(
                WavePort(
                    index=index, attributes=(index + 2,), behind=INSIDE, excited=True, voltage=line
                )
                for index, line in enumerate(lines, start=1)
            )
        )

    def tables(self, directory):
        wrote(directory, out=(1, 2), driven=(1, 2))
        powers(directory, out=(1, 2), driven=(1, 2))

    def test_each_port_given_a_line_states_its_own_and_no_other_does(self, tmp_path):
        self.tables(tmp_path)
        impedances(tmp_path, stated=(2,))
        answer = scattering(tmp_path, self.run(None, self.LINE))

        assert np.isnan(answer.impedance[:, 0]).all()
        assert answer.impedance[:, 1] == pytest.approx(402.0 + np.arange(13), abs=0.0)

    def test_each_port_given_no_line_is_named_as_signed_by_palaces_rule(self, tmp_path):
        """The line is what turns a mode so its voltage is positive, so a port
        without one carries the sign Palace chose."""
        self.tables(tmp_path)
        impedances(tmp_path, stated=(2,))
        assert scattering(tmp_path, self.run(None, self.LINE)).sign_by_rule == (1,)

    def test_a_run_that_gave_no_port_a_line_states_nothing_and_asks_no_table(self, tmp_path):
        self.tables(tmp_path)
        assert scattering(tmp_path, self.run(None, None)).impedance is None

    def test_a_run_that_asked_and_wrote_no_table_is_refused(self, tmp_path):
        self.tables(tmp_path)
        with pytest.raises(ResultsError, match="asked for one at port 1"):
            scattering(tmp_path, self.run(self.LINE, None))

    @pytest.mark.parametrize("read", [0.0, -3.0, float("nan")])
    def test_a_line_that_read_no_voltage_is_refused_naming_the_port(self, tmp_path, read):
        """Palace reads nothing where the line leaves the mesh, and a line that
        missed the face reads zero."""
        self.tables(tmp_path)
        impedances(tmp_path, stated=(1, 2), given={2: read})
        with pytest.raises(ResultsError, match="port 2 read an impedance"):
            scattering(tmp_path, self.run(self.LINE, self.LINE))

    def test_a_table_of_another_band_is_not_this_runs(self, tmp_path):
        self.tables(tmp_path)
        impedances(tmp_path, stated=(1,), start=21.0)
        with pytest.raises(ResultsError, match="not one run's"):
            scattering(tmp_path, self.run(self.LINE, None))


def figure(out, port):
    """One number per entry, so no two of them agree."""
    return 10.0 * out + port


def expected(out, port):
    return 10.0 ** (-figure(out, port) / 20.0) * math.e ** (1j * math.radians(figure(out, port)))


class TestWherePalaceIsStartedFrom:
    """The one convention that makes a relative path in the configuration work.

    Palace resolves a path in its configuration against the directory it was
    started in rather than against the file, so what the adapter writes there is
    only correct while the run starts beside it. Nothing else in this file can
    see that: the stages test stands the run in for, and a stand-in that writes
    where it was told encodes the convention instead of checking it.
    """

    def _binary(self, tmp_path):
        """A stand-in for Palace that records where it was started."""
        stub = tmp_path / "palace-stub"
        stub.write_text("#!/bin/sh\npwd > where-it-ran.txt\n", encoding="utf-8")
        stub.chmod(0o755)
        return stub

    def test_it_runs_in_the_directory_the_configuration_is_in(self, tmp_path):
        where = tmp_path / "run"
        where.mkdir()
        config = configure(driven(), where)
        elsewhere = tmp_path / "somewhere-else"
        elsewhere.mkdir()
        solve(config, 1, solver=self._binary(tmp_path))
        assert (where / "where-it-ran.txt").is_file()
        assert Path((where / "where-it-ran.txt").read_text().strip()).resolve() == where.resolve()
        assert not (elsewhere / "where-it-ran.txt").exists()


class TestTheStagesInOrder:
    """One study, taken through every stage to a matrix.

    Each stage is measured against the one below it elsewhere in this file.
    What is tested here is the order and the joins - what each stage is handed,
    where the run is written, and what a drawing the mesher will not mesh
    becomes on the way up.
    """

    @staticmethod
    def _absent(named=None):
        raise SolverNotFound("Palace was not found")

    def run(
        self,
        tmp_path,
        monkeypatch,
        mesher=None,
        palace=None,
        version=None,
        drawn=None,
        processes=2,
        absent=(),
        carried=ONE_MODE,
        faced=None,
        mode_failure=None,
        taken=None,
        log="",
        mode_log="",
        shifted=0.0,
        check_failure=None,
        **passed,
    ):
        """The pipeline, with the processes it starts stood in for.

        :param log: what the driven run prints.
        :param mode_log: what each mode run prints.

        :param carried: the propagation constants each port's mode run finds, or
            what gives them from the frequency in GHz.
        :param faced: what the mesh comes back as, in place of :func:`meshed`'s
            own, given the problem.
        :param taken: what each port's face takes beyond what the matrix says,
            by port and excitation, as :func:`powers` takes it.
        :param shifted: how many decibels below the driven run's the run that
            checks an adaptive sweep's model answers each magnitude.
        :param check_failure: what that run raises, in place of an answer.
        """
        found = drawn if drawn is not None else guide()[0]
        described = problem(found)
        asked = {"order": [], "modes": []}

        monkeypatch.setattr(pipeline.run, "find_solver", lambda named=None: named or "palace")
        monkeypatch.setattr(pipeline.run, "find_launcher", lambda named=None: named or "mpirun")

        def supported(binary):
            asked["version of"] = binary
            asked["order"].append("version")
            if version is not None:
                raise version
            return STATED

        def meshing(request, interpreter=None, cancel=None):
            asked["order"].append("mesh")
            asked["request"] = request
            asked["interpreter"] = interpreter
            asked["mesher cancel"] = cancel
            if mesher is not None:
                raise mesher
            # Where the real mesher writes it: Microwave/Gmsh/mesh.py joins the
            # request's directory and its name, so a stand-in that answers with
            # a path of its own would not be standing in for anything.
            made = meshed(described, absent=absent) if faced is None else faced(described)
            return replace(made, path=str(Path(request.directory) / f"{request.name}.msh"))

        def solving(config, processes, solver=None, on_output=None, cancel=None, launcher=None):
            if Path(config).name.startswith(MODES_NAME):
                if mode_failure is not None:
                    raise mode_failure
                written = json.loads(Path(config).read_text())
                asked["order"].append("modes")
                asked["modes"].append(
                    {
                        "config": written,
                        "processes": processes,
                        "cancel": cancel,
                        "solver": solver,
                        "launcher": launcher,
                    }
                )
                answered = Path(config).parent / written["Problem"]["Output"]
                found = carried
                if callable(carried):
                    found = carried(written["Solver"]["BoundaryMode"]["Freq"])
                if found is not None:
                    mode_table(answered, found)
                return mode_log
            if Path(config).name.startswith(CHECKED_NAME):
                written = json.loads(Path(config).read_text())
                asked["order"].append("checked")
                asked["checked"] = written
                if check_failure is not None:
                    raise check_failure
                (samples,) = written["Solver"]["Driven"]["Samples"]
                at = samples.get("Freq") or [samples["MinFreq"], samples["MaxFreq"]]
                answered = Path(config).parent / written["Problem"]["Output"]
                answered.mkdir(parents=True, exist_ok=True)
                spaced = {"points": len(at), "start": at[0], "step": (at[-1] - at[0]) or 1.0}
                wrote(answered, out=(1, 2), driven=(1, 2), shift=shifted, **spaced)
                powers(answered, out=(1, 2), driven=(1, 2), **spaced)
                lined = [
                    port["Index"]
                    for port in written["Boundaries"].get("WavePort", [])
                    if "VoltagePath" in port
                ]
                if lined:
                    impedances(answered, stated=tuple(lined), **spaced)
                return ""
            asked["order"].append("palace")
            asked["config"] = Path(config)
            asked["palace cancel"] = cancel
            asked["processes"] = processes
            asked["solver"] = solver
            asked["launcher"] = launcher
            if on_output is not None:
                on_output("Palace said something")
            if palace is not None:
                raise palace
            (Path(config).parent / OUTPUT_NAME).mkdir(parents=True, exist_ok=True)
            wrote(Path(config).parent / OUTPUT_NAME, out=(1, 2), driven=(1, 2))
            powers(Path(config).parent / OUTPUT_NAME, out=(1, 2), driven=(1, 2), taken=taken)
            lined = [
                port["Index"]
                for port in json.loads(Path(config).read_text())["Boundaries"].get("WavePort", [])
                if "VoltagePath" in port
            ]
            if lined:
                impedances(Path(config).parent / OUTPUT_NAME, stated=tuple(lined))
            return log

        monkeypatch.setattr(pipeline.gmsh_meshing, "mesh", meshing)
        monkeypatch.setattr(pipeline.run, "solve", solving)
        monkeypatch.setattr(pipeline.run, "supported", supported)
        asked["answer"] = pipeline.solve(found, tmp_path, processes, **passed)
        return asked

    def test_a_warning_after_which_the_answer_stands_is_said_beside_it(self, tmp_path, monkeypatch):
        """From the driven run and from each mode run, once each."""
        heard = []
        stopped = estimated(ESTIMATOR_ITERATIONS)
        asked = self.run(
            tmp_path, monkeypatch, log=stopped, mode_log=stopped, on_output=heard.append
        )
        assert asked["answer"].matrix.shape[1:] == (2, 2)
        stated = [line for line in heard if "The answer stands" in line]
        assert all(line.startswith(balance.WARNING) for line in stated)
        modes = [line for line in stated if "The mode run for " in line]
        assert len(modes) == len(asked["modes"]) > 0
        assert len(stated) == len(modes) + 1

    def test_an_adaptive_sweep_that_took_every_solve_fails_the_run(self, tmp_path, monkeypatch):
        with pytest.raises(run_module.SolverFailed, match="^'Port2': the adaptive sweep"):
            self.run(
                tmp_path,
                monkeypatch,
                drawn=guide(NumFrequencyPoints=41)[0],
                log=CONVERGED + REACHED,
            )

    def adaptive(self, **study):
        """The guide swept adaptively over the thirteen points the stand-in
        table holds, both ports driven and each sweep converged."""
        found, _ = guide(NumFrequencyPoints=13, **study)
        found.Group = [
            palace_solver(SweepSolves=FEWEST_SOLVES) if kind(member) == "EMSolverPalace" else member
            for member in found.Group
        ]
        return found

    def test_an_adaptive_sweep_is_solved_again_where_its_model_is_likeliest_wrong(
        self, tmp_path, monkeypatch
    ):
        heard = []
        asked = self.run(
            tmp_path,
            monkeypatch,
            drawn=self.adaptive(),
            log=CONVERGED + CONVERGED_AT_THE_CAP,
            on_output=heard.append,
        )
        assert asked["order"][-2:] == ["palace", "checked"]
        driven_ = asked["checked"]["Solver"]["Driven"]
        assert "AdaptiveTol" not in driven_
        solving, checked = [line for line in heard if "reduced model" in line][-2:]
        assert solving.startswith("The band is solved again in full at 20 and 26 GHz")
        assert "the reduced model is off the full solve by 0 " in checked
        assert not checked.startswith(balance.WARNING)

    def checked(self, tmp_path, monkeypatch, **passed):
        heard = []
        study = passed.pop("study", {})
        self.run(
            tmp_path,
            monkeypatch,
            drawn=self.adaptive(**study),
            log=CONVERGED + CONVERGED_AT_THE_CAP,
            on_output=heard.append,
            **passed,
        )
        (*_, checked) = [line for line in heard if "reduced model" in line]
        return checked, heard

    def test_a_model_off_by_more_than_the_study_allows_is_warned_of(self, tmp_path, monkeypatch):
        checked, _ = self.checked(tmp_path, monkeypatch, shifted=1.0)
        assert checked.startswith(balance.WARNING)
        assert "SweepTolerance" in checked

    def test_the_bar_follows_the_smallest_response_the_study_reads(self, tmp_path, monkeypatch):
        """A hundredth of a decibel on a term near -11 dB is an error of a few
        parts in ten thousand: inside the bar at full scale, and past it for a
        study reading down to -40 dB."""
        at_full_scale, _ = self.checked(tmp_path / "full", monkeypatch, shifted=0.01)
        reading_down, _ = self.checked(
            tmp_path / "deep", monkeypatch, shifted=0.01, study={"SmallestResponse": -40.0}
        )
        assert not at_full_scale.startswith(balance.WARNING)
        assert reading_down.startswith(balance.WARNING)

    def test_a_check_that_fails_leaves_the_band_its_answer(self, tmp_path, monkeypatch):
        failed = run_module.SolverFailed("Palace stopped", log="")
        checked, heard = self.checked(tmp_path, monkeypatch, check_failure=failed)
        assert checked.startswith(balance.WARNING)
        assert "are not checked: Palace stopped" in checked

    def test_a_discrete_sweep_is_not_solved_again(self, tmp_path, monkeypatch):
        asked = self.run(tmp_path, monkeypatch)
        assert "checked" not in asked["order"]

    def test_a_run_carrying_lossy_metal_keeps_its_processes_and_says_why(
        self, tmp_path, monkeypatch
    ):
        """Palace decides per rank whether a port's mode solve makes a collective
        assembly, and a loss tangent on every material decides it alike on every
        rank."""
        heard = []
        asked = self.run(tmp_path, monkeypatch, drawn=lossy_guide(), on_output=heard.append)
        assert asked["processes"] == 2
        written = json.loads(asked["config"].read_text())["Domains"]["Materials"]
        assert {material["LossTan"] for material in written} == {RANK_LOSS_TANGENT}
        assert [
            line
            for line in heard
            if line.startswith(
                f"Each lossless material is given a loss tangent of {RANK_LOSS_TANGENT:g}"
            )
        ]

    def test_a_run_carrying_lossy_metal_is_one_model_on_any_count_of_processes(
        self, tmp_path, monkeypatch
    ):
        written = []
        for count in (1, 2):
            where = tmp_path / str(count)
            where.mkdir()
            asked = self.run(where, monkeypatch, drawn=lossy_guide(), processes=count)
            assert asked["processes"] == count
            written.append(asked["config"].read_text())
        assert written[0] == written[1]

    def test_a_run_carrying_lossy_metal_says_the_rest_of_the_boundary_is_perfect(
        self, tmp_path, monkeypatch
    ):
        heard = []
        self.run(tmp_path, monkeypatch, drawn=lossy_guide(), on_output=heard.append)
        assert "The rest of the boundary, 'wall', is a perfect conductor" in heard

    def test_a_run_says_which_bindings_it_holds_as_one_metal(self, tmp_path, monkeypatch):
        heard = []
        found = metal_bodies(monkeypatch, "Block", "Via", shared=[("Block", "Via")])
        self.run(tmp_path, monkeypatch, drawn=found, on_output=heard.append)
        assert [line for line in heard if line.endswith("as one metal under 'BlockMetal'")]

    def test_a_run_whose_whole_boundary_was_drawn_says_nothing_of_the_rest(
        self, tmp_path, monkeypatch
    ):
        """Where ports and sheets cover every face, the mesher makes no wall."""
        heard = []
        self.run(
            tmp_path, monkeypatch, drawn=lossy_guide(), absent=("wall",), on_output=heard.append
        )
        assert not [line for line in heard if "rest of the boundary" in line]

    def test_a_run_with_perfect_metal_alone_says_neither(self, tmp_path, monkeypatch):
        heard = []
        asked = self.run(tmp_path, monkeypatch, on_output=heard.append)
        assert asked["processes"] == 2
        assert not [line for line in heard if "perfect conductor" in line or "process" in line]

    def test_the_lossy_metal_is_written_into_the_configuration_palace_runs(
        self, tmp_path, monkeypatch
    ):
        asked = self.run(tmp_path, monkeypatch, drawn=lossy_guide())
        written = json.loads(asked["config"].read_text())
        assert [metal["Conductivity"] for metal in written["Boundaries"]["Conductivity"]] == [
            1.57e7
        ]

    def test_the_answer_says_how_each_lossy_material_was_given_to_palace(
        self, tmp_path, monkeypatch
    ):
        """The record is held against the configuration Palace ran, so what the
        result says it modelled is what was solved."""
        drawn, _ = guide(material=dielectric(LossTangent=0.02))
        drawn.Group = [*drawn.Group, metal_binding(septum(), material=sheet_metal())]
        asked = self.run(tmp_path, monkeypatch, drawn=drawn)
        written = json.loads(asked["config"].read_text())
        (filling,) = [
            material for material in written["Domains"]["Materials"] if "LossTan" in material
        ]
        (metal,) = written["Boundaries"]["Conductivity"]
        assert asked["answer"].modelled == (
            {"material": "Air", "held": "loss tangent", "loss_tangent": filling["LossTan"]},
            {
                "material": "Brass",
                "sheet": "surface impedance",
                "conductivity": metal["Conductivity"],
                "thickness": 0.0,
                "faces": ["boundary"],
            },
        )

    def test_the_palace_found_is_asked_its_version_before_the_mesh_is_built(
        self, tmp_path, monkeypatch
    ):
        """The mesh is the stage that costs minutes, and an older Palace would
        refuse the run after it."""
        asked = self.run(tmp_path, monkeypatch, solver=tmp_path / "chosen")
        assert asked["version of"] == tmp_path / "chosen"
        assert asked["order"] == ["version", "mesh", *["modes"] * 4, "palace"]

    def test_an_older_palace_stops_the_run_with_what_it_was_told(self, tmp_path, monkeypatch):
        with pytest.raises(SolverUnsupported, match="older"):
            self.run(tmp_path, monkeypatch, version=SolverUnsupported("older"))

    def test_the_mesher_is_asked_for_what_the_problem_describes(self, tmp_path, monkeypatch):
        asked = self.run(tmp_path, monkeypatch)
        described = problem(guide()[0])
        request = asked["request"]
        assert request.demand == described.demand
        assert request.profile is described.profile
        assert request.directory == str(tmp_path)
        assert request.name == MESH_NAME
        assert len(request.pieces) == sum(len(row[2]) for row in described.labelled)
        # The wall is asked for by name and drawn as nothing. Left unnamed, the
        # mesher refuses the drawing for a boundary nothing claims - and a
        # drawing whose whole boundary the adapter did claim is the one that
        # cannot tell the outside of the model from the face between two bodies.
        assert request.remainder == described.wall

    def test_every_shape_reached_a_file_before_the_mesher_was_asked(self, tmp_path, monkeypatch):
        asked = self.run(tmp_path, monkeypatch)
        for piece in asked["request"].pieces:
            assert Path(piece.file).is_file()
            assert Path(piece.file).parent == tmp_path

    def test_palace_is_run_on_the_configuration_that_was_written(self, tmp_path, monkeypatch):
        asked = self.run(tmp_path, monkeypatch)
        assert asked["config"] == tmp_path / CONFIG_NAME
        assert asked["config"].is_file()
        assert asked["processes"] == 2

    def test_the_configuration_names_the_mesh_beside_it(self, tmp_path, monkeypatch):
        """Palace resolves a path in the configuration against the directory it
        was started in, which is the configuration's own. Named that way the
        whole directory moves as one, which is what an attached bug report is;
        named by where it was written it moves and stops running."""
        asked = self.run(tmp_path, monkeypatch)
        written = json.loads(asked["config"].read_text())
        assert written["Model"]["Mesh"] == Path(meshed(problem(guide()[0])).path).name
        assert not Path(written["Model"]["Mesh"]).is_absolute()

    def test_a_relative_directory_is_resolved_before_anything_is_written(
        self, tmp_path, monkeypatch
    ):
        """A relative path names one place to this process and another to the
        one Palace runs in, and Palace runs in the configuration's directory."""
        monkeypatch.chdir(tmp_path)
        asked = self.run(Path("under/here"), monkeypatch)
        assert asked["config"] == tmp_path / "under/here" / CONFIG_NAME

    def test_an_earlier_runs_tables_are_gone_before_this_one_starts(self, tmp_path, monkeypatch):
        """A table left where this run writes is otherwise read as this run's,
        and the reader's rule that a missing table says something about the run
        stops holding."""
        stale = tmp_path / OUTPUT_NAME
        stale.mkdir()
        (stale / "left-behind.csv").write_text("from an earlier solve\n", encoding="utf-8")
        self.run(tmp_path, monkeypatch)
        assert not (stale / "left-behind.csv").exists()

    def test_a_rank_count_that_is_not_one_is_refused_before_anything_is_drawn(
        self, tmp_path, monkeypatch
    ):
        found, _ = guide()
        with pytest.raises(ValueError, match="at least one process"):
            pipeline.solve(found, tmp_path, 0)
        assert list(tmp_path.iterdir()) == []

    def test_a_machine_without_palace_is_told_before_the_mesh_is_built(self, tmp_path, monkeypatch):
        """The mesh is the stage that costs minutes, and the binary is known at
        the call."""

        def never(request, interpreter=None, cancel=None):
            raise AssertionError("the mesher was asked for a grid")

        monkeypatch.setattr(pipeline.gmsh_meshing, "mesh", never)
        monkeypatch.setattr(pipeline.run, "find_solver", self._absent)
        found, _ = guide()
        with pytest.raises(SolverNotFound):
            pipeline.solve(found, tmp_path, 1)
        assert list(tmp_path.iterdir()) == []

    def test_a_machine_without_mpirun_is_told_before_the_mesh_is_built(self, tmp_path, monkeypatch):
        def never(request, interpreter=None, cancel=None):
            raise AssertionError("the mesher was asked for a grid")

        def absent(named=None):
            raise SolverNotFound("no mpirun on the PATH")

        monkeypatch.setattr(pipeline.gmsh_meshing, "mesh", never)
        monkeypatch.setattr(pipeline.run, "find_solver", lambda named=None: named or "palace")
        monkeypatch.setattr(pipeline.run, "find_launcher", absent)
        found, _ = guide()
        with pytest.raises(SolverNotFound, match="mpirun"):
            pipeline.solve(found, tmp_path, 1)

    def test_the_documents_own_fault_is_reported_before_the_machines(self, tmp_path, monkeypatch):
        """A study that does not translate is what the user is here to fix; a
        machine with no Palace leaves them nothing to do about it."""
        monkeypatch.setattr(pipeline.run, "find_solver", self._absent)
        found, _ = guide()
        found.Group = [member for member in found.Group if kind(member) != "EMSolverPalace"]
        with pytest.raises(TranslationError, match="holds no Palace solver"):
            pipeline.solve(found, tmp_path, 1)

    def test_the_matrix_is_read_from_where_palace_was_told_to_write(self, tmp_path, monkeypatch):
        asked = self.run(tmp_path, monkeypatch)
        written = json.loads(asked["config"].read_text())
        assert written["Problem"]["Output"] == OUTPUT_NAME
        assert (tmp_path / OUTPUT_NAME / TABLE).is_file()
        assert asked["answer"].matrix.shape == (13, 2, 2)

    def test_the_binary_and_the_interpreter_are_passed_through(self, tmp_path, monkeypatch):
        """Neither is searched for where a caller named one, so a wrong path
        fails where it was given rather than being quietly replaced."""
        asked = self.run(
            tmp_path, monkeypatch, solver="/opt/palace/bin/palace", interpreter="/venv/python3"
        )
        assert asked["solver"] == "/opt/palace/bin/palace"
        assert asked["interpreter"] == "/venv/python3"

    def test_what_palace_prints_while_it_works_reaches_the_caller(self, tmp_path, monkeypatch):
        """A solve is minutes, and its log is the only thing that says how it
        is going. It also carries the warnings that mean the answer is about a
        different model, which ``run.complaints`` reads."""
        said = []
        self.run(tmp_path, monkeypatch, on_output=said.append)
        assert "Palace said something" in said

    def test_what_the_matrix_leaves_unaccounted_for_is_said_after_palace(
        self, tmp_path, monkeypatch
    ):
        """Once per driven port, naming the face that took the most, and the
        matrix returned however much it is."""
        said = []
        asked = self.run(tmp_path, monkeypatch, on_output=said.append, taken={(2, 1): 0.25})
        after = said[said.index("Palace said something") + 1 :]
        assert len(after) == 2
        assert all(line.startswith(balance.WARNING) for line in after)
        assert "Driven from 'Port1'" in after[0] and "through 'Port2'" in after[0]
        assert "Driven from 'Port2'" in after[1]
        assert asked["answer"].matrix.shape == (13, 2, 2)

    def test_a_model_that_dissipates_is_credited_its_heat_on_the_way_through(
        self, tmp_path, monkeypatch
    ):
        """The stand-in's matrix keeps little of the power, and its faces say
        where the rest went. Read as the lossy guide's run, all but what port 2
        took is heat."""
        said = []
        self.run(
            tmp_path,
            monkeypatch,
            drawn=lossy_guide(),
            processes=1,
            on_output=said.append,
            taken={(2, 1): 5e-4},
        )
        (first,) = [line for line in said if "Driven from 'Port1'" in line]
        assert not first.startswith(balance.WARNING)
        assert "0.05% of the power" in first

    def test_the_mesh_says_what_it_came_out_as_before_the_solve(self, tmp_path, monkeypatch):
        """The stage after the mesh is the one that costs minutes, and the size
        the mesh reached is the answer to the length the document stated. Said
        before, so a size the drawing bounded rather than the request is
        readable while it still decides anything.

        The whole of what reaches the caller is asserted, so a line nobody meant
        to say is caught here rather than read as the mesher's."""
        said = []
        asked = self.run(tmp_path, monkeypatch, on_output=said.append)
        described = problem(guide()[0])
        labels = {feed.number: feed.label for feed in described.feeds}
        assert said == [
            STATED,
            *gmsh_mesh_report.describe(meshed(described), described.demand),
            *unlaid_said(described.unlaid, meshed(described)),
            sweeping(described.sweep),
            *(
                judged(feed.label, ONE_MODE, frequency, where)
                for feed in described.feeds
                for frequency, where in (
                    (described.sweep.start, "the bottom of the band"),
                    (described.sweep.stop, "the top of the band"),
                )
            ),
            "Palace said something",
            *(balance.said(short, labels) for short in balance.shortfalls(asked["answer"])),
        ]
        assert said[1].startswith("Mesh: elements from")

    def test_a_stop_reaches_the_mesher_and_palace_alike(self, tmp_path, monkeypatch):
        """Both are processes of minutes, and a panel stopping the run does not
        know which of them is running."""
        cancel = Cancellation()
        asked = self.run(tmp_path, monkeypatch, cancel=cancel)
        assert asked["mesher cancel"] is cancel
        assert asked["palace cancel"] is cancel
        assert all(run["cancel"] is cancel for run in asked["modes"])

    def test_each_ports_modes_are_asked_before_the_band_on_one_process(self, tmp_path, monkeypatch):
        """Each is one face at one frequency, run on one process whatever the
        driven run is given."""
        asked = self.run(tmp_path, monkeypatch, processes=2, solver=tmp_path / "chosen")
        assert asked["order"][-5:] == [*["modes"] * 4, "palace"]
        ports = json.loads(asked["config"].read_text())["Boundaries"]["WavePort"]
        asked_at = [
            (mode["Attributes"], mode["Freq"])
            for mode in (run["config"]["Solver"]["BoundaryMode"] for run in asked["modes"])
        ]
        assert asked_at == [
            (port["Attributes"], frequency) for port in ports for frequency in (20.0, 26.0)
        ]
        assert {run["processes"] for run in asked["modes"]} == {1}
        assert {run["solver"] for run in asked["modes"]} == {tmp_path / "chosen"}
        assert asked["processes"] == 2

    def test_the_mpi_launcher_named_starts_the_modes_and_the_band_alike(
        self, tmp_path, monkeypatch
    ):
        asked = self.run(tmp_path, monkeypatch, launcher="/bin/mpirun")
        assert asked["launcher"] == "/bin/mpirun"
        assert {run["launcher"] for run in asked["modes"]} == {"/bin/mpirun"}

    def test_a_sheet_clear_of_the_ports_is_left_out_of_their_mode_runs(self, tmp_path, monkeypatch):
        asked = self.run(tmp_path, monkeypatch, drawn=lossy_guide())
        assert json.loads(asked["config"].read_text())["Boundaries"]["Conductivity"]
        assert all("Conductivity" not in run["config"]["Boundaries"] for run in asked["modes"])

    def test_a_sheet_round_a_ports_face_is_kept_in_that_ports_mode_run(self, tmp_path, monkeypatch):
        """Walls of finite conductivity are the walls of the face's own guide."""
        asked = self.run(
            tmp_path,
            monkeypatch,
            drawn=lossy_guide(),
            faced=lambda described: cross_section(
                described, "Port1", {WHOLE: ROUND}, ROUND_END, {"SeptumMetal": ROUND}
            ),
        )
        first = asked["modes"][:2]
        second = asked["modes"][2:]
        assert all(run["config"]["Boundaries"]["Conductivity"] for run in first)
        assert all("Conductivity" not in run["config"]["Boundaries"] for run in second)

    def test_the_refusal_names_the_sheet_that_divides_the_face(self, tmp_path, monkeypatch):
        pair = (complex(459.05, 0.0), complex(459.05, 0.0), complex(0.0, 692.8))
        found, _ = guide()
        found.Group = [*found.Group, metal_binding(septum())]
        with pytest.raises(TranslationError, match="a face 'SeptumMetal' divides"):
            self.run(
                tmp_path,
                monkeypatch,
                drawn=found,
                carried=pair,
                faced=lambda described: cross_section(
                    described,
                    "Port1",
                    HALVES,
                    HALVES_END,
                    {described.wall: HALVES_ROUND, "SeptumMetal": (ACROSS,)},
                ),
            )

    def test_a_band_starting_below_the_guides_cutoff_is_refused_at_its_bottom(
        self, tmp_path, monkeypatch
    ):
        """The port solves every frequency of the band, and below cutoff it has
        no wave to take; the top of the band alone does not say so."""
        dying = (complex(0.0, 205.6), complex(0.0, 548.5), complex(0.0, 600.0))

        def carried(frequency):
            return ONE_MODE if frequency > 20.0 else dying

        with pytest.raises(TranslationError, match="no mode at 20 GHz, the bottom of the band"):
            self.run(tmp_path, monkeypatch, carried=carried)

    def test_a_port_whose_face_reflects_its_own_mode_is_refused_at_the_bottom(
        self, tmp_path, monkeypatch
    ):
        """The loss is the largest share of the constant's square at the bottom
        of the band, so that is where the face reflects most."""
        lossy = (complex(370.3, -160.1), complex(204.6, -434.8), complex(148.6, -598.6))

        def carried(frequency):
            return ONE_MODE if frequency > 20.0 else lossy

        with pytest.raises(TranslationError, match="at 20 GHz, the bottom of the band. Palace"):
            self.run(tmp_path, monkeypatch, carried=carried)

    def test_a_lossy_filling_is_named_where_a_face_carries_no_mode(self, tmp_path, monkeypatch):
        dying = (complex(0.0, 205.6), complex(0.0, 548.5), complex(0.0, 600.0))
        drawn, _ = guide(material=dielectric(Permittivity=2.0, LossTangent=0.6))
        with pytest.raises(TranslationError, match="A lossy filling can also hide"):
            self.run(tmp_path, monkeypatch, drawn=drawn, carried=dying)

    @pytest.mark.parametrize("lossy_metal", [False, True], ids=["nothing lossy", "lossy metal"])
    def test_a_lossless_filling_is_not_named_where_a_face_carries_no_mode(
        self, tmp_path, monkeypatch, lossy_metal
    ):
        """Metal of finite conductivity dissipates too, and has no filling to change."""
        dying = (complex(0.0, 205.6), complex(0.0, 548.5), complex(0.0, 600.0))
        drawn = lossy_guide() if lossy_metal else guide()[0]
        with pytest.raises(TranslationError) as refused:
            self.run(tmp_path, monkeypatch, drawn=drawn, carried=dying)
        assert "filling" not in str(refused.value)

    def test_a_band_of_one_frequency_is_asked_at_the_frequency_it_solves(
        self, tmp_path, monkeypatch
    ):
        """One point is written as the start of the band, wherever its stop."""
        asked = self.run(tmp_path, monkeypatch, drawn=guide(NumFrequencyPoints=1)[0])
        assert [run["config"]["Solver"]["BoundaryMode"]["Freq"] for run in asked["modes"]] == [
            20.0,
            20.0,
        ], "one run a port at the one frequency the band solves"

    def test_a_mode_run_that_fails_says_which_it_was_and_what_stopped_it(
        self, tmp_path, monkeypatch
    ):
        """Its lines are not shown as they arrive, so what stopped it goes with it."""
        said = ("Palace stopped: MFEM abort: Unable to open mesh file",)
        with pytest.raises(SolverFailed) as failed:
            self.run(
                tmp_path,
                monkeypatch,
                mode_failure=SolverFailed("Palace exited with code 1", said, "the whole log"),
            )
        assert "the mode run for 'Port1', modes-1-bottom.json" in str(failed.value)
        assert failed.value.said == said
        assert failed.value.log == "the whole log"

    def test_a_port_carrying_two_modes_stops_the_run_before_the_band(self, tmp_path, monkeypatch):
        pair = (complex(459.05, 0.0), complex(459.05, 0.0), complex(0.0, 692.8))
        with pytest.raises(TranslationError, match="'Port1' stands on a face carrying 2 modes"):
            self.run(tmp_path, monkeypatch, carried=pair)
        assert not (tmp_path / OUTPUT_NAME).exists(), "the band was solved after the refusal"

    def test_a_mode_run_that_found_too_few_is_no_answer(self, tmp_path, monkeypatch):
        with pytest.raises(ResultsError, match="found 1 of the"):
            self.run(tmp_path, monkeypatch, carried=ONE_MODE[:1])

    def test_a_table_an_earlier_mode_run_left_is_not_read_as_this_ones(self, tmp_path, monkeypatch):
        """Palace also warns about an output directory it is going to write
        into, and a warning here is a failed run."""
        for port in (1, 2):
            for edge in ("bottom", "top"):
                mode_table(tmp_path / MODES_NAME / str(port) / edge, ONE_MODE)
        with pytest.raises(ResultsError, match="left no table"):
            self.run(tmp_path, monkeypatch, carried=None)

    def test_preparing_a_run_starts_no_process(self, tmp_path, monkeypatch):
        """The stage a GUI runs on its own thread, so it must cost a translation
        and the files and nothing that takes minutes."""

        def never(*unused, **ignored):
            raise AssertionError("a process was started")

        monkeypatch.setattr(pipeline.gmsh_meshing, "mesh", never)
        monkeypatch.setattr(pipeline.run, "solve", never)
        monkeypatch.setattr(pipeline.run, "find_solver", lambda named=None: named or "palace")
        prepared = pipeline.prepare(guide()[0], tmp_path)
        assert prepared.pieces
        assert all(Path(piece.file).is_file() for piece in prepared.pieces)

    def test_what_a_refinement_region_names_reaches_the_mesher_as_marks(
        self, tmp_path, monkeypatch
    ):
        """Each is a file of its own, beside the pieces, and claims nothing."""
        rod = obj("Part::Feature", "Rod", Shape=Boxed("rod", (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
        asked = self.run(tmp_path, monkeypatch, drawn=with_members(refinement([(rod, [""])])))
        (mark,) = asked["request"].marks
        assert mark.name == "Ring on Rod" and mark.dimension == 3
        assert Path(mark.file).read_text(encoding="utf-8") == "rod"
        assert mark.name not in {piece.label for piece in asked["request"].pieces}

    def test_what_is_handed_on_carries_no_shape(self, tmp_path, monkeypatch):
        """The stages after the first run on a thread the kernel's objects do
        not belong to, so they are handed the files and none of the shapes."""
        monkeypatch.setattr(pipeline.run, "find_solver", lambda named=None: named or "palace")
        found = lossy_guide()
        found.Group = [*found.Group, metal_binding(septum(name="Iris"))]
        prepared = pipeline.prepare(found, tmp_path)
        assert prepared.problem.conductors
        assert prepared.problem.lossy
        assert all(not shapes for _, _, shapes in prepared.problem.labelled)
        rod = obj("Part::Feature", "Rod", Shape=Boxed("rod", (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
        prepared = pipeline.prepare(with_members(refinement([(rod, [""])])), tmp_path / "marked")
        assert prepared.marks and all(mark.shape is None for mark in prepared.problem.marks)

    def test_a_drawing_the_mesher_refuses_is_the_documents_own_fault(self, tmp_path, monkeypatch):
        """``Refused`` is the mesher's own and is not the type a panel catches
        to know the model is at fault, so it would reach a user as a traceback.
        Its complaints name what somebody drew, which is what a user is shown."""
        with pytest.raises(TranslationError, match="two labels claim one face"):
            self.run(
                tmp_path,
                monkeypatch,
                mesher=Refused(["two labels claim one face"]),
            )

    def test_a_drawing_gmsh_could_not_fill_is_too(self, tmp_path, monkeypatch):
        with pytest.raises(TranslationError, match="the guide has no elements in it"):
            self.run(
                tmp_path,
                monkeypatch,
                mesher=Unmeshed(["the guide has no elements in it"], ["Error: self-intersection"]),
            )

    def test_a_part_gmsh_could_not_fill_is_told_how_to_make_its_elements_finer(
        self, tmp_path, monkeypatch
    ):
        """Where the drawing curves or narrows tighter than the elements laid
        there, finer elements there are what fill it."""
        with pytest.raises(TranslationError) as refusal:
            self.run(
                tmp_path,
                monkeypatch,
                mesher=Unmeshed(["elements turned inside out in 'Rod'"]),
            )
        said = str(refusal.value)
        assert said.startswith("elements turned inside out in 'Rod'. ")
        assert "raise ElementsPerTurn on the Gmsh mesh from 6," in said
        assert "set MinElementSize on the mesh policy below the floor of " in said
        assert "lay a Mesh Refinement there" in said

    def test_shapes_the_kernel_could_not_cut_are_told_nothing_about_the_element_size(
        self, tmp_path, monkeypatch
    ):
        with pytest.raises(TranslationError) as refusal:
            self.run(
                tmp_path,
                monkeypatch,
                mesher=Uncut(["Gmsh could not cut the shapes drawn for 'Guide', 'Post'"]),
            )
        assert str(refusal.value) == "Gmsh could not cut the shapes drawn for 'Guide', 'Post'"

    def test_what_gmsh_logged_reaches_the_caller_and_not_the_sentence(self, tmp_path, monkeypatch):
        """Gmsh writes to a terminal a caller running beside its document has
        not got, and it is many lines. The sentence is what somebody reads."""
        said = []
        with pytest.raises(TranslationError) as refusal:
            self.run(
                tmp_path,
                monkeypatch,
                mesher=Unmeshed(["nothing filled the guide"], ["Error: self-intersection"]),
                on_output=said.append,
            )
        assert said == [STATED, "Error: self-intersection"]
        assert "self-intersection" not in str(refusal.value)

    def test_a_mesh_made_and_not_solved_is_refused_as_the_run_would_be(self, tmp_path, monkeypatch):
        """A panel can make the mesh alone, and a mesh it reports as made has to
        be one the run would take. A port on a face inside the model is found
        once the mesh exists, so the stage that made it asks, before it says
        what the mesh came out as."""
        monkeypatch.setattr(pipeline.run, "find_solver", lambda named=None: named or "palace")
        prepared = pipeline.prepare(guide()[0], tmp_path)
        monkeypatch.setattr(
            pipeline.gmsh_meshing,
            "mesh",
            lambda request, interpreter=None, cancel=None: meshed(
                prepared.problem, inside=("Port2",)
            ),
        )
        said = []
        with pytest.raises(TranslationError, match="'Port2' stands on a face inside the model"):
            pipeline.meshed(prepared, on_output=said.append)
        assert said == []

    def test_a_study_it_cannot_translate_never_reaches_the_mesher(self, tmp_path, monkeypatch):
        def never(request, interpreter=None, cancel=None):
            raise AssertionError("the mesher was asked for a grid")

        monkeypatch.setattr(pipeline.gmsh_meshing, "mesh", never)
        found, _ = guide()
        found.Group = [member for member in found.Group if kind(member) != "EMSolverPalace"]
        with pytest.raises(TranslationError, match="holds no Palace solver"):
            pipeline.solve(found, tmp_path, 1)

    def test_it_insists_on_being_told_how_many_processes(self, tmp_path):
        """A default here is a second copy of a figure the caller edits, and it
        is the one every caller that said nothing would get."""
        with pytest.raises(TypeError):
            pipeline.solve(guide()[0], tmp_path)


class TestReadingTheTable:
    def test_the_matrix_is_the_size_of_the_run(self, tmp_path):
        run = driven()
        wrote(tmp_path, out=(1, 2), driven=(1, 2))
        powers(tmp_path, out=(1, 2), driven=(1, 2))
        answer = scattering(tmp_path, run)
        assert answer.matrix.shape == (13, 2, 2)
        assert answer.out == (1, 2)
        assert answer.driven == (1, 2)

    def test_a_magnitude_and_a_phase_become_one_number(self, tmp_path):
        wrote(tmp_path, out=(1, 2), driven=(1, 2))
        powers(tmp_path, out=(1, 2), driven=(1, 2))
        answer = scattering(tmp_path, driven())
        assert answer.matrix[0, 1, 0] == pytest.approx(expected(2, 1))
        assert answer.matrix[0, 0, 1] == pytest.approx(expected(1, 2))

    def test_a_band_of_one_frequency_is_one_row_and_not_a_run_that_stopped(self, tmp_path):
        """Palace spaces a linear block by dividing by one less than the count,
        so a single sample is asked for as a frequency rather than as a block -
        and what the table then holds is one row.
        """
        run = driven(sweep=Sweep(20e9, 20e9, 13))
        wrote(tmp_path, out=(1, 2), driven=(1, 2), rows=1)
        powers(tmp_path, out=(1, 2), driven=(1, 2), rows=1)
        assert scattering(tmp_path, run).matrix.shape == (1, 2, 2)

    @pytest.mark.parametrize("written", ["+nan", "+inf"])
    def test_a_term_that_is_not_a_finite_number_is_refused_naming_where(self, tmp_path, written):
        """Palace writes a solve that produced no answer as nan and exits cleanly."""
        path = wrote(tmp_path, out=(1, 2), driven=(1, 2))
        powers(tmp_path, out=(1, 2), driven=(1, 2))
        lines = path.read_text(encoding="utf-8").splitlines()
        cells = lines[3].split(",")
        cells[5] = written
        lines[3] = ",".join(cells)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(ResultsError, match="not a finite number at 21 GHz"):
            scattering(tmp_path, driven())

    def test_the_band_comes_back_in_hertz(self, tmp_path):
        wrote(tmp_path, out=(1, 2), driven=(1, 2))
        powers(tmp_path, out=(1, 2), driven=(1, 2))
        answer = scattering(tmp_path, driven())
        assert answer.frequency[0] == pytest.approx(20e9)
        assert answer.frequency[-1] == pytest.approx(26e9)

    def test_a_column_is_named_by_two_port_numbers(self, tmp_path):
        """A document whose ports are numbered otherwise than from one is the
        case that separates the port's number from its place in the matrix.
        """
        run = driven(
            ports=(
                WavePort(index=3, attributes=(3,), behind=INSIDE, excited=True),
                WavePort(index=7, attributes=(4,), behind=INSIDE, excited=True),
            )
        )
        wrote(tmp_path, out=(3, 7), driven=(3, 7))
        powers(tmp_path, out=(3, 7), driven=(3, 7))
        answer = scattering(tmp_path, run)
        assert answer.out == (3, 7)
        assert answer.driven == (3, 7)
        assert answer.matrix[0, 1, 0] == pytest.approx(expected(7, 3))

    def test_a_port_nothing_drives_is_a_row_and_not_a_column(self, tmp_path):
        run = driven(
            ports=(
                WavePort(index=1, attributes=(3,), behind=INSIDE, excited=True),
                WavePort(index=2, attributes=(4,), behind=INSIDE),
            )
        )
        wrote(tmp_path, out=(1, 2), driven=(1,))
        powers(tmp_path, out=(1, 2), driven=(1,))
        answer = scattering(tmp_path, run)
        assert answer.matrix.shape == (13, 2, 1)
        assert answer.driven == (1,)

    def test_no_table_says_what_leaves_none(self, tmp_path):
        """A run whose excitations do not drive one port each writes every other
        table and not this one, and says nothing about it.
        """
        with pytest.raises(ResultsError, match="exactly one port"):
            scattering(tmp_path, driven())

    def test_a_table_shorter_than_the_band_is_a_run_that_stopped(self, tmp_path):
        """Every table is rewritten whole at each step with nothing marking it
        partial, so a stopped run leaves a well-formed file.
        """
        wrote(tmp_path, out=(1, 2), driven=(1, 2), rows=4)
        with pytest.raises(ResultsError, match="stopped"):
            scattering(tmp_path, driven())

    def test_a_table_the_run_did_not_ask_for_is_refused_by_the_column(self, tmp_path):
        wrote(tmp_path, out=(1,), driven=(1,))
        with pytest.raises(ResultsError, match="no column"):
            scattering(tmp_path, driven())

    def test_a_table_holding_only_a_header_is_refused(self, tmp_path):
        wrote(tmp_path, out=(1, 2), driven=(1, 2), rows=0)
        with pytest.raises(ResultsError, match="no samples"):
            scattering(tmp_path, driven())

    def test_a_table_carrying_something_that_is_not_a_number_is_refused(self, tmp_path):
        path = wrote(tmp_path, out=(1, 2), driven=(1, 2))
        path.write_text(path.read_text().replace("2.00000000e+01", "nowhere"), encoding="utf-8")
        with pytest.raises(ResultsError, match="not a number"):
            scattering(tmp_path, driven())

    def test_an_empty_file_is_refused(self, tmp_path):
        (tmp_path / TABLE).write_text("", encoding="utf-8")
        with pytest.raises(ResultsError, match="empty"):
            scattering(tmp_path, driven())


#: What no module of this adapter may pull in. FreeCAD and the GUI because the
#: whole translation is duck typing; the other backend's engine because an
#: adapter never imports another; Gmsh because the mesher stands outside every
#: adapter and behind a process boundary; and the result layer's own
#: dependencies because assembling a result is the neutral layer's.
BANNED = (
    "FreeCAD",
    "FreeCADGui",
    "openEMS",
    "CSXCAD",
    "gmsh",
    "skrf",
    "scipy",
    "pandas",
    "typing_extensions",
    "matplotlib",
)

#: The GUI, which shares its root with the code under test and so cannot be
#: named by root at all.
GUI = ("FreeCADGui", "Microwave.Gui", "Microwave.ViewProviders")


class TestWhatTheAdapterMayReach:
    """Measured in a child interpreter, because this suite imports the adapter
    itself: asking ``sys.modules`` in process would answer about everything the
    session has already loaded.
    """

    def leaks(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        code = (
            "import importlib, sys\n"
            f"banned = {BANNED!r}\n"
            f"gui = {GUI!r}\n"
            "def loaded():\n"
            "    found = set()\n"
            "    for n in list(sys.modules):\n"
            "        if n.split('.')[0] in banned:\n"
            "            found.add(n.split('.')[0])\n"
            "        found |= {p for p in gui if n.startswith(p)}\n"
            "    return found\n"
            "seen = loaded()\n"
            f"for name in {import_sweep.swept('Microwave.Solvers.palace')!r}:\n"
            "    importlib.import_module('Microwave.Solvers.palace.' + name)\n"
            "    now = loaded()\n"
            "    print(name + ':' + ','.join(sorted(now - seen)))\n"
            "    seen |= now\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=root
        )
        assert done.returncode == 0, done.stderr
        return {
            line.split(":")[0]: [name for name in line.split(":")[1].split(",") if name]
            for line in done.stdout.splitlines()
        }

    def test_no_module_of_it_needs_freecad_a_solver_or_a_mesher(self):
        brought = {name: what for name, what in self.leaks().items() if what}
        assert not brought, f"importing these reached what the adapter may not: {brought}"

    def test_the_sweep_still_finds_this_adapter(self):
        """A floor on names rather than on a count. A module that dropped out of
        the discovery would take its own leak test with it and say nothing.
        """
        assert {
            "attributes",
            "capabilities",
            "config",
            "document",
            "policy",
            "problem",
            "read",
            "run",
            "write",
        } <= set(import_sweep.swept("Microwave.Solvers.palace"))

    def test_the_script_ci_runs_sweeps_this_adapter_too(self):
        """The rule is one rule, and an adapter left out of the sweep keeps it
        only by promise.
        """
        assert "Microwave.Solvers.palace" in import_sweep.PACKAGES


class TestHowManyModesAPortCarries:
    """A wave port absorbs its own mode and nothing else, so a face carrying
    more than one, or none, is refused on the mode run's own count."""

    def test_one_mode_is_said_with_how_fast_the_next_dies(self):
        said = judged("Port1", ONE_MODE, 26e9, "the top of the band")
        assert said.startswith("'Port1' carries one mode at 26 GHz, 459.06 per metre")
        assert said.endswith(f"of the others found falls by e in {1e3 / 218.82:.3g} mm")

    @pytest.mark.parametrize(
        "found",
        [
            (complex(459.05, 0.0), complex(459.05, 0.0), complex(0.0, 692.8)),
            (complex(579.59, 0.0), complex(277.98, 0.0), complex(0.0, 444.9)),
        ],
        ids=["a degenerate pair", "a second mode above its cutoff"],
    )
    def test_a_second_propagating_mode_is_refused_naming_both(self, found):
        with pytest.raises(TranslationError) as refused:
            judged("Port1", found, 26e9, "the top of the band")
        said = str(refused.value)
        assert "'Port1' stands on a face carrying 2 modes at 26 GHz" in said
        assert all(f"{k.real:.2f}" in said for k in found if k.real)

    def test_what_divides_the_face_is_named_where_something_does(self):
        pair = (complex(459.05, 0.0), complex(459.05, 0.0), complex(0.0, 692.8))
        with pytest.raises(TranslationError, match="a face 'Sheet' divides, whose parts carry 2"):
            judged("Port1", pair, 26e9, "the top of the band", "'Sheet'")

    def test_a_face_carrying_nothing_is_refused_with_how_fast_its_first_dies(self):
        below = (complex(0.0, 700.0), complex(0.0, 300.0), complex(0.0, 500.0))
        with pytest.raises(TranslationError, match=f"falls by e in {1e3 / 300.0:.3g} mm"):
            judged("Port1", below, 26e9, "the top of the band")

    def test_a_face_where_every_mode_asked_for_propagates_may_carry_more(self):
        every = (complex(515.1, 0.0), complex(459.1, 0.0), complex(152.7, 0.0))
        with pytest.raises(TranslationError, match=f"carrying at least {MODES} modes"):
            judged("Port1", every, 26e9, "the top of the band")

    def test_a_run_that_found_fewer_than_asked_says_so_rather_than_counting(self):
        with pytest.raises(SolvedShort, match=f"found 2 of the {MODES}"):
            judged("Port1", ONE_MODE[:2], 26e9, "the top of the band")

    @pytest.mark.parametrize(
        ("constant", "carries"),
        [
            (complex(459.09, -0.0375), True),
            (complex(7.2, 9.4), False),
            (complex(5.0, 5.0), False),
            (complex(1.9e-6, -726.7), False),
        ],
        ids=["a lossy wall's TE10", "a mode past cutoff in a lossy filling", "on cutoff", "dying"],
    )
    def test_a_mode_propagates_where_its_square_is_positive(self, constant, carries):
        """Palace's constant is the principal root, so the real part is never
        negative and the square is positive exactly where it exceeds the
        imaginary one."""
        assert propagates(constant) is carries

    @pytest.mark.parametrize(
        "constant",
        [complex(370.3, -160.1), complex(459.09, -0.0375), complex(5.0, -400.0)],
        ids=["a lossy filling", "lossy walls", "near cutoff"],
    )
    def test_the_share_is_the_mismatch_of_two_lines_of_those_constants(self, constant):
        """A TE mode's impedance is omega mu over its constant, so a line of the
        constant's real part meets the guide's with this reflection."""
        mismatch = (constant.real - constant) / (constant.real + constant)
        assert reflected(constant) == pytest.approx(abs(mismatch) ** 2, rel=1e-12)

    def test_a_lossless_mode_is_not_reflected(self):
        assert reflected(complex(459.06, 0.0)) == 0.0

    def test_a_mode_its_face_reflects_past_the_bar_is_refused_naming_the_share(self):
        lossy = (complex(370.3, -160.1), complex(204.6, -434.8), complex(148.6, -598.6))
        with pytest.raises(TranslationError) as refused:
            judged("Port1", lossy, 15e9, "the bottom of the band")
        said = str(refused.value)
        assert "'Port1' carries a mode of 370.30-160.10i per metre at 15 GHz" in said
        assert f"reflects {reflected(lossy[0]):.3g} of the power" in said

    @pytest.mark.parametrize(("loss", "refused"), [(12.5, False), (12.8, True)])
    def test_the_bar_is_the_one_the_run_holds_its_power_to(self, loss, refused):
        """Two faces reflecting in phase make four times one face's share, so
        that is what is held to the bar: either side of it here."""
        constant = complex(400.0, -loss)
        assert (4 * reflected(constant) > balance.BAR) is refused
        found = (constant, complex(0.0, -486.67), complex(0.0, 218.82))
        if refused:
            with pytest.raises(TranslationError, match="reflects"):
                judged("Port1", found, 26e9, "the top of the band")
        else:
            said = judged("Port1", found, 26e9, "the top of the band")
            assert said.endswith(
                "Palace matches the port to the real part of the constant, so as a TE mode "
                f"its face reflects {reflected(constant):.2g} of the power, and two faces in "
                "phase 4 times that"
            )

    def test_a_mode_lossless_to_the_solves_own_tolerance_is_said_as_before(self):
        """An imaginary part under a millionth of the real one is the solve's."""
        rounded = complex(459.06, -2e-4)
        said = judged("Port1", (rounded, *ONE_MODE[1:]), 26e9, "the top of the band")
        assert "reflects" not in said

    def test_a_mode_losing_to_its_walls_is_said_with_what_its_face_reflects(self):
        """Brass walls: TE10 at 459.09-0.0375i per metre, a loss far under the
        bar and far past the solve's own rounding."""
        walls = complex(459.09, -0.0375)
        said = judged("Port1", (walls, *ONE_MODE[1:]), 26e9, "the top of the band")
        assert f"its face reflects {reflected(walls):.2g} of the power" in said

    def test_a_lossy_filling_carrying_nothing_says_it_may_hide_the_guides_mode(self):
        """At 5 S/m the mode run found TE01, TE20 and TE11 and not TE10, which
        propagates there."""
        found = (complex(331.3, -716.0), complex(389.0, -609.0), complex(310.0, -765.0))
        with pytest.raises(TranslationError, match="can also hide the guide's own mode"):
            judged("Port1", found, 12e9, "the bottom of the band", lossy=True)

    def test_lossy_walls_carrying_nothing_are_not_told_about_a_filling(self):
        """Below cutoff between walls of finite conductivity, a constant carries
        both parts, and there is no filling to change."""
        found = (complex(0.1, -151.0), complex(0.2, -480.0), complex(0.3, -700.0))
        with pytest.raises(TranslationError) as refused:
            judged("Port1", found, 12e9, "the bottom of the band")
        assert "filling" not in str(refused.value)

    def test_a_lossless_face_carrying_nothing_is_not_told_about_loss(self):
        below = (complex(0.0, 700.0), complex(0.0, 300.0), complex(0.0, 500.0))
        with pytest.raises(TranslationError) as refused:
            judged("Port1", below, 26e9, "the top of the band")
        assert "lossy" not in str(refused.value)

    def test_a_mode_that_does_not_fall_at_all_is_said_rather_than_divided_by(self):
        said = judged(
            "Port1", (complex(459.06, 0.0), 0j, complex(0.0, 692.8)), 26e9, "the top of the band"
        )
        assert said.endswith("no distance a number states")


class TestAPortReadsItsVoltageAcrossTheMiddleOfItsBroadSide:
    """Where Palace reads a port's mode voltage, and so states its power-voltage
    impedance, and where it does not.

    The face is given here as the mesher measures it, since the line is placed
    on the face as meshed and not as drawn.
    """

    def across(self, inward, lower, upper, size=None, **changed):
        described = replace(problem(guide()[0]), **changed)
        feed = replace(described.feeds[0], inward=inward)
        described = replace(described, feeds=(feed, *described.feeds[1:]))
        mesh = meshed(described)
        spans = sorted(high - low for low, high in zip(lower, upper, strict=True))[1:]
        labels = dict(mesh.labels)
        labels[feed.label] = replace(
            labels[feed.label],
            lower=lower,
            upper=upper,
            size=spans[0] * spans[1] if size is None else size,
        )
        return voltage_across(feed, described, replace(mesh, labels=labels))

    @pytest.mark.parametrize(
        "inward, lower, upper, line",
        [
            # Along z, broad on x and then on y.
            ((0, 0, 1), (0, 0, 5), (10.7, 4.3, 5), ((5.35, 0, 5), (5.35, 4.3, 5))),
            ((0, 0, -1), (0, 0, 5), (4.3, 10.7, 5), ((0, 5.35, 5), (4.3, 5.35, 5))),
            # Along x, broad on y and then on z.
            ((1, 0, 0), (2, 1, 1), (2, 11.7, 5.3), ((2, 6.35, 1), (2, 6.35, 5.3))),
            ((-1, 0, 0), (2, 1, 1), (2, 5.3, 11.7), ((2, 1, 6.35), (2, 5.3, 6.35))),
            # Along y, broad on x and then on z.
            ((0, 1, 0), (0, 3, 0), (10.7, 3, 4.3), ((5.35, 3, 0), (5.35, 3, 4.3))),
            ((0, -1, 0), (0, 3, 0), (4.3, 3, 10.7), ((0, 3, 5.35), (4.3, 3, 5.35))),
        ],
    )
    def test_the_line_crosses_the_narrow_side_upwards_at_the_middle_of_the_broad(
        self, inward, lower, upper, line
    ):
        """Upwards along the global axis whichever way the port faces, so every
        port's mode is turned by one rule."""
        got = self.across(inward, lower, upper)
        assert got is not None
        assert np.allclose(got, line, rtol=0.0, atol=1e-12)

    def test_a_square_states_none(self):
        """Its broad side is not decided, and neither is the way its mode's field
        stands."""
        assert self.across((0, 0, 1), (0, 0, 5), (4.3, 4.3, 5)) is None

    def test_a_face_that_does_not_fill_its_rectangle_states_none(self):
        """The middle of the rectangle need not be the middle of the guide, and a
        line that leaves the face reads nothing there."""
        assert self.across((0, 0, 1), (0, 0, 5), (10.7, 4.3, 5), size=10.7 * 4.3 * 0.9) is None

    def test_a_model_that_dissipates_states_none(self):
        """The impedance is complex there, and Palace states ``V V*``, real by
        construction."""
        described = problem(guide()[0])
        lossy = tuple(
            replace(region, filling=Filling(permittivity=2.2, loss_tangent=1e-3))
            for region in described.regions
        )
        assert self.across((0, 0, 1), (0, 0, 5), (10.7, 4.3, 5), regions=lossy) is None

    def test_a_face_the_mesher_did_not_measure_is_refused_naming_the_port(self):
        """A port written without its line has its mode turned by the solver's
        own rule, so the line is never left out for want of a figure."""
        described = problem(guide()[0])
        mesh = meshed(described)
        feed = described.feeds[0]
        labels = dict(mesh.labels)
        labels[feed.label] = replace(labels[feed.label], lower=None, upper=None, size=None)
        with pytest.raises(TranslationError, match=r"'Port1'.* the box round its face"):
            voltage_across(feed, described, replace(mesh, labels=labels))

    def test_a_face_no_region_is_said_to_stand_on_is_refused_naming_the_port(self):
        described = problem(guide()[0])
        mesh = self.measured(described, meshed(described))
        feed = described.feeds[0]
        labels = dict(mesh.labels)
        labels[feed.label] = replace(labels[feed.label], beside=())
        with pytest.raises(TranslationError, match=r"'Port1'.* which regions stand on its face"):
            voltage_across(feed, described, replace(mesh, labels=labels))

    @staticmethod
    def measured(described, mesh):
        """Every port's face measured as a WR-42 end, flat along its own axis."""
        labels = dict(mesh.labels)
        for feed in described.feeds:
            normal = next(axis for axis, part in enumerate(feed.inward) if part)
            sides = iter((10.7, 4.3))
            upper = tuple(0.0 if axis == normal else next(sides) for axis in range(3))
            labels[feed.label] = replace(
                labels[feed.label], lower=(0.0, 0.0, 0.0), upper=upper, size=10.7 * 4.3
            )
        return replace(mesh, labels=labels)

    def test_the_driven_run_carries_each_ports_line(self):
        described = problem(guide()[0])
        run = configured(described, self.measured(described, meshed(described)), "results")
        assert all(port.voltage is not None for port in run.ports)

    def test_a_face_divided_into_parts_states_none(self):
        """Each part carries a mode of its own, and a line across the whole face
        reads none of them alone."""
        described = with_a_sheet()
        mesh = cross_section(
            described,
            "Port1",
            HALVES,
            HALVES_END,
            {described.wall: HALVES_ROUND, "SeptumMetal": (ACROSS,)},
        )
        (feed,) = [feed for feed in described.feeds if feed.label == "Port1"]
        assert voltage_across(feed, described, self.measured(described, mesh)) is None

    def test_a_fin_on_the_face_states_none(self):
        """The ridged guide it makes is taken, and the line across the middle
        of its broad side would lie partly on the fin."""
        described = with_a_sheet()
        floor, fin = (101, 108), 105
        mesh = cross_section(
            described,
            "Port1",
            {WHOLE: (*floor, 102, 103, 104, fin)},
            {**ROUND_END, 101: (1001, 1007), 108: (1007, 1002), fin: (1007, 1008)},
            {described.wall: (*floor, 102, 103, 104), "SeptumMetal": (fin,)},
        )
        (feed,) = [feed for feed in described.feeds if feed.label == "Port1"]
        assert configured(described, mesh, "results").ports
        assert voltage_across(feed, described, self.measured(described, mesh)) is None

    def test_a_face_cut_into_pieces_in_one_filling_carries_the_line(self):
        """A shape drawn across the face cuts it where it meets it, and the
        pieces together are still the face: the line is the uncut face's."""
        described = problem(guide()[0])
        mesh = self.measured(described, meshed(described))
        feed = described.feeds[0]
        whole = voltage_across(feed, described, mesh)
        labels = dict(mesh.labels)
        labels[feed.label] = replace(
            labels[feed.label], entities=(*labels[feed.label].entities, 77)
        )
        rims = {**mesh.rims, 2: {**mesh.rims[2], 77: ((1, 77),)}}
        assert whole is not None
        assert voltage_across(feed, described, replace(mesh, labels=labels, rims=rims)) == whole

    def test_a_face_standing_on_two_fillings_states_none(self):
        """The mode's field need not stand across the narrow side there."""
        described = problem(guide()[0])
        (region,) = described.regions
        other = replace(region, label="Slab", filling=Filling(permittivity=4.4))
        described = replace(described, regions=(region, other))
        mesh = self.measured(described, meshed(described))
        feed = described.feeds[0]
        labels = dict(mesh.labels)
        labels[feed.label] = replace(labels[feed.label], beside=(region.label, "Slab"))
        assert voltage_across(feed, described, replace(mesh, labels=labels)) is None

    def test_a_face_standing_on_two_regions_of_one_filling_carries_the_line(self):
        """Two bindings of one material are one filling to the mode."""
        described = problem(guide()[0])
        (region,) = described.regions
        described = replace(described, regions=(region, replace(region, label="Twin")))
        mesh = self.measured(described, meshed(described))
        feed = described.feeds[0]
        labels = dict(mesh.labels)
        labels[feed.label] = replace(labels[feed.label], beside=(region.label, "Twin"))
        assert voltage_across(feed, described, replace(mesh, labels=labels)) is not None

    def test_a_sheet_of_finite_conductivity_anywhere_states_none(self):
        """It dissipates whether or not it meets the face."""
        described = with_a_sheet(material=sheet_metal())
        feed = described.feeds[0]
        assert voltage_across(feed, described, self.measured(described, meshed(described))) is None

    def test_a_conducting_filling_states_none(self):
        described = problem(guide()[0])
        conducting = tuple(
            replace(region, filling=Filling(conductivity=1e-2)) for region in described.regions
        )
        assert self.across((0, 0, 1), (0, 0, 5), (10.7, 4.3, 5), regions=conducting) is None

    def test_the_line_reaches_palace_and_the_mode_run_leaves_it_out(self):
        """The mode run asks which modes the face carries, and nothing of it is
        read along a line."""
        port = WavePort(
            index=1, attributes=(2,), behind=(0, 0, 1), voltage=((0, 0, 0), (0, 0, 4.3))
        )
        assert port.to_dict(1)["VoltagePath"] == [[0, 0, 0], [0, 0, 4.3]]
        described = problem(guide()[0])
        run = replace(configured(described, meshed(described), "results"))
        run = replace(
            run, ports=tuple(replace(p, voltage=((0, 0, 0), (0, 0, 1))) for p in run.ports)
        )
        (mine,) = run.port_modes(1, 26e9, 3, "modes/1", frozenset())["Boundaries"]["WavePort"]
        assert "VoltagePath" not in mine


class TestTheModeRunWritten:
    """A port's face, cut out of the driven run's own model, with the other
    ports taken as the port's own mode solve takes them."""

    def driven(self, **changed):
        """The guide's driven run, as the adapter configures it."""
        described = problem(guide()[0])
        return replace(configured(described, meshed(described), "results"), **changed)

    def test_the_port_under_check_is_the_one_port_left(self):
        run = self.driven()
        written = run.port_modes(2, 26e9, 3, "modes/2", frozenset())
        (port,) = written["Boundaries"]["WavePort"]
        assert port["Index"] == 2
        assert written["Solver"]["BoundaryMode"] == {
            "Freq": 26.0,
            "N": 3,
            "Attributes": port["Attributes"],
        }
        assert written["Problem"]["Type"] == "BoundaryMode"
        assert written["Problem"]["Output"] == "modes/2"

    def test_every_other_port_is_a_perfect_conductor(self):
        """Left out, its faces would be named by no condition, which Palace takes
        for a magnetic wall with a warning."""
        run = self.driven()
        written = run.port_modes(1, 26e9, 3, "modes/1", frozenset())
        (other,) = [port for port in run.ports if port.index == 2]
        assert set(other.attributes) <= set(written["Boundaries"]["PEC"]["Attributes"])
        assert set(run.perfect_conductor) <= set(written["Boundaries"]["PEC"]["Attributes"])

    def test_a_model_with_no_perfect_conductor_gains_one_in_the_other_ports(self):
        run = self.driven(perfect_conductor=(), conducting=(SurfaceConductivity((9,), 1.57e7),))
        written = run.port_modes(1, 26e9, 3, "modes/1", frozenset({9}))
        (other,) = [port for port in run.ports if port.index == 2]
        assert written["Boundaries"]["PEC"]["Attributes"] == sorted(other.attributes)

    def test_metal_of_finite_conductivity_is_kept_where_it_meets_the_face(self):
        """A sheet standing clear of the port stands nowhere on the face the
        problem is solved on, and a condition on an attribute the face does not
        carry is a warning, which a run here fails on."""
        walls, sheet = SurfaceConductivity((7,), 1.57e7), SurfaceConductivity((8,), 1.57e7)
        run = self.driven(conducting=(walls, sheet))
        written = run.port_modes(1, 26e9, 3, "modes/1", frozenset({7}))
        assert written["Boundaries"]["Conductivity"] == [walls.to_dict()]
        bare = run.port_modes(1, 26e9, 3, "modes/1", frozenset())
        assert "Conductivity" not in bare["Boundaries"]

    def test_metal_where_the_model_ends_clear_of_the_face_is_a_wall_in_the_mode_run(self):
        """Left out, that outer face is named by no condition, which Palace takes
        for a magnetic wall with a warning; it stands nowhere on the face, so a
        perfect conductor there changes nothing the problem solves."""
        outer = SurfaceConductivity((8,), 2e5, thickness=0.05, external=True)
        # A sheet standing both inside and where the model ends is written
        # with no side, and its outer faces are as bare as a wall's.
        both = SurfaceConductivity((9,), 2e5)
        run = self.driven(conducting=(outer, both))
        written = run.port_modes(1, 26e9, 3, "modes/1", frozenset())
        walls = written["Boundaries"]["PEC"]["Attributes"]
        assert {8, 9} <= set(walls)
        assert "Conductivity" not in written["Boundaries"]
        meeting = run.port_modes(1, 26e9, 3, "modes/1", frozenset({8}))
        assert 8 not in meeting["Boundaries"]["PEC"]["Attributes"]
        assert meeting["Boundaries"]["Conductivity"] == [outer.to_dict()]

    def test_the_driven_run_is_left_as_it_was(self):
        run = self.driven()
        before = run.to_dict()
        run.port_modes(1, 26e9, 3, "modes/1", frozenset())
        assert run.to_dict() == before

    def test_each_port_is_written_to_a_file_and_a_directory_of_its_own(self, tmp_path):
        run = self.driven()
        made = {
            (port, edge): configure_modes(run, port, 26e9, 3, tmp_path, frozenset(), edge)
            for port in (1, 2)
            for edge in ("bottom", "top")
        }
        assert len({path for path, _ in made.values()}) == len(made)
        assert len({answered for _, answered in made.values()}) == len(made)
        first, answered = made[(1, "top")]
        assert (
            json.loads(first.read_text())["Problem"]["Output"]
            == answered.relative_to(tmp_path).as_posix()
        )


class TestTheModeTableRead:
    def test_each_constant_comes_back_complex_in_the_order_written(self, tmp_path):
        mode_table(tmp_path, ONE_MODE)
        assert modes(tmp_path) == ONE_MODE

    def test_a_missing_table_is_no_answer(self, tmp_path):
        with pytest.raises(ResultsError, match="left no table"):
            modes(tmp_path)


def scattered(magnitudes, flux, out=(1, 2), driven=(1,), dissipates=False, frequency=(26e9,)):
    """A run's answer as the reader hands one on, from the magnitudes and the
    powers through the faces a run returned, each indexed by sample, row and
    column. The phases change nothing any figure here is read from."""
    return Scattering(
        frequency=np.asarray(frequency, dtype=float),
        out=out,
        driven=driven,
        matrix=np.asarray(magnitudes, dtype=complex),
        flux=np.asarray(flux, dtype=float),
        dissipates=dissipates,
    )


#: What Palace returned for WR-42 driven from port 1 at 26 GHz, order 3 and six
#: elements a wavelength, with an off-centre post of perfect conductor a
#: millimetre from port 1: |S11| and |S21|, and the power leaving through each
#: face.
CLOSE_POST = scattered([[[0.163969], [0.915269]]], [[[-0.8311349076398], [0.8384417795567]]])

#: The same with the post centred and 20 mm from each port, at 27.9 GHz, where
#: nothing is left unaccounted for and the powers through the faces carry only
#: their own error.
CENTRED_POST = scattered(
    [[[0.579189], [0.815193]]], [[[-0.6649139436488], [0.6656246600318]]], frequency=(27.9e9,)
)

#: The guide filled with a dielectric of loss tangent 0.02, at 12, 14 and 16 GHz,
#: order 2 and six elements a wavelength: a fifth of the power turns to heat, and
#: the powers through the faces are measured high enough to account for more
#: than went in.
COARSE_FILLING = scattered(
    [[[0.023205], [0.886412]], [[0.002705], [0.888846]], [[0.013141], [0.884244]]],
    [
        [[-1.006857993496], [0.7904619713459]],
        [[-1.014544475718], [0.8001597600712]],
        [[-1.022356593103], [0.7978761110625]],
    ],
    dissipates=True,
    frequency=(12e9, 14e9, 16e9),
)

#: The empty guide with brass walls at 20, 23 and 26 GHz, order 3 and six
#: elements a wavelength, as Palace 0.18.0 returned it: the most unaccounted for
#: is at the bottom of the band.
BRASS_WALLS = scattered(
    [[[0.000181], [0.998039]], [[0.000048], [0.998230]], [[0.000183], [0.998319]]],
    [
        [[-0.9998482796438], [0.9963582822449]],
        [[-1.000002431916], [0.9968691626092]],
        [[-1.000270015746], [0.997298405101]],
    ],
    dissipates=True,
    frequency=(20e9, 23e9, 26e9),
)

NAMES = {1: "Port1", 2: "Port2", 3: "Port3", 7: "Port7"}


class TestWhatTheMatrixLeavesUnaccountedFor:
    """The power a port absorbs besides its mode, read off the power through
    its face and the matrix that went with it."""

    def test_a_model_that_dissipates_nothing_is_held_to_its_matrix_alone(self):
        """The powers through the faces sum to their own error where nothing
        turns power into heat, and counting them would report it."""
        (short,) = balance.shortfalls(CENTRED_POST)
        power = 0.579189**2 + 0.815193**2
        assert short.share == pytest.approx(1.0 - power, abs=1e-12)

    def test_a_model_that_dissipates_is_credited_the_heat_its_faces_say(self):
        """What the faces' powers sum to is heat, which separates the walls' own
        loss from a port's: read as dissipating nothing, the same run leaves
        more than the bar unaccounted for."""
        (short,) = balance.shortfalls(BRASS_WALLS)
        heat = -(-0.9998482796438 + 0.9963582822449)
        power = 0.000181**2 + 0.998039**2
        assert short.share == pytest.approx(1.0 - power - heat, abs=1e-12)
        assert abs(short.share) <= balance.BAR
        (bare,) = balance.shortfalls(replace(BRASS_WALLS, dissipates=False))
        assert bare.share > balance.BAR

    def test_the_port_that_took_it_is_named(self):
        (short,) = balance.shortfalls(CLOSE_POST)
        assert short.excitation == 1
        assert short.port == 1
        assert short.share == pytest.approx(1.0 - 0.163969**2 - 0.915269**2, abs=1e-12)

    def test_the_watt_that_came_in_is_added_back_at_the_port_that_drove_it(self):
        """Matched by the port's number and not by where its column stands: a
        run driven from port 2 alone has that port's column first."""
        (short,) = balance.shortfalls(
            scattered(
                [[[0.915269], [0.163969]]],
                [[[0.8384417795567], [-0.8311349076398]]],
                driven=(2,),
            )
        )
        assert short.excitation == 2
        assert short.port == 2

    def test_ports_numbered_otherwise_are_named_by_their_numbers(self):
        (short,) = balance.shortfalls(replace(CLOSE_POST, out=(3, 7), driven=(3,)))
        assert (short.excitation, short.port) == (3, 3)

    def test_the_sample_reported_is_the_one_leaving_the_most_either_way(self):
        """The coarse filling accounts for more than went in at every sample,
        and most at the top of the band."""
        (short,) = balance.shortfalls(COARSE_FILLING)
        assert short.frequency == 16e9
        heat = -(-1.022356593103 + 0.7978761110625)
        assert short.share == pytest.approx(1.0 - 0.013141**2 - 0.884244**2 - heat, abs=1e-12)
        assert short.share < 0

    def test_the_sample_reported_need_not_be_the_last(self):
        (short,) = balance.shortfalls(BRASS_WALLS)
        assert short.frequency == 20e9

    def test_more_accounted_for_than_went_in_names_the_port_short_of_it(self):
        """Where the share is negative the port named is the one whose face
        carried less than its term of the matrix says: the driven port, here."""
        (short,) = balance.shortfalls(COARSE_FILLING)
        assert short.port == 1

    def test_each_driven_port_is_answered_for_on_its_own(self):
        both = scattered(
            [[[0.163969, 0.915269], [0.915269, 0.163969]]],
            [[[-0.8311349076398, 0.8384417795567], [0.8384417795567, -0.8311349076398]]],
            driven=(1, 2),
        )
        assert [(short.excitation, short.port) for short in balance.shortfalls(both)] == [
            (1, 1),
            (2, 2),
        ]


class TestWhatTheRunSaysOfIt:
    def short(self, share, port=2):
        return balance.Shortfall(excitation=1, frequency=27.9e9, share=share, port=port)

    def test_a_share_within_the_bar_is_stated_and_not_warned_of(self):
        """The port that took the most is not named: within the bar, which port
        it is follows the measurement's own error."""
        line = balance.said(self.short(balance.BAR), NAMES)
        assert not line.startswith(balance.WARNING)
        assert "Driven from 'Port1'" in line and "0.1%" in line and "27.9 GHz" in line
        assert "'Port2'" not in line

    def test_a_share_past_the_bar_is_a_warning_naming_both_ports_and_the_figure(self):
        line = balance.said(self.short(0.0599), NAMES)
        assert line.startswith(balance.WARNING)
        assert "5.99% of the power unaccounted for" in line
        assert "Driven from 'Port1'" in line and "through 'Port2'" in line

    def test_more_accounted_for_than_went_in_past_the_bar_is_a_warning_too(self):
        line = balance.said(self.short(-6.54e-3), NAMES)
        assert line.startswith(balance.WARNING)
        assert "for 0.654% more power than went in" in line

    def test_nothing_unaccounted_for_is_said_as_nothing_left_over(self):
        line = balance.said(self.short(0.0), NAMES)
        assert "leaves 0% of the power unaccounted for" in line

    def test_more_accounted_for_within_the_bar_is_stated(self):
        line = balance.said(self.short(-1e-4), NAMES)
        assert not line.startswith(balance.WARNING)
        assert "0.01% more power" in line

    def test_the_bar_is_the_one_the_palace_gates_hold_power_to(self):
        assert balance.BAR == WAVEGUIDE_POWER == SEPTUM_POWER


class TestThePowerThroughThePortsRead:
    def test_each_face_is_read_under_its_port_and_the_port_that_drove_it(self, tmp_path):
        wrote(tmp_path, out=(1, 2), driven=(1, 2))
        powers(tmp_path, out=(1, 2), driven=(1, 2), taken={(2, 1): 0.25})
        answer = scattering(tmp_path, driven())
        assert answer.flux.shape == (13, 2, 2)
        assert answer.flux[0, 1, 0] == pytest.approx(abs(expected(2, 1)) ** 2 + 0.25)
        assert answer.flux[0, 0, 1] == pytest.approx(abs(expected(1, 2)) ** 2)
        assert answer.flux[0, 1, 1] == pytest.approx(abs(expected(2, 2)) ** 2 - 1.0)

    def test_a_run_driven_from_one_port_names_no_excitation(self, tmp_path):
        run = driven(
            ports=(
                WavePort(index=1, attributes=(3,), behind=INSIDE, excited=True),
                WavePort(index=2, attributes=(4,), behind=INSIDE),
            )
        )
        wrote(tmp_path, out=(1, 2), driven=(1,))
        path = powers(tmp_path, out=(1, 2), driven=(1,))
        assert "Φ_pow[2] (W)" in path.read_text(encoding="utf-8")
        assert scattering(tmp_path, run).flux.shape == (13, 2, 1)

    def test_whether_the_model_dissipates_is_the_runs(self, tmp_path):
        wrote(tmp_path, out=(1, 2), driven=(1, 2))
        powers(tmp_path, out=(1, 2), driven=(1, 2))
        assert not scattering(tmp_path, driven()).dissipates
        lossy = driven(conducting=(SurfaceConductivity((9,), 1.57e7),))
        assert scattering(tmp_path, lossy).dissipates

    def test_a_run_that_left_no_table_of_them_is_no_answer(self, tmp_path):
        wrote(tmp_path, out=(1, 2), driven=(1, 2))
        with pytest.raises(ResultsError, match="power through the ports"):
            scattering(tmp_path, driven())

    def test_a_table_of_them_shorter_than_the_band_is_a_run_that_stopped(self, tmp_path):
        wrote(tmp_path, out=(1, 2), driven=(1, 2))
        powers(tmp_path, out=(1, 2), driven=(1, 2), rows=4)
        with pytest.raises(ResultsError, match="4 of the 13 samples"):
            scattering(tmp_path, driven())

    def test_a_table_of_them_over_another_band_is_not_this_runs(self, tmp_path):
        wrote(tmp_path, out=(1, 2), driven=(1, 2))
        powers(tmp_path, out=(1, 2), driven=(1, 2), start=21.0)
        with pytest.raises(ResultsError, match="not one run's"):
            scattering(tmp_path, driven())

    def test_a_port_missing_from_it_is_refused_by_the_column(self, tmp_path):
        wrote(tmp_path, out=(1, 2), driven=(1, 2))
        powers(tmp_path, out=(1,), driven=(1, 2))
        with pytest.raises(ResultsError, match=r"no column 'Φ_pow\[2\]\[1\] \(W\)'"):
            scattering(tmp_path, driven())


class TestThePowerAskedFor:
    def test_every_port_asks_for_the_power_through_its_face_from_behind_it(self):
        ports = (
            WavePort(index=3, attributes=(5, 6), behind=(1.0, 2.0, 3.0), excited=True),
            WavePort(index=7, attributes=(4,), behind=INSIDE),
        )
        written = driven(ports=ports).to_dict()["Boundaries"]["Postprocessing"]["SurfaceFlux"]
        assert written == [
            {"Index": 3, "Attributes": [5, 6], "Type": "Power", "Center": [1.0, 2.0, 3.0]},
            {"Index": 7, "Attributes": [4], "Type": "Power", "Center": list(INSIDE)},
        ]

    def test_a_mode_run_asks_for_none(self):
        """It is a question about the driven field, and a mode run would build
        it on a model that has none."""
        written = driven().port_modes(1, 26e9, 3, "modes/1", frozenset())
        assert "Postprocessing" not in written["Boundaries"]

    def test_a_port_measured_from_nowhere_is_refused(self):
        with pytest.raises(ValueError, match="three finite coordinates"):
            WavePort(index=1, attributes=(3,), behind=(0.0, math.nan, 0.0))
        with pytest.raises(ValueError, match="three finite coordinates"):
            WavePort(index=1, attributes=(3,), behind=(0.0, 0.0))

    def test_a_model_of_perfect_metal_and_vacuum_dissipates_nothing(self):
        assert not driven().dissipates

    @pytest.mark.parametrize(
        "changed",
        [
            {"materials": (Material((1,), Filling(permittivity=4.3, loss_tangent=0.02)),)},
            {"materials": (Material((1,)), Material((5,), Filling(conductivity=0.1)))},
            {"conducting": (SurfaceConductivity((9,), 1.57e7),)},
        ],
        ids=["loss tangent", "conductivity", "finite metal"],
    )
    def test_anything_that_turns_power_into_heat_dissipates(self, changed):
        assert driven(**changed).dissipates


class TestWherePortsAreMeasuredFrom:
    @pytest.mark.parametrize(
        "axis, moved",
        [("Y", (0.0, 5.35 + 11.5, 2.15)), ("-Z", (0.0, 5.35, 2.15 - 11.5))],
    )
    def test_each_way_in_moves_the_point_along_itself(self, axis, moved):
        found, body = guide()
        body.Shape.Faces[0] = Drawn("pipe.face1", reach=11.5, spans=square_to(axis))
        (first,) = [member for member in found.Group if getattr(member, "Number", 0) == 1]
        first.PropagationAxis = axis
        (feed,) = [feed for feed in problem(found).feeds if feed.number == 1]
        assert feed.behind == pytest.approx(moved)

    def test_the_point_stands_into_the_model_by_the_faces_own_reach(self):
        """Down the port's way in from the face's centre of mass: for a port at
        the far end of a guide 20 mm long, back along x."""
        found, body = guide()
        body.Shape.Faces[1] = Drawn("pipe.face2", centre=(20.0, 5.35, 2.15), reach=11.5)
        (far,) = [member for member in found.Group if getattr(member, "Number", 0) == 2]
        far.PropagationAxis = "-X"
        (second,) = [feed for feed in problem(found).feeds if feed.number == 2]
        assert second.behind == pytest.approx((20.0 - 11.5, 5.35, 2.15))

    def test_the_point_is_handed_to_the_port_palace_is_given(self):
        described = problem(guide()[0])
        run = configured(described, meshed(described), "results")
        by_number = {feed.number: feed.behind for feed in described.feeds}
        assert {port.index: port.behind for port in run.ports} == by_number


class Boxed(Drawn):
    """A solid standing between two corners, with faces a pick can name."""

    def __init__(self, name, lower, upper, faces=()):
        super().__init__(name)
        self.BoundBox = SimpleNamespace(
            XMin=lower[0],
            YMin=lower[1],
            ZMin=lower[2],
            XMax=upper[0],
            YMax=upper[1],
            ZMax=upper[2],
        )
        self.Faces = list(faces)
        self.Solids = [self]

    def getElement(self, name):
        return self.Faces[int(name[len("Face") :]) - 1]


def flat(name, lower, upper):
    """A face standing between two corners: it holds itself and no solid."""
    face = Boxed(name, lower, upper)
    face.Solids = []
    face.Faces = [face]
    return face


def refinement(references, **overrides):
    """A mesh refinement region as the document carries one."""
    properties = dict(
        Mode="Refine",
        References=references,
        ElementSize=0.5,
        MinElementsAcross=0,
        Enabled=True,
    )
    properties.update(overrides)
    return obj("EMMeshRegion", "Ring", **properties)


def with_members(*added, settings=None, recipe=None):
    """The guide, with ``added`` among its members and its policy or its recipe
    replaced where one is given."""
    found, _ = guide()
    members = list(found.Group)
    for replacing, wanted in ((settings, "EMMeshPolicy"), (recipe, "EMGmshMesh")):
        if replacing is not None:
            members = [member for member in members if kind(member) != wanted]
            members.append(replacing)
    found.Group = [*members, *added]
    return found


#: What the Gmsh mesh declares rather than the mesh policy. An override below
#: goes to the object that declares it, so a test names the property and not
#: the object it happens to sit on today.
OF_THE_RECIPE = ("ElementsPerWavelength", "EdgeRefinement", "MaxGrowthRatio")


def two_sheets(**asked):
    """The guide with a perfect conductor and a sheet of brass drawn in it."""
    return with_members(
        metal_binding(septum("Septum")),
        metal_binding(septum("Film"), material=sheet_metal()),
        settings=mesh_settings(
            **{name: value for name, value in asked.items() if name not in OF_THE_RECIPE}
        ),
        recipe=gmsh_mesh(**{name: value for name, value in asked.items() if name in OF_THE_RECIPE}),
    )


class TestTheSizeAtTheEdgeOfMetal:
    """``EdgeRefinement`` sizes the edges of every metal the document binds where
    the room turns round them past half a turn, and nothing else."""

    def test_each_metal_has_its_edges_sized_finer_than_the_bulk(self):
        asked = problem(two_sheets(EdgeRefinement=6.0)).demand
        assert {place.label for place in asked.places} == {"SeptumMetal", "FilmMetal"}
        assert all(isinstance(place, AtRim) and place.reentrant for place in asked.places)
        assert all(place.size == pytest.approx(asked.coarsest / 6.0) for place in asked.places)

    def test_every_metal_and_the_wall_end_the_room_and_each_plane_mirrors_it(self):
        """What the mesher reads an opening against is every metal, whether or not
        a size is laid at it, so a coarsening moves no other metal's edges."""
        asked = problem(two_sheets(EdgeRefinement=6.0)).demand
        assert set(asked.walls) == {"SeptumMetal", "FilmMetal", WALL}
        assert asked.mirrors == ()
        bare = problem(plates())
        coarsened = problem(plates((["Face1"], bare.demand.coarsest)))
        assert "PlateMetal" not in {place.label for place in coarsened.demand.places}
        assert coarsened.demand.walls == bare.demand.walls
        assert "PlateMetal" in coarsened.demand.walls

    def test_a_finer_refinement_asks_for_a_finer_rim(self):
        six = problem(two_sheets(EdgeRefinement=6.0)).demand
        twelve = problem(two_sheets(EdgeRefinement=12.0)).demand
        assert twelve.places[0].size == pytest.approx(six.places[0].size / 2.0)

    def test_a_refinement_of_one_asks_for_no_place(self):
        assert problem(two_sheets(EdgeRefinement=1.0)).demand.places == ()

    def test_a_refinement_below_one_is_refused(self):
        with pytest.raises(TranslationError, match="EdgeRefinement is 0.5"):
            problem(two_sheets(EdgeRefinement=0.5))

    def test_a_model_with_no_metal_asks_for_the_bulk_alone(self):
        """The rest of the boundary is the sides of a box, which the room stands
        inside, so no crease of it asks a size."""
        asked = problem(guide()[0]).demand
        assert (asked.places, asked.growth) == ((), 0.0)

    def test_the_growth_is_the_policys(self):
        assert problem(two_sheets(MaxGrowthRatio=1.7)).demand.growth == 1.7

    def test_a_growth_of_one_or_less_is_refused_where_a_place_is_asked_for(self):
        for growth in (1.0, 0.8):
            with pytest.raises(TranslationError, match=f"MaxGrowthRatio is {growth:g}"):
                problem(two_sheets(MaxGrowthRatio=growth))
        assert problem(two_sheets(MaxGrowthRatio=1.0, EdgeRefinement=1.0)).demand.places == ()

    def test_a_rim_finer_than_the_floor_is_laid_at_the_floor_and_said(self):
        """The floor is as fine as the policy lets any element be, so the rim is
        laid there, and the mesher is never handed a size below its floor."""
        coarsest = problem(two_sheets(EdgeRefinement=6.0)).demand.coarsest
        floor = coarsest / 3.0
        described = problem(two_sheets(EdgeRefinement=6.0, MinElementSize=floor))
        assert {place.size for place in described.demand.places} == {floor}
        assert coverage.asked(described.demand, PROFILE) == []
        (line,) = [
            line
            for line in unlaid_said(described.unlaid, answered(described))
            if "EdgeRefinement" in line
        ]
        assert f"asks for {coarsest / 6.0:.4g} mm" in line
        assert f"laid at {floor:.4g} mm, the floor under every element" in line

    def test_a_rim_above_the_floor_is_said_nothing_about_it(self):
        _, said = TestWhatIsStatedAndNotLaid().lines(two_sheets(MinElementSize=0.01))
        assert not any("floor under every element" in line for line in said)


def marked(asked):
    """The places a refinement region lays, which the rims of metal and of the
    wall are not."""
    return tuple(place for place in asked.places if not isinstance(place, AtRim))


class TestARefinementRegion:
    """A region set to refine sizes the box round each thing it names."""

    LOWER, UPPER = (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)

    def rod(self):
        return obj(
            "Part::Feature",
            "Rod",
            Shape=Boxed("rod", self.LOWER, self.UPPER, faces=[flat("top", (1, 2, 6), (4, 5, 6))]),
        )

    def test_it_is_laid_at_what_it_names_at_its_size(self):
        """The shape is handed over as a mark and the size is asked near it, so a
        body is sized throughout itself and nowhere its box adds."""
        rod = self.rod()
        described = problem(with_members(refinement([(rod, [""])])))
        (place,) = marked(described.demand)
        (mark,) = described.marks
        assert isinstance(place, Near)
        assert (place.label, place.size) == (mark.name, 0.5)
        assert (mark.dimension, mark.shape, mark.thinnest) == (3, rod.Shape, 3.0)
        assert "Ring" in place.name and "Rod" in place.name

    def test_a_face_it_names_is_a_mark_of_its_own_dimension(self):
        described = problem(with_members(refinement([(self.rod(), ["Face1"])])))
        (mark,) = described.marks
        assert (mark.dimension, mark.thinnest) == (2, 0.0)
        assert "Face1" in mark.name

    def test_each_thing_it_names_is_a_mark_of_its_own(self):
        rod = self.rod()
        described = problem(with_members(refinement([(rod, [""]), (rod, ["Face1"])])))
        assert len({place.name for place in marked(described.demand)}) == 2
        assert [mark.name for mark in described.marks] == [
            place.label for place in marked(described.demand)
        ]

    def test_a_region_left_out_of_the_mesh_asks_for_nothing(self):
        """Nothing about it is read, so a size it never set is not refused."""
        region = refinement([(self.rod(), [""])], Enabled=False, ElementSize=0.0)
        assert marked(problem(with_members(region)).demand) == ()

    @pytest.mark.parametrize(
        ("changed", "said"),
        [
            (dict(ElementSize=0.0), "has no element size set"),
            (dict(References=[]), "refines nothing"),
            (dict(Mode="Sideways"), "Mode is 'Sideways'"),
            (dict(MinElementsAcross=-1), "MinElementsAcross is -1"),
            (dict(Mode="Coarsen", MinElementsAcross=3), "set to Coarsen and asks for 3"),
        ],
    )
    def test_a_region_that_describes_no_refinement_is_refused_by_name(self, changed, said):
        region = refinement([(self.rod(), [""])], **changed)
        with pytest.raises(TranslationError, match="'Ring'") as refused:
            problem(with_members(region))
        assert said in str(refused.value)

    def test_a_size_no_finer_than_the_bulk_lays_nothing_and_is_said(self):
        """It asks for what is met everywhere already. The other backend's advice
        for a conductor too narrow for its grid is such a region with a count,
        and a document following it is not refused here."""
        coarsest = problem(guide()[0]).demand.coarsest
        for size in (coarsest, 2.0 * coarsest):
            region = refinement([(self.rod(), [""])], ElementSize=size, MinElementsAcross=4)
            described = problem(with_members(region))
            assert (marked(described.demand), described.marks) == ((), ())
            (line,) = [
                line
                for line in unlaid_said(described.unlaid, answered(described))
                if line.startswith("'Ring'")
            ]
            assert f"ElementSize {size:.4g} mm is not laid" in line
            assert f"asks for {coarsest:.4g} mm everywhere" in line
            assert "MinElementsAcross 4 is not laid" in line
            assert "from 0.6 mm to 3.1 mm" in line

    def test_a_size_below_the_floor_is_laid_at_the_floor_and_said(self):
        coarsest = problem(guide()[0]).demand.coarsest
        floor = coarsest / 4.0
        region = refinement([(self.rod(), [""])], ElementSize=floor / 2.0)
        described = problem(with_members(region, settings=mesh_settings(MinElementSize=floor)))
        (place,) = marked(described.demand)
        assert place.size == floor
        assert coverage.asked(described.demand, PROFILE) == []
        mesh = answered(
            described,
            **{
                place.name: Reached(
                    asked=floor, reached=0.7, standing=1.1, elements=30, dimension=3
                )
            },
        )
        (line,) = [
            line for line in unlaid_said(described.unlaid, mesh) if line.startswith("'Ring'")
        ]
        assert f"ElementSize {floor / 2.0:.4g} mm is laid at {floor:.4g} mm" in line
        assert "MinElementSize" in line and "0.7 mm" in line

    def test_a_size_at_the_floor_is_laid_as_stated_and_not_said(self):
        """The floor lays nothing finer than it, so a size standing on it is laid
        as it was asked for."""
        coarsest = problem(guide()[0]).demand.coarsest
        floor = coarsest / 4.0
        region = refinement([(self.rod(), [""])], ElementSize=floor)
        described = problem(with_members(region, settings=mesh_settings(MinElementSize=floor)))
        (place,) = marked(described.demand)
        assert place.size == floor
        mesh = answered(
            described,
            **{
                place.name: Reached(
                    asked=floor, reached=0.7, standing=1.1, elements=30, dimension=3
                )
            },
        )
        assert not [
            line for line in unlaid_said(described.unlaid, mesh) if line.startswith("'Ring'")
        ]

    def test_every_box_is_laid_under_a_name_of_its_own(self):
        """A reference listed twice is one box. Two regions carrying one label and
        naming one thing, and a box named as a sheet of metal is, are numbered,
        so the mesher is never handed two places under one name."""
        rod = self.rod()
        twice = refinement([(rod, ["Face1"]), (rod, ["Face1"])])
        assert len(marked(problem(with_members(twice)).demand)) == 1
        other = refinement([(rod, ["Face1"])], ElementSize=0.4)
        asked = problem(with_members(twice, other)).demand
        assert len({place.name for place in asked.places}) == len(asked.places) == 2
        assert coverage.asked(asked, PROFILE) == []
        sheet = metal_binding(septum("Septum"))
        sheet.Label = "Ring on Rod"
        asked = problem(with_members(sheet, refinement([(rod, [""])]))).demand
        assert {place.name for place in asked.places} == {"Ring on Rod", "Ring on Rod 2"}

    @pytest.mark.parametrize("kind_named", ["EMMaterialBinding", "EMPortRectWaveguide"])
    def test_a_mark_is_named_clear_of_every_label(self, kind_named):
        """A region of dielectric and a port are labels the mesher is handed as
        a sheet of metal is, and a place names a label or a mark by one name."""
        found = with_members(refinement([(self.rod(), [""])]))
        (named, *_) = [member for member in found.Group if kind(member) == kind_named]
        named.Label = "Ring on Rod"
        described = problem(found)
        (mark,) = described.marks
        assert mark.name == "Ring on Rod 2"
        assert mark.name not in {name for name, _, _ in described.labelled}

    def test_a_coarsening_adds_no_mark(self):
        """It names what a binding sizes, and nothing of it is handed over."""
        region = refinement([(self.rod(), [""])], Mode="Coarsen", ElementSize=5.0)
        found = with_members(region)
        for member in found.Group:
            if kind(member) == "EMMaterialBinding":
                member.References = [(self.rod(), [""])]
        described = problem(found)
        assert (marked(described.demand), described.marks) == ((), ())

    def test_a_coarsening_of_what_no_binding_names_is_refused_in_both_backends_words(self):
        """Only bound geometry has a size to settle for, and a sub-element an edit
        has left behind is matched against the bindings rather than resolved."""
        stale = refinement([(self.rod(), ["Face9"])], Mode="Coarsen", ElementSize=5.0)
        with pytest.raises(TranslationError) as refused:
            problem(with_members(stale))
        assert "'Ring' coarsens 'Rod:Face9', which no material binding names" in str(refused.value)
        with pytest.raises(TranslationError) as other:
            check_relaxations_were_used(relaxations([stale]), set())
        assert str(other.value) == str(refused.value)

    def test_a_reference_whose_shape_holds_nothing_is_refused_by_name(self):
        """FreeCAD answers a shape that holds nothing with a bounding box whose
        least corner stands past its greatest, which is no box to refine in."""
        empty = obj(
            "Part::Feature",
            "Empty",
            Shape=Boxed("empty", (sys.float_info.max,) * 3, (-sys.float_info.max,) * 3),
        )
        with pytest.raises(TranslationError, match="'Ring'") as refused:
            problem(with_members(refinement([(empty, [""])])))
        assert "'Empty', whose shape holds nothing" in str(refused.value)

    def test_a_reference_whose_shape_holds_no_element_is_refused_by_name(self):
        """What a mark is is read off what its shape holds, and a shape holding no
        solid, face, edge or vertex is nothing to lay a size at."""
        bare = Boxed("bare", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        bare.Solids = []
        hollow = obj("Part::Feature", "Hollow", Shape=bare)
        with pytest.raises(TranslationError, match="'Ring'") as refused:
            problem(with_members(refinement([(hollow, [""])])))
        assert "'Hollow', whose shape holds nothing" in str(refused.value)


class Rim:
    """An edge of a sheet: its length, how far it stands from other sheets by
    their names, and whether it is where a face meets itself."""

    def __init__(self, length, gaps=None, seam=False):
        self.Length = length
        self.gaps = dict(gaps or {})
        self.seam = seam

    def isSeam(self, face):
        return self.seam

    def isSame(self, other):
        return other is self

    def hashCode(self):
        return id(self)

    def distToShape(self, shape):
        return (self.gaps.get(shape.name, 1000.0), [], [])


class Plate(Drawn):
    """A sheet of faces each bounded by edges, two faces sharing one edge between
    them, so that edge is no part of the rim. The first edge of a face stands at
    the gaps given and the others further off."""

    def __init__(self, name, gaps=None, faces=1):
        super().__init__(name)
        shared = Rim(4.0)
        further = {other: gap + 5.0 for other, gap in (gaps or {}).items()}
        self.Faces = []
        for index in range(faces):
            face = Drawn(f"{name}.face{index + 1}")
            face.Edges = [Rim(10.0, gaps), Rim(4.0, further), Rim(10.0, further)]
            face.Edges.append(shared if faces > 1 else Rim(4.0, further))
            face.Faces = [face]
            self.Faces.append(face)
        self.Solids = []

    def getElement(self, name):
        return self.Faces[int(name[len("Face") :]) - 1]


def plates(*coarsenings, faces=1, named=None, gap=3.0, closed=False, **asked):
    """The guide with two sheets of metal drawn in it, a plate ``gap`` from a
    strip, and the coarsenings given, each as the faces of the plate it names and
    its size. A closed plate is two faces sharing every edge."""
    drawn = Plate("plate", {"strip": gap}, faces=faces)
    if closed:
        for face in drawn.Faces:
            face.Edges = drawn.Faces[0].Edges
    plate = obj("Part::Plane", "Plate", Shape=drawn)
    strip = obj("Part::Plane", "Strip", Shape=Plate("strip", {"plate": gap}))
    bound = named if named is not None else [f"Face{index + 1}" for index in range(faces)]
    regions = []
    for number, (names, size) in enumerate(coarsenings, start=1):
        region = refinement([(plate, list(names))], Mode="Coarsen", ElementSize=size)
        region.Label = f"Coarsen{number}"
        regions.append(region)
    return with_members(
        metal_binding(plate, sub=bound),
        metal_binding(strip),
        *regions,
        recipe=gmsh_mesh(**asked),
    )


class TestACoarseningOfMetal:
    """A region set to coarsen that names a binding of metal whole lays that
    binding's rim at the size it settles for."""

    def rims(self, described):
        return {place.label: place.size for place in described.demand.places if place.label != WALL}

    def lines(self, described, **reached):
        mesh = answered(described, **reached)
        return [line for line in unlaid_said(described.unlaid, mesh) if line.startswith("'Coarsen")]

    def test_the_rim_settles_for_a_size_between_the_edge_and_the_bulk(self):
        bare = problem(plates())
        edge, bulk = self.rims(bare)["PlateMetal"], bare.demand.coarsest
        size = (edge + bulk) / 2.0
        described = problem(plates((["Face1"], size)))
        assert self.rims(described) == {"PlateMetal": size, "StripMetal": edge}
        reached = Reached(asked=size, reached=size, standing=size, elements=9, dimension=1)
        (line,) = self.lines(described, PlateMetal=reached)
        assert line.startswith(f"'Coarsen1': Mode Coarsen at {size:.4g} mm lays the rim of")
        assert f"'PlateMetal' at {size:.4g} mm rather than {edge:.4g} mm" in line
        assert "Its edges are 28 mm long, and they come within 3 mm of 'StripMetal'" in line

    def test_a_rim_settling_at_the_bulk_lays_nothing_and_is_said(self):
        bare = problem(plates())
        described = problem(plates((["Face1"], bare.demand.coarsest)))
        assert self.rims(described) == {"StripMetal": self.rims(bare)["StripMetal"]}
        (line,) = self.lines(described)
        assert "lays no size at the rim of 'PlateMetal'" in line
        said = unlaid_said(described.unlaid, answered(described))
        assert not [line for line in said if "EdgeRefinement" in line]

    def test_every_rim_settling_at_the_bulk_is_said_to_be_why_none_is_laid(self):
        bare = problem(plates())
        plate = obj("Part::Plane", "Plate", Shape=Plate("plate", {"strip": 3.0}))
        strip = obj("Part::Plane", "Strip", Shape=Plate("strip", {"plate": 3.0}))
        regions = [
            refinement([(sheet, ["Face1"])], Mode="Coarsen", ElementSize=bare.demand.coarsest)
            for sheet in (plate, strip)
        ]
        described = problem(
            with_members(
                *(metal_binding(sheet, sub=["Face1"]) for sheet in (plate, strip)),
                *regions,
                recipe=gmsh_mesh(),
            )
        )
        assert self.rims(described) == {}
        said = unlaid_said(described.unlaid, answered(described))
        (creases,) = [line for line in said if "EdgeRefinement" in line]
        assert "since a coarsening settles the edges of 'PlateMetal', 'StripMetal'" in creases
        assert "no sheet or body is bound" not in creases

    def test_a_finer_refinement_of_the_same_metal_is_said_beside_the_coarsening(self):
        """Both are laid, and the finer size decides the elements where the two
        meet, so the line naming the coarsening says which won there."""
        bare = problem(plates())
        edge, bulk = self.rims(bare)["PlateMetal"], bare.demand.coarsest
        size = (edge + bulk) / 2.0
        found = plates((["Face1"], size))
        (plate,) = [
            member.References[0][0]
            for member in found.Group
            if kind(member) == "EMMaterialBinding" and member.Label == "PlateMetal"
        ]
        plate.Shape.Faces[0].BoundBox = SimpleNamespace(
            XMin=0.0, YMin=0.0, ZMin=1.0, XMax=10.0, YMax=4.0, ZMax=1.0
        )
        finer = refinement([(plate, ["Face1"])], ElementSize=edge / 2.0)
        finer.Label = "Fine"
        found.Group = [*found.Group, finer]
        (line,) = self.lines(problem(found))
        assert f"'Fine' at {edge / 2.0:.4g} mm refines some of the same metal finer" in line
        (unrefined,) = self.lines(problem(plates((["Face1"], size))))
        assert "refines" not in unrefined
        finer.ElementSize = (size + bulk) / 2.0
        (coarser,) = self.lines(problem(found))
        assert "refines" not in coarser
        # A refinement at or above the size everywhere lays nothing, even where
        # the coarsening asks for more still.
        (coarsening,) = [member for member in found.Group if member.Label == "Coarsen1"]
        coarsening.ElementSize = 2.0 * bulk
        finer.ElementSize = 1.5 * bulk
        (unlaid,) = self.lines(problem(found))
        assert "refines" not in unlaid

    def test_the_rim_laid_is_said_beside_the_edges_drawn(self):
        """The mesher lays the rim along the edges the room turns round past half a
        turn, and measures it."""
        bare = problem(plates())
        size = (self.rims(bare)["PlateMetal"] + bare.demand.coarsest) / 2.0
        described = problem(plates((["Face1"], size)))
        reached = Reached(
            asked=size, reached=size, standing=size, elements=9, dimension=1, extent=20.0
        )
        (line,) = self.lines(described, PlateMetal=reached)
        assert "Its edges are 28 mm long, the rim laid along them 20 mm" in line

    def test_metal_the_room_turns_round_no_edge_of_is_said_to_carry_nothing(self):
        bare = problem(plates())
        size = (self.rims(bare)["PlateMetal"] + bare.demand.coarsest) / 2.0
        described = problem(plates((["Face1"], size)))
        reached = Reached(
            asked=size, reached=None, standing=None, elements=0, dimension=1, laid=False, left=4
        )
        (line,) = self.lines(described, PlateMetal=reached)
        assert line.endswith(
            "lays nothing at 'PlateMetal', since the room turns round none of its edges by "
            "more than half a turn"
        )

    def test_a_rim_touching_another_conductor_is_said_to_touch_it(self):
        bare = problem(plates())
        size = (self.rims(bare)["PlateMetal"] + bare.demand.coarsest) / 2.0
        (line,) = self.lines(problem(plates((["Face1"], size), gap=0.0)))
        assert line.endswith(", and they touch 'StripMetal'")
        assert "gap" not in line

    def test_a_sheet_closed_on_itself_is_measured_along_its_edges(self):
        """The room turns round the outside of a closed sheet as it does round a
        body, so its edges are measured as a body's are, each once."""
        bare = problem(plates(faces=2))
        described = problem(
            plates((["Face1", "Face2"], bare.demand.coarsest), faces=2, closed=True)
        )
        (line,) = self.lines(described)
        assert "Its edges are 28 mm long" in line

    def test_a_sheet_with_no_edge_but_a_seam_is_said_to_have_none(self):
        found = plates((["Face1"], 1.0))
        (plate,) = [
            member.References[0][0]
            for member in found.Group
            if kind(member) == "EMMaterialBinding" and member.Label == "PlateMetal"
        ]
        plate.Shape.Faces[0].Edges = [Rim(7.0, seam=True)]
        (line,) = self.lines(problem(found))
        assert line.endswith("lays nothing at 'PlateMetal', which has no edge but a seam")

    def test_the_rim_of_a_body_is_every_edge_of_it(self):
        """A body's skin closes, and every edge of it but a seam is one the mesher
        reads the room round, so every such edge is measured."""
        # Two faces sharing every edge: a skin that closes, as a body's does.
        block = Plate("block", {"strip": 3.0}, faces=2)
        block.Faces[1].Edges = block.Faces[0].Edges
        block.Solids = [block]
        # A seam is where one face meets itself, and no edge of the body.
        block.Edges = [*block.Faces[0].Edges, Rim(7.0, seam=True)]
        body = obj("Part::Box", "Block", Shape=block)
        strip = obj("Part::Plane", "Strip", Shape=Plate("strip", {"block": 3.0}))
        bare = problem(with_members(metal_binding(body), metal_binding(strip)))
        size = (self.rims(bare)["BlockMetal"] + bare.demand.coarsest) / 2.0
        region = refinement([(body, [""])], Mode="Coarsen", ElementSize=size)
        region.Label = "Coarsen1"
        described = problem(with_members(metal_binding(body), metal_binding(strip), region))
        reached = Reached(asked=size, reached=size, standing=size, elements=9, dimension=1)
        (line,) = self.lines(described, BlockMetal=reached)
        assert "Its edges are 28 mm long, and they come within 3 mm of 'StripMetal'" in line

    def test_a_coarsening_finer_than_the_edge_leaves_the_rim(self):
        bare = problem(plates())
        edge = self.rims(bare)["PlateMetal"]
        described = problem(plates((["Face1"], edge / 2.0)))
        assert self.rims(described) == self.rims(bare)
        reached = Reached(asked=edge, reached=edge, standing=edge, elements=9, dimension=1)
        (line,) = self.lines(described, PlateMetal=reached)
        assert f"leaves the rim of 'PlateMetal' at {edge:.4g} mm" in line

    def test_part_of_a_binding_is_not_coarsened_and_is_said(self):
        """As the other backend relaxes a reference only where every element it
        names is coarsened."""
        bare = problem(plates(faces=2))
        described = problem(plates((["Face1"], bare.demand.coarsest), faces=2))
        assert self.rims(described) == self.rims(bare)
        (line,) = self.lines(described)
        assert line.startswith("'Coarsen1': Mode Coarsen at")
        assert "not laid where it names some faces of 'PlateMetal' and not all" in line

    def test_two_coarsenings_naming_a_binding_between_them_settle_it_at_the_finer(self):
        bare = problem(plates(faces=2))
        edge, bulk = self.rims(bare)["PlateMetal"], bare.demand.coarsest
        finer, coarser = edge + (bulk - edge) / 3.0, edge + 2.0 * (bulk - edge) / 3.0
        described = problem(plates((["Face1"], coarser), (["Face2"], finer), faces=2))
        assert self.rims(described)["PlateMetal"] == finer
        (line,) = self.lines(described)
        assert line.startswith(f"'Coarsen2': Mode Coarsen at {finer:.4g} mm")
        # Every edge of its faces once, the one they share included.
        assert "Its edges are 52 mm long" in line

    def test_nothing_is_said_where_edge_refinement_lays_no_rim(self):
        described = problem(plates((["Face1"], 1.0), EdgeRefinement=1.0))
        assert described.demand.places == ()
        (line,) = self.lines(described)
        assert "lays nothing at the rim of 'PlateMetal', which EdgeRefinement leaves" in line


def answered(described, **reached):
    """The mesh of a described run, each region label holding elements of a known
    size in a known box, the wall as well, and the places given."""
    labels = {
        region.label: Label(
            dimension=3,
            tag=number,
            entities=(number,),
            sits=None,
            lower=(0.0, 0.0, 0.0),
            upper=(20.0, 10.7, 4.3),
            edges=Edges(shortest=0.7, longest=2.9),
        )
        for number, region in enumerate(described.regions, start=1)
    }
    labels[described.wall] = Label(
        dimension=2, tag=99, entities=(99,), sits=FRONTIER, edges=Edges(shortest=0.8, longest=2.1)
    )
    return Mesh(
        path="/run/model.msh",
        labels=labels,
        worst_quality={},
        edges=Edges(shortest=0.6, longest=3.1),
        settled=(),
        version="",
        algorithm={},
        rims={},
        reached=reached,
    )


class TestWhatIsStatedAndNotLaid:
    """A size the document states and this backend does not lay is said once the
    mesh can answer it, with the figure it answers it with - and changes nothing
    the mesher is asked."""

    def lines(self, found, **reached):
        described = problem(found)
        return described, unlaid_said(described.unlaid, answered(described, **reached))

    def test_a_count_across_each_region_is_said_with_its_elements_and_its_box(self):
        _, said = self.lines(guide()[0])
        (line,) = [line for line in said if "MinElementsAcross" in line]
        assert line.startswith("'MeshSettings': MinElementsAcross 9 is not laid in 'AirFill'")
        assert "from 0.7 mm to 2.9 mm" in line
        assert "the smallest side of the box round it is 4.3 mm" in line

    def test_a_count_across_is_asked_of_nothing_the_mesher_is_given(self):
        counted = problem(with_members(settings=mesh_settings(MinElementsAcross=9))).demand
        uncounted = problem(with_members(settings=mesh_settings(MinElementsAcross=0))).demand
        assert counted == uncounted

    def test_no_count_says_nothing_about_one_and_the_least_count_is_said(self):
        _, said = self.lines(with_members(settings=mesh_settings(MinElementsAcross=0)))
        assert not any("MinElementsAcross" in line for line in said)
        _, said = self.lines(with_members(settings=mesh_settings(MinElementsAcross=1)))
        assert any("MinElementsAcross 1 is not laid" in line for line in said)

    def test_nothing_is_said_where_every_metal_carries_its_edges(self):
        """The mesh's own line for each place says what it laid there."""
        _, said = self.lines(two_sheets())
        assert not [line for line in said if "EdgeRefinement" in line]

    def test_metal_the_room_turns_round_no_edge_of_is_said(self):
        """The mesher lays nothing at metal whose every edge the room turns round
        by no more than half a turn, and the line names the property that asked
        for it and says why."""
        described = problem(two_sheets(EdgeRefinement=6.0))
        reached = {
            "SeptumMetal": Reached(
                asked=0.3, reached=None, standing=None, elements=0, dimension=1, laid=False, left=4
            ),
            "FilmMetal": Reached(asked=0.3, reached=0.32, standing=0.6, elements=50, dimension=1),
        }
        (line,) = [
            line
            for line in unlaid_said(described.unlaid, answered(described, **reached))
            if "EdgeRefinement" in line
        ]
        assert line.startswith("'GmshMesh': EdgeRefinement 6 is laid at the edges of 'FilmMetal'")
        assert (
            "no edge of 'SeptumMetal', since the room turns round none of them by more than "
            "half a turn"
        ) in line
        assert "creases" not in line.split(". ")[0]

    def test_a_refinement_of_one_says_nothing_about_creases(self):
        _, said = self.lines(two_sheets(EdgeRefinement=1.0))
        assert not any("EdgeRefinement" in line for line in said)

    def test_a_regions_count_is_said_with_the_size_its_body_reached(self):
        rod = obj("Part::Feature", "Rod", Shape=Boxed("rod", (0, 0, 0), (8, 2, 1)))
        found = with_members(refinement([(rod, [""])], MinElementsAcross=4))
        described = problem(found)
        (place,) = marked(described.demand)
        reached = Reached(asked=0.5, reached=0.61, standing=0.9, elements=40, dimension=3)
        said = unlaid_said(described.unlaid, answered(described, **{place.name: reached}))
        (line,) = [line for line in said if line.startswith("'Ring'")]
        assert "MinElementsAcross 4 is not laid" in line
        assert "the median mean edge of the 40 elements in" in line and "0.61 mm" in line
        assert "the smallest side of the box round it is 1 mm" in line

    def test_a_count_on_a_face_is_said_with_the_size_along_it(self):
        """A face has no elements in it, so the figure is the one along it."""
        face = flat("top", (0, 0, 1), (8, 2, 1))
        rod = obj("Part::Feature", "Rod", Shape=Boxed("rod", (0, 0, 0), (8, 2, 1), faces=[face]))
        found = with_members(refinement([(rod, ["Face1"])], MinElementsAcross=4))
        described = problem(found)
        (place,) = marked(described.demand)
        reached = Reached(
            asked=0.5, reached=0.7, standing=1.3, elements=60, dimension=2, along=0.55
        )
        said = unlaid_said(described.unlaid, answered(described, **{place.name: reached}))
        (line,) = [line for line in said if line.startswith("'Ring'")]
        assert "the median longest edge of the elements along" in line and "0.55 mm" in line

    def test_a_coarsening_of_a_region_is_said_with_the_size_of_the_elements_everywhere(self):
        """A region's size is the one everywhere, so a coarsening of what it is
        made of settles for nothing here."""
        found, body = guide()
        region = refinement([(body, [])], Mode="Coarsen", ElementSize=5.0)
        _, said = self.lines(with_members(region))
        (line,) = [line for line in said if line.startswith("'Ring'")]
        assert "Mode Coarsen at 5 mm is not laid where it names what 'AirFill' is made of" in line
        assert "from 0.6 mm to 3.1 mm" in line


class TestEveryPropertyOfTheMeshObjectsIsReadHere:
    """A property this backend never reads is a knob with nothing on the other end
    of it, and the net is the adapter's own code: each property the mesh policy
    and the refinement region declare is read somewhere under it.

    Read means read by the code, and not named in a docstring, a comment or a
    message. So the source is parsed rather than searched: a property is read
    where an attribute of that name is taken, where ``getattr`` is asked for it
    by name, or where ``getattr`` is asked for a name an f-string builds and the
    property's name starts with the f-string's leading text followed by a
    capital.

    The adapter's own code includes the region reader both backends share, which
    reads a refinement region for this one."""

    SOLVERS = pathlib.Path(Microwave.__file__).resolve().parent / "Solvers"
    SOURCES = (*sorted((SOLVERS / "palace").glob("*.py")), SOLVERS / "mesh_regions.py")

    @staticmethod
    def reads(sources):
        """The names the code takes as attributes, and the leading text of each
        name it builds to ask ``getattr`` for."""
        names, starts = set(), set()
        for source in sources:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Attribute):
                    names.add(node.attr)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                ):
                    asked = node.args[1]
                    if isinstance(asked, ast.Constant) and isinstance(asked.value, str):
                        names.add(asked.value)
                    elif (
                        isinstance(asked, ast.JoinedStr)
                        and asked.values
                        and isinstance(asked.values[0], ast.Constant)
                    ):
                        starts.add(asked.values[0].value)
        return names, starts

    def unread(self, names, sources=None):
        if sources is None:
            sources = [path.read_text(encoding="utf-8") for path in self.SOURCES]
        read, starts = self.reads(sources)
        return sorted(
            name
            for name in names
            if name not in read
            and not any(
                name.startswith(start) and name[len(start) :][:1].isupper() for start in starts
            )
        )

    def declared(self, doc, made):
        return list(made(doc).PropertiesList)

    def test_the_policy_the_recipe_and_the_region_are_read_whole(self, doc):
        from Microwave.Objects.mesh import (
            createEMGmshMesh,
            createEMMeshPolicy,
            createEMMeshRegion,
        )

        names = (
            self.declared(doc, createEMMeshPolicy)
            + self.declared(doc, createEMGmshMesh)
            + self.declared(doc, createEMMeshRegion)
        )
        assert "EdgeRefinement" in names and "ElementSize" in names
        assert self.unread(names) == []

    def test_a_name_in_prose_is_not_a_read(self):
        """A docstring, a comment and a message each name the property, and none of
        them reads it."""
        source = (
            "def f(settings):\n"
            '    """MaxGrowthRatio is how fast the size grows."""\n'
            "    # MaxGrowthRatio\n"
            '    raise ValueError("Raise MaxGrowthRatio above 1")\n'
        )
        assert self.unread(["MaxGrowthRatio"], [source]) == ["MaxGrowthRatio"]

    def test_each_way_of_reading_a_property_is_a_read(self):
        source = (
            "def f(settings, axis, side):\n"
            "    settings.EdgeRefinement\n"
            '    getattr(settings, "Enabled", True)\n'
            '    getattr(settings, f"Padding{axis}{side}")\n'
        )
        assert self.unread(["EdgeRefinement", "Enabled", "PaddingXMin"], [source]) == []
        assert self.unread(["Paddington", "InventedForTheNet"], [source]) == [
            "InventedForTheNet",
            "Paddington",
        ]

    def test_the_growth_is_read_by_the_code_and_not_by_its_messages(self):
        """The adapter's own source with the one read of the growth taken out: the
        name is still in its docstrings and its refusal, and the net catches it."""
        sources = [path.read_text(encoding="utf-8") for path in self.SOURCES]
        read = "value(recipe.MaxGrowthRatio)"
        assert sum(source.count(read) for source in sources) == 1
        cut = [source.replace(read, "1.3") for source in sources]
        assert any("MaxGrowthRatio" in source for source in cut)
        assert self.unread(["MaxGrowthRatio"], cut) == ["MaxGrowthRatio"]
