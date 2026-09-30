# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where a mesh property lives, and whether it reaches the run that reads it.

``EMMeshPolicy`` holds what the device asks of any mesh, and ``EMMeshRegion``
what it asks at one place; ``EMYeeGrid`` and ``EMGmshMesh`` each hold the
primitives of one pipeline's discretisation. The cut is made by what a number is rather than by
which backends are installed, so a third solver arriving adds an object and
moves none of the others.

Written the way ``tests/test_ports.py`` holds the port kinds, and for the same
reason: a rule stated in prose is a rule nothing re-derives.

* one table, so a property leaking back onto another kind is caught by that
  kind's own row rather than by whichever test its author had in mind;
* a coarse static net per adapter, so a property no adapter reads is a silent
  no-op caught here;
* a behavioural case per demand, because the net cannot see *which kind*
  reaches a read, nor whether a read it finds can fire at all;
* a guard on the net, because a net that has stopped catching anything is
  invisible.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from Microwave.Objects.mesh import (
    EMGmshMesh,
    EMMeshPolicy,
    EMMeshRegion,
    EMYeeGrid,
    createEMGmshMesh,
    createEMMeshPolicy,
    createEMMeshRegion,
    createEMYeeGrid,
)

FACES = tuple(f"{axis}{side}" for axis in "XYZ" for side in ("Min", "Max"))

#: The air round an open face as a count of cells, which the policy's
#: ``Clearance`` replaced and no kind may carry again.
RETIRED_AIR_CELLS = tuple(f"AirCells{face}" for face in FACES)

#: What every recipe spells the same way. One number of elements per wavelength
#: is a different accuracy on the two methods, and the two do not share a shape
#: of dependence on it - so a user comparing them states the same numbers to
#: both and reads the difference as the physics rather than as the settings.
SHARED_BY_EVERY_RECIPE = ("ElementsPerWavelength", "EdgeRefinement", "MaxGrowthRatio")

#: ``(factory, class, everything it carries, the value each arrives at, what it
#: must not carry)``. The fourth column is every property rather than a sample:
#: a default is what a user gets without typing anything, and this delivery
#: moved the properties without moving one of them.
KINDS = [
    (
        createEMMeshPolicy,
        EMMeshPolicy,
        {
            "MinElementsAcross",
            "CurveTolerance",
            "MinElementSize",
            "Clearance",
            "Medium",
            *(f"Padding{f}" for f in FACES),
        },
        {
            "MinElementsAcross": 9,
            "CurveTolerance": 0.0,
            "MinElementSize": 0.0,
            "Clearance": 0.0,
            "Medium": None,
            **{f"Padding{face}": "Air" for face in FACES},
        },
        {*SHARED_BY_EVERY_RECIPE, *RETIRED_AIR_CELLS},
    ),
    (
        createEMYeeGrid,
        EMYeeGrid,
        set(SHARED_BY_EVERY_RECIPE),
        {
            "ElementsPerWavelength": 20.0,
            "EdgeRefinement": 6.0,
            "MaxGrowthRatio": 1.3,
        },
        # The air round the structure is a length on the policy. A count of
        # cells was this pipeline standing in for it.
        {
            "MinElementsAcross",
            "CurveTolerance",
            "MinElementSize",
            "Clearance",
            "Medium",
            *RETIRED_AIR_CELLS,
            *(f"Padding{f}" for f in FACES),
        },
    ),
    (
        createEMGmshMesh,
        EMGmshMesh,
        {*SHARED_BY_EVERY_RECIPE, "ElementsPerTurn"},
        {
            "ElementsPerWavelength": 20.0,
            "EdgeRefinement": 8.0,
            "ElementsPerTurn": 6,
            "MaxGrowthRatio": 1.6,
        },
        # No air. This pipeline meshes bodies somebody drew, and what bounds an
        # open problem is a condition on a face of one of them.
        {
            "MinElementsAcross",
            "CurveTolerance",
            "MinElementSize",
            "Clearance",
            "Medium",
            *RETIRED_AIR_CELLS,
            *(f"Padding{f}" for f in FACES),
        },
    ),
]

#: What a refinement region carries. A length and a count at a place the user
#: named are statements about the device, so the region is on the problem and
#: both backends read it through one reader.
REGION = ("Mode", "References", "ElementSize", "MinElementsAcross", "Enabled")

KINDS.append(
    (
        createEMMeshRegion,
        EMMeshRegion,
        set(REGION),
        {"Mode": "Refine", "ElementSize": 0.0, "MinElementsAcross": 0},
        {
            *SHARED_BY_EVERY_RECIPE,
            *RETIRED_AIR_CELLS,
            "CurveTolerance",
            "MinElementSize",
            "Clearance",
            "Medium",
            *(f"Padding{f}" for f in FACES),
        },
    )
)

kinds = pytest.mark.parametrize(
    "factory, proxy, carries, defaults, must_not_have",
    KINDS,
    ids=[proxy.__name__ for _, proxy, *_ in KINDS],
)


class TestEachKindDeclaresItsOwnSurface:
    """One table: what each mesh kind has, what it arrives at, and what it must not.

    ``must_not_have`` is where this entry's rule becomes a check. A property
    only one pipeline can value is that pipeline's, and the way to state it is
    to keep it off the kinds that ignore it - a setting the user can change
    that reaches nothing is a silent no-op, and the kind it sits on is the one
    thing that tells a reader which of the two objects a number is about.
    """

    @kinds
    def test_the_factory_makes_the_kind_it_is_named_for(
        self, doc, factory, proxy, carries, defaults, must_not_have
    ):
        made = factory(doc)
        assert isinstance(made.Proxy, proxy)

    @kinds
    def test_a_kind_carries_what_the_table_says_and_nothing_else(
        self, doc, factory, proxy, carries, defaults, must_not_have
    ):
        """Equality rather than containment, so a property added tomorrow lands
        in this table before it lands in the property editor. The stub's
        ``PropertiesList`` holds exactly what ``addProperty`` was given, so
        there are no FreeCAD built-ins to filter out.
        """
        assert set(factory(doc).PropertiesList) == carries

    @kinds
    def test_a_kind_does_not_carry_another_kinds_property(
        self, doc, factory, proxy, carries, defaults, must_not_have
    ):
        """The row that catches a leak back. It is stated as well as the
        equality above, because a name moving between two rows of one table
        would otherwise pass by being spelt in the column it moved to.
        """
        made = factory(doc)
        carried = [name for name in sorted(must_not_have) if hasattr(made, name)]
        assert not carried, f"{proxy.__name__} carries {carried}, which belongs to another kind"

    @kinds
    def test_a_kind_arrives_at_the_values_it_was_argued_into(
        self, doc, factory, proxy, carries, defaults, must_not_have
    ):
        made = factory(doc)
        for name, wanted in defaults.items():
            found = getattr(made, name)
            if wanted is None:
                assert found is None, name
            elif isinstance(wanted, str):
                assert str(found) == wanted, name
            else:
                assert float(found) == pytest.approx(wanted), name

    def test_every_layer_spells_a_padding_value_the_same_way(self, doc):
        """One fact in four places that may not import each other.

        ``Objects/mesh.py`` sits behind ``import FreeCAD`` and neither adapter
        reaches any FreeCAD at all, so each adapter states the words again. A
        copy that drifted would leave a face the user set matched by nobody: the
        openEMS translation would refuse the value as unknown, and the Palace
        refusal would stop firing on the one value it is for.
        """
        from Microwave.Objects import mesh as objects
        from Microwave.Solvers.openems import policy as fdtd
        from Microwave.Solvers.palace import policy as fem

        offered = createEMMeshPolicy(doc).getEnumerationsOfProperty("PaddingXMin")
        assert list(offered) == [objects.AIR, objects.THROUGH, objects.ENDS]
        assert tuple(offered) == (fdtd.AIR_FACE, fdtd.THROUGH_FACE, fdtd.ENDS_FACE)
        assert (fem.AIR, fem.ENDS) == (objects.AIR, objects.ENDS)

    def test_the_two_recipes_spell_their_shared_demands_the_same_way(self, doc):
        """Deliberate, and the reason is in ``SHARED_BY_EVERY_RECIPE``. What
        differs between them is the tooltip, which names the pipeline."""
        yee = set(createEMYeeGrid(doc).PropertiesList)
        gmsh = set(createEMGmshMesh(doc).PropertiesList)
        assert yee & gmsh == set(SHARED_BY_EVERY_RECIPE)


# ---------------------------------------------------------------------------
# The net
# ---------------------------------------------------------------------------

ADAPTERS = Path(__file__).resolve().parents[1] / "Microwave" / "Solvers"


def _reads(where):
    """``(names read, prefixes an f-string builds a name from)`` under ``where``.

    Parsed rather than grepped. Asking whether a name appears anywhere in the
    adapter as a substring passes on prose and on any identifier that merely
    contains it. Two dozen property names are built with an f-string, which no
    reader sees as an attribute at all, so the prefix before the first
    replacement is collected beside the plain reads.
    """
    names: set[str] = set()
    prefixes: set[str] = set()
    for path in sorted(where.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
            elif isinstance(node, ast.JoinedStr) and node.values:
                first = node.values[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    prefixes.add(first.value)
    return names, prefixes


def _named(properties, where):
    """Which of ``properties`` appear in the adapter under ``where``, by a plain
    read or through a prefix an f-string builds a name from."""
    names, prefixes = _reads(where)
    return sorted(
        name
        for name in properties
        if name in names
        or any(name.startswith(prefix) and name[len(prefix) :][:1].isupper() for prefix in prefixes)
    )


def _unread(properties, where):
    """Which of ``properties`` the adapter under ``where`` never reads."""
    return sorted(set(properties) - set(_named(properties, where)))


def _imports_mesh_regions(where):
    """Whether a module under ``where`` imports the shared region reader."""
    for path in sorted(where.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (
                (node.module or "").split(".")[-1] == "mesh_regions"
                or any(alias.name == "mesh_regions" for alias in node.names)
            ):
                return True
    return False


def _declared(doc, factory):
    return list(factory(doc).PropertiesList)


class TestNoMeshPropertyIsANoOp:
    """Every property a mesh kind offers is read by the adapter that owns it.

    An unsupported thing is a refusal that names it, never a silent no-op. A
    property in the editor that reaches no solver is one of those.

    This net is coarse on purpose, and what it cannot see is *which kind*
    reaches a read: both recipes spell three properties the same way, so a name
    read for the Yee grid counts as read for the Gmsh mesh. The class below is
    what separates them.
    """

    def test_the_policy_is_read_by_every_adapter(self, doc):
        """It states what the device asks, so each backend answers it - by
        honouring it, by refusing it, or by saying with a figure what the mesh
        did instead.

        What this cannot see is whether the read can *fire*. A name behind a
        condition that never holds passes here exactly as one on the straight
        path does, and a read gated on another object being in the study was
        what taught that: it satisfied this net while reaching nothing on the
        study a user of that backend actually builds. The class below is what
        separates the two.
        """
        declared = _declared(doc, createEMMeshPolicy)
        assert declared, "the policy declares nothing, so this checks nothing"
        for backend in ("openems", "palace"):
            assert _unread(declared, ADAPTERS / backend) == [], backend

    def test_the_refinement_region_is_read_once_for_every_adapter(self):
        """``Solvers/mesh_regions.py`` reads each property, and each adapter
        reaches the region through it rather than reading one of its own."""
        shared = (ADAPTERS / "mesh_regions.py").read_text(encoding="utf-8")
        assert [name for name in REGION if name not in shared] == []
        for backend in ("openems", "palace"):
            assert _imports_mesh_regions(ADAPTERS / backend), backend

    def test_the_yee_grid_is_read_by_the_openems_adapter(self, doc):
        assert _unread(_declared(doc, createEMYeeGrid), ADAPTERS / "openems") == []

    def test_the_gmsh_mesh_is_read_by_the_palace_adapter(self, doc):
        assert _unread(_declared(doc, createEMGmshMesh), ADAPTERS / "palace") == []

    @pytest.mark.parametrize("backend", ["openems", "palace"])
    def test_the_net_itself_notices_a_property_nobody_reads(self, backend):
        """The guard on the guard.

        A no-op test that has stopped detecting no-ops is invisible. So it is
        asserted to fail on a property invented for the purpose, and on one
        whose name merely begins with a prefix the adapter does build names
        from.
        """
        invented = ["ThoroughlyImaginary", "Paddingfor"]
        assert _unread(invented, ADAPTERS / backend) == sorted(invented)

    def test_no_adapter_names_a_property_of_the_other_pipelines_recipe(self, doc):
        """The net run the other way, and the one the split needed.

        Each adapter finds its own kinds by kind. Run declared-property to
        source alone, an adapter reaching into the other pipeline's recipe is
        invisible - and one did: the Palace air check read ``AirCells*`` off an
        ``EMYeeGrid``, satisfied every net in this file, and made what that
        backend solves depend on which objects the other backend had left in the
        study.

        Only the names that separate the two recipes count. Both spell three
        properties the same way on purpose, and each adapter reads those off its
        own object.
        """
        owned = {
            "openems": (createEMMeshPolicy, createEMYeeGrid),
            "palace": (createEMMeshPolicy, createEMGmshMesh),
        }
        foreign = {"openems": createEMGmshMesh, "palace": createEMYeeGrid}
        distinctive = {
            backend: set(_declared(doc, foreign[backend])).difference(
                *(_declared(doc, factory) for factory in factories)
            )
            for backend, factories in owned.items()
        }
        if not any(distinctive.values()):
            # Nothing tells the two recipes apart by name, so the loop below
            # runs over nothing. That is allowed only while the table says so.
            recipes = [
                carries for _, proxy, carries, *_ in KINDS if proxy in (EMYeeGrid, EMGmshMesh)
            ]
            assert all(carries == set(SHARED_BY_EVERY_RECIPE) for carries in recipes), (
                "a recipe carries a name of its own, and this check ran over none"
            )
        named = {
            backend: _named(properties, ADAPTERS / backend)
            for backend, properties in distinctive.items()
            if _named(properties, ADAPTERS / backend)
        }
        assert not named, f"an adapter reaches into another pipeline's recipe: {named}"

    def test_that_net_sees_a_name_an_adapter_does_build(self, doc):
        """Its own guard, and the half that is easy to lose.

        ``Padding{face}`` is built with an f-string, so a reader matching
        attributes alone would find it nowhere and pass whatever any adapter
        did with it. It is found in an adapter that builds it, which is what
        says the same read would be found in one that does not own it.
        """
        faces = sorted(f"Padding{face}" for face in FACES)
        assert _named(faces, ADAPTERS / "openems") == faces
        assert _named(["ThoroughlyImaginary"], ADAPTERS / "openems") == []


class TestAStudyWithoutItsRecipeIsRefusedByName:
    """A file saved before the recipe became an object of its own.

    Refused at translation rather than mended quietly on open. A document that
    gained a recipe by itself would be solved at settings nobody typed, and the
    user would read the answer as one about the model they saved. The message
    names the command that puts the object back, which is the route that keeps
    the solver's own settings.
    """

    def test_the_openems_adapter_names_the_command_that_adds_the_grid(self):
        from Microwave.Solvers.openems import document

        from .test_openems_document_translation import model, removes

        doc = model()
        removes(doc, "YeeGrid")
        with pytest.raises(document.TranslationError, match="holds no Yee grid") as raised:
            document.problem(doc.Objects[0])
        assert "Add openEMS Solver" in str(raised.value)

    def test_the_palace_adapter_names_the_command_that_adds_the_mesh(self):
        from Microwave.Solvers.errors import TranslationError
        from Microwave.Solvers.palace.document import contents, kind

        from .test_palace_adapter import guide

        study, _ = guide()
        study.Group = [obj for obj in study.Group if kind(obj) != "EMGmshMesh"]
        with pytest.raises(TranslationError, match="holds no Gmsh mesh") as raised:
            contents(study)
        assert "Add Palace Solver" in str(raised.value)


# ---------------------------------------------------------------------------
# A case per demand
# ---------------------------------------------------------------------------


class TestEachPropertyMovesTheOpenemsRun:
    """Set it, translate, and watch the answer move.

    The only check that tells "read somewhere" from "read for this kind". The
    static net cannot: both recipes spell three properties the same way, so a
    name read for one counts as read for the other.
    """

    def params(self, **overrides):
        from Microwave.Solvers.openems import document

        from .test_openems_document_translation import model

        return document.problem(model(**overrides).Objects[0]).grid.params

    def test_elements_per_wavelength_sizes_every_cell(self):
        from .test_openems_document_translation import yee_grid

        coarse = self.params()
        fine = self.params(grid=yee_grid(ElementsPerWavelength=40.0))
        assert fine["dielectric_res"] == pytest.approx(coarse["dielectric_res"] / 2.0)

    def test_edge_refinement_sizes_the_cell_at_a_conductor_edge(self):
        from .test_openems_document_translation import yee_grid

        params = self.params(grid=yee_grid(EdgeRefinement=12.0))
        assert params["dielectric_res"] / params["metal_res"] == pytest.approx(12.0)

    def test_the_growth_ratio_bounds_the_step_between_adjacent_cells(self):
        from .test_openems_document_translation import yee_grid

        assert self.params(grid=yee_grid(MaxGrowthRatio=1.15))["max_ratio"] == [1.15] * 3

    def test_the_clearance_pads_every_air_face_by_the_length_stated(self):
        from .test_openems_document_translation import mesh_settings

        params = self.params(
            settings=mesh_settings(Clearance=4.5, **{f"Padding{face}": "Air" for face in FACES}),
        )
        assert [[float(face) for face in axis] for axis in params["padding"]] == [[4.5, 4.5]] * 3

    def test_the_medium_sets_the_coarsest_cell_and_the_derived_clearance(self):
        """The medium fills every cell no solid asks a size of, so the coarsest
        cell is the bulk size in it, and the air a face is padded by is derived
        from the wavelength in it."""
        from .test_openems_document_translation import EPS_R, fr4, mesh_settings

        vacuum = self.params()
        filled = self.params(settings=mesh_settings(Medium=fr4()))
        assert filled["cap"] == pytest.approx(vacuum["cap"] / math.sqrt(EPS_R), rel=1e-12)
        assert float(filled["padding"][1][1]) == pytest.approx(
            float(vacuum["padding"][1][1]) / math.sqrt(EPS_R), rel=1e-12
        )

    def test_the_padding_mode_decides_whether_the_face_is_padded_at_all(self):
        """One face, because the fixture is a line: padded on every face, the
        absorber eats the board's own thickness and the mesher refuses it.

        Each of the three values reaches the domain as its own thing. ``Ends``
        is air with no room in it - the domain stops on the structure and the
        absorber is added beyond - where ``Through`` takes the absorber out of
        the structure's own extent, and the mesher lays a different grid at the
        wall for each.
        """
        from .test_openems_document_translation import mesh_settings

        air = self.params()["padding"][1][1]
        assert float(air) > 0.0
        assert self.params()["padding"][1] == [air, air]
        through = self.params(settings=mesh_settings(PaddingYMin="Through", PaddingYMax="Air"))
        assert through["padding"][1] == ["through", air]
        ends = self.params(settings=mesh_settings(PaddingYMin="Ends", PaddingYMax="Air"))
        assert ends["padding"][1] == ["0", air]

    def test_a_padding_value_no_backend_knows_is_refused_by_name(self):
        """FreeCAD stores an enumeration's whole list in the document and
        restores it from there rather than from the class, so a file written by a
        build that offered a fourth value goes on offering it. Taken silently as
        air, it would pad a domain the user closed."""
        from Microwave.Solvers.openems import document

        from .test_openems_document_translation import mesh_settings, model

        doc = model(settings=mesh_settings(PaddingYMin="Snug"))
        with pytest.raises(document.TranslationError, match="PaddingYMin is 'Snug'"):
            document.problem(doc.Objects[0])

    def test_the_count_across_a_dielectric_reaches_the_mesher(self):
        from .test_openems_document_translation import mesh_settings

        assert self.params(settings=mesh_settings(MinElementsAcross=15))["min_lines"] == 15

    def test_the_floor_on_element_size_reaches_the_mesher(self):
        from .test_openems_document_translation import mesh_settings

        assert self.params(settings=mesh_settings(MinElementSize=0.004))["min_cell"] == (
            pytest.approx(0.004)
        )

    def test_the_curve_tolerance_reaches_the_shape_the_run_is_given(self):
        """A distance no ladder can reach, so the refusal naming the figure is a
        state the translation gets into only by having carried the number down
        to the triangulation."""
        from Microwave.Solvers.openems import document

        from .test_openems_document_translation import (
            BOARD,
            HEIGHT,
            LENGTH,
            Shape,
            mesh_settings,
            model,
            part,
            replaces,
            substrate,
        )

        board = substrate()
        board.Shape = Shape(
            (-LENGTH / 2, -BOARD / 2, 0.0),
            (LENGTH / 2, BOARD / 2, HEIGHT),
            fill=0.71,
            curves=0.1,
            coarsens=1e-6,
        )
        doc = model(settings=mesh_settings(CurveTolerance=1e-9))
        replaces(doc, "Substrate", board)
        part(doc, "DielectricBinding").References = [(board, [])]

        with pytest.raises(document.TranslationError, match="1e-09"):
            document.problem(doc.Objects[0])
        assert board.Shape.asked_deflection is not None, "the ladder never ran"


class TestEachPropertyMovesThePalaceRun:
    """The same, on the backend whose recipe is the tetrahedral mesh."""

    def asked(self, **overrides):
        from Microwave.Solvers.palace.policy import demand

        from .test_palace_adapter import gmsh_mesh, mesh_settings

        settings = overrides.pop("settings", None) or mesh_settings()
        recipe = overrides.pop("recipe", None) or gmsh_mesh()
        slowest = overrides.pop("slowest", 1.0)
        return demand(settings, recipe, stop=30e9, slowest=slowest, **overrides)

    def test_elements_per_wavelength_sizes_every_element(self):
        from .test_palace_adapter import gmsh_mesh

        coarse = self.asked()
        fine = self.asked(recipe=gmsh_mesh(ElementsPerWavelength=40.0))
        assert fine.coarsest == pytest.approx(coarse.coarsest / 2.0)

    def test_edge_refinement_sizes_the_element_at_a_rim(self):
        from .test_palace_adapter import gmsh_mesh

        asked = self.asked(recipe=gmsh_mesh(EdgeRefinement=12.0), rims=["Foil"])
        (place,) = asked.places
        assert place.size == pytest.approx(asked.coarsest / 12.0)

    def test_the_growth_ratio_sets_how_fast_a_refined_size_grows_back(self):
        from .test_palace_adapter import gmsh_mesh

        asked = self.asked(recipe=gmsh_mesh(MaxGrowthRatio=1.7), rims=["Foil"])
        assert asked.growth == pytest.approx(1.7)

    def test_the_floor_on_element_size_reaches_the_mesher(self):
        from .test_palace_adapter import mesh_settings

        assert self.asked(settings=mesh_settings(MinElementSize=0.01)).finest == 0.01

    def test_the_count_across_a_body_is_answered_with_the_size_it_reached(self):
        """No size on tetrahedra is a count, so the run says what it laid
        instead. That is the answer this demand gets, and it is an answer rather
        than a silence."""
        from Microwave.Solvers.palace.policy import unlaid

        from .test_palace_adapter import gmsh_mesh, mesh_settings

        (record,) = unlaid(
            mesh_settings(MinElementsAcross=15),
            gmsh_mesh(EdgeRefinement=1.0),
            self.asked(),
            ["AirFill"],
            (),
            (),
            "wall",
        )
        assert (record.count, record.labels) == (15, ("AirFill",))

    def test_the_curve_tolerance_is_refused_by_name(self):
        """The surface is meshed rather than sent as facets, so a distance from
        the drawing reaches nothing here and is said so rather than ignored."""
        from Microwave.Solvers.errors import TranslationError

        from .test_palace_adapter import mesh_settings

        with pytest.raises(TranslationError, match="CurveTolerance"):
            self.asked(settings=mesh_settings(CurveTolerance=0.001))

    def test_a_padded_face_is_asked_for_nothing_by_the_demand(self):
        """Air outside the drawing is reserved by the adapter, and the demand is
        asked for a size per region only by a caller that reserved it. The
        policy decides alone whether a study is open, with no Yee grid anywhere.
        """
        from Microwave.Solvers.palace.policy import opens

        from .test_palace_adapter import mesh_settings

        settings = mesh_settings(PaddingYMax="Air")
        assert opens(settings) == ("YMax",)
        assert self.asked(settings=settings) == self.asked()

    def test_a_study_left_at_the_defaults_is_open_rather_than_walled_in(self, doc):
        """Against the real document classes, because it is what a command makes.

        Every face defaults to ``Air``, which describes a radiating structure. A
        study answered by this backend alone holds no Yee grid at all, and it is
        open on every face: a domain the policy says is open, solved against a
        perfect conducting wall, is a clean run about another device.
        """
        from Microwave.Objects.mesh import createEMMeshPolicy
        from Microwave.Solvers.palace.policy import FACES, opens

        assert opens(createEMMeshPolicy(doc)) == FACES

    def test_the_medium_sets_the_size_the_reserved_air_is_laid_at(self):
        """The reserved air is the medium, so a study reserving it lays it at the
        medium's own size."""
        vacuum = self.asked(slowest=4.0, reserving=True)
        filled = self.asked(slowest=4.0, reserving=True, medium=4.0)
        assert filled.coarsest == pytest.approx(vacuum.coarsest / 2.0, rel=1e-12)

    def test_a_face_that_ends_on_the_drawing_is_taken(self):
        """The value that says the domain stops where the structure does. It is
        what this backend builds, so nothing is missing and nothing is said.
        """
        assert math.isfinite(self.asked().coarsest)
