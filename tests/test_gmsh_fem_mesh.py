# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A mesh put in the study as FreeCAD's own mesh object.

``Fem`` is a stand-in here, holding groups the way FreeCAD's reader hands them
back: numbered in the order it met them, under the names the file carried.
Whether FreeCAD reads the copy that way, draws the object, and keeps it through
a save is asked of a real FreeCAD by ``tests/test_palace_mesh_shown.py``.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from Microwave.Gmsh.vocabulary import Label
from Microwave.Gui import fem_mesh
from Microwave.Objects.analysis import createEMAnalysis
from Microwave.Objects.fem_mesh import BUILT_FROM, DIGEST, IDENTITY, MADE_FOR, STATUS
from Microwave.Objects.kinds import kind_of
from Microwave.Objects.materials import createEMMaterialBinding
from Microwave.Objects.mesh import createEMGmshMesh
from Microwave.Objects.preview import CURRENT, OUT_OF_DATE
from Microwave.Objects.results import SOLVED_ON


class Read:
    """What ``Fem.read`` hands back: groups under the names the file carried,
    numbered by the reader rather than by the file."""

    def __init__(self, names, nodes=10):
        self.NodeCount = nodes
        self.names = {number: name for number, name in enumerate(names, start=1)}

    @property
    def Groups(self):
        return tuple(self.names)

    def getGroupName(self, group):
        return self.names[group]

    def renameGroup(self, group, name):
        self.names[group] = name


def mesher_answer(**labels):
    """The mesher's answer, as far as the copy is read against it."""
    return SimpleNamespace(
        numbered="/run/model.unv",
        labels={
            name: Label(dimension=dimension, tag=tag, entities=(), sits=None)
            for name, (dimension, tag) in labels.items()
        },
    )


GUIDE = {"Guide Air": (3, 1), "Port 1 (in)": (2, 2), "wall": (2, 3)}


@pytest.fixture
def reader(monkeypatch):
    """A ``Fem`` whose ``read`` hands back what the test put in ``made``."""
    made = {}

    def read(path):
        made.setdefault("paths", []).append(path)
        if "raises" in made:
            raise made["raises"]
        return made["mesh"]

    monkeypatch.setitem(__import__("sys").modules, "Fem", SimpleNamespace(read=read))
    return made


class TestReadingTheCopy:
    def test_each_group_comes_back_under_the_label_its_number_stands_for(self, reader):
        # Out of the file's order, as the reader numbers them.
        reader["mesh"] = Read(["3", "1", "2"])
        shown = fem_mesh.read(mesher_answer(**GUIDE))
        assert sorted(shown.names.values()) == sorted(GUIDE)
        assert shown.names[1] == "wall"
        assert reader["paths"] == ["/run/model.unv"]

    @pytest.mark.parametrize(
        ("groups", "complaint"),
        [
            (["1", "2"], "no group came back for 'wall'"),
            (["1", "2", "3", "4"], "groups Gmsh did not write: ['4']"),
            (["1", "2", "3", "3"], "two groups under one number"),
            (["Guide_Air", "2", "3"], "no group came back for 'Guide Air'"),
        ],
    )
    def test_a_copy_that_does_not_hold_the_labels_is_refused(self, reader, groups, complaint):
        reader["mesh"] = Read(groups)
        with pytest.raises(fem_mesh.Unshown) as raised:
            fem_mesh.read(mesher_answer(**GUIDE))
        assert complaint in str(raised.value)

    @pytest.mark.parametrize("breaks", ["\n", "\r"])
    def test_a_label_a_saved_document_cannot_read_back_is_refused(self, reader, breaks):
        reader["mesh"] = Read(["1"])
        with pytest.raises(fem_mesh.Unshown) as raised:
            fem_mesh.read(mesher_answer(**{f"Port{breaks}1": (2, 1)}))
        assert str(raised.value).endswith(
            "contain a line break, which FreeCAD loses when it saves the document. "
            "Remove the line break from the label"
        )

    def test_nothing_is_renamed_when_the_copy_is_refused(self, reader):
        reader["mesh"] = Read(["1", "2"])
        with pytest.raises(fem_mesh.Unshown):
            fem_mesh.read(mesher_answer(**GUIDE))
        assert reader["mesh"].names == {1: "1", 2: "2"}

    def test_an_empty_mesh_is_refused_rather_than_shown(self, reader):
        """FreeCAD's reader hands some formats back empty and raises nothing."""
        reader["mesh"] = Read(["1", "2", "3"], nodes=0)
        with pytest.raises(fem_mesh.Unshown, match="no mesh in it"):
            fem_mesh.read(mesher_answer(**GUIDE))

    def test_a_file_freecad_cannot_read_is_refused_naming_it(self, reader):
        reader["raises"] = RuntimeError("Unknown extension")
        with pytest.raises(fem_mesh.Unshown, match="could not read /run/model.unv"):
            fem_mesh.read(mesher_answer(**GUIDE))

    def test_an_answer_with_no_copy_is_refused_before_anything_is_read(self, reader):
        answer = mesher_answer(**GUIDE)
        answer.numbered = ""
        with pytest.raises(fem_mesh.Unshown, match="no copy"):
            fem_mesh.read(answer)
        assert "paths" not in reader


def meshes_in(analysis):
    return [member for member in analysis.Group if member.TypeId == fem_mesh.TYPE]


class TestPuttingItInTheStudy:
    def test_it_is_filed_in_the_study_under_the_backend(self, doc):
        analysis = createEMAnalysis(doc)
        made = fem_mesh.put(analysis, "Palace", "a mesh", key=fem_mesh.inputs(analysis))
        assert meshes_in(analysis) == [made]
        assert made.FemMesh == "a mesh"
        assert made.Label == "Mesh (Palace)"
        assert made.MeshedFor == "Palace"
        assert made.getEditorMode(fem_mesh.MADE_FOR) == ["ReadOnly"]
        assert made.State == ["Valid"]

    def test_it_records_what_it_was_made_from_and_nothing_can_edit_that(self, doc):
        analysis = createEMAnalysis(doc)
        key, linked = fem_mesh.inputs(analysis), fem_mesh.built_from(analysis)
        made = fem_mesh.put(analysis, "Palace", "a mesh", key=key, identity="abc", linked=linked)
        assert (made.Status, made.Digest, made.Identity) == (CURRENT, key, "abc")
        assert made.MeshedFrom == linked
        for name in (MADE_FOR, BUILT_FROM, STATUS, DIGEST, IDENTITY):
            assert made.getEditorMode(name) == ["ReadOnly"], name
        # So that marking it stale does not reach it again through a recompute.
        for name in (MADE_FOR, STATUS, DIGEST, IDENTITY):
            assert "Output" in made.getPropertyStatus(name), name

    def test_it_is_built_from_its_own_backends_regions_and_from_none_of_the_others(self, doc):
        from Microwave.Objects.mesh import createEMMeshRegion

        analysis = createEMAnalysis(doc)
        gmsh = createEMGmshMesh(doc)
        analysis.addObject(gmsh)
        (grid,) = [one for one in analysis.Group if kind_of(one) == "EMYeeGrid"]
        ours, theirs = createEMMeshRegion(doc, gmsh), createEMMeshRegion(doc, grid)
        analysis.addObject(ours)
        analysis.addObject(theirs)
        linked = fem_mesh.built_from(analysis)
        assert any(one is ours for one in linked)
        assert not any(one is theirs for one in linked)

    def test_a_mesh_of_a_study_that_moved_while_it_was_made_is_put_in_out_of_date(self, doc):
        analysis = createEMAnalysis(doc)
        key = fem_mesh.inputs(analysis)
        analysis.FrequencyStop = 20e9
        made = fem_mesh.put(analysis, "Palace", "a mesh", key=key)
        assert made.Status == OUT_OF_DATE
        assert made.Label == "Mesh (Palace)", "the tree says it by an overlay, not the label"

    def test_a_label_the_user_wrote_is_carried_to_the_mesh_that_replaces_it(self, doc):
        analysis = createEMAnalysis(doc)
        fem_mesh.put(analysis, "Palace", "first").Label = "Coarse"
        assert fem_mesh.put(analysis, "Palace", "second").Label == "Coarse"

    def test_a_label_beginning_as_the_workbench_s_is_the_workbench_s(self, doc):
        """A mesh saved while the workbench still marked a stale mesh by its
        label is replaced under the plain label."""
        analysis = createEMAnalysis(doc)
        fem_mesh.put(analysis, "Palace", "first").Label = "Mesh (Palace) - out of date"
        assert fem_mesh.put(analysis, "Palace", "second").Label == "Mesh (Palace)"

    def test_it_carries_the_proxy_that_redraws_its_icon(self, doc):
        from Microwave.Objects.fem_mesh import FemMeshProxy

        made = fem_mesh.put(createEMAnalysis(doc), "Palace", "a mesh")
        assert isinstance(made.Proxy, FemMeshProxy)
        assert made.Proxy.allowDuplicateLabel(made)

    def test_a_mesh_given_no_key_is_put_in_out_of_date(self, doc):
        analysis = createEMAnalysis(doc)
        assert fem_mesh.put(analysis, "Palace", "a mesh").Status == OUT_OF_DATE

    def test_it_replaces_the_backends_own_and_nothing_else(self, doc):
        analysis = createEMAnalysis(doc)
        other = fem_mesh.put(analysis, "Elsewhere", "their mesh")
        first = fem_mesh.put(analysis, "Palace", "first")
        second = fem_mesh.put(analysis, "Palace", "second")
        assert meshes_in(analysis) == [other, second]
        assert first not in doc.Objects
        assert fem_mesh.find(analysis, "Palace") is second
        assert fem_mesh.find(analysis, "Elsewhere") is other

    def test_a_copy_in_the_study_is_replaced_with_the_one_it_was_copied_from(self, doc):
        analysis = createEMAnalysis(doc)
        first = fem_mesh.put(analysis, "Palace", "first")
        copy = doc.addObject(fem_mesh.TYPE, "Copy")
        copy.addProperty("App::PropertyString", fem_mesh.MADE_FOR)
        copy.MeshedFor = "Palace"
        analysis.addObject(copy)
        made = fem_mesh.put(analysis, "Palace", "second")
        assert meshes_in(analysis) == [made]
        assert first not in doc.Objects and copy not in doc.Objects

    def test_one_taken_out_of_the_study_is_left_where_it_is(self, doc):
        analysis = createEMAnalysis(doc)
        first = fem_mesh.put(analysis, "Palace", "first")
        analysis.Group = [member for member in analysis.Group if member is not first]
        made = fem_mesh.put(analysis, "Palace", "second")
        assert meshes_in(analysis) == [made]
        assert first in doc.Objects

    def test_a_mesh_nobody_made_for_a_backend_is_left_alone(self, doc):
        """A FEM mesh a user dropped into the study carries no backend."""
        analysis = createEMAnalysis(doc)
        theirs = doc.addObject(fem_mesh.TYPE, "TheirMesh")
        analysis.addObject(theirs)
        fem_mesh.put(analysis, "Palace", "ours")
        assert theirs in analysis.Group
        assert fem_mesh.find(analysis, "Palace") is not theirs

    def test_a_mesh_that_cannot_be_made_whole_leaves_the_old_one_and_no_other(
        self, doc, monkeypatch
    ):
        """A caller filing this inside a step of its own may commit that step
        after a failure here, and what it commits is then the document as this
        left it."""
        analysis = createEMAnalysis(doc)
        old = fem_mesh.put(analysis, "Palace", "old")
        before = list(doc.Objects)

        def refuses(obj):
            raise RuntimeError("the group refused it")

        monkeypatch.setattr(analysis, "addObject", refuses)
        with pytest.raises(RuntimeError):
            fem_mesh.put(analysis, "Palace", "new")
        assert fem_mesh.find(analysis, "Palace") is old
        assert doc.Objects == before

    @pytest.mark.parametrize("visible", [False, True])
    def test_the_replacement_is_shown_as_the_old_one_was(self, doc, monkeypatch, visible):
        analysis = createEMAnalysis(doc)
        adding = doc.addObject

        def with_a_view(type, name):
            made = adding(type, name)
            object.__setattr__(made, "ViewObject", SimpleNamespace(Visibility=True))
            return made

        monkeypatch.setattr(doc, "addObject", with_a_view)
        first = fem_mesh.put(analysis, "Palace", "first")
        first.ViewObject.Visibility = visible
        second = fem_mesh.put(analysis, "Palace", "second")
        assert second.ViewObject.Visibility is visible

    def test_recording_it_is_one_undo_step_named_for_what_appears(self, doc):
        analysis = createEMAnalysis(doc)
        doc.transactions.clear()
        fem_mesh.record(analysis, "Palace", "a mesh")
        assert doc.transactions == [("Store Mesh", "commit")]

    def test_recording_it_is_one_undo_step_and_marks_no_run(self, doc):
        """A run keeps the mesh it was solved on in its matrix's record, and a
        mark written into a label would land on every older run."""
        from Microwave.Gui import results as glue

        analysis = createEMAnalysis(doc)
        result = glue.put(analysis, solved_on_mesh("first"))
        run = glue.run_of(result)
        labels = (run.Label, result.Label)
        doc.transactions.clear()
        made = fem_mesh.record(analysis, "Palace", "a mesh", identity="second")
        assert doc.transactions == [("Store Mesh", "commit")]
        assert made in analysis.Group
        assert (run.Label, result.Label) == labels


def solved_on_mesh(identity):
    """A Palace matrix recording that it was solved on the mesh ``identity``."""
    from Microwave.Results.sparameters import SParameters

    return fem_mesh.solved_on(
        SParameters(
            frequency=np.array([1e9, 2e9]),
            s=np.zeros((2, 1, 1), dtype=complex),
            port_numbers=(1,),
            reference=np.full((2, 1), 50.0),
            measured_impedance=np.full((2, 1), 50.0 + 0j),
            provenance={"solver": "Palace"},
        ),
        identity,
    )


class TestWhetherItStillDescribesTheStudy:
    """The mark first, then a change awaiting a recompute, then the key."""

    def made(self, analysis):
        return fem_mesh.put(analysis, "Palace", "a mesh", key=fem_mesh.inputs(analysis))

    def test_a_mesh_just_made_matches(self, doc):
        analysis = createEMAnalysis(doc)
        self.made(analysis)
        assert fem_mesh.staleness(analysis, "Palace") is None

    def test_a_study_with_no_mesh_says_so(self, doc):
        assert fem_mesh.staleness(createEMAnalysis(doc), "Palace") == "no mesh yet"

    def test_a_mesh_that_records_nothing_is_not_said_to_match(self, doc):
        """A mesh object carrying the backend it was made for and nothing else."""
        analysis = createEMAnalysis(doc)
        old = doc.addObject(fem_mesh.TYPE, "Mesh")
        old.addProperty("App::PropertyString", MADE_FOR)
        old.MeshedFor = "Palace"
        analysis.addObject(old)
        assert fem_mesh.staleness(analysis, "Palace") == (
            "the mesh does not record what it was made from"
        )

    def test_the_mark_is_taken_at_its_word(self, doc, monkeypatch):
        """Rather than re-derived: an edit put back still reads out of date."""
        analysis = createEMAnalysis(doc)
        made = self.made(analysis)
        made.Status = OUT_OF_DATE
        monkeypatch.setattr(fem_mesh, "inputs", lambda analysis: pytest.fail("derived"))
        assert fem_mesh.staleness(analysis, "Palace") == fem_mesh.CHANGED

    def test_a_shape_edited_and_not_yet_recomputed_is_reported(self, doc):
        """A body a binding names is linked beside the binding, so its touched
        mark is read before a recompute moves it to the mesh."""
        analysis = createEMAnalysis(doc)
        body = doc.addObject("Part::Box", "Body")
        object.__setattr__(body, "OutListRecursive", [])
        binding = createEMMaterialBinding(doc=doc)
        binding.References = [(body, [""])]
        object.__setattr__(binding, "OutList", [body])
        analysis.addObject(binding)
        made = fem_mesh.put(
            analysis,
            "Palace",
            "a mesh",
            key=fem_mesh.inputs(analysis),
            linked=fem_mesh.built_from(analysis),
        )
        body.purgeTouched()
        binding.purgeTouched()
        assert fem_mesh.staleness(analysis, "Palace") is None
        # What an edit to it leaves, as FreeCAD's recompute has not yet run.
        object.__setattr__(body, "_touched", True)
        assert any(obj is body for obj in made.MeshedFrom)
        assert fem_mesh.staleness(analysis, "Palace") == fem_mesh.CHANGED

    def test_a_change_no_edit_announced_is_found_by_the_key(self, doc):
        analysis = createEMAnalysis(doc)
        self.made(analysis)
        # Written past the mark, as a change FreeCAD never announced would be.
        object.__getattribute__(analysis, "_props")["FrequencyStart"] = 2e9
        assert fem_mesh.staleness(analysis, "Palace") == fem_mesh.CHANGED

    def test_a_study_that_cannot_be_read_is_not_said_to_match(self, doc, monkeypatch):
        analysis = createEMAnalysis(doc)
        self.made(analysis)

        def unreadable(analysis):
            raise RuntimeError("a shape the kernel will not measure")

        monkeypatch.setattr(fem_mesh, "inputs", unreadable)
        assert "cannot be checked" in fem_mesh.staleness(analysis, "Palace")


class TestTheKey:
    """What a mesh is built from moves it, and what no mesh reads does not."""

    def test_the_band_moves_it_at_either_end(self, doc):
        """The element is sized off the band's top, and a material's catalog
        row is read at its centre."""
        analysis = createEMAnalysis(doc)
        for name in ("FrequencyStart", "FrequencyStop"):
            before = fem_mesh.inputs(analysis)
            setattr(analysis, name, float(getattr(analysis, name)) * 1.5)
            assert fem_mesh.inputs(analysis) != before, name

    @pytest.mark.parametrize("name", ["FrequencyPoints", "ResponseFloor"])
    def test_what_the_study_declares_as_moving_no_cell_does_not_move_it(self, doc, name):
        analysis = createEMAnalysis(doc)
        before = fem_mesh.inputs(analysis)
        setattr(analysis, name, 7)
        assert fem_mesh.inputs(analysis) == before

    def test_the_other_backends_settings_do_not_move_it(self, doc):
        analysis = createEMAnalysis(doc)
        before = fem_mesh.inputs(analysis)
        grid = next(m for m in analysis.Group if kind_of(m) == "EMYeeGrid")
        grid.ElementsPerWavelength = 40.0
        assert fem_mesh.inputs(analysis) == before

    def test_its_own_recipe_moves_it(self, doc):
        analysis = createEMAnalysis(doc)
        recipe = createEMGmshMesh(doc)
        analysis.addObject(recipe)
        before = fem_mesh.inputs(analysis)
        recipe.ElementsPerWavelength = 40.0
        assert fem_mesh.inputs(analysis) != before

    def test_a_mesh_in_the_study_does_not_move_it(self, doc):
        analysis = createEMAnalysis(doc)
        before = fem_mesh.inputs(analysis)
        fem_mesh.put(analysis, "Palace", "a mesh")
        assert fem_mesh.inputs(analysis) == before


class TestAMatrixRecordsItsMesh:
    def test_a_matrix_records_the_mesh_it_was_solved_on_beside_what_it_had(self):
        @dataclass(frozen=True)
        class Matrix:
            provenance: dict

        matrix = fem_mesh.solved_on(Matrix({"solver": "Palace"}), "abc")
        assert matrix.provenance == {"solver": "Palace", SOLVED_ON: "abc"}


class TestTheIdentityOfAMesh:
    def test_it_is_the_digest_of_the_file(self, tmp_path):
        one, other = tmp_path / "one.msh", tmp_path / "other.msh"
        one.write_bytes(b"$MeshFormat 2.2")
        other.write_bytes(b"$MeshFormat 2.2")
        assert fem_mesh.identity(one) == fem_mesh.identity(other) != ""
        other.write_bytes(b"$MeshFormat 4.1")
        assert fem_mesh.identity(one) != fem_mesh.identity(other)

    def test_a_file_that_cannot_be_read_records_none(self, tmp_path):
        assert fem_mesh.identity(tmp_path / "gone.msh") == ""
