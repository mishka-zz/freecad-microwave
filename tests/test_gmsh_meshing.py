# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The mesher in a process of its own, and what crosses the boundary.

The boundary is meant to be invisible: a drawing the mesher refuses raises the
same exception here as it would in the process, and so does one Gmsh could not
fill. What is the boundary's own is a child that never answered.

Most of what is below needs no Gmsh, because a wire format is a value. The one
class that meshes for real skips itself where the module is missing.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import threading
import time
from dataclasses import replace

import pytest

from Microwave.Gmsh.vocabulary import (
    FRONTIER,
    AtRim,
    Demand,
    Edges,
    Label,
    LeftOut,
    Mark,
    Mesh,
    Near,
    Part,
    Piece,
    Profile,
    Reached,
    Refused,
    Settled,
    Trimmed,
    Uncut,
    Unmeshed,
    Within,
)
from Microwave.Solvers import gmsh_mesh_report
from Microwave.Solvers.cancellation import Cancellation, Cancelled
from Microwave.Solvers.gmsh_meshing import (
    ANSWER_NAME,
    INTERPRETER_ENV_VAR,
    REQUEST_NAME,
    MesherNotFound,
    MeshFailed,
    Request,
    addon_packages,
    find_interpreter,
    from_answer,
    has_gmsh,
    mesh,
    to_answer,
    to_refusal,
    to_unmeshed,
)
from Microwave.Solvers.interpreters import PACKAGE_ROOT, child_environment
from tests.conftest import needed
from tests.drawings import drawings
from tests.processes import gone_within, killed_after_first_line

#: Where ``tests/drawings_probe.py`` wrote the drawings, under a real FreeCAD.
DRAWN = pathlib.Path(__file__).resolve().parent / "_drawings"

#: One drawing with a port face standing on a wall the wall label covers too,
#: which is the shape a Palace run has: a volume, its boundary, and a face of
#: that boundary claimed by something else.
DRAWING = "boundary_port"

#: What that drawing is meshed to. Coarse, because what is asked here is
#: whether the answer survived a process boundary rather than what it holds.
DEMAND = Demand(coarsest=8.0, finest=2.0)
VOLUME = Profile(top=3, element_order=1, curved=False)

#: A demand stating a place of every kind, a growth, and the walls and mirrors
#: a rim's opening is read against, which is what a crossing that dropped any of
#: them would come back without.
PLACED = Demand(
    coarsest=8.0,
    finest=0.5,
    growth=1.4,
    places=(
        Near("near", "port", 1.0),
        AtRim("rim", "walls", 2.0),
        AtRim("turned", "walls", 2.0, reentrant=True),
        Within("within", "air", 3.0),
    ),
    walls=("walls",),
    mirrors=("port",),
    per_turn=6,
)

#: A ring, and an element coarse against its own thickness.
RING = "curving_turns_elements_over"
RING_ELEMENT = 60.0

#: Elements round each turn of that ring, enough that the curving pass leaves
#: none turned over. The pass does not answer the same way on every run, so a
#: change to this value is checked over repeated runs.
RING_PER_TURN = 12

#: A thinner ring, where the pass that pulls back what curving turned over fails
#: and says so, as the corpus draws and meshes it.
FAILING = next(drawing for drawing in drawings() if drawing.name == "curving_pass_fails")

#: A profile whose every field differs from what one built with defaults holds,
#: so a crossing that dropped one is a request for a different mesh.
CURVED = Profile(top=3, element_order=2, curved=True, written="msh41", connected=True)


def pieces(name):
    """One drawing, as a caller would hand it over."""
    said = json.loads((DRAWN / f"{name}.manifest.json").read_text(encoding="utf-8"))
    return tuple(
        Piece(piece["label"], piece["dim"], str(DRAWN / piece["file"]), piece["priority"])
        for piece in said["pieces"]
    )


def built(**changed):
    """A mesh as the far side would hand one back."""
    fields = dict(
        path="/run/model.msh",
        labels={
            "air": Label(dimension=3, tag=1, entities=(1, 2), sits=None, elements=560),
            "walls": Label(
                dimension=2,
                tag=2,
                entities=(3, 4),
                sits=FRONTIER,
                lower=(0.0, 0.0, 0.0),
                upper=(4.0, 3.0, 2.0),
                size=52.0,
                edges=Edges(shortest=0.3, longest=1.9),
                elements=96,
            ),
        },
        worst_quality={2: 0.61, 3: 0.42},
        edges=Edges(shortest=0.21, longest=0.87),
        settled=(
            Settled(
                dimension=2,
                tag=4,
                took="port",
                gave_up=("walls",),
                place="(0, 0, 0) to (4, 3, 2)",
            ),
        ),
        version="4.13.1",
        algorithm={2: 6, 3: 1},
        rims={
            2: {3: ((1, 7),), 4: ((0, 11), (1, 7), (1, 8))},
            1: {7: ((0, 9),), 8: ((0, 9), (0, 10))},
        },
        bounds={1: {7: ((0.0, 0.0, 0.0), (0.0, 3.0, 0.0)), 8: ((0.0, 0.0, 0.0), (4.0, 0.0, 2.0))}},
        numbered="/run/model.unv",
        elements=560,
        left_out=(LeftOut(behind="port", place="(0, 0, 0) to (5, 3, 2)", held=("air",)),),
        reached={
            "rim": Reached(
                asked=0.25,
                reached=0.27,
                standing=0.61,
                elements=140,
                dimension=1,
                along=0.26,
                extent=14.0,
            ),
            "dot": Reached(asked=0.25, reached=None, standing=0.66, elements=12, dimension=0),
            "closed": Reached(
                asked=0.25, reached=None, standing=None, elements=0, dimension=1, laid=False
            ),
            "turned": Reached(
                asked=0.25,
                reached=None,
                standing=None,
                elements=0,
                dimension=1,
                laid=False,
                left=5,
            ),
        },
        parted=(
            Part(labels=("air", "port", "walls"), place="(0, 0, 0) to (4, 3, 2)"),
            Part(labels=("air", "skin"), place="(1, 1, 1) to (2, 2, 2)"),
        ),
    )
    fields.update(changed)
    return Mesh(**fields)


def answering(tmp_path, status, says=None):
    """A stand-in for the far side: it writes an answer and exits like that.

    The driver writes its answer and exits zero, so a child that answered and
    then ended badly cannot be reached through it. What ends a process after it
    has finished its work is a library's own teardown, and there is no way to
    ask Gmsh for one on demand.
    """
    kept = tmp_path / "kept-answer.json"
    kept.write_text(json.dumps(to_answer(built())) if says is None else says, encoding="utf-8")
    script = tmp_path / "stand-in"
    script.write_text(
        f'#!/bin/sh\ncat "{kept}" > "$(dirname "$3")/{ANSWER_NAME}"\nexit {status}\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def asked(directory, **changed):
    fields = dict(
        pieces=(
            Piece("air", 3, "/drawn/air.brep", 0),
            Piece("port", 2, "/drawn/port.brep", 1, inward=(1.0, 0.0, 0.0)),
            Piece("skin", 2, "/drawn/skin.brep", 1, divides=True),
            Piece("post", 3, "/drawn/post.brep", 2, divides=True, leaves=True),
        ),
        demand=DEMAND,
        profile=VOLUME,
        directory=str(directory),
        name="model",
    )
    fields.update(changed)
    return Request(**fields)


class TestWhatCrossesTheBoundary:
    def test_a_request_survives_being_written_and_read(self, tmp_path):
        """Every field of it decides the mesh, so one dropped in the crossing is
        a mesh of something else.
        """
        # Every field is set away from its default, a field left at one being
        # the one a crossing can drop and still come back equal.
        asking = asked(
            tmp_path,
            demand=PLACED,
            profile=CURVED,
            remainder="wall",
            numbered_as="unv",
            marks=(Mark("Ring on Rod:Face1", 2, "/drawn/face.brep"),),
        )
        assert Request.from_dict(json.loads(json.dumps(asking.to_dict()))) == asking

    def test_a_request_carrying_a_kind_of_place_not_laid_here_is_refused_by_it(self, tmp_path):
        """A request written by an older build carries a box, and the answer names
        it rather than failing on a lookup."""
        said = asked(tmp_path).to_dict()
        said["demand"]["places"] = [
            {"kind": "inside", "name": "Ring on Arm1_Top", "lower": [0, 0, 0], "upper": [1, 1, 1]}
        ]
        manifest = tmp_path / REQUEST_NAME
        manifest.write_text(json.dumps(said), encoding="utf-8")
        needed("gmsh", "gmsh is not on this interpreter, so the mesher is unreachable")
        from Microwave.Solvers import gmsh_mesh_driver

        assert gmsh_mesh_driver.main([str(manifest)]) == 0
        with pytest.raises(Refused) as refused:
            from_answer(json.loads((tmp_path / ANSWER_NAME).read_text(encoding="utf-8")))
        (complaint,) = refused.value.complaints
        assert "'Ring on Arm1_Top' is of kind 'inside'" in complaint
        assert "near, rim, within" in complaint

    def test_a_request_carrying_no_marks_is_refused_by_what_it_lacks(self, tmp_path):
        """A request written before marks crossed the boundary has no list of
        them, and the answer names it rather than the child dying on a lookup."""
        said = asked(tmp_path).to_dict()
        del said["marks"]
        manifest = tmp_path / REQUEST_NAME
        manifest.write_text(json.dumps(said), encoding="utf-8")
        needed("gmsh", "gmsh is not on this interpreter, so the mesher is unreachable")
        from Microwave.Solvers import gmsh_mesh_driver

        assert gmsh_mesh_driver.main([str(manifest)]) == 0
        with pytest.raises(Refused) as refused:
            from_answer(json.loads((tmp_path / ANSWER_NAME).read_text(encoding="utf-8")))
        (complaint,) = refused.value.complaints
        assert complaint.startswith("the request carries no 'marks'")

    def test_a_mesh_survives_being_written_and_read(self):
        answer = built(
            trimmed=(Trimmed(label="walls", by=("post",), place="(1, 1, 0) to (2, 2, 0)"),)
        )
        assert from_answer(json.loads(json.dumps(to_answer(answer)))) == answer

    def test_where_a_label_sits_survives_it(self):
        """It is the answer to the one question a caller cannot ask of the
        drawing, so a crossing that dropped it would leave the question unasked.
        """
        answer = built()
        assert from_answer(to_answer(answer)).labels["walls"].sits == FRONTIER

    def test_a_refusal_arrives_as_a_refusal(self):
        with pytest.raises(Refused) as raised:
            from_answer(json.loads(json.dumps(to_refusal(["the model ends at these"]))))
        assert raised.value.complaints == ("the model ends at these",)

    def test_a_drawing_gmsh_could_not_fill_arrives_as_itself(self):
        """What Gmsh said about it goes with it. It is written to a terminal the
        caller does not have, and it is the whole of what there is to read.
        """
        crossed = json.loads(json.dumps(to_unmeshed(["a region is empty"], ["Error: no mesh"])))
        with pytest.raises(Unmeshed) as raised:
            from_answer(crossed)
        assert raised.value.complaints == ("a region is empty",)
        assert raised.value.said == ("Error: no mesh",)
        assert type(raised.value) is Unmeshed

    def test_a_drawing_the_kernel_could_not_cut_arrives_as_itself(self):
        """A caller advises on the element size for every other drawing Gmsh
        could not fill, and on none of this one's."""
        crossed = json.loads(
            json.dumps(to_unmeshed(["could not cut"], ["Error: Fragments failed"], uncut=True))
        )
        with pytest.raises(Uncut) as raised:
            from_answer(crossed)
        assert raised.value.complaints == ("could not cut",)
        assert raised.value.said == ("Error: Fragments failed",)


class TestWhenTheChildNeverAnswers:
    def test_a_child_that_writes_nothing_is_not_a_mesh(self, tmp_path):
        with pytest.raises(MeshFailed, match="left no answer"):
            mesh(asked(tmp_path), interpreter=shutil.which("true"))

    def test_an_answer_an_earlier_request_left_is_not_read_as_this_one(self, tmp_path):
        """A directory is re-used between runs, and a stale answer read as a
        fresh one is a mesh of the drawing before the edit.
        """
        (tmp_path / ANSWER_NAME).write_text(json.dumps(to_answer(built())), encoding="utf-8")
        with pytest.raises(MeshFailed):
            mesh(asked(tmp_path), interpreter=shutil.which("true"))
        assert not (tmp_path / ANSWER_NAME).exists()

    def test_a_child_that_ends_badly_says_what_it_said(self, tmp_path):
        with pytest.raises(MeshFailed, match="exited with code"):
            mesh(asked(tmp_path), interpreter=shutil.which("false"))

    def test_an_interpreter_that_is_not_there_is_named(self, tmp_path):
        with pytest.raises(MeshFailed, match="could not be started"):
            mesh(asked(tmp_path), interpreter=tmp_path / "no-such-python")

    def test_the_request_is_left_where_it_can_be_re_run_by_hand(self, tmp_path):
        with pytest.raises(MeshFailed):
            mesh(asked(tmp_path), interpreter=shutil.which("true"))
        written = json.loads((tmp_path / REQUEST_NAME).read_text(encoding="utf-8"))
        assert Request.from_dict(written) == asked(tmp_path)

    def test_a_child_that_ends_badly_after_answering_has_still_answered(self, tmp_path):
        """A library that ends its own process decides the status and nothing
        else, and the answer was written before it did.
        """
        assert mesh(asked(tmp_path), interpreter=answering(tmp_path, 1)) == built()

    def test_an_answer_that_is_not_one_is_the_boundarys_own_failure(self, tmp_path):
        """The answer is what says the child finished, so a child that died part
        way through writing it must not reach a caller as a parse error.
        """
        stand_in = answering(tmp_path, 0, says="{ this is not an answer")
        with pytest.raises(MeshFailed, match="not an answer"):
            mesh(asked(tmp_path), interpreter=stand_in)

    def test_the_directory_is_made_where_there_is_none(self, tmp_path):
        where = tmp_path / "run" / "mesh"
        with pytest.raises(MeshFailed):
            mesh(asked(where), interpreter=shutil.which("true"))
        assert (where / REQUEST_NAME).is_file()


class TestAMeshCanBeStopped:
    """A drawing Gmsh cannot settle holds its process as long as it likes, and
    the panel stopping it runs on another thread. A real child, because the
    whole subject is what happens to one."""

    def hanging(self, tmp_path):
        """A stand-in for the far side that says its pid and never answers."""
        script = tmp_path / "hanging-python"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys, time\n"
            "sys.stderr.write('pid=%d\\n' % os.getpid())\n"
            "sys.stderr.flush()\n"
            # Only as the far side, which is handed the request. As a probe it
            # is handed a statement, and has nowhere to write.
            "if sys.argv[-1].endswith('.json'):\n"
            "    open(sys.argv[-1] + '.pid', 'w').write(str(os.getpid()))\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return script

    def test_a_stop_ends_the_mesher_and_there_is_no_answer(self, tmp_path):
        cancel = Cancellation()
        stand_in = self.hanging(tmp_path)
        pid_file = tmp_path / f"{REQUEST_NAME}.pid"
        stopper = threading.Thread(target=lambda: (_wait_for(pid_file), cancel.cancel()))
        stopper.start()
        began = time.monotonic()
        with pytest.raises(Cancelled):
            mesh(asked(tmp_path), interpreter=stand_in, cancel=cancel)
        stopper.join()
        assert time.monotonic() - began < 30.0
        assert not _alive(int(pid_file.read_text()))

    def test_a_mesh_ends_with_the_process_that_started_it(self, tmp_path):
        """FreeCAD can end without stopping anything, and Gmsh can mesh for a
        long while without printing, so no write to a closed pipe ends it."""
        stand_in = self.hanging(tmp_path)
        pid_file = tmp_path / f"{REQUEST_NAME}.pid"
        killed_after_first_line(
            "import threading, time\n"
            "from pathlib import Path\n"
            "from Microwave.Solvers.gmsh_meshing import mesh\n"
            "from tests.test_gmsh_meshing import asked\n"
            f"pid = Path({str(pid_file)!r})\n"
            f"threading.Thread(target=mesh, args=(asked(Path({str(tmp_path)!r})), "
            f"{str(stand_in)!r}), daemon=True).start()\n"
            "while not pid.exists():\n"
            "    time.sleep(0.05)\n"
            "print(pid.read_text(), flush=True)\n"
            "time.sleep(60)\n"
        )
        gone_within(int(pid_file.read_text()), 10.0)

    def test_a_request_that_arrives_first_starts_no_process_at_all(self, tmp_path):
        """The interpreter does not exist, so starting one would be the
        boundary's own failure. Getting ``Cancelled`` is the proof."""
        cancel = Cancellation()
        cancel.cancel()
        with pytest.raises(Cancelled):
            mesh(asked(tmp_path), interpreter=tmp_path / "no-such-python", cancel=cancel)

    def test_a_stopped_search_says_so_rather_than_that_no_mesher_is_installed(self, tmp_path):
        """``MesherNotFound`` would send the user to install a Gmsh that is
        there and was never really asked."""
        cancel = Cancellation()
        cancel.cancel()
        began = time.monotonic()
        with pytest.raises(Cancelled):
            find_interpreter(self.hanging(tmp_path), cancel=cancel)
        # A probe that is not watched is waited out to its own timeout, and the
        # search still ends by saying it was stopped - so the time is the proof
        # that the stop reached it.
        assert time.monotonic() - began < 10.0

    def test_the_search_for_an_interpreter_is_handed_the_stop(self, tmp_path, monkeypatch):
        """A blank setting means probing each candidate for up to a minute, and
        a stop that could not reach the probes would be waited out."""
        from Microwave.Solvers import gmsh_meshing

        seen = []

        def search(explicit=None, cancel=None):
            seen.append(cancel)
            return shutil.which("true")

        monkeypatch.setattr(gmsh_meshing, "find_interpreter", search)
        cancel = Cancellation()
        with pytest.raises(MeshFailed):
            mesh(asked(tmp_path), cancel=cancel)
        assert seen == [cancel]


def _wait_for(path, patience=30.0):
    """Until ``path`` exists, which is the child saying it has started."""
    deadline = time.monotonic() + patience
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} never appeared")
        time.sleep(0.05)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestFindingAMesher:
    def test_a_candidate_is_proved_by_importing_rather_than_by_exiting_well(self):
        """``/bin/echo`` takes ``-c <source>``, prints the source and exits zero,
        so a check on the status alone would take it for a Python.
        """
        assert has_gmsh("/bin/echo") is False

    def test_an_interpreter_that_is_not_there_is_not_one(self, tmp_path):
        assert has_gmsh(tmp_path / "no-such-python") is False

    def test_what_was_tried_is_named_where_none_of_it_answered(self, monkeypatch, tmp_path):
        """Every candidate is a real path here and none of them is a Python, so
        what is being read is the search rather than a stand-in for it.
        """
        stated = tmp_path / "stated-python"
        monkeypatch.setenv(INTERPRETER_ENV_VAR, str(stated))
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "executable", str(tmp_path / "beside" / "python"))
        with pytest.raises(MesherNotFound) as raised:
            find_interpreter(explicit=tmp_path / "asked-python")
        said = str(raised.value)
        assert "asked-python" in said
        assert str(stated) in said
        assert INTERPRETER_ENV_VAR in said

    def test_the_python_beside_this_one_is_a_candidate(self, monkeypatch, tmp_path):
        """Under FreeCAD the running interpreter is the application itself, and
        the module is installed beside it rather than into it.
        """
        needed("gmsh", "gmsh is not on this interpreter, so the mesher is unreachable")
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.delenv(INTERPRETER_ENV_VAR, raising=False)
        assert find_interpreter().parent == pathlib.Path(sys.executable).parent

    def test_python3_on_the_path_is_the_last_thing_tried(self, monkeypatch, tmp_path):
        """The route for a host that installed the module somewhere of its own
        and put that Python on the path.
        """
        needed("gmsh", "gmsh is not on this interpreter, so the mesher is unreachable")
        monkeypatch.setenv("PATH", str(pathlib.Path(sys.executable).parent))
        monkeypatch.delenv(INTERPRETER_ENV_VAR, raising=False)
        monkeypatch.setattr(sys, "executable", str(tmp_path / "gone" / "python"))
        assert find_interpreter() == pathlib.Path(shutil.which("python3"))


class TestWhatAChildIsStartedWith:
    def test_the_child_reaches_this_workbench_by_path(self):
        assert child_environment({})["PYTHONPATH"] == str(PACKAGE_ROOT)

    def test_the_hosts_own_interpreter_is_not_forced_on_it(self):
        """FreeCAD exports PYTHONHOME at its own bundled Python. Inherited, a
        child under any other interpreter loads FreeCAD's standard library and
        dies during startup with an encodings error.
        """
        assert "PYTHONHOME" not in child_environment({"PYTHONHOME": "/somewhere/else"})

    def test_the_rest_of_the_environment_is_the_callers(self):
        assert child_environment({"HOME": "/home/someone"})["HOME"] == "/home/someone"


class TestWhatFreeCADsOwnPythonIsHanded:
    """The Addon Manager installs a package an addon declares - this workbench
    declares Gmsh - into a directory FreeCAD adds to its own path at startup. A
    child under FreeCAD's Python runs none of that startup, so it is handed the
    directory: and only that child, since the packages there were built for that
    interpreter."""

    def freecad(self, monkeypatch, tmp_path, made=True):
        """This process as FreeCAD, with its own Python beside it."""
        import FreeCAD

        home = tmp_path / "user"
        packages = home / "AdditionalPythonPackages"
        version = packages / f"py{sys.version_info.major}{sys.version_info.minor}"
        if made:
            version.mkdir(parents=True)
        monkeypatch.setattr(FreeCAD, "USER_APP_DATA_DIR", str(home), raising=False)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").write_text("", encoding="utf-8")
        monkeypatch.setattr(sys, "executable", str(bin_dir / "FreeCAD"))
        return bin_dir / "python", version, packages

    def test_freecads_own_python_is_handed_what_the_addon_manager_installed(
        self, monkeypatch, tmp_path
    ):
        beside, version, packages = self.freecad(monkeypatch, tmp_path)
        assert addon_packages(beside) == [str(version), str(packages)]

    def test_any_other_python_is_handed_nothing(self, monkeypatch, tmp_path):
        """A binary package built for one interpreter and loaded into another is
        a crash rather than a refusal."""
        self.freecad(monkeypatch, tmp_path)
        assert addon_packages(sys.base_prefix) == []

    def test_nothing_is_handed_where_nothing_was_installed(self, monkeypatch, tmp_path):
        beside, _, _ = self.freecad(monkeypatch, tmp_path, made=False)
        assert addon_packages(beside) == []

    def test_outside_freecad_there_is_nothing_to_hand(self, monkeypatch, tmp_path):
        beside, _, _ = self.freecad(monkeypatch, tmp_path)
        monkeypatch.delitem(sys.modules, "FreeCAD")
        assert addon_packages(beside) == []

    def test_the_directories_reach_the_child_after_the_workbench(self):
        env = child_environment({}, also=["/a", "/b"])
        assert env["PYTHONPATH"].split(os.pathsep) == [str(PACKAGE_ROOT), "/a", "/b"]

    def test_the_probe_and_the_mesher_are_both_handed_them(self, monkeypatch, tmp_path):
        """A probe that saw the package and a mesher that did not would pass the
        search and fail the mesh."""
        from Microwave.Solvers import gmsh_meshing, interpreters

        beside, version, _ = self.freecad(monkeypatch, tmp_path)
        # A program, or the mesher is refused before anything is started.
        beside.chmod(0o755)
        seen = []
        monkeypatch.setattr(
            interpreters.subprocess,
            "Popen",
            lambda command, **kw: (
                seen.append(kw["env"]["PYTHONPATH"]) or (_ for _ in ()).throw(OSError())
            ),
        )
        assert not gmsh_meshing.has_gmsh(beside)
        with pytest.raises(MeshFailed):
            mesh(asked(tmp_path / "run"), interpreter=beside)
        assert len(seen) == 2
        assert all(str(version) in path for path in seen)


class TestTheMeshItselfComesBack:
    """The real mesher, in a real process. Skipped where Gmsh is missing."""

    @pytest.fixture(autouse=True)
    def _needs_gmsh(self):
        needed("gmsh", "gmsh is not on this interpreter, so the mesher is unreachable")

    def test_a_curving_pass_that_fails_arrives_as_unmeshed_rather_than_a_dead_child(self, tmp_path):
        """Gmsh's optimiser reports its failure inside a parallel loop, and an
        error thrown there ends the process on a signal."""
        ring = Profile(top=FAILING.top, element_order=FAILING.element_order, curved=FAILING.curved)
        coarse = Demand(coarsest=FAILING.coarsest, finest=FAILING.finest, per_turn=FAILING.per_turn)
        with pytest.raises(Unmeshed) as raised:
            mesh(asked(tmp_path, pieces=pieces(FAILING.name), demand=coarse, profile=ring))
        assert type(raised.value) is Unmeshed
        stopped, turned = raised.value.complaints
        assert stopped.startswith("Gmsh stopped: ") and "critical value" in stopped
        assert "turned inside out" in turned and "in 'air' at (" in turned

    def test_shapes_the_kernel_could_not_cut_arrive_as_themselves(self, tmp_path):
        drawn = tuple(
            replace(piece, leaves=piece.label == "post") for piece in pieces("curve_inside_a_post")
        )
        with pytest.raises(Uncut) as raised:
            mesh(asked(tmp_path, pieces=drawn))
        assert "could not cut the shapes drawn for 'air', 'curve', 'post', 'walls'" in str(
            raised.value
        )

    def test_elements_sized_round_each_turn_let_a_coarse_ring_curve(self, tmp_path):
        ring = Profile(top=3, element_order=3, curved=True)
        turning = Demand(coarsest=RING_ELEMENT, finest=0.0, per_turn=RING_PER_TURN)
        answer = mesh(asked(tmp_path, pieces=pieces(RING), demand=turning, profile=ring))
        assert min(answer.worst_quality.values()) > 0.0

    def test_a_drawing_meshes_across_the_boundary(self, tmp_path):
        answer = mesh(asked(tmp_path, pieces=pieces(DRAWING)))
        assert pathlib.Path(answer.path).is_file()
        assert set(answer.labels) == {"air", "walls", "port"}
        assert answer.labels["port"].sits == FRONTIER

    def test_the_answer_says_what_built_it(self, tmp_path):
        """A version and an algorithm are chosen below the caller and decide the
        elements, so a crossing that dropped them would leave two runs alike in
        everything stated and different in what they meshed.
        """
        answer = mesh(asked(tmp_path, pieces=pieces(DRAWING)))
        assert answer.version
        assert set(answer.algorithm) == {2, 3}
        assert min(answer.worst_quality.values()) > 0.0

    def test_a_drawing_that_describes_no_mesh_is_refused_across_it(self, tmp_path):
        drawn = [piece for piece in pieces(DRAWING) if piece.label != "walls"]
        with pytest.raises(Refused) as raised:
            mesh(asked(tmp_path, pieces=tuple(drawn)))
        assert raised.value.complaints

    def test_a_remainder_named_here_reaches_the_mesher_on_the_far_side(self, tmp_path):
        """The one field the caller states that the drawing does not carry.

        Dropped anywhere in the crossing it reads as a caller that named none,
        and the same drawing is then refused for a boundary nothing claims - so
        this is driven end to end rather than read off the request.
        """
        drawn = [piece for piece in pieces(DRAWING) if piece.label != "walls"]
        answer = mesh(asked(tmp_path, pieces=tuple(drawn), remainder="left over"))
        assert answer.labels["left over"].sits == FRONTIER
        assert answer.labels["left over"].entities

    def test_a_numbered_copy_asked_for_here_is_written_on_the_far_side(self, tmp_path):
        """Dropped in the crossing it reads as a caller that asked for none, and
        the mesh comes back whole with nothing beside it."""
        answer = mesh(asked(tmp_path, pieces=pieces(DRAWING), numbered_as="unv"))
        assert pathlib.Path(answer.numbered).is_file()
        assert pathlib.Path(answer.numbered).parent == pathlib.Path(answer.path).parent

    def test_which_label_took_a_piece_two_were_drawn_over_comes_back(self, tmp_path):
        """The drawing puts the port face where the wall label covers it too, so
        the caller's own order is what settles it.
        """
        answer = mesh(asked(tmp_path, pieces=pieces(DRAWING)))
        assert [one.took for one in answer.settled] == ["port"]


class TestWhatTheMeshSaysAboutItself:
    """The demand is a length, and these are the lines that answer it.

    Each is content: what size, what shape, what overlapped. That they are said
    before the solve is the pipeline's own subject.
    """

    def said(self, **changed):
        mesh = built(**changed)
        return gmsh_mesh_report.describe(mesh, Demand(coarsest=2.0, finest=0.0))

    def test_it_states_the_size_reached_beside_the_size_asked_for(self):
        line = self.said(edges=Edges(shortest=0.4, longest=2.5))[0]
        assert "from 0.4 mm to 2.5 mm" in line
        assert "asked for elements around 2 mm" in line

    def test_what_was_left_out_is_said_with_where_and_what_it_held(self):
        assert [line for line in self.said() if "left out" in line] == [
            "Mesh: left out what stands behind 'port', at (0, 0, 0) to (5, 3, 2), drawn as 'air'"
        ]

    def test_a_mesh_that_left_nothing_out_says_nothing_about_it(self):
        assert not [line for line in self.said(left_out=()) if "left out" in line]

    def test_what_a_label_lost_inside_a_body_is_said_with_where(self):
        trimmed = (Trimmed(label="walls", by=("post",), place="(1, 1, 0) to (2, 2, 0)"),)
        assert [line for line in self.said(trimmed=trimmed) if "inside" in line] == [
            "Mesh: the part of 'walls' inside 'post', at (1, 1, 0) to (2, 2, 0), leaves the "
            "model with it"
        ]
        assert not [line for line in self.said() if "inside" in line]

    def test_a_model_cut_into_parts_says_so_and_then_says_where_each_is(self):
        """The reader has to learn that the model came apart before being handed
        a part, or the first line reads as a fact about the whole model."""
        assert [line for line in self.said() if "part" in line] == [
            "Mesh: no field crosses from one part of the model to another. Each part:",
            "Mesh: a part at (0, 0, 0) to (4, 3, 2), on 'air', 'port', 'walls'",
            "Mesh: a part at (1, 1, 1) to (2, 2, 2), on 'air', 'skin'",
        ]

    def test_a_model_the_cuts_left_whole_says_nothing_about_parts(self):
        """One part is what a model that stayed whole under its cuts comes back
        as, and saying so on every mesh would bury the line that matters."""
        one = built().parted[:1]
        assert not [line for line in self.said(parted=one) if "part" in line]
        assert not [line for line in self.said(parted=()) if "part" in line]

    def test_a_floor_that_was_asked_for_is_named_and_one_that_was_not_is_left_out(self):
        assert "below" not in self.said()[0]
        mesh = built()
        stated = gmsh_mesh_report.describe(mesh, Demand(coarsest=2.0, finest=0.25))[0]
        assert "none below 0.25 mm" in stated

    def test_every_dimension_carrying_a_quality_is_reported_in_dimension_order(self):
        quality = [line for line in self.said(worst_quality={3: 0.42, 2: 0.61}) if "qual" in line]
        assert quality == [
            "Mesh: worst element quality at dimension 2 is 0.61",
            "Mesh: worst element quality at dimension 3 is 0.42",
        ]

    def test_a_piece_two_labels_were_drawn_over_names_the_winner_first(self):
        """Which label took it is the whole content of the line, so a sentence
        that would read the same with the two swapped says nothing."""
        line = self.said(
            settled=(
                Settled(
                    dimension=3,
                    tag=7,
                    took="air",
                    gave_up=("block",),
                    place="(0, 0, 0) to (30, 20, 10)",
                ),
            ),
            left_out=(),
            parted=(),
        )[-1]
        assert line == (
            "Mesh: 'air' took a piece also drawn over by 'block', at (0, 0, 0) to (30, 20, 10)"
        )

    def test_each_label_says_the_edges_of_its_own_elements(self):
        """Under its own name, and only where its elements were measured."""
        lines = self.said()
        assert "Mesh: 'walls' elements from 0.3 mm to 1.9 mm" in lines
        assert not any(line.startswith("Mesh: 'air' elements") for line in lines)

    def test_it_states_how_many_elements_fill_the_model_and_each_label_holds(self):
        """Over the labels of the dimension filled, each with its share."""
        assert self.said()[1] == "Mesh: 560 elements fill the model: 'air' 560 (100%)"

    def test_a_place_says_the_size_asked_beside_what_the_mesh_holds_there(self):
        """The size along it first, since that is the figure set against the ask,
        and the elements touching it after, which the growth sets."""
        mesh = built()
        demand = Demand(coarsest=2.0, finest=0.0, growth=1.5, places=(AtRim("rim", "walls", 0.25),))
        (line,) = [line for line in gmsh_mesh_report.describe(mesh, demand) if "'rim'" in line]
        assert line == (
            "Mesh: at 'rim' asked 0.25 mm, growing to 2 mm over 3.5 mm: along its 14 mm the "
            "median longest edge of the elements lying on it is 0.26 mm and the longest 0.27 mm; "
            "the 140 elements touching it are 25% of the model, and their median longest "
            "edge is 0.61 mm"
        )

    def test_a_place_of_points_reads_the_elements_at_it(self):
        """No element lies along a point, so the figure beside the size asked is
        the longest edge of the elements at it, and the line says which it is."""
        labels = dict(built().labels)
        labels["port"] = Label(dimension=0, tag=5, entities=(5,), sits=None)
        reached = {
            "dot": Reached(asked=0.25, reached=0.44, standing=0.66, elements=12, dimension=0)
        }
        mesh = built(labels=labels, reached=reached)
        demand = Demand(coarsest=2.0, finest=0.0, growth=1.5, places=(Near("dot", "port", 0.25),))
        (line,) = [line for line in gmsh_mesh_report.describe(mesh, demand) if "'dot'" in line]
        assert "the longest edge of the elements at it is 0.44 mm" in line
        assert "the 12 elements touching it are 2.14% of the model" in line
        assert "0.66 mm" in line

    def test_a_place_of_surfaces_says_its_area(self):
        reached = {
            "face": Reached(
                asked=0.5,
                reached=0.7,
                standing=1.1,
                elements=56,
                dimension=2,
                along=0.55,
                extent=48.0,
            )
        }
        demand = Demand(coarsest=2.0, finest=0.0, growth=1.5, places=(Near("face", "port", 0.5),))
        (line,) = [
            line
            for line in gmsh_mesh_report.describe(built(reached=reached), demand)
            if "'face'" in line
        ]
        assert "over its 48 square mm the median longest edge" in line
        assert "are 10% of the model" in line

    def test_a_size_throughout_a_body_says_the_mean_edge_in_it(self):
        """The elements counted are the ones in the body, whether the size is
        laid throughout it alone or near it, and never the ones standing on it."""
        reached = {"slab": Reached(asked=0.5, reached=0.62, standing=0.8, elements=90, dimension=3)}
        for place, opening in (
            (Within("slab", "air", 0.5), "Mesh: throughout 'slab' asked 0.5 mm: "),
            (Near("slab", "air", 0.5), "Mesh: at 'slab' asked 0.5 mm, growing to 2 mm over 3 mm: "),
        ):
            demand = Demand(coarsest=2.0, finest=0.0, growth=1.5, places=(place,))
            (line,) = [
                line
                for line in gmsh_mesh_report.describe(built(reached=reached), demand)
                if "'slab'" in line
            ]
            assert line.startswith(opening)
            assert "median mean edge is 0.62 mm" in line and "0.8 mm" in line
            assert "the 90 elements in it are 16.1% of the model" in line
            assert "standing" not in line

    def test_a_rim_of_a_label_closed_on_itself_says_nothing_is_laid(self):
        """Under the label it asked a rim of, since that is what the caller drew."""
        mesh = built()
        demand = Demand(
            coarsest=2.0, finest=0.0, growth=1.5, places=(AtRim("closed", "walls", 0.25),)
        )
        (line,) = [line for line in gmsh_mesh_report.describe(mesh, demand) if "'closed'" in line]
        assert line == (
            "Mesh: at 'closed' asked 0.25 mm: nothing is laid, since 'walls' is closed on "
            "itself and has no rim"
        )

    def test_a_rim_the_room_turns_round_no_edge_of_says_why_nothing_is_laid(self):
        """Under the label it asked a rim of, with the count of its edges."""
        mesh = built()
        for left, why in (
            (5, "the room turns round no curve of 'walls' by more than half a turn: 5 in all"),
            (0, "'walls' holds no curve but a seam"),
        ):
            reached = dict(mesh.reached)
            reached["turned"] = replace(reached["turned"], left=left)
            demand = Demand(
                coarsest=2.0,
                finest=0.0,
                growth=1.5,
                places=(AtRim("turned", "walls", 0.25, reentrant=True),),
            )
            (line,) = [
                line
                for line in gmsh_mesh_report.describe(replace(mesh, reached=reached), demand)
                if "'turned'" in line
            ]
            assert line == f"Mesh: at 'turned' asked 0.25 mm: nothing is laid, since {why}"

    def test_a_rim_laid_at_some_edges_says_how_many_carry_nothing(self):
        mesh = built()
        reached = dict(mesh.reached)
        reached["rim"] = replace(reached["rim"], left=3)
        demand = Demand(
            coarsest=2.0,
            finest=0.0,
            growth=1.5,
            places=(AtRim("rim", "walls", 0.25, reentrant=True),),
        )
        (line,) = [
            line
            for line in gmsh_mesh_report.describe(replace(mesh, reached=reached), demand)
            if "'rim'" in line
        ]
        assert line.endswith(
            ". Nothing is laid at the curves of 'walls' the room turns round by no more than "
            "half a turn: 3 in all"
        )

    def test_a_place_no_element_is_at_says_so(self):
        reached = {
            "speck": Reached(asked=0.5, reached=None, standing=None, elements=0, dimension=1)
        }
        demand = Demand(coarsest=2.0, finest=0.0, growth=1.5, places=(Near("speck", "walls", 0.5),))
        (line,) = [
            line
            for line in gmsh_mesh_report.describe(built(reached=reached), demand)
            if "'speck'" in line
        ]
        assert line.endswith("no element of the dimension filled is at it")

    def test_a_mark_outside_the_model_says_nothing_is_laid(self):
        reached = {
            "beyond": Reached(
                asked=0.5, reached=None, standing=None, elements=0, dimension=2, laid=False
            )
        }
        demand = Demand(coarsest=2.0, finest=0.0, growth=1.5, places=(Near("beyond", "ring", 0.5),))
        (line,) = [
            line
            for line in gmsh_mesh_report.describe(built(reached=reached), demand)
            if "'beyond'" in line
        ]
        assert line == (
            "Mesh: at 'beyond' asked 0.5 mm: nothing is laid, since 'ring' stands outside the model"
        )

    def test_a_mesh_nothing_overlapped_says_nothing_about_overlaps(self):
        """Held against the same mesh with an overlap in it, so an
        implementation that never speaks of overlaps fails the pair."""
        settled = Settled(
            dimension=3, tag=7, took="air", gave_up=("block",), place="(0, 0, 0) to (1, 1, 1)"
        )
        overlapped = self.said(settled=(settled,), left_out=(), parted=())
        assert self.said(settled=(), left_out=(), parted=()) == overlapped[:-1]
