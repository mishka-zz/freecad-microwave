# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Several catalogs at once, which is the whole feature.

A board house's laminates beside the generic nominal ones, plus whatever the
user has been sent. It works because a material is named ``catalog:material``,
so two catalogs both defining FR4 is the normal case rather than a clash.
"""

import os
import pathlib
import sys

import FreeCAD
import pytest

from Microwave.Materials.library import load_library
from Microwave.Materials.locations import (
    BUNDLED,
    ENV_VAR,
    PARAMETER_KEY,
    configured_paths,
    freecad_user_dir,
    installed,
    search_paths,
)
from Microwave.Materials.model import MaterialRef


def catalog(identifier, *materials, name=None, version="1"):
    body = f'schema = 1\n[catalog]\nid = "{identifier}"\n'
    body += f'name = "{name or identifier.title()}"\nversion = "{version}"\n'
    for material in materials:
        body += f'[[material]]\nid = "{material}"\nkind = "pec"\n'
    return body


def write(directory, filename, text):
    path = pathlib.Path(directory) / filename
    path.write_text(text)
    return path


class TestSeveralAtOnce:
    def test_two_catalogs_both_load(self, tmp_path):
        write(tmp_path, "a.toml", catalog("generic", "fr4"))
        write(tmp_path, "b.toml", catalog("jlcpcb", "fr4"))
        library = load_library([tmp_path])
        assert sorted(c.id for c in library.catalogs) == ["generic", "jlcpcb"]
        assert not library.failures

    def test_the_same_material_name_in_two_catalogs_is_not_a_clash(self, tmp_path):
        """This is why references are qualified, and it is the point of the design."""
        write(tmp_path, "a.toml", catalog("generic", "fr4"))
        write(tmp_path, "b.toml", catalog("jlcpcb", "fr4"))
        library = load_library([tmp_path])
        assert library.lookup(MaterialRef("generic", "fr4")) is not None
        assert library.lookup(MaterialRef("jlcpcb", "fr4")) is not None

    def test_a_single_file_is_a_path_too(self, tmp_path):
        """An emailed catalog should work from wherever it landed."""
        path = write(tmp_path, "jlcpcb.toml", catalog("jlcpcb", "fr4"))
        assert [c.id for c in load_library([path]).catalogs] == ["jlcpcb"]

    def test_a_directory_that_is_not_there_is_not_an_error(self, tmp_path):
        library = load_library([tmp_path / "nowhere"])
        assert library.catalogs == () and library.failures == ()


class TestOneBadFileDoesNotCostTheOthers:
    def test_the_good_ones_still_load(self, tmp_path):
        write(tmp_path, "good.toml", catalog("generic", "fr4"))
        write(tmp_path, "bad.toml", "schema = = 1")
        library = load_library([tmp_path])
        assert [c.id for c in library.catalogs] == ["generic"]
        assert len(library.failures) == 1

    def test_the_failure_names_the_file(self, tmp_path):
        write(tmp_path, "bad.toml", "schema = = 1")
        failure = load_library([tmp_path]).failures[0]
        assert failure.path.endswith("bad.toml")
        assert "not valid TOML" in failure.message

    def test_loading_never_raises(self, tmp_path):
        write(tmp_path, "bad.toml", "[nonsense]\n")
        load_library([tmp_path])

    def test_a_directory_that_cannot_be_read_is_a_failure_not_a_traceback(self, tmp_path):
        """Listing the directory sat outside the try, so a mode nobody meant to
        set took the whole picker down - against a docstring promising this
        never raises."""
        good = tmp_path / "good"
        good.mkdir()
        write(good, "good.toml", catalog("generic", "fr4"))
        shut = tmp_path / "shut"
        shut.mkdir()
        shut.chmod(0o000)
        try:
            library = load_library([shut, good])
        finally:
            shut.chmod(0o755)

        assert [c.id for c in library.catalogs] == ["generic"]
        assert len(library.failures) == 1
        assert library.failures[0].path == str(shut)
        assert "cannot be read" in library.failures[0].message


class TestADuplicateCatalogIdIsAFailureNotAShadow:
    """First wins, and the loser is reported.

    Last-wins would let a forgotten copy in a downloads folder redefine what FR4
    means with nothing said, and "why did my permittivity change?" would have no
    answer. Refusing both would be worse still: one stale file would delete FR4
    from the picker entirely.
    """

    def setup_files(self, tmp_path):
        first = write(tmp_path, "a-first.toml", catalog("generic", "fr4", version="1"))
        second = write(tmp_path, "b-second.toml", catalog("generic", "ptfe", version="2"))
        return first, second

    def test_the_first_one_keeps_the_id(self, tmp_path):
        self.setup_files(tmp_path)
        library = load_library([tmp_path])
        assert len(library.catalogs) == 1
        assert library.catalog("generic").version == "1"

    def test_the_loser_is_reported_and_both_files_are_named(self, tmp_path):
        first, second = self.setup_files(tmp_path)
        failure = load_library([tmp_path]).failures[0]
        assert failure.path == str(second)
        assert str(first) in failure.message
        assert "already loaded" in failure.message

    def test_the_bundled_catalog_cannot_be_shadowed(self, tmp_path):
        """Search order puts what ships with the workbench first, on purpose."""
        write(tmp_path, "mine.toml", catalog("generic", "something", version="mine"))
        library = load_library(search_paths(user_dir=tmp_path, env={}), bundled=BUNDLED)
        assert library.catalog("generic").bundled
        assert library.catalog("generic").get("fr4") is not None


class TestSearch:
    def library(self, tmp_path):
        write(
            tmp_path,
            "a.toml",
            'schema = 1\n[catalog]\nid = "jlcpcb"\nname = "JLCPCB"\nversion = "1"\n'
            '[[material]]\nid = "fr4-tg155"\nname = "FR-4 TG155"\nkind = "dielectric"\n'
            'epsilon_r = 4.4\ndescription = "high glass transition"\n',
        )
        write(tmp_path, "b.toml", catalog("generic", "ptfe"))
        return load_library([tmp_path])

    def test_the_catalog_name_is_searchable(self, tmp_path):
        """ "Which board house is this from" is the question being asked."""
        found = self.library(tmp_path).search("jlcpcb")
        assert [entry.id for _, entry in found] == ["fr4-tg155"]

    def test_so_is_the_description(self, tmp_path):
        assert self.library(tmp_path).search("glass transition")

    def test_an_empty_search_is_everything(self, tmp_path):
        assert len(self.library(tmp_path).search("  ")) == 2


class TestWhereToLook:
    def test_the_order_is_bundled_then_user_then_configured_then_environment(self):
        paths = search_paths(
            bundled=pathlib.Path("/bundled"),
            user_dir=pathlib.Path("/user"),
            configured=[pathlib.Path("/configured")],
            env={ENV_VAR: "/from-env"},
        )
        assert [str(path) for path in paths] == ["/bundled", "/user", "/configured", "/from-env"]

    def test_the_environment_variable_takes_a_list(self):
        paths = search_paths(bundled=None, env={ENV_VAR: os.pathsep.join(["/a", "/b"])})
        assert [str(path) for path in paths] == ["/a", "/b"]

    def test_an_empty_environment_variable_adds_nothing(self):
        assert search_paths(bundled=None, env={ENV_VAR: ""}) == ()

    def test_no_user_directory_is_not_an_empty_path(self):
        assert search_paths(bundled=None, user_dir=None, env={}) == ()

    def test_a_path_named_twice_is_searched_once(self):
        """$MICROWAVE_MATERIAL_PATH pointing at the user directory is the easy
        way in, and undeduplicated it loads every catalog there twice - reporting the
        second as colliding with the first, a failure whose two halves are the
        same file."""
        paths = search_paths(bundled=None, user_dir=pathlib.Path("/user"), env={ENV_VAR: "/user"})
        assert [str(path) for path in paths] == ["/user"]

    def test_it_keeps_the_first_mention_because_the_order_is_precedence(self):
        paths = search_paths(
            bundled=pathlib.Path("/bundled"),
            user_dir=pathlib.Path("/user"),
            env={ENV_VAR: os.pathsep.join(["/user", "/other"])},
        )
        assert [str(path) for path in paths] == ["/bundled", "/user", "/other"]

    def test_a_trailing_slash_is_the_same_directory(self):
        paths = search_paths(bundled=None, user_dir=pathlib.Path("/user"), env={ENV_VAR: "/user/"})
        assert len(paths) == 1


class TestTheDocumentedInstallPath:
    """Where a user is told to put a catalog, and whether one put there loads.

    ``search_paths`` above is the ordering, and it is pure. These are the parts
    that ask FreeCAD instead - the user data directory, the parameter store,
    and the load that joins them - and together they are what the install
    instructions describe.

    Three failures are distinct and are tested apart, because they arrive at
    three different lines: FreeCAD absent (the import raises), FreeCAD present
    but too old to carry the attribute, and a parameter store that answers with
    an exception. Both names are patched with ``raising=True``, so a rename on
    either side fails here rather than being invented by the patch.
    """

    def patched(self, monkeypatch, name, answer):
        """Patch the stub's *class*, never the instance.

        ``FreeCAD`` is one long-lived stub object. ``monkeypatch`` restores an
        attribute by assigning the old value back, so patching the instance
        leaves a bound method sitting on it as instance state - which then
        shadows the class for every later test, including one that deletes the
        class attribute to play a FreeCAD too old to have it.
        """
        monkeypatch.setattr(type(FreeCAD), name, lambda self, *args: answer)

    def user_dir(self, monkeypatch, root):
        self.patched(monkeypatch, "getUserAppDataDir", str(root))
        monkeypatch.delenv(ENV_VAR, raising=False)
        directory = root / "Microwave" / "materials"
        directory.mkdir(parents=True)
        return directory

    def test_the_user_directory_hangs_off_freecad_s_own(self, monkeypatch, tmp_path):
        """Spelled out rather than built from ``USER_SUBDIRECTORY``: the two
        names are what the install instructions tell a user to type, so they are
        the contract, and a test that reads the constant would follow it to
        wherever it moved and call that agreement."""
        self.patched(monkeypatch, "getUserAppDataDir", str(tmp_path))
        assert freecad_user_dir() == tmp_path / "Microwave" / "materials"

    def test_the_trailing_separator_freecad_returns_is_absorbed(self, monkeypatch, tmp_path):
        """FreeCAD hands back a path that ends in a separator. ``joinpath`` does
        not care and string concatenation would, so the two spellings have to
        land on the same directory."""
        self.patched(monkeypatch, "getUserAppDataDir", f"{tmp_path}{os.sep}")
        assert freecad_user_dir() == tmp_path / "Microwave" / "materials"

    def test_a_freecad_too_old_to_answer_is_not_a_traceback(self, monkeypatch):
        """The attribute is read rather than called blind, so the picker gets a
        ``None`` to report instead of an ``AttributeError`` to crash on."""
        monkeypatch.delattr(type(FreeCAD), "getUserAppDataDir")
        assert freecad_user_dir() is None

    def test_outside_freecad_there_is_no_user_directory(self, monkeypatch):
        """``None`` in ``sys.modules`` is what an interpreter without FreeCAD
        looks like from inside the function - ``import FreeCAD`` raises. It
        reaches these two only because both import it in the body rather than at
        module scope, which is the property under test as much as the answer."""
        monkeypatch.setitem(sys.modules, "FreeCAD", None)
        assert freecad_user_dir() is None
        assert configured_paths() == ()

    def test_the_parameter_store_takes_a_list(self, monkeypatch):
        class Group:
            def GetString(self, key, default):
                return os.pathsep.join(["/one", "/two"]) if key == PARAMETER_KEY else default

        self.patched(monkeypatch, "ParamGet", Group())
        assert [str(path) for path in configured_paths()] == ["/one", "/two"]

    def test_a_parameter_store_that_will_not_answer_is_not_a_traceback(self, monkeypatch):
        """Defensive rather than observed: FreeCAD 1.1 creates a group it has
        never seen rather than refusing one. The guard costs a line and the
        alternative is a picker that cannot open."""

        def refuse(self, group):
            raise RuntimeError("no parameter store")

        monkeypatch.setattr(type(FreeCAD), "ParamGet", refuse)
        assert configured_paths() == ()

    def test_a_catalog_dropped_in_the_user_directory_is_loaded(self, monkeypatch, tmp_path):
        """The install story end to end, and the point of this class. Beside the
        bundled catalog rather than instead of it, and marked as not bundled -
        the picker shows that distinction, and it is decided here."""
        write(self.user_dir(monkeypatch, tmp_path), "house.toml", catalog("housebrand", "fr4"))

        library = installed()
        assert library.lookup(MaterialRef("housebrand", "fr4")) is not None
        assert library.failures == ()
        assert not library.catalog("housebrand").bundled
        assert library.catalog("generic").bundled

    def test_a_path_the_user_configured_is_searched_too(self, monkeypatch, tmp_path):
        """The parameter store's paths reach the load, not just ``search_paths``.
        Driven through ``ParamGet`` rather than by replacing ``configured_paths``,
        so the wiring between the two is what is being tested."""
        self.user_dir(monkeypatch, tmp_path)
        elsewhere = tmp_path / "downloads"
        elsewhere.mkdir()
        write(elsewhere, "sent.toml", catalog("vendor", "fr4"))

        class Group:
            def GetString(self, key, default):
                return str(elsewhere) if key == PARAMETER_KEY else default

        self.patched(monkeypatch, "ParamGet", Group())

        assert installed().lookup(MaterialRef("vendor", "fr4")) is not None

    def test_a_path_the_user_typed_wrong_is_still_reported(self, monkeypatch, tmp_path):
        """A path the user *chose* is required, so a typo in it is a failure
        rather than a directory that happens to be empty. That distinction is
        made inside ``installed``, where nothing was watching it."""
        self.user_dir(monkeypatch, tmp_path)
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "nowhere"))

        assert [str(failure.path) for failure in installed().failures] == [
            str(tmp_path / "nowhere")
        ]


class TestTheBundledCatalog:
    """The one that ships. Tested as data, against the rules tested elsewhere."""

    def library(self):
        return load_library([BUNDLED], bundled=BUNDLED)

    def test_it_loads_with_no_failures(self):
        library = self.library()
        assert not library.failures, [f.message for f in library.failures]

    def test_it_is_marked_as_bundled(self):
        assert self.library().catalog("generic").bundled

    @pytest.mark.parametrize("material", ["fr4", "copper", "brass", "pvc", "air"])
    def test_it_has_the_basics(self, material):
        assert self.library().lookup(MaterialRef("generic", material)) is not None

    def test_its_copper_is_the_one_the_acceptance_case_is_built_from(self):
        """The shipped catalog and the shipped example must agree here.

        ``Generic.py`` had copper as a PEC with no conductivity, while
        ``examples/microstrip_50ohm.py`` builds it as a conducting sheet at
        5.8e7 S/m and 35 um - so the catalog could not express the single most
        common material in the workbench, and its version was the one no gate
        could check.
        """
        copper = self.library().lookup(MaterialRef("generic", "copper"))
        assert copper.kind == "conducting_sheet"
        assert copper.conductivity == 5.8e7
        assert copper.thickness == 0.035

    def test_every_lossy_material_says_where_its_loss_was_measured(self):
        for _catalog, entry in self.library().entries():
            if entry.loss_tangent > 0:
                assert entry.measured_at > 0, entry.id


class TestFailuresThatCouldPassInSilence:
    def test_a_file_that_is_not_utf8_is_a_failure_not_an_exception(self, tmp_path):
        """``UnicodeDecodeError`` is a ValueError, not an OSError, so a handler
        catching only the latter lets it escape ``load_library`` and take down
        every other catalog - which is what that function promises cannot
        happen."""
        (tmp_path / "latin.toml").write_bytes(
            'schema = 1\n[catalog]\nid = "x"\nname = "Caf\xe9"\nversion = "1"\n'.encode("latin-1")
        )
        write(tmp_path, "good.toml", catalog("generic", "fr4"))
        library = load_library([tmp_path])
        assert [c.id for c in library.catalogs] == ["generic"]
        assert "not UTF-8" in library.failures[0].message

    def test_a_path_the_user_typed_wrong_is_reported(self, tmp_path):
        """The bundled catalog always loads, so "no catalogs" can never fire -
        without this, MICROWAVE_MATERIAL_PATH=/typo produced no feedback at all."""
        library = load_library([], required=[tmp_path / "typo.toml"])
        assert library.failures[0].message.startswith("no such file")

    def test_a_directory_that_may_legitimately_be_absent_is_not(self, tmp_path):
        """The FreeCAD user directory does not exist until somebody puts a
        catalog in it, so complaining about it would be noise."""
        assert load_library([tmp_path / "nowhere"]).failures == ()

    def test_an_uppercase_suffix_is_still_a_catalog(self, tmp_path):
        write(tmp_path, "Rogers.TOML", catalog("rogers", "ro4350b"))
        assert [c.id for c in load_library([tmp_path]).catalogs] == ["rogers"]


class TestWhereACatalogCameFrom:
    """``bundled`` drives what the picker shows as the catalog's origin, so it
    has to be measured rather than assumed."""

    def test_a_file_outside_the_bundled_directory_is_not_bundled(self, tmp_path):
        write(tmp_path, "mine.toml", catalog("mine", "fr4"))
        library = load_library([tmp_path], bundled=BUNDLED)
        assert not library.catalog("mine").bundled

    def test_a_file_inside_it_is(self):
        assert load_library([BUNDLED], bundled=BUNDLED).catalog("generic").bundled

    def test_with_no_bundled_directory_nothing_claims_to_be(self, tmp_path):
        write(tmp_path, "mine.toml", catalog("mine", "fr4"))
        assert not load_library([tmp_path]).catalog("mine").bundled


class TestSearchingByCatalog:
    def test_the_catalog_id_matches_as_well_as_its_name(self, tmp_path):
        write(tmp_path, "a.toml", catalog("jlcpcb", "fr4", name="Shenzhen Board Co"))
        library = load_library([tmp_path])
        assert library.search("jlcpcb")
        assert library.search("shenzhen")
