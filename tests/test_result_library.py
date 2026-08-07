# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The vendored scikit-rf, and how the result layer gets hold of it.

The interesting cases are not "does it import" - they are the two ways the
choice can go wrong: silently using a copy that does not work, and silently
hijacking the name from a user who has their own.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from Microwave import VENDOR
from Microwave.Results import _skrf

WORKBENCH = Path(__file__).resolve().parent.parent


def in_fresh_interpreter(body):
    """Run ``body`` in a clean interpreter and return its stdout.

    ``sys.path`` and ``sys.modules`` are process-global, and this module's whole
    subject is what happens to them at import time. Testing that in-process
    would mean each test seeing the leftovers of the last.
    """
    script = textwrap.dedent(body)
    finished = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(WORKBENCH),
    )
    assert finished.returncode == 0, finished.stderr
    return finished.stdout.strip()


def facts_from(body):
    """``label=value`` lines out of one clean interpreter, as a dict.

    Several properties below are facts about *one* process at different moments.
    Each still gets its own assertion, so a failure names one thing.

    Not a runtime saving, measured: this file costs 8.2 s either way. What it
    costs is importing scikit-rf, 1.4 s a time, and the four children that do
    that each need their own interpreter - a different ``sys.path``, which is
    the whole subject. The two merged below were the cheap ones.
    """
    return dict(
        line.split("=", 1) for line in in_fresh_interpreter(body).splitlines() if "=" in line
    )


@pytest.fixture(scope="module")
def bootstrapped():
    """One interpreter, read at the three moments that matter, in order.

    ``import Microwave`` must leave ``_vendor`` last on ``sys.path`` and must not
    import scikit-rf; asking the result layer for the library is what imports it,
    and it must then say whose copy it got.

    One child rather than three because it is *stricter*, not because it is
    faster: the order is the order a real session goes in, so "skrf is not
    imported" is established in the same process that then imports it, rather than
    in one where it never could have been.
    """
    return facts_from(
        """
        import sys
        import Microwave
        print(f"vendor_index={sys.path.index(Microwave.VENDOR)}")
        print(f"path_length={len(sys.path)}")
        print(f"skrf_after_bootstrap={'skrf' in sys.modules}")
        from Microwave.Results import _skrf
        print(f"origin={_skrf.module().__file__}")
        print(f"source={_skrf.source()}")
        """
    )


@pytest.fixture(scope="module")
def with_numpy_and_scipy_first():
    """The vendored copy imported after numpy and scipy, and what numpy looked like.

    ``numpy_typing_absent`` is recorded *before* skrf is imported, because that
    is the precondition for the failure 1.13.0 was chosen over - and recording
    it rather than asserting it in the child is what lets one process serve both
    the plain import check and the numpy 1.x reproduction. On numpy 1 that is a
    child saved; on numpy 2 the reproduction skips and it is a wash.
    """
    return facts_from(
        f"""
        import sys
        sys.path.insert(0, {str(VENDOR)!r})
        import numpy, scipy
        print("numpy_typing_absent=" + str(not hasattr(numpy, "typing")))
        import skrf
        print("version=" + skrf.__version__)
        print("numpy_version=" + numpy.__version__)
        """
    )


class TestTheVendoredCopy:
    def test_it_is_there_and_imports_after_numpy_and_scipy(self, with_numpy_and_scipy_first):
        """1.13.0, not the newest release, and it survives the import order
        that chose it. See ``_vendor/README.md``.

        The version is pinned in a test because the reason is invisible from
        the tree: 2.0.x looks like an upgrade and breaks on the numpy the
        official FreeCAD ships. The failure has a shape - scikit-rf 2.0.x
        raises ``AttributeError: module 'numpy' has no attribute 'typing'``,
        because a bare ``import scipy`` does not pull ``numpy.typing`` in and
        nothing else had - so the fixture imports numpy and scipy *first*,
        and the plain import this replaces was a strict subset of it, in its own
        1.58 s subprocess.
        """
        assert with_numpy_and_scipy_first["version"] == "1.13.0"

    @pytest.mark.skipif(
        __import__("numpy").__version__.split(".")[0] != "1",
        reason="numpy 2 exposes numpy.typing lazily, so the 2.0.x failure cannot be reproduced",
    )
    def test_the_numpy_one_trap_is_really_reproduced(self, with_numpy_and_scipy_first):
        """On numpy 1.x, prove the precondition held while skrf imported.

        Without this, the test above passes for the wrong reason on the
        interpreter that matters. The state is only reachable on numpy 1.x -
        numpy 2 resolves ``numpy.typing`` through a lazy ``__getattr__``, so the
        precondition cannot be created there. FreeCAD 1.1.1 ships numpy 1.26.4,
        where it bites.
        """
        assert with_numpy_and_scipy_first["numpy_typing_absent"] == "True"

    def test_it_carries_its_licence(self):
        """BSD-3 requires the copyright notice to travel with the source."""
        licence = (Path(VENDOR) / "LICENSE.skrf.txt").read_text(encoding="utf-8")
        assert "Alexander Arsenovic" in licence
        assert "scikit-rf Developers" in licence
        assert "Redistribution and use in source and binary forms" in licence


class TestTheUsersOwnCopyWins:
    def test_vendor_is_appended_not_inserted(self, bootstrapped):
        """Anything already on the path outranks us.

        Inserting would hijack ``import skrf`` for the user's own scripts inside
        FreeCAD - their copy would stop being the one they get. The direction
        of this comparison is the whole point of the test.

        Dead last, exactly. Asking only that the index be near the end is
        satisfied by ``insert(len(sys.path) - 1, ...)``, which lands ``_vendor``
        *ahead of site-packages* - the one wrong answer that matters.
        """
        index = int(bootstrapped["vendor_index"])
        assert index == int(bootstrapped["path_length"]) - 1, (
            "vendor must be the last entry on sys.path"
        )

    def test_importing_the_workbench_does_not_import_skrf(self, bootstrapped):
        """A second of import time must not be paid at FreeCAD start.

        1.13.0 pulls scipy and pandas eagerly, and matplotlib wherever it is
        installed. Deferring it is the reason the bootstrap only touches
        ``sys.path``.
        """
        assert bootstrapped["skrf_after_bootstrap"] == "False"


class TestChoosingBetweenThem:
    def test_a_broken_installed_copy_falls_back_to_ours(self, tmp_path):
        """The case we expect to hit, not a hypothetical.

        scikit-rf 2.0.x is the current PyPI release and it fails on the numpy
        official FreeCAD ships, so "the user has their own" and "their own is
        broken" coincide. Simulated with a stub that raises the way 2.0.x does:
        an ``AttributeError`` from module scope, which is *not* an ImportError
        and would slip past a narrower ``except``.
        """
        # A *package* that imports a submodule before raising, which is the
        # shape scikit-rf 2.0.x actually has: it does ``from .frequency import
        # *`` and reaches the ``np.typing`` line afterwards. A single-file stub
        # leaves nothing behind in ``sys.modules``, so it cannot exercise the
        # cleanup in ``_import_vendored`` - and with that cleanup deleted the
        # whole suite stayed green. The leftover ``skrf.frequency`` is what
        # poisons the retry.
        broken = tmp_path / "broken"
        (broken / "skrf").mkdir(parents=True)
        # Deliberately *without* a ``Frequency``: the vendored package's
        # ``__init__`` does ``from .frequency import Frequency``, so a stale
        # submodule that happens to define the name would satisfy it and the
        # poisoning would go unnoticed. This is what makes the retry fail if the
        # ``sys.modules`` cleanup is removed.
        (broken / "skrf" / "frequency.py").write_text("# nothing useful here\n", encoding="utf-8")
        (broken / "skrf" / "__init__.py").write_text(
            "from . import frequency\n"
            "raise AttributeError(\"module 'numpy' has no attribute 'typing'\")\n",
            encoding="utf-8",
        )

        report = in_fresh_interpreter(
            f"""
            import sys
            sys.path.insert(0, {str(broken)!r})
            import Microwave
            from Microwave.Results import _skrf
            print(_skrf.module().__version__, _skrf.source(),
                  sys.path.index(Microwave.VENDOR), len(sys.path),
                  sys.path.count(Microwave.VENDOR))
            """
        )
        version, source, index, length, occurrences = report.split()
        assert source == "vendored"
        assert version == "1.13.0"

        # And it must not have promoted us while doing it. ``_import_vendored``
        # puts ``_vendor`` at the front for the duration of one import; failing
        # to take it away again leaves us outranking the user's own packages
        # for the rest of the session - the exact hijack that appending
        # rather than inserting exists to prevent. Without this assertion,
        # deleting the restoration leaves the suite green.
        #
        # Asserted here rather than in a second subprocess with an
        # ``ImportError`` stub: ``_skrf.module`` catches bare ``Exception``, so
        # both stubs take the identical path, and this one is strictly stronger
        # - a package that leaves ``skrf.frequency`` in ``sys.modules`` before
        # raising, which is what poisons the retry.
        assert int(occurrences) == 1, "the temporary entry was left behind"
        assert int(index) == int(length) - 1, "vendor outranks site-packages after a fallback"

    def test_a_working_installed_copy_is_preferred(self, tmp_path):
        """When their copy works, they keep it - we do not override."""
        theirs = tmp_path / "theirs"
        theirs.mkdir()
        (theirs / "skrf.py").write_text("__version__ = '99.0.0-theirs'", encoding="utf-8")

        report = in_fresh_interpreter(
            f"""
            import sys
            sys.path.insert(0, {str(theirs)!r})
            import Microwave
            from Microwave.Results import _skrf
            print(_skrf.module().__version__, _skrf.source())
            """
        )
        assert report == "99.0.0-theirs installed"

    def test_it_reports_which_copy_it_used(self):
        """Provenance: a Touchstone file should say what wrote it."""
        assert _skrf.module() is not None
        assert _skrf.source() in {"installed", "vendored"}
        assert _skrf.description().startswith("scikit-rf ")
        assert _skrf.source() in _skrf.description()

    def test_our_own_copy_is_not_reported_as_the_users(self, bootstrapped):
        """The label answers "whose copy", not "did the first import work".

        ``_vendor`` is on ``sys.path``, so with nothing else installed a plain
        ``import skrf`` succeeds *from our tree*. Reading that as "installed"
        was a real bug - it put a false claim in provenance, and it was
        invisible until the module was asked where it came from.
        """
        if bootstrapped["origin"].startswith(str(VENDOR)):
            assert bootstrapped["source"] == "vendored", (
                f"our own copy reported as {bootstrapped['source']!r}"
            )
        else:
            assert bootstrapped["source"] == "installed"

    def test_a_failure_of_our_own_copy_is_loud(self, monkeypatch):
        """No silent no-op. If nothing imports, say so and name it."""
        monkeypatch.setattr(_skrf, "_module", None)
        monkeypatch.setattr(
            _skrf, "_import_vendored", lambda: (_ for _ in ()).throw(ImportError("gone"))
        )
        monkeypatch.setitem(sys.modules, "skrf", None)

        with pytest.raises(_skrf.NoResultLibrary, match="scikit-rf"):
            _skrf.module()
