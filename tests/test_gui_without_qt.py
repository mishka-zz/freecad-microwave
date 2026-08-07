# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the drawing modules do when there is nothing to draw with.

A plot module that raised ``NameError`` at import shipped behind a green suite,
and the reason is worth stating exactly, because the obvious explanation is
wrong. ``conftest`` installs a ``MagicMock`` for ``PySide``, and that did *not*
make the module's ``HAS_GUI`` true - the guarded block went on to import
matplotlib's Qt backend, which fails against a mock, so under the suite the
fallback branch ran. What the mock did was bind the name ``QtWidgets``
**successfully, before the failure**. So a class defined at module scope as
``QtWidgets.QDialog`` resolved against a mock and imported cleanly, while the
same statement on a machine whose *first* failing import came earlier had no
such name to resolve against.

Two consequences. The suite could not see this class of fault by running
normally; and which side of ``if HAS_GUI:`` it exercised depended on whether the
developer happened to have a real Qt binding, so it differed machine to machine.

**The trap is gone rather than guarded.** Charts are values now, drawn by
``Gui/charts.py`` into FreeCAD's own plot window, and nothing anywhere is
defined conditionally on a library being importable - so there is no statement
left whose meaning depends on how far an import got. What is tested here is the
property that made it matter: **importing must not raise**, because
``Gui/task_panel.py`` takes :func:`show_matrix` at module scope and an import
that raises there costs the whole task panel. Everything the workbench does
apart from drawing works without any of these libraries.

The interpreter below has none of matplotlib, Qt, FreeCAD or ``Plot`` - it is a
bare Python with the workbench on its path, which is both stronger than naming
the libraries one at a time and simpler to state.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

WORKBENCH = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def facts():
    """``label=value`` lines from an interpreter with nothing to draw with."""
    script = textwrap.dedent(
        """
        from Microwave.Gui import charts, plot_s_params, plot_tdr

        print("imported=yes")
        print(f"has_show_matrix={hasattr(plot_s_params, 'show_matrix')}")
        print(f"has_show_trace={hasattr(plot_tdr, 'show_trace')}")

        try:
            charts.module()
        except charts.ChartUnavailable as error:
            print(f"refusal={error}")

        # The path both entry points funnel through, with a chart that holds
        # nothing: what refuses here is the absence of anywhere to draw, which
        # is the condition under test and the only one a bare interpreter has.
        try:
            charts.render(charts.Chart(), "nothing to draw with")
        except charts.ChartUnavailable as error:
            print(f"render_refusal={error}")
        except Exception as error:
            print(f"render_wrong_type={type(error).__name__}: {error}")
        """
    )
    finished = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(WORKBENCH),
    )
    assert finished.returncode == 0, finished.stderr
    return dict(line.split("=", 1) for line in finished.stdout.strip().splitlines() if "=" in line)


def test_the_modules_import_with_nothing_to_draw_with(facts):
    """The whole point. The task panel imports these at module scope, so an
    import that raises here is a workbench whose analysis object cannot be
    opened at all - double-clicking it does nothing, and the traceback names a
    drawing symbol rather than the library that is actually missing."""
    assert facts["imported"] == "yes"


def test_the_entry_points_still_exist(facts):
    """``task_panel`` binds :func:`show_matrix` at import. A module that defines
    its entry point only in the working case turns a missing chart into a
    missing panel."""
    assert facts["has_show_matrix"] == "True"
    assert facts["has_show_trace"] == "True"


def test_drawing_refuses_at_the_call_and_says_what_is_missing(facts):
    """The refusal arrives when a chart is asked for, not at import, and it is
    one exception type - so a caller catching it for one chart catches it for
    every other. Both entry points reach it through :func:`charts.render`, which
    is what is exercised here rather than either of them: a chart of an
    S-matrix and a chart of a trace differ in what they compute and not at all
    in where they are drawn."""
    assert "render_wrong_type" not in facts, facts.get("render_wrong_type")
    assert facts["render_refusal"] == facts["refusal"]


def test_the_refusal_quotes_the_import_error_it_got(facts):
    """What has to be installed is decided by whichever import failed, and the
    refusal is the only place the user meets it. Asserting on the *quoted* text
    rather than on a library name: the message names FreeCAD's Plot module in
    its own static prose, so a check for that word passes with the interpolation
    deleted - and the interpolation is the whole point. It is not always the
    same name, either: ``Plot`` present and matplotlib absent reports matplotlib
    instead, which is why the text is passed through rather than interpreted."""
    assert "No module named" in facts["refusal"]


def test_the_task_panel_imports_with_qt_present_and_matplotlib_absent():
    """The reported symptom itself, and the configuration a real FreeCAD is in.

    FreeCAD ships Qt and does not always ship matplotlib. ``task_panel`` binds
    :func:`show_matrix` at module scope, so a plot module that raises under those
    conditions is a workbench whose analysis object cannot be opened.

    Qt comes from ``conftest``'s own stub, loaded here deliberately: it is what
    supplies a Qt that imports, which is half the condition being stated.
    """
    script = textwrap.dedent(
        """
        import sys, importlib.util

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "matplotlib":
                    raise ImportError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, Blocker())

        spec = importlib.util.spec_from_file_location("conftest", "tests/conftest.py")
        conftest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conftest)

        import Microwave.Gui.task_panel
        print("task_panel=imported")
        """
    )
    finished = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(WORKBENCH),
    )
    assert finished.returncode == 0, finished.stderr
    assert "task_panel=imported" in finished.stdout
