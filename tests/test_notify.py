# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where a message lands. The strings themselves are asserted elsewhere."""

import pytest

from Microwave.Gui import notify
from tests.qt_recording import recording_qt, shown


@pytest.fixture
def qt(monkeypatch):
    """``QMessageBox``, recording, with a main window behind it."""
    boxes, _ = recording_qt(monkeypatch)
    return boxes


class TestARefusalGoesToBothChannels:
    """Nothing happened, so the sentence is the whole outcome of the press."""

    def test_the_console_keeps_it_and_the_box_delivers_it(self, qt, capsys):
        notify.refused("Update Mesh", "'Trace' is not axis-aligned")

        assert capsys.readouterr().out == "ERROR: Microwave: 'Trace' is not axis-aligned\n"
        assert shown(qt) == (qt.Warning, "Update Mesh", "'Trace' is not axis-aligned")

    def test_the_box_belongs_to_the_main_window(self, qt):
        """Nothing else holds a reference: the parent owns the box once this
        returns, and closing it is what deletes it."""
        notify.refused("Update Mesh", "no")

        qt.assert_called_once_with("the main window")
        qt.return_value.setAttribute.assert_called_once()
        qt.return_value.setStandardButtons.assert_called_once()

    def test_the_window_title_is_set_as_well_as_the_headline(self, qt):
        """macOS drops it, which is why the headline carries the name too.
        Everywhere else it is what the window is called."""
        notify.refused("Update Mesh", "no")

        qt.return_value.setWindowTitle.assert_called_once_with("Update Mesh")

    def test_the_box_is_shown_and_not_executed(self, qt):
        """What keeps a real FreeCAD driveable without a human: the modal
        calls run an event loop until somebody presses OK."""
        notify.refused("Update Mesh", "no")

        qt.return_value.show.assert_called_once()
        qt.return_value.exec_.assert_not_called()
        for modal in ("warning", "information", "critical"):
            getattr(qt, modal).assert_not_called()

    def test_with_no_window_there_is_only_the_console(self, qt, monkeypatch, capsys):
        """Building a widget with no QApplication aborts the process."""
        monkeypatch.setattr(notify, "main_window", lambda: None)

        notify.refused("Update Mesh", "'Trace' is not axis-aligned")

        assert "not axis-aligned" in capsys.readouterr().out
        qt.assert_not_called()


class TestANoteIsAboutSomethingThatDidHappen:
    """The object is in the tree and looks finished; the note says it is not."""

    def test_every_note_is_logged_and_all_of_them_share_one_box(self, qt, capsys):
        """One press, one answer. A box per note is a queue to dismiss."""
        notify.noted("Add Microstrip Port", ["Port: no ground", "Port: axes at defaults"])

        assert capsys.readouterr().out == (
            "WARNING: Microwave: Port: no ground\nWARNING: Microwave: Port: axes at defaults\n"
        )
        qt.assert_called_once()
        icon, _, detail = shown(qt)
        assert icon is qt.Information
        assert "no ground" in detail and "axes at defaults" in detail

    def test_nothing_to_say_says_nothing(self, qt, capsys):
        notify.noted("Add Microstrip Port", [])

        assert capsys.readouterr().out == ""
        qt.assert_not_called()

    def test_notes_arriving_lazily_are_read_once(self, qt, capsys):
        """The callers pass generators, and the notes are wanted twice."""
        notify.noted("Bind Material", (note for note in ("no material", "no geometry")))

        assert capsys.readouterr().out.count("WARNING") == 2
        _, _, detail = shown(qt)
        assert "no material" in detail and "no geometry" in detail
