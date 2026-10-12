# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the impedance chart draws, and what it says it drew.

A chart is a value, so an axis in the wrong units or a basis naming a
wrong reference is a thing to assert on rather than a picture to squint at.
Everything here runs with no display and no matplotlib.
"""

import numpy as np
import pytest

from Microwave.Gui import plot_tdr, views
from Microwave.Results import tdr
from Microwave.Results.overlay import Table

from .test_tdr import SECTION, V, against, line, measured, reflected, shorted
from .test_views import LENGTH, ground, kept, port, two_port

TRACE = tdr.step_response(reflected(line([50.0, 75.0, 50.0])), 1)
OWN = tdr.step_response(against(line([50.0, 75.0, 50.0]), measured()), 1)

#: A line with one impedance from end to end, where nothing stands behind
#: anything - so a chart of it still carries the note about readings behind
#: a discontinuity.
PLAIN = tdr.step_response(reflected(line([50.0, 50.0, 50.0])), 1)


class TestTheAxisIsTheUsersChoice:
    def test_both_axes_are_offered(self):
        assert plot_tdr.AXES == (plot_tdr.DISTANCE, plot_tdr.TIME)

    def test_time_is_the_round_trip_in_nanoseconds(self):
        axis = plot_tdr.axis_of(TRACE, plot_tdr.TIME)
        assert axis.values == pytest.approx(1e9 * TRACE.time, rel=0.0, abs=0.0)
        assert "ns" in axis.label

    def test_distance_is_one_way_in_millimetres(self):
        axis = plot_tdr.axis_of(TRACE, plot_tdr.DISTANCE, speed=V)
        assert axis.values == pytest.approx(1e3 * tdr.distance(TRACE, V), rel=0.0, abs=0.0)
        assert "mm" in axis.label

    def test_the_two_axes_differ_by_the_velocity_and_the_round_trip(self):
        """The one relationship a reader cannot check by looking at the chart."""
        time = plot_tdr.axis_of(TRACE, plot_tdr.TIME).values
        along = plot_tdr.axis_of(TRACE, plot_tdr.DISTANCE, speed=V).values
        assert along == pytest.approx(0.5 * V * time * 1e-6, rel=1e-12, abs=0.0)

    def test_the_discontinuity_sits_where_the_drawing_put_it(self):
        """Not just self-consistent units: the section boundary is at 20 mm on
        the distance axis, which is where it was built."""
        along = plot_tdr.axis_of(TRACE, plot_tdr.DISTANCE, speed=V).values
        rising = along[np.nanargmax(np.gradient(TRACE.impedance, along))]
        assert rising == pytest.approx(1e3 * SECTION, abs=2.0)

    def test_distance_without_a_velocity_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="without a propagation velocity"):
            plot_tdr.axis_of(TRACE, plot_tdr.DISTANCE)

    def test_time_needs_no_velocity(self):
        assert plot_tdr.axis_of(TRACE, plot_tdr.TIME).values.size == TRACE.time.size

    def test_an_axis_it_does_not_draw_is_named_in_the_refusal(self):
        with pytest.raises(ValueError, match="no such axis"):
            plot_tdr.axis_of(TRACE, "frequency", speed=V)


def drawn(trace, speed=None, name="Run 1 (openEMS)"):
    return plot_tdr.Drawn(name, trace, speed, "" if speed is not None else "no velocity")


def imported(result, name="line.s2p"):
    """A table no run kept, as a Touchstone file comes in."""
    return Table(result, name)


class TestWhatTheChartSaysItIs:
    def test_the_reference_impedance_is_named(self):
        """A trace against 50 ohm and the same solve against 75 are different
        charts, and nothing in the curve says which."""
        assert "50 ohm" in plot_tdr.chart_text([drawn(TRACE)]).basis

    def test_the_port_is_named(self):
        assert "Port 1" in plot_tdr.chart_text([drawn(TRACE)]).basis

    def test_a_reference_the_port_measured_says_so(self):
        """The number then came out of this solve rather than being chosen, so
        the line under the port reads it back and the first plateau is not
        evidence about the line. Nothing in the curve says which it was."""
        assert "measured" in plot_tdr.chart_text([drawn(OWN)]).basis
        assert "measured" not in plot_tdr.chart_text([drawn(TRACE)]).basis

    def test_the_rise_time_is_named_with_what_set_it(self):
        basis = plot_tdr.chart_text([drawn(TRACE)]).basis
        quoted = float(basis.split("rise ", 1)[1].split(" ps")[0])
        assert quoted == pytest.approx(TRACE.rise * 1e12, rel=1e-2)
        assert "the band's fastest" in basis
        assert "as chosen" in plot_tdr.chart_text([drawn(TRACE)], chosen=True).basis

    def test_several_traces_name_the_table_whose_band_set_their_rise(self):
        basis = plot_tdr.chart_text(
            [drawn(TRACE), drawn(TRACE, name="line.s2p")], setter="line.s2p"
        ).basis
        assert "the fastest line.s2p allows" in basis

    def test_the_velocity_is_named_only_when_it_was_used(self):
        assert "mm/ns" not in plot_tdr.chart_text([drawn(TRACE)]).basis
        assert "mm/ns" in plot_tdr.chart_text([drawn(TRACE, V)]).basis

    def test_a_time_axis_does_not_claim_a_distance_basis(self):
        """The velocity is what turned time into distance. Quoting it under a
        time axis names a basis that was not used, and a reader takes the
        horizontal scale for something it is not."""
        assert "mm/ns" not in plot_tdr.chart_text([drawn(TRACE, V)], axis=plot_tdr.TIME).basis
        assert "mm/ns" in plot_tdr.chart_text([drawn(TRACE, V)], axis=plot_tdr.DISTANCE).basis

    def test_the_velocity_is_quoted_in_the_units_it_is_labelled_with(self):
        """A substring check passes with the exponent wrong, and this is the one
        number in the basis a reader would take at face value."""
        basis = plot_tdr.chart_text([drawn(TRACE, V)]).basis
        quoted = float(basis.rsplit("distance at ", 1)[1].split(" mm/ns")[0])
        # 1 m/s is 1e-6 mm/ns, and V is a real guided velocity, so this pins the
        # exponent rather than the arithmetic.
        assert quoted == pytest.approx(V * 1e-6, rel=1e-3)

    def test_what_the_conversion_does_to_every_reflection_but_the_first(self):
        """A plateau standing behind a discontinuity is not the impedance it was
        drawn at, and the curve looks exactly as it would if it were. That is a
        fact about the vertical axis rather than about the board, so it is said
        on every impedance chart and under either axis."""
        for axis in plot_tdr.AXES:
            assert plot_tdr.MASKING in plot_tdr.chart_text([drawn(TRACE, V)], axis=axis).basis
        assert plot_tdr.MASKING in plot_tdr.chart_text([drawn(PLAIN)]).basis

    def test_a_reflection_axis_makes_no_claim_about_impedance(self):
        text = plot_tdr.chart_text([drawn(TRACE)], vertical=plot_tdr.REFLECTION)
        assert plot_tdr.MASKING not in text.basis
        assert "reflection" in text.heading

    def test_what_the_velocity_carries_besides_distance(self):
        """The velocity is the delay averaged across the band, and what a
        structure stores is delay with no distance in it. Said with the number
        it qualifies, so it goes wherever that number goes and nowhere else."""
        assert plot_tdr.STORAGE in plot_tdr.chart_text([drawn(TRACE, V)]).basis
        assert plot_tdr.STORAGE not in plot_tdr.chart_text([drawn(TRACE)]).basis
        assert (
            plot_tdr.STORAGE not in plot_tdr.chart_text([drawn(TRACE, V)], axis=plot_tdr.TIME).basis
        )

    def test_the_study_titles_the_chart_and_does_not_take_the_basis(self):
        text = plot_tdr.chart_text([drawn(TRACE, V)], title="Board A")
        assert text.heading.startswith("Board A")
        assert "Board A" not in text.basis

    def test_an_untitled_study_still_gets_its_basis(self):
        """Which quantity is plotted can be read off the axes; the basis cannot
        be read off anything."""
        text = plot_tdr.chart_text([drawn(TRACE)])
        assert text.heading and not text.heading.startswith(" ")
        assert text.basis


class TestTheChartThatGetsDrawn:
    """What :mod:`~Microwave.Gui.charts` is handed. The selectors rebuild
    through :func:`~Microwave.Gui.plot_tdr.chart` on every change, so this is
    the whole of what changing a selector does."""

    def choices(self, built, label):
        (selector,) = [found for found in built.selectors if found.label == label]
        return [value for _, value in selector.choices], selector.chosen

    def test_the_curve_is_the_impedance_against_the_chosen_axis(self):
        table = two_port()
        built = plot_tdr.chart([table], axis=plot_tdr.DISTANCE)
        (curve,) = built.series
        trace = tdr.step_response(table.result, 1)
        assert curve.x == pytest.approx(
            plot_tdr.axis_of(trace, plot_tdr.DISTANCE, built_speed(table)).values
        )
        assert curve.y == pytest.approx(trace.impedance, nan_ok=True)

    def test_the_reflection_is_drawn_when_it_is_chosen(self):
        table = two_port()
        built = plot_tdr.chart([table], vertical=plot_tdr.REFLECTION)
        assert built.series[0].y == pytest.approx(
            np.real(tdr.step_response(table.result, 1).reflection)
        )
        assert built.y_label == "Reflection coefficient"

    def test_the_horizontal_label_follows_the_axis(self):
        assert "mm" in plot_tdr.chart([two_port()], axis=plot_tdr.DISTANCE).x_label
        assert "ns" in plot_tdr.chart([two_port()], axis=plot_tdr.TIME).x_label

    def test_the_view_starts_at_the_reference_plane(self):
        """The window's skirt reaches back before it. It is a diagnostic and not
        a reading, so it stays in the numbers the toolbar exports and out of the
        frame."""
        assert plot_tdr.chart([two_port()]).left == 0.0

    def test_the_ohms_are_named_on_the_vertical(self):
        assert plot_tdr.chart([two_port()]).y_label == "Impedance (ohm)"

    def test_a_velocity_puts_both_axes_in_the_selector(self):
        assert self.choices(plot_tdr.chart([two_port()]), "Horizontal axis") == (
            list(plot_tdr.AXES),
            plot_tdr.DISTANCE,
        )

    def test_a_table_no_run_kept_offers_only_time(self):
        """A greyed-out entry would invite the question the chart cannot answer,
        and an ungreyed one would put the refusal in the axes."""
        built = plot_tdr.chart([imported(two_port().result)], axis=plot_tdr.DISTANCE)
        assert self.choices(built, "Horizontal axis") == ([plot_tdr.TIME], plot_tdr.TIME)

    def test_the_missing_axis_is_a_warning_naming_the_table(self):
        """The whole reason goes to the console, and the warning says that the
        scale is a time and whose."""
        built = plot_tdr.chart([two_port(), imported(two_port().result)])
        assert f"{plot_tdr.NO_VELOCITY} line.s2p" in "\n".join(built.warnings)
        assert not any(plot_tdr.NO_VELOCITY in note for note in built.notes)

    def test_nothing_is_said_when_there_is_nothing_missing(self):
        built = plot_tdr.chart([two_port()])
        assert plot_tdr.NO_VELOCITY not in "\n".join(built.warnings)
        assert plot_tdr.BLANK not in "\n".join(built.warnings)

    def test_an_impedance_with_no_value_points_to_the_reflection(self):
        """At a short or an open the impedance has none, and the reflection
        reads there."""
        built = plot_tdr.chart([imported(shorted())])
        assert plot_tdr.BLANK in "\n".join(built.warnings)
        assert (
            plot_tdr.BLANK
            not in plot_tdr.chart([imported(shorted())], vertical=plot_tdr.REFLECTION).warnings
        )

    def test_several_traces_carry_their_velocities_in_a_table_and_not_the_notes(self):
        plane = ground()
        near = kept(
            two_port().result, port(1, 0.0, plane), port(2, LENGTH / 2, plane), name="Run 2"
        )
        built = plot_tdr.chart([two_port(), near])
        assert not any("mm/ns" in note for note in built.notes)
        assert [row[0] for row in built.table.rows] == ["Run 1 (openEMS)", "Run 2"]
        speeds = [float(row[1].split(" mm/ns")[0]) for row in built.table.rows]
        assert speeds[1] == pytest.approx(speeds[0] / 2, rel=1e-2)

    def test_one_trace_has_no_table(self):
        assert plot_tdr.chart([two_port()]).table is None

    def test_each_table_is_named_in_its_curve(self):
        built = plot_tdr.chart([two_port(), imported(two_port().result)])
        assert [curve.name for curve in built.series] == ["Run 1 (openEMS)", "line.s2p"]


def built_speed(table):
    from Microwave.Gui import views

    return views.velocity(table, 1)


class TestSeveralTablesShareOneRiseTime:
    """Two traces of one structure drawn at different rise times differ in
    the height of every narrow feature, and the difference reads as one
    between the structures."""

    NARROW = imported(reflected(line([50.0, 75.0, 50.0], points=60, fmax=8e9)), "narrow.s1p")
    WIDE = imported(reflected(line([50.0, 75.0, 50.0], points=300, fmax=20e9)), "wide.s1p")

    def test_every_trace_rises_in_the_time_the_slowest_band_allows(self):
        drawn = plot_tdr.traced(
            [self.NARROW, self.WIDE], 1, plot_tdr.edge([self.NARROW, self.WIDE])[0]
        )
        assert drawn[0].trace.rise == drawn[1].trace.rise
        assert drawn[0].trace.rise == pytest.approx(
            tdr.rise_time(tdr.fastest(self.NARROW.result)), rel=1e-12
        )

    def test_the_edge_is_named_by_the_table_that_set_it(self):
        assert plot_tdr.edge([self.WIDE, self.NARROW])[1] == "narrow.s1p"

    def test_every_trace_is_against_the_first_table_s_reference(self):
        other = imported(reflected(line([75.0, 75.0], port_z=75.0), reference=75.0), "75.s1p")
        drawn = plot_tdr.traced([self.WIDE, other], 1, tdr.fastest(self.WIDE.result))
        assert {found.trace.reference for found in drawn} == {50.0}

    def test_a_port_one_table_cannot_trace_is_offered_for_none(self):
        both = two_port()
        assert plot_tdr.ports([both], tdr.fastest(both.result)) == [1, 2]
        assert plot_tdr.ports([both, self.WIDE], tdr.fastest(self.WIDE.result)) == [1]

    def test_nothing_traceable_is_refused_naming_each_table(self):
        undriven = imported(reflected(line([50.0]), driven=(2,)), "undriven.s2p")
        with pytest.raises(Exception, match="undriven.s2p: cannot take a step response"):
            plot_tdr.chart([undriven, self.WIDE])


class TestAPortAskedForByName:
    """The menu asks for a port by name, and a chart of another port drawn in
    its place would answer a different question."""

    def test_a_port_the_chart_would_not_offer_is_refused_naming_why(self):
        from .test_views import guide_port

        plane = ground()
        table = kept(two_port().result, guide_port(1, 0.0, "X", 0.0), port(2, LENGTH, plane))
        with pytest.raises(Exception, match="it is a waveguide port"):
            plot_tdr.chart([table], 1, strict=True)

    def test_without_asking_by_name_the_first_offered_is_drawn(self):
        from .test_views import guide_port

        plane = ground()
        table = kept(two_port().result, guide_port(1, 0.0, "X", 0.0), port(2, LENGTH, plane))
        assert plot_tdr.chart([table], 1).selectors[0].chosen == 2


class TestOneFrequencyPoint:
    def test_it_is_refused_by_the_transform_s_own_sentence(self):
        one = reflected(line([50.0], points=2))
        from Microwave.Results.sparameters import SParameters

        single = SParameters(
            frequency=one.frequency[:1],
            s=one.s[:1],
            port_numbers=one.port_numbers,
            reference=one.reference[:1],
            measured_impedance=one.measured_impedance[:1],
            driven=one.driven,
        )
        with pytest.raises(Exception, match="holds 1 frequency point"):
            plot_tdr.chart([imported(single)])


class TestTheRiseTimesOffered:
    def test_the_fastest_comes_first_and_the_rest_are_slower(self):
        fastest = 10e9
        offered = plot_tdr.rises(fastest)
        assert offered[0][1] is None
        assert all(seconds > tdr.rise_time(fastest) for _, seconds in offered[1:])

    def test_none_is_far_slower_than_the_band_allows(self):
        fastest = 10e9
        assert all(
            seconds <= plot_tdr.SLOWEST * tdr.rise_time(fastest)
            for _, seconds in plot_tdr.rises(fastest)[1:]
        )

    def test_a_chosen_rise_time_is_the_one_drawn(self):
        table = two_port()
        offered = plot_tdr.rises(tdr.fastest(table.result))
        _, slower = offered[2]
        built = plot_tdr.chart([table], rise=slower)
        [basis] = built.notes
        assert "as chosen" in basis
        quoted = basis.split("rise ", 1)[1].split(" 10-90%")[0]
        assert quoted == plot_tdr._ps(slower)

    def test_a_slow_rise_is_quoted_in_nanoseconds(self):
        assert plot_tdr._ps(2e-9) == "2 ns"
        assert plot_tdr._ps(350e-12) == "350 ps"

    def test_a_rise_time_not_offered_falls_back_to_the_fastest(self):
        built = plot_tdr.chart([two_port()], rise=1e-15)
        assert "the band's fastest" in built.notes[0]


class TestAGuideIsRefusedByName:
    def test_the_chart_says_a_guide_has_no_low_pass_trace(self):
        from .test_views import guide_port

        table = kept(
            two_port().result, guide_port(1, 0.0, "X", 0.0), guide_port(2, LENGTH, "-X", 0.0)
        )
        with pytest.raises(
            Exception, match="a waveguide port, which carries nothing below its cutoff"
        ):
            plot_tdr.chart([table])


def test_a_table_carried_by_a_run_with_its_far_port_moved_is_drawn_where_the_run_was():
    """The distance comes from the run's own ports. A second run of the same
    numbers kept at half the length puts every feature at half the distance."""
    plane = ground()
    near = kept(two_port().result, port(1, 0.0, plane), port(2, LENGTH / 2, plane))
    far = plot_tdr.chart([two_port()]).series[0].x
    assert plot_tdr.chart([near]).series[0].x == pytest.approx(far / 2, rel=1e-6)


class TestATableExportedAndReadBack:
    """A Touchstone file written from a run and imported again holds the run's
    numbers and none of its ports, so it draws the same trace against time
    alone."""

    @pytest.fixture
    def read_back(self, tmp_path):
        from Microwave.Results import touchstone

        table = two_port()
        written = table.result.write_touchstone(tmp_path / "line")
        return table, Table(
            touchstone.read(written.name, written.read_bytes()).result, written.name
        )

    def test_it_draws_the_run_s_trace(self, read_back):
        run, imported_again = read_back
        drawn = plot_tdr.traced([run, imported_again], 1, tdr.fastest(run.result))
        assert drawn[1].trace.reflection == pytest.approx(drawn[0].trace.reflection, abs=1e-9)

    def test_it_is_drawn_against_time_naming_the_file(self, read_back):
        run, imported_again = read_back
        built = plot_tdr.chart([run, imported_again])
        assert [value for _, value in built.selectors[-1].choices] == [plot_tdr.TIME]
        assert f"{plot_tdr.NO_VELOCITY} {imported_again.name}" in "\n".join(built.warnings)


class TestATableTooCoarseForTheOthers:
    """Its band reaches higher than another's, and its step is too coarse to
    be drawn as slowly as the other needs."""

    FINE = imported(reflected(line([50.0, 75.0, 50.0], points=100, fmax=1e9)), "fine.s1p")
    COARSE = imported(reflected(line([50.0, 75.0, 50.0], points=21, fmax=19.95e9)), "coarse.s1p")

    def test_it_is_refused_by_name(self):
        with pytest.raises(Exception, match="coarse.s1p: cannot take a step response"):
            plot_tdr.chart([self.FINE, self.COARSE])

    def test_the_refusal_names_the_table_that_sets_the_rise(self):
        with pytest.raises(Exception, match="the fastest rise fine.s1p allows"):
            plot_tdr.chart([self.FINE, self.COARSE])

    def test_no_rise_time_is_offered_that_one_table_cannot_be_drawn_at(self):
        """The fine table sets the fastest edge, and the coarse one's step sets
        the slowest: a rise time slower than that is not offered."""
        fine = imported(reflected(line([50.0, 75.0, 50.0], points=200, fmax=5e9)), "fine.s1p")
        coarse = imported(reflected(line([50.0, 75.0, 50.0], points=40, fmax=20e9)), "coarse.s1p")
        tables = [fine, coarse]
        offered = plot_tdr.rises(plot_tdr.edge(tables)[0], plot_tdr.lowest(tables))[1:]
        assert offered
        assert len(offered) < len(plot_tdr.rises(plot_tdr.edge(tables)[0])[1:])
        for _, seconds in offered:
            plot_tdr.traced(tables, 1, tdr.edge_for(seconds))


def test_the_menu_s_port_is_drawn_or_refused_and_never_swapped(monkeypatch):
    from Microwave.Gui import charts

    from .test_views import guide_port

    drawn = []
    monkeypatch.setattr(charts, "render", lambda built, *_args, **_kwargs: drawn.append(built))
    plane = ground()
    table = kept(two_port().result, guide_port(1, 0.0, "X", 0.0), port(2, LENGTH, plane))
    with pytest.raises(Exception, match="it is a waveguide port"):
        plot_tdr.show_traces([table], 1)
    plot_tdr.show_traces([table], 2)
    assert drawn[0].selectors[0].chosen == 2


def test_nothing_selected_is_refused_saying_what_to_select():
    with pytest.raises(Exception, match="select a run"):
        plot_tdr.chart([])


class TestARefusalOverSeveralTables:
    """Each table's reason is read on its own, and nothing is said about a rise
    where no trace was refused for it."""

    def guide(self):
        from .test_views import guide_port

        return kept(
            two_port().result,
            guide_port(1, 0.0, "X", 0.0),
            guide_port(2, LENGTH, "X", 0.0),
            name="Run 4 (openEMS)",
        )

    def far(self):
        """A file whose sweep starts far above DC, as the guide's export is."""
        return imported(reflected(line([50.0], fmin=5e9, fmax=10e9, points=101)), "guide.s2p")

    def refusal(self, tables):
        with pytest.raises(Exception) as refused:
            plot_tdr.chart(tables)
        return str(refused.value)

    def test_each_tables_reason_stands_on_its_own_under_one_line(self):
        said = self.refusal([self.far(), self.guide()]).split("\n\n")
        assert said[0] == "no port can be traced in every table:"
        assert [line.split(": ")[0] for line in said[1:]] == ["guide.s2p", "Run 4 (openEMS)"]

    def test_no_rise_is_named_where_no_table_was_refused_for_it(self):
        assert "fastest rise" not in self.refusal([self.far(), self.guide()])

    def test_the_rise_is_named_where_a_table_is_refused_at_it_alone(self):
        fine = imported(reflected(line([50.0, 75.0, 50.0], points=100, fmax=1e9)), "fine.s1p")
        coarse = imported(
            reflected(line([50.0, 75.0, 50.0], points=21, fmax=19.95e9)), "coarse.s1p"
        )
        said = self.refusal([fine, coarse]).split("\n\n")
        assert said[-1] == "Every trace is drawn at the fastest rise fine.s1p allows."

    def test_a_files_sweep_is_said_as_what_a_step_response_needs(self):
        said = self.refusal([self.far(), self.guide()])
        assert "A step response needs a sweep from " in said
        assert "Sweep " not in said

    def test_no_rise_is_named_where_a_table_beside_the_refused_one_traces(self):
        traced = imported(reflected(line([50.0, 75.0, 50.0])), "line.s2p")
        said = self.refusal([traced, self.guide()])
        assert said.split("\n\n")[1:] == [said.split("\n\n")[1]]
        assert "fastest rise" not in said

    def test_tables_that_share_no_port_are_said_to(self):
        first = imported(reflected(line([50.0])), "first.s2p")
        second = imported(reflected(line([50.0]), ports=(3, 4), driven=(3,)), "second.s2p")
        assert self.refusal([first, second]) == "no port is measured in every table"

    def test_one_tables_reason_is_said_alone(self):
        said = self.refusal([self.guide()])
        assert said.startswith("cannot take a step response at port 1: ")
        assert said.endswith("Read its return loss and group delay off the S-parameter chart")

    def test_the_reasons_are_at_the_shared_port_fewest_tables_refuse(self):
        """Each table traces the port the other refuses, so either port comes as
        close, and the lower is said."""
        from .test_views import guide_port

        plane = ground()
        result = two_port().result
        first = kept(result, port(1, 0.0, plane), guide_port(2, LENGTH, "X", 0.0), name="A")
        second = kept(result, guide_port(1, 0.0, "X", 0.0), port(2, LENGTH, plane), name="B")
        said = self.refusal([first, second]).split("\n\n")
        assert said[1:] == [f"B: {views.guide_refusal(second, 1)}"]

    def test_a_port_no_table_traces_is_not_said_where_another_comes_closer(self):
        from .test_views import guide_port

        plane = ground()
        result = two_port().result
        first = kept(result, guide_port(1, 0.0, "X", 0.0), port(2, LENGTH, plane), name="A")
        second = kept(
            result, guide_port(1, 0.0, "X", 0.0), guide_port(2, LENGTH, "X", 0.0), name="B"
        )
        said = self.refusal([first, second]).split("\n\n")
        assert said[1:] == [f"B: {views.guide_refusal(second, 2)}"]

    def test_tables_of_one_name_are_each_said(self):
        said = self.refusal([self.far(), self.far()]).split("\n\n")
        assert [line.split(": ")[0] for line in said[1:]] == ["guide.s2p", "guide.s2p"]

    def test_a_band_of_one_point_is_said_as_its_tables(self):
        from Microwave.Results.sparameters import SParameters

        one = reflected(line([50.0], points=2))
        single = SParameters(
            frequency=one.frequency[:1],
            s=one.s[:1],
            port_numbers=one.port_numbers,
            reference=one.reference[:1],
            measured_impedance=one.measured_impedance[:1],
            driven=one.driven,
        )
        traced = imported(reflected(line([50.0, 75.0, 50.0])), "line.s2p")
        said = self.refusal([traced, imported(single, "one.s2p")]).split("\n\n")
        assert said[0] == "no port can be traced in every table:"
        assert [line.split(": ")[0] for line in said[1:]] == ["one.s2p"]

    def test_a_port_asked_for_by_name_is_said_a_table_to_a_paragraph(self):
        from .test_views import guide_port

        plane = ground()
        result = two_port().result
        first = kept(result, guide_port(1, 0.0, "X", 0.0), port(2, LENGTH, plane), name="A")
        second = kept(result, guide_port(1, 0.0, "X", 0.0), port(2, LENGTH, plane), name="B")
        with pytest.raises(Exception) as refused:
            plot_tdr.chart([first, second], 1, strict=True)
        said = str(refused.value).split("\n\n")
        assert said[0] == "port 1 cannot be traced in every table:"
        assert [line.split(": ")[0] for line in said[1:]] == ["A", "B"]

    def test_a_table_refused_at_the_shared_rise_is_told_what_it_needs(self):
        fine = imported(reflected(line([50.0, 75.0, 50.0], points=100, fmax=1e9)), "fine.s1p")
        coarse = imported(
            reflected(line([50.0, 75.0, 50.0], points=21, fmax=19.95e9)), "coarse.s1p"
        )
        said = self.refusal([fine, coarse]).split("\n\n")[1]
        assert said.startswith("coarse.s1p: ")
        assert said.endswith("That rise time needs a finer frequency step")

    def test_the_rise_is_named_from_the_port_the_reasons_are_at(self):
        """Port 1 is driven in neither table, so the reasons are at port 2,
        where the coarse table is refused at the fine one's rise alone."""

        def at_two(network, name):
            return imported(reflected(network, ports=(2, 1), driven=(2,)), name)

        fine = at_two(line([50.0, 75.0, 50.0], points=100, fmax=1e9), "fine.s2p")
        coarse = at_two(line([50.0, 75.0, 50.0], points=21, fmax=19.95e9), "coarse.s2p")
        said = self.refusal([fine, coarse]).split("\n\n")
        assert said[1].startswith("coarse.s2p: cannot take a step response at port 2")
        assert said[-1] == "Every trace is drawn at the fastest rise fine.s2p allows."

    def test_one_table_asked_for_a_port_by_name_is_told_its_reason_alone(self):
        from .test_views import guide_port

        table = kept(two_port().result, guide_port(1, 0.0, "X", 0.0), port(2, LENGTH, ground()))
        with pytest.raises(Exception) as refused:
            plot_tdr.chart([table], 1, strict=True)
        assert str(refused.value) == views.guide_refusal(table, 1)

    def test_a_port_asked_for_by_name_is_the_port_refused_where_none_is_offered(self):
        from .test_views import guide_port

        plane = ground()
        result = reflected(line([50.0]))
        tables = [
            kept(result, guide_port(1, 0.0, "X", 0.0), port(2, LENGTH, plane), name=name)
            for name in ("A", "B")
        ]
        with pytest.raises(Exception) as refused:
            plot_tdr.chart(tables, 2, strict=True)
        said = str(refused.value).split("\n\n")
        assert said[0] == "port 2 cannot be traced in every table:"
        assert all("at port 2: the port was not driven" in line for line in said[1:])
