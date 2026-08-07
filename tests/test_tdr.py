# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Impedance against time, gated against lines whose sections are declared.

The reference here is exact rather than approximate: a synthetic network is
*built* from sections of stated impedance, so what the transform gets back can
be compared against the number that went in. What that leaves as the only source
of error is the transform's own band-limiting, which is the property worth
pinning - :class:`TestBandwidthSetsWhatIsResolved` measures it directly.

No solver and no FreeCAD. The derivation reads an
:class:`~Microwave.Results.sparameters.SParameters` and nothing else, so a
network assembled by hand exercises every line of it.
"""

import numpy as np
import pytest

from Microwave.Results import _skrf, tdr
from Microwave.Results.sparameters import ResultError, SParameters

C = 299792458.0

#: A plausible microstrip's effective permittivity. Nothing here rests on the
#: value - it only has to be the same one on both sides of the comparison.
EPS_EFF = 2.7
V = C / np.sqrt(EPS_EFF)

#: Each section of the fixture line, in metres.
SECTION = 0.020

#: The sweep the design settled on: the top of the band, and a point count that
#: puts the first measured point one step above DC, which is the one invented
#: bin the transform wants. The trace then reaches ``v / (4 * f_step)`` one way -
#: a quarter metre here, against a structure measured in centimetres. A quarter
#: and not a half: the transform's own axis spans ``1 / f_step`` of round trip
#: about zero, so half of it is the causal side and half of *that* is one way.
FMAX = 20e9
POINTS = 110


def _media(frequency, z0, port_z=50.0, nepers_per_m=0.0):
    skrf = _skrf.module()
    return skrf.media.DefinedGammaZ0(
        frequency=frequency,
        gamma=nepers_per_m + 1j * 2 * np.pi * frequency.f / V,
        z0_port=port_z,
        z0=z0,
    )


def _frequency(points=POINTS, fmax=FMAX, fmin=None):
    """A linear sweep starting one step above DC, which is what the transform
    wants: ``f_start == f_step`` leaves exactly one invented bin."""
    skrf = _skrf.module()
    fmin = fmax / points if fmin is None else fmin
    return skrf.Frequency.from_f(np.linspace(fmin, fmax, points), unit="hz")


def line(sections, load="match", points=POINTS, fmax=FMAX, port_z=50.0, fmin=None):
    """A chain of :data:`SECTION`-long sections, as a one-port ``skrf.Network``.

    ``port_z`` is what the reflection is referenced to, and it is a parameter so
    the conversion's ``Z_ref`` factor is exercised by something other than the
    50 ohm that would make a hard-coded one indistinguishable from it.
    """
    frequency = _frequency(points, fmax, fmin)
    network = _media(frequency, sections[0], port_z).line(SECTION, unit="m")
    for z0 in sections[1:]:
        network = network ** _media(frequency, z0, port_z).line(SECTION, unit="m")
    terminator = _media(frequency, port_z, port_z)
    return network ** getattr(terminator, load)()


def at_frequencies(f, sections, port_z=50.0):
    """The same chain of sections, over an arbitrary frequency axis."""
    skrf = _skrf.module()
    frequency = skrf.Frequency.from_f(np.asarray(f, dtype=float), unit="hz")
    network = _media(frequency, sections[0], port_z).line(SECTION, unit="m")
    for z0 in sections[1:]:
        network = network ** _media(frequency, z0, port_z).line(SECTION, unit="m")
    return network ** _media(frequency, port_z, port_z).match()


def reflected(network, reference=50.0, ports=(1, 2), driven=(1,)):
    """A one-port network as the driven column of a study with a second port.

    Two ports and one driven on purpose: that is what a time-domain solve
    produces, and the undriven column is the thing
    :meth:`SParameters.network` refuses over.
    """
    frequency = np.asarray(network.f, dtype=float)
    size = (frequency.size, len(ports), len(ports))
    s = np.full(size, np.nan, dtype=complex)
    s[:, 0, 0] = np.asarray(network.s, dtype=complex).ravel()
    z = np.full((frequency.size, len(ports)), reference, dtype=complex)
    return SParameters(
        frequency=frequency,
        s=s,
        port_numbers=tuple(ports),
        reference=z,
        measured_impedance=z,
        driven=tuple(driven),
    )


def _two_port(network):
    frequency = np.asarray(network.f, dtype=float)
    z = np.full((frequency.size, 2), 50.0, dtype=complex)
    return SParameters(
        frequency=frequency,
        s=np.asarray(network.s, dtype=complex),
        port_numbers=(1, 2),
        reference=z,
        measured_impedance=z,
    )


def _at(result, frequency):
    """The same matrix, hung on a different frequency axis."""
    return SParameters(
        frequency=np.asarray(frequency, dtype=float),
        s=result.s,
        port_numbers=result.port_numbers,
        reference=result.reference,
        measured_impedance=result.measured_impedance,
    )


def through(z0=50.0, sections=3, points=POINTS, fmax=FMAX):
    """A uniform line as a complete two-port, for measuring a velocity."""
    frequency = _frequency(points, fmax)
    return _two_port(_media(frequency, z0).line(SECTION * sections, unit="m"))


def stepped(impedances, points=POINTS, fmax=FMAX, nepers_per_m=0.0):
    """A chain of :data:`SECTION`-long sections as a two-port, one velocity throughout.

    Every section carries :data:`V` whatever its impedance, so what a velocity
    measured from this must return is *declared* rather than approximated. The
    sections still reflect off each other, and what that stores is delay with no
    distance in it - which is the whole of what this fixture is here to put in
    the way.
    """
    frequency = _frequency(points, fmax)

    def section(z0):
        return _media(frequency, z0, nepers_per_m=nepers_per_m).line(SECTION, unit="m")

    network = section(impedances[0])
    for z0 in impedances[1:]:
        network = network ** section(z0)
    return _two_port(network)


def stubbed(stub_lengths, points=POINTS, fmax=FMAX, z0=50.0):
    """A line with open stubs shunted along it, one velocity throughout.

    Each stub is a transmission zero, and a structure's phase runs backwards
    either side of one - so the honest steps of this fixture span more than a
    whole turn between them, and no single point in that range holds all of
    them. It is what separates unwrapping the phase from re-centring it.

    ``stub_lengths`` are in metres and there is a :data:`SECTION` of line before
    each stub and after the last.
    """
    frequency = _frequency(points, fmax)
    media = _media(frequency, z0)
    network = media.line(SECTION, unit="m")
    for length in stub_lengths:
        stub = media.line(length, unit="m") ** media.open()
        network = network ** media.shunt(stub) ** media.line(SECTION, unit="m")
    return _two_port(network)


def plateau(trace, low, high, speed=V):
    """Mean impedance between two distances along the line, in millimetres."""
    along = 1e3 * tdr.distance(trace, speed)
    inside = (along > low) & (along < high)
    assert inside.any(), f"no samples between {low} and {high} mm"
    return float(np.nanmean(trace.impedance[inside]))


class TestASectionReadsBackAtWhatItWasBuiltFrom:
    """The closed form: a section's impedance is the number it was declared with."""

    @pytest.mark.parametrize("middle", (25.0, 75.0, 120.0))
    def test_the_middle_section_is_recovered(self, middle):
        trace = tdr.step_response(reflected(line([50.0, middle, 50.0])), 1)
        # A section four resolution cells long is fully resolved, so what is left
        # is the window's ripple on the plateau. One percent is far outside it and
        # far inside the error a section this size would show if the transform
        # were band-limited differently than it is.
        assert plateau(trace, 25.0, 35.0) == pytest.approx(middle, rel=0.01)

    def test_the_feed_section_reads_the_reference(self):
        trace = tdr.step_response(reflected(line([50.0, 75.0, 50.0])), 1)
        assert plateau(trace, 2.0, 15.0) == pytest.approx(50.0, rel=0.01)

    @pytest.mark.parametrize(("port_z", "middle"), ((75.0, 120.0), (25.0, 40.0), (100.0, 60.0)))
    def test_it_scales_with_the_reference_and_does_not_assume_fifty(self, port_z, middle):
        """``Z = Z_ref (1 + rho) / (1 - rho)``, and ``Z_ref`` is the port's, not
        a constant. A fixture referenced to 50 cannot tell the two apart, and 50
        is what every other fixture here uses."""
        trace = tdr.step_response(
            reflected(line([port_z, middle, port_z], port_z=port_z), reference=port_z), 1
        )
        assert trace.reference == port_z
        assert plateau(trace, 2.0, 15.0) == pytest.approx(port_z, rel=0.01)
        assert plateau(trace, 25.0, 35.0) == pytest.approx(middle, rel=0.01)

    @pytest.mark.parametrize(
        ("load", "expected"),
        (("open", lambda z: z > 1e4), ("short", lambda z: abs(z) < 1.0)),
    )
    def test_a_singular_termination_reads_as_one(self, load, expected):
        trace = tdr.step_response(reflected(line([50.0], load=load)), 1)
        along = 1e3 * tdr.distance(trace, V)
        past = (along > 25.0) & (along < 60.0)
        assert expected(float(np.nanmedian(trace.impedance[past])))


class TestBandwidthSetsWhatIsResolved:
    """Close enough together and two features arrive as one, and no step count
    changes how close that is. Held as a property and not as a formula, because
    the separable width depends on the window as well as the bandwidth: a
    section wide enough for the bandwidth is recovered, the same section is not
    once the bandwidth alone is cut, and points bought at one bandwidth buy
    nothing."""

    def test_a_section_narrower_than_the_resolution_is_smeared_away(self):
        wide = tdr.step_response(reflected(line([50.0, 75.0, 50.0], fmax=FMAX)), 1)
        # Same point count, so the same unambiguous depth and the same trace
        # length: only the bandwidth moves, and with it the resolution.
        narrow = tdr.step_response(reflected(line([50.0, 75.0, 50.0], fmax=2.5e9)), 1)
        assert plateau(wide, 25.0, 35.0) == pytest.approx(75.0, rel=0.01)
        assert plateau(narrow, 25.0, 35.0) < 0.9 * 75.0

    def test_more_points_at_one_bandwidth_do_not_sharpen_it(self):
        few = tdr.step_response(reflected(line([50.0, 75.0, 50.0], fmax=2.5e9, points=60)), 1)
        many = tdr.step_response(reflected(line([50.0, 75.0, 50.0], fmax=2.5e9, points=400)), 1)
        # Both wrong, and wrong by the same amount: the sweep bought depth and
        # smoothness, and neither is resolution.
        assert plateau(many, 25.0, 35.0) == pytest.approx(plateau(few, 25.0, 35.0), rel=0.05)


class TestTheInventedBandIsBounded:
    """Everything below the first measured point is fabricated by the
    extrapolation, and a step response is carried by exactly those frequencies.
    Wide enough, and the fabrication replaces the answer rather than blurring
    it - smoothly, in range, with no blank sample to notice."""

    def bins(self, invented, points=201, fmax=10e9):
        """The 75 ohm section, swept so exactly ``invented`` bins sit under it.

        Solved for rather than approached: ``f_stop = (k + N - 1) * f_step``, so
        asking for the ratio directly is the only way the test says what it
        means. Setting ``f_start`` to ``k`` times a step computed from some other
        sweep moves the step as well, and lands beside the bar rather than on it.
        """
        step = fmax / (invented + points - 1)
        return self.at(invented * step, points, fmax)

    def at(self, fmin, points=201, fmax=10e9):
        return reflected(line([50.0, 75.0, 50.0], points=points, fmax=fmax, fmin=fmin))

    def test_a_sweep_starting_at_its_own_step_is_accurate(self):
        assert plateau(tdr.step_response(self.bins(1), 1), 25.0, 35.0) == pytest.approx(
            75.0, rel=0.01
        )

    def test_everything_up_to_the_bar_is_still_accurate(self):
        """The bar is a conservative proxy for a condition about the structure,
        so what it has to be is *safe*, not tight: every sweep it admits reads
        the section correctly."""
        for invented in range(1, tdr.INVENTED_BINS + 1):
            trace = tdr.step_response(self.bins(invented), 1)
            assert plateau(trace, 25.0, 35.0) == pytest.approx(75.0, rel=0.01), invented

    def test_the_answer_it_withholds_is_worth_withholding(self, monkeypatch):
        """The guard lifted, so the trace it refuses can be looked at.

        Outside the tolerance every admitted sweep meets, and arriving as an
        ordinary number rather than as a blank - which is the whole reason this
        is refused up front rather than reported afterwards. *How far* outside
        is not asserted, because it grows with the structure's length: that
        dependence is why the bar is a proxy and not a measurement.
        """
        default_band = self.at(1e9, points=501)
        with pytest.raises(ResultError):
            tdr.step_response(default_band, 1)

        monkeypatch.setattr(tdr, "INVENTED_BINS", float("inf"))
        withheld = plateau(tdr.step_response(default_band, 1), 25.0, 35.0)
        assert withheld != pytest.approx(75.0, rel=0.01)
        assert np.isfinite(withheld), "and it arrives as a number, not as a blank"

    def test_the_workbenchs_own_default_band_is_refused_rather_than_answered(self):
        """1 to 10 GHz over 501 points is what ``EMAnalysis`` starts life with,
        and it reads a 75 ohm section as several hundred. That has to refuse."""
        with pytest.raises(ResultError, match="would be invented"):
            tdr.step_response(self.at(1e9, points=501), 1)

    def test_the_refusal_names_a_sweep_that_would_work(self):
        with pytest.raises(ResultError) as refused:
            tdr.step_response(self.at(1e9, points=501), 1)
        # The suggestion has to be the sweep the transform actually wants, so it
        # is read back out of the message and run.
        suggested = float(str(refused.value).rsplit("start at about ", 1)[1].split(" GHz")[0]) * 1e9
        trace = tdr.step_response(self.at(suggested, points=501), 1)
        assert plateau(trace, 25.0, 35.0) == pytest.approx(75.0, rel=0.01)

    @pytest.mark.parametrize("shape", ("log", "jittered"))
    def test_a_sweep_that_is_not_linear_is_refused_rather_than_resampled(self, shape):
        """``extrapolate_to_dc`` ends by interpolating onto a ``linspace``, so
        anything downstream of it sees a uniform sweep whatever went in. The
        check has to be on the input, and without it a log sweep is silently
        resampled and answered."""
        even = np.linspace(20e9 / 110, 20e9, 110)
        f = (
            np.logspace(np.log10(1.8e8), np.log10(20e9), 110)
            if shape == "log"
            else np.sort(even * (1 + 0.4 * np.sin(np.arange(110))))
        )
        with pytest.raises(ResultError, match="evenly spaced whatever they are"):
            tdr.step_response(reflected(at_frequencies(f, [50.0, 75.0, 50.0])), 1)

    def test_a_sweep_that_descends_is_refused(self):
        result = self.bins(1)
        with pytest.raises(ResultError, match="do not ascend"):
            tdr.step_response(
                SParameters(
                    frequency=result.frequency[::-1],
                    s=result.s,
                    port_numbers=result.port_numbers,
                    reference=result.reference,
                    measured_impedance=result.measured_impedance,
                    driven=result.driven,
                ),
                1,
            )

    def test_a_sweep_of_one_point_has_nothing_to_transform(self):
        result = self.at(1e9)
        one = SParameters(
            frequency=result.frequency[:1],
            s=result.s[:1],
            port_numbers=result.port_numbers,
            reference=result.reference[:1],
            measured_impedance=result.measured_impedance[:1],
            driven=result.driven,
        )
        with pytest.raises(ResultError, match="more than 1 frequency point"):
            tdr.step_response(one, 1)


class TestTheSingularityIsBlankedRatherThanClipped:
    def test_unity_reflection_gives_no_number(self):
        blank = tdr._ohms(np.array([0.5, 1.0, 1.0001, -1.0]), 50.0)
        assert np.isnan(blank[1:]).all()
        assert blank[0] == pytest.approx(150.0)

    def test_nothing_is_pulled_to_a_bound(self):
        """A clip would put a plausible impedance where there is no answer."""
        trace = tdr.step_response(reflected(line([50.0], load="open")), 1)
        finite = trace.impedance[np.isfinite(trace.impedance)]
        # An open genuinely reaches into the hundreds of kilohms; a clip would
        # stand the whole trace on whatever bound it chose.
        assert finite.max() > 1e4
        assert np.isnan(trace.impedance).any()

    def test_a_passive_line_needs_no_blanking_at_all(self):
        trace = tdr.step_response(reflected(line([50.0, 75.0, 50.0])), 1)
        assert np.isfinite(trace.impedance).all()


class TestTheWindowIsAChoiceWithConsequences:
    """It is fixed rather than offered, but not because it does nothing: the
    docstring on :data:`~Microwave.Results.tdr.WINDOW` claims a trade, and these
    hold both ends of it so the claim cannot quietly stop being true."""

    STRONG = ([50.0], "open")

    def test_the_default_is_the_one_that_is_shipped(self):
        assert tdr.WINDOW == "hamming"
        assert tdr.step_response(reflected(line([50.0, 75.0, 50.0])), 1).impedance == pytest.approx(
            tdr.step_response(reflected(line([50.0, 75.0, 50.0])), 1, window=tdr.WINDOW).impedance,
            nan_ok=True,
        )

    @pytest.mark.parametrize("window", ("hann", "blackman", "boxcar"))
    def test_a_plateau_survives_any_of_them(self, window):
        """The section is read correctly whichever window is used - that is what
        makes fixing one a presentation choice rather than a physics one."""
        trace = tdr.step_response(reflected(line([50.0, 75.0, 50.0])), 1, window=window)
        assert plateau(trace, 25.0, 35.0) == pytest.approx(75.0, rel=0.01)

    def test_a_strong_reflection_is_where_it_shows(self):
        """Ringing carries |rho| past unity around a singular termination, and
        how much of the trace that blanks depends on the window. This is the
        half of the trade the plateau test cannot see."""
        blanked = {
            window: int(
                np.isnan(
                    tdr.step_response(reflected(line(*self.STRONG)), 1, window=window).impedance
                ).sum()
            )
            for window in ("hamming", "blackman")
        }
        assert blanked["hamming"] != blanked["blackman"]


class TestPaddingDrawsAndDoesNotResolve:
    SECTIONS = [50.0, 75.0, 50.0]

    @pytest.mark.parametrize("spectrum", (256, 2048))
    def test_more_padding_draws_the_same_trace_with_more_samples(self, spectrum):
        trace = tdr.step_response(reflected(line(self.SECTIONS)), 1, spectrum=spectrum)
        assert trace.time.size > spectrum
        assert plateau(trace, 25.0, 35.0) == pytest.approx(75.0, rel=0.01)

    def test_padding_buys_samples_and_not_separation(self):
        coarse = tdr.step_response(reflected(line(self.SECTIONS)), 1, spectrum=256)
        fine = tdr.step_response(reflected(line(self.SECTIONS)), 1, spectrum=4096)
        assert fine.time.size > 4 * coarse.time.size
        # The edge sits in the same place and rises over the same distance: the
        # extra samples drew it, they did not sharpen it.
        edges = [
            tdr.distance(t, V)[np.nanargmax(np.gradient(t.impedance, tdr.distance(t, V)))]
            for t in (coarse, fine)
        ]
        assert edges[1] == pytest.approx(edges[0], abs=1e-3)


class TestWhatItRefuses:
    def test_a_self_referenced_port_is_refused_as_circular(self):
        result = reflected(line([50.0, 75.0, 50.0]))
        circular = SParameters(
            frequency=result.frequency,
            s=result.s,
            port_numbers=result.port_numbers,
            reference=result.reference,
            measured_impedance=result.measured_impedance,
            driven=result.driven,
            self_referenced=(1,),
        )
        with pytest.raises(ResultError, match="own measured impedance"):
            tdr.step_response(circular, 1)

    def test_a_reference_that_varies_across_the_band_is_refused(self):
        result = reflected(line([50.0, 75.0, 50.0]))
        varying = np.asarray(result.reference, dtype=complex).copy()
        varying[:, 0] = np.linspace(48.0, 52.0, varying.shape[0])
        with pytest.raises(ResultError, match="one number for the whole trace"):
            tdr.step_response(
                SParameters(
                    frequency=result.frequency,
                    s=result.s,
                    port_numbers=result.port_numbers,
                    reference=varying,
                    measured_impedance=result.measured_impedance,
                    driven=result.driven,
                ),
                1,
            )

    def test_a_complex_reference_is_refused_for_being_complex(self):
        """A constant complex reference is not a *varying* one, and telling that
        user their reference moves across the band would quote them one number
        twice and send them to a setting that is already fixed."""
        result = reflected(line([50.0]))
        reactive = np.full_like(np.asarray(result.reference, dtype=complex), 50.0 + 3.0j)
        with pytest.raises(ResultError, match="real by construction"):
            tdr.step_response(
                SParameters(
                    frequency=result.frequency,
                    s=result.s,
                    port_numbers=result.port_numbers,
                    reference=reactive,
                    measured_impedance=result.measured_impedance,
                    driven=result.driven,
                ),
                1,
            )

    def test_a_hole_in_the_band_is_refused_rather_than_transformed(self):
        result = reflected(line([50.0, 75.0, 50.0]))
        holed = np.asarray(result.s, dtype=complex).copy()
        holed[5, 0, 0] = np.nan
        with pytest.raises(ResultError, match="hold no number"):
            tdr.step_response(
                SParameters(
                    frequency=result.frequency,
                    s=holed,
                    port_numbers=result.port_numbers,
                    reference=result.reference,
                    measured_impedance=result.measured_impedance,
                    driven=result.driven,
                ),
                1,
            )

    def test_an_undriven_column_does_not_block_the_port_that_was_driven(self):
        """The case ``SParameters.network`` refuses, and this must not."""
        result = reflected(line([50.0, 75.0, 50.0]))
        assert result.unmeasured == (2,)
        assert tdr.step_response(result, 1).reference == 50.0

    def test_the_one_port_it_builds_says_what_it_is_referenced_to(self):
        """The transform reads only ``s``, so a wrong ``z0`` here changes no
        number this module returns - and would still be a network that lies
        about itself to anything that later cascades or renormalises it."""
        result = reflected(line([75.0], port_z=75.0), reference=75.0)
        assert np.asarray(tdr._one_port(result, 1, 75.0).z0) == pytest.approx(75.0)

    def test_a_port_nobody_drove_is_told_that_and_not_that_it_is_circular(self):
        """``SParameters`` folds every undriven port into ``self_referenced``,
        which is true bookkeeping and the wrong thing to say here: nothing was
        restated, the column is ``nan``, and the fix is to drive the port rather
        than to change what it is referenced to."""
        result = reflected(line([50.0, 75.0, 50.0]))
        assert 2 in result.self_referenced and 2 in result.unmeasured
        with pytest.raises(ResultError, match="nobody drove it") as refused:
            tdr.step_response(result, 2)
        assert "own measured impedance" not in str(refused.value)

    def test_a_reference_of_zero_is_refused(self):
        result = reflected(line([50.0, 75.0, 50.0]), reference=0.0)
        with pytest.raises(ResultError, match="measured against a positive one"):
            tdr.step_response(result, 1)


class TestVelocity:
    def test_a_known_line_gives_back_the_velocity_it_was_built_with(self):
        speed = tdr.velocity(through(), 2, 1, separation=3 * SECTION)
        assert speed == pytest.approx(V, rel=1e-3)

    def test_it_belongs_to_the_medium_and_not_to_the_length(self):
        """Twice the line, twice the delay, one velocity - which is what makes it
        the right thing to put a distance axis on."""
        short = tdr.velocity(through(sections=6), 2, 1, separation=6 * SECTION)
        long_ = tdr.velocity(through(sections=12), 2, 1, separation=12 * SECTION)
        assert short == pytest.approx(long_, rel=1e-6)

    @pytest.mark.parametrize(("sections", "points"), ((60, 12), (3, 4), (30, 20), (30, 110)))
    def test_a_sweep_too_coarse_to_unwrap_is_refused(self, sections, points):
        """A fold shows as a delay too small for the length, which is a speed
        nothing travels at. Nothing in the phase itself gives it away: it stays
        as straight as an honest one, so the check has to be physical."""
        with pytest.raises(ResultError, match="not a speed"):
            tdr.velocity(
                through(sections=sections, points=points), 2, 1, separation=sections * SECTION
            )

    def test_a_line_right_up_to_the_fold_is_still_measured(self):
        """The refusal must not be a blanket one on long lines: this one turns
        wraps several times across the band and is answered exactly."""
        result = through(sections=12)
        assert tdr.velocity(result, 2, 1, separation=12 * SECTION) == pytest.approx(V, rel=1e-3)

    def test_ports_at_one_place_are_refused(self):
        with pytest.raises(ResultError, match="different places"):
            tdr.velocity(through(), 2, 1, separation=0.0)

    def test_a_band_with_no_width_is_refused(self):
        result = through()
        flat = np.full_like(np.asarray(result.frequency, dtype=float), 5e9)
        assert tdr.velocity(result, 2, 1, separation=3 * SECTION) > 0.0
        with pytest.raises(ResultError, match="Sweep a band"):
            tdr.velocity(_at(result, flat), 2, 1, separation=3 * SECTION)

    def test_a_sweep_that_runs_backwards_is_refused(self):
        """A whole measurement in reverse, matrix and axis together, which the
        arithmetic here would answer correctly - the phase and the band change
        sign as one. It is refused anyway, because ``_check_the_sweep`` refuses a
        descending axis for the step response and a result that can be drawn
        against distance one way and not the other would be the worse surprise.
        """
        result = through()
        backwards = SParameters(
            frequency=np.asarray(result.frequency, dtype=float)[::-1],
            s=np.asarray(result.s, dtype=complex)[::-1],
            port_numbers=result.port_numbers,
            reference=result.reference,
            measured_impedance=result.measured_impedance,
        )
        with pytest.raises(ResultError, match="Sweep a band"):
            tdr.velocity(backwards, 2, 1, separation=3 * SECTION)

    @pytest.mark.parametrize(("sections", "points"), ((12, 60), (20, 110), (25, 110)))
    def test_a_line_whose_phase_wraps_is_still_followed(self, sections, points):
        """Lines long enough that the phase turns several times across the band.
        Read as it comes rather than unwrapped, each of these gives a delay
        either negative or faster than light, so the guard below turns a line
        that is perfectly measurable into a refusal.
        """
        result = through(sections=sections, points=points)
        turns = np.ptp(np.unwrap(np.angle(np.asarray(result.parameter(2, 1), complex))))
        assert turns > 2 * np.pi, "the fixture has to wrap for this to test anything"
        assert tdr.velocity(result, 2, 1, separation=sections * SECTION) == pytest.approx(
            V, rel=1e-3
        )

    def test_a_hole_in_the_transmission_term_is_refused(self):
        """Both bounds below are comparisons, and a ``nan`` fails a comparison
        by being false - so without this the answer is ``nan`` and neither guard
        says a word."""
        result = through()
        holed = np.asarray(result.s, dtype=complex).copy()
        holed[7, 1, 0] = np.nan
        with pytest.raises(ResultError, match="not numbers"):
            tdr.velocity(
                SParameters(
                    frequency=result.frequency,
                    s=holed,
                    port_numbers=result.port_numbers,
                    reference=result.reference,
                    measured_impedance=result.measured_impedance,
                ),
                2,
                1,
                separation=3 * SECTION,
            )


class TestOneWildPointCostsAtMostOneTurn:
    """A resonance where the extraction collapses puts a single absurd phase in
    the band, and a delay taken across the whole band is
    two measured phases with the unwrapping between them, so that one point is
    carried rather than rejected.

    Between the ends a spike is *two* steps, onto the bad point and off it again,
    equal and opposite: they cancel outright unless one lands past half a turn
    and is unwrapped as its own complement, and the damage is then one whole turn
    of the band. At an end there is only one step, nothing cancels, and what is
    left is the spoilt phase itself - under half a turn, since that is all a
    phase can be out by.

    Either way it is one turn of the band at the very worst, which is the bound
    the whole estimator rests on: two absurd samples are two turns, and a hundred
    of them are not a hundred.
    """

    SECTIONS = 6

    #: The last index of the fixture's own sweep, so the two ends are named
    #: rather than counted.
    LAST = POINTS - 1

    def separation(self):
        return self.SECTIONS * SECTION

    def turns_lost(self, index, wild):
        """What spoiling one point costs the delay, counted in turns of phase."""
        clean = through(sections=self.SECTIONS)
        spoilt = np.asarray(clean.s, dtype=complex).copy()
        spoilt[index, 1, 0] *= np.exp(1j * wild)
        wobbly = SParameters(
            frequency=clean.frequency,
            s=spoilt,
            port_numbers=clean.port_numbers,
            reference=clean.reference,
            measured_impedance=clean.measured_impedance,
        )
        delay = self.separation() / tdr.velocity(wobbly, 2, 1, separation=self.separation())
        was = self.separation() / tdr.velocity(clean, 2, 1, separation=self.separation())
        # One turn of phase across the band is 1 / band of delay, so the
        # difference comes out in turns and the assertions need no tolerance of
        # their own.
        return (delay - was) * float(np.ptp(np.asarray(clean.frequency, dtype=float)))

    @pytest.mark.parametrize("index", (1, 40, 108))
    @pytest.mark.parametrize("wild", (0.5, -0.5, 2.5, -2.5, 3.0, np.pi))
    def test_between_the_ends_it_costs_nothing_or_one_whole_turn(self, index, wild):
        """Nothing in between, because the two steps are equal and opposite and
        the only thing that can happen to either of them is a whole turn."""
        assert self.turns_lost(index, wild) in (
            pytest.approx(0.0, abs=1e-12),
            pytest.approx(1.0, abs=1e-9),
            pytest.approx(-1.0, abs=1e-9),
        )

    @pytest.mark.parametrize("index", (1, 40, 108))
    @pytest.mark.parametrize("wild", (0.5, -0.5))
    def test_a_spike_the_unwrapping_can_follow_costs_nothing_at_all(self, index, wild):
        assert self.turns_lost(index, wild) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("wild", (0.5, -0.5, 2.5, -2.5, 3.0))
    def test_on_an_end_it_costs_the_phase_that_was_spoilt(self, wild):
        """There is one step at an end rather than two, so the spoilt phase
        stays in the answer instead of cancelling - give or take the whole turn
        that one step can still be moved by. The far end enters with the opposite
        sign, the delay being read from the bottom of the band to the top.
        """
        for index, sign in ((0, 1.0), (self.LAST, -1.0)):
            left = self.turns_lost(index, wild) - sign * wild / (2 * np.pi)
            assert left == pytest.approx(round(left), abs=1e-9)

    @pytest.mark.parametrize("index", (0, 1, 40, 108, POINTS - 1))
    @pytest.mark.parametrize("wild", (0.5, 2.5, -2.5, 3.0, np.pi))
    def test_nothing_a_single_point_can_hold_costs_more_than_one_turn(self, index, wild):
        assert abs(self.turns_lost(index, wild)) <= 1.0 + 1e-9


class TestWhatTheStructureStoresIsNotDistance:
    """A structure that reflects also stores, and stored energy is delay with no
    distance in it: an axis calibrated with it is slow, and every feature on the
    board is drawn nearer the port than it was cut.

    Every section of :func:`stepped` carries the same declared :data:`V`, so the
    answer is exact and known however hard the sections reflect off each other -
    which makes this the gate on the one decision :func:`~Microwave.Results.tdr.velocity`
    makes. An average of the *local* group delays lands wherever the chain
    happens to be ringing and is out by tens of percent on every row here.
    """

    #: Alternating high and low sections, which is a stepped-impedance low-pass
    #: and the strongest store this fixture can build - the reflection at each
    #: interface is what fills it.
    FILTER = (50.0, 20.0, 110.0, 20.0, 110.0, 20.0, 50.0)

    def separation(self):
        return len(self.FILTER) * SECTION

    @pytest.mark.parametrize("fmax", (10e9, 15e9, 20e9))
    def test_a_chain_of_mismatched_sections_reads_the_velocity_it_carries(self, fmax):
        """At three tops of the band, because a user picks that for the
        S-parameters and the board does not move when they do.

        All three are wide enough that the storage has averaged out. Taken down
        towards the chain's own ripple this fails, by design and not by accident
        - which is what a finite band costs rather than a fault, and the
        tolerance here is what the band being finite costs rather than anything
        about a solver.
        """
        result = stepped(self.FILTER, fmax=fmax)
        assert tdr.velocity(result, 2, 1, separation=self.separation()) == pytest.approx(
            V, rel=5e-2
        )

    #: Three open stubs of different lengths on one line, so the band holds
    #: three transmission zeros and the phase runs backwards at each. Written
    #: down rather than swept because it is a regression fixture: it is the
    #: shape that separates unwrapping the phase from re-centring it.
    STUBS = (0.017, 0.009, 0.023)

    def test_the_phase_is_unwrapped_and_not_re_centred(self):
        """Centring each step on the band's own median advance makes a wild
        point cancel exactly, and is the plausible-looking change that must not
        come back - see :func:`~Microwave.Results.tdr.velocity`. Where a
        structure has more than one transmission zero, the honest steps span
        more than a whole turn between them, so a centre taken from the middle
        of that range relocates the outliers instead of the outlier.
        """
        result = stubbed(self.STUBS)
        separation = (1 + len(self.STUBS)) * SECTION
        assert tdr.velocity(result, 2, 1, separation=separation) == pytest.approx(V, rel=5e-2)

    @pytest.mark.parametrize("nepers_per_m", (0.0, 1.0, 5.0))
    def test_the_answer_does_not_ride_on_the_loss(self, nepers_per_m):
        """The same chain with more and less attenuation on it. How lossy the
        copper is decides how deep the stopband goes and how long the ringing
        lasts, and it is not a fact about how far apart the ports are - so an
        estimator that moves with it is reading the ringing.
        """
        result = stepped(self.FILTER, nepers_per_m=nepers_per_m)
        assert tdr.velocity(result, 2, 1, separation=self.separation()) == pytest.approx(
            V, rel=2e-2
        )

    def test_the_sections_really_do_get_in_the_way(self):
        """The fixture has to store something for any of the above to test
        anything, and what it stores shows as a reflection."""
        assert np.abs(np.asarray(stepped(self.FILTER).parameter(1, 1), complex)).max() > 0.5


class TestDistance:
    def test_it_halves_the_round_trip(self):
        trace = tdr.step_response(reflected(line([50.0])), 1)
        assert tdr.distance(trace, V) == pytest.approx(0.5 * V * trace.time, rel=0.0, abs=0.0)

    def test_a_reflection_lands_at_the_discontinuity_that_caused_it(self):
        """Where the 75 ohm section starts, in millimetres, is what the axis says."""
        trace = tdr.step_response(reflected(line([50.0, 75.0, 50.0])), 1)
        along = 1e3 * tdr.distance(trace, V)
        rising = along[np.nanargmax(np.gradient(trace.impedance, along))]
        assert rising == pytest.approx(1e3 * SECTION, abs=1e3 * V / (2 * FMAX))
