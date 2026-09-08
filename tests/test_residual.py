# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the end of a port's record is worth, and whether the bar for it is real.

The first class is the one that matters. It builds a ringing mode that really
is cut off, transforms it the way openEMS does, and checks both halves of
:data:`WANTED`'s argument: that what the measurement reports is what a tenth
less record would have answered, and that what was never recorded is the factor
above it the constant claims. Without those two the bar is a number somebody
liked.

Every record here is a drive going in and a mode coming back out, because the
measurement reads the two apart. A component that only passes through a port is
charged by what that port reflects rather than in full, and a record holding
one total voltage cannot say which it had.
"""

from __future__ import annotations

import re
import weakref

import numpy as np
import pytest

from Microwave.Solvers.openems import residual

#: A mode to ring, and a band around it. Several cycles fit in a short record,
#: so a truncation can be placed anywhere along the decay.
RESONANCE = 5e9
BAND = np.linspace(4e9, 6e9, 21)

#: Samples per cycle of the resonance. Well past Nyquist, so the discrete sum
#: below is the integral it stands for rather than an artefact of the rate.
PER_CYCLE = 200


#: What the records below are referenced to, in ohms. A lumped port declares a
#: real one; a microstrip port measures a complex one off its own record, and the
#: second value here is that case - two ports of one run need not share either
#: the value or the kind.
REFERENCE = 50.0
MEASURED_REFERENCE = 63.0 - 4.0j

#: How far the module's answer may stand from the same measure written out by
#: hand below, as an absolute error in S. The two are no longer one expression
#: written twice: the module sums a prefix as the whole record less the window
#: dropped, where the lines below sum the prefix from its own start. What
#: separates them is the rounding of a transform, so it follows the size of the
#: parameters being differenced rather than the size of the difference - under
#: a hundred units in the last place of a parameter of one, and every record
#: here reads below one. A band reaching below where the drive carries anything
#: divides by a vanishing number and answers far above one, and this figure
#: would not cover that.
SETTLED = 1e-14

#: How far the current probe sits behind the voltage probe, as a share of the
#: step between samples. A Yee scheme staggers the two by half a step in time,
#: and each probe file carries the instants it was written at.
STAGGER = 0.5


def drive_of(times, amplitude: float = 1.0, echo: float = 0.0):
    """An impulse whose transform is ``amplitude``, so a share reads as a share.

    One sample wide, which makes it flat across the band and leaves it alone
    when the record is truncated: what a shorter record moves is the wave that
    came back. ``echo`` repeats it half a period of the band's middle frequency
    later, which puts a notch there and nowhere else.
    """
    step = times[1] - times[0]
    drive = np.zeros_like(times)
    drive[0] = amplitude / (2 * step)
    if echo:
        drive[int(round(0.5 / BAND[BAND.size // 2] / step))] = echo * amplitude / (2 * step)
    return drive


def ringing(
    cycles: float,
    decay: float,
    drive: float = 1.0,
    echo: float = 0.0,
    reference=REFERENCE,
    stagger: float = STAGGER,
):
    """A drive, and a mode ringing after it that is cut off at ``decay``.

    The mode leaves the port and the drive enters it, so the current carries the
    two with opposite signs - which is what gives the record an incident and a
    reflected wave rather than one total.
    """
    duration = cycles / RESONANCE
    times = np.linspace(0.0, duration, int(cycles * PER_CYCLE) + 1)
    tau = -duration / np.log(decay)
    outgoing = np.exp(-times / tau) * np.cos(2 * np.pi * RESONANCE * times)
    incoming = drive_of(times, drive, echo)
    return residual.Record(
        voltage=residual.Probe(times, incoming + outgoing),
        current=residual.Probe(
            times + stagger * (times[1] - times[0]), (incoming - outgoing) / REFERENCE
        ),
        reference=reference,
    )


def transform(probe, samples: int):
    """openEMS' own: ``2 * dt * sum(u exp(-i 2 pi f t))``, on the probe's own axis."""
    times, values = probe.times[:samples], probe.values[:samples]
    kernel = np.exp(-2j * np.pi * np.outer(BAND, times))
    return 2 * (times[1] - times[0]) * (kernel * values).sum(axis=1)


def _with(record, direction: float):
    """``record`` plus a component that never decays, travelling ``direction``.

    ``+1`` makes it incident and ``-1`` outgoing; it is the same size in the
    voltage either way, which is what a measurement reading the voltage alone
    cannot distinguish.

    As large as the mode was at its loudest, and no larger: the drive is an
    impulse standing orders above everything else in the record, and a component
    that size would leave nothing but rounding for the split to cancel.
    """
    steady = np.full_like(record.voltage.times, np.max(np.abs(record.voltage.values[1:])))
    return residual.Record(
        voltage=residual.Probe(record.voltage.times, record.voltage.values + steady),
        current=residual.Probe(
            record.current.times, record.current.values + direction * steady / REFERENCE
        ),
        reference=record.reference,
    )


def _flat(samples: int):
    """A record of ``samples`` samples, both probes on one axis and nothing in it."""
    times = np.linspace(0.0, 1e-9, samples)
    probe = residual.Probe(times, times)
    return residual.Record(voltage=probe, current=probe, reference=REFERENCE)


def parameter(record, driving, samples: int) -> np.ndarray:
    """The S-parameter these two records give over their first ``samples``, by hand.

    Each port's own reference: the wave coming back is split against the port it
    came back out of, and the wave that drove the run against the port that drove.
    """
    reflected = (
        transform(record.voltage, samples) - transform(record.current, samples) * record.reference
    )
    incident = (
        transform(driving.voltage, samples)
        + transform(driving.current, samples) * driving.reference
    )
    return reflected / incident


def share(record, driving) -> float:
    """What the run's measure reports for ``record``, driven by ``driving``.

    :func:`residual.tail_shares` weighs a whole device at once, and every test
    here is about one port of one. The driving port is given a number of its
    own so that the two records stay distinguishable even where they are the
    same record.
    """
    return residual.tail_shares({0: driving, 1: record}, 0, BAND)[1]


class TestWhatTheBarBuys:
    @pytest.mark.parametrize("decay", [1e-1, 1e-2, 1e-3])
    def test_it_reports_what_the_tail_contributes(self, decay):
        """The measurement against the thing it stands for: what this run
        answered, and what it would have answered a tenth of a record early."""
        record = ringing(cycles=40, decay=decay, reference=MEASURED_REFERENCE)
        driving = ringing(cycles=40, decay=decay)
        samples = record.voltage.times.size
        early = samples - int(samples * residual.TAIL)

        reported = share(record, driving)
        want = np.max(
            np.abs(parameter(record, driving, samples) - parameter(record, driving, early))
        )
        assert reported == pytest.approx(want, rel=0.0, abs=SETTLED), (
            "the two are one measure reached two ways, so they agree to the "
            "rounding of a transform or one of them is not the expression it "
            "says it is"
        )

    @pytest.mark.parametrize("decay", [1e-1, 1e-2, 1e-3])
    def test_what_was_never_recorded_is_the_factor_the_constant_claims(self, decay):
        """``1/(exp(T/(10 tau)) - 1)``, which is about two at the bar. The whole
        reason the bar is not simply the error: the run is judged on the last
        thing it kept, and what it dropped is larger than that."""
        record = ringing(cycles=40, decay=decay)
        reported = share(record, record)
        # The same mode allowed to run on, so what it adds is exactly what the
        # short record never saw.
        longer = ringing(cycles=400, decay=decay**10)
        missed = np.max(
            np.abs(
                parameter(longer, longer, longer.voltage.times.size)
                - parameter(record, record, record.voltage.times.size)
            )
        )

        claimed = 1.0 / (np.exp(-np.log(decay) * residual.TAIL) - 1.0)
        assert missed / reported == pytest.approx(claimed, rel=0.05)

    def test_the_crossover_the_constant_names(self):
        """Seven time constants, where the part never recorded and the part
        measured are the same size. Moving TAIL without moving the sentence
        fails this."""
        assert 1.0 / (np.exp(7.0 * residual.TAIL) - 1.0) == pytest.approx(1.0, rel=0.05)


class TestReadingOneRecord:
    def test_a_record_cut_early_is_worth_far_more_than_one_that_ran_on(self):
        cut_early = ringing(cycles=40, decay=0.5)
        ran_on = ringing(cycles=40, decay=1e-4)
        assert share(cut_early, cut_early) > 100 * share(ran_on, ran_on)

    def test_a_tail_leaving_the_port_is_charged_in_full(self):
        """A component that never dies, added to the record as an outgoing wave.
        It is in the reflected term, so what a shorter record drops of it is an
        error in S and this says so."""
        record = ringing(cycles=40, decay=1e-4, stagger=0.0)
        outward = _with(record, -1.0)
        assert share(outward, record) > 100 * share(record, record)

    def test_and_the_same_tail_only_passing_through_is_not(self):
        """The same component satisfying ``u = i R`` instead, which makes it
        incident and nothing else. It cancels out of the reflected term, so with
        the wave that drove the run held fixed it leaves nothing behind but the
        rounding of the cancellation - where added the other way round it is the
        whole answer. The two records have identical voltages, so a measurement
        reading the total voltage cannot tell them apart at all.

        Both probes on one axis here. What separates the two waves is the split,
        and it separates them exactly only where they were sampled at one
        instant: staggered, a component constant in time still leaves the phase
        between the two axes behind, which is a fact about the scheme rather
        than about this measurement.
        """
        record = ringing(cycles=40, decay=1e-4, stagger=0.0)
        through, outward = _with(record, 1.0), _with(record, -1.0)
        assert np.array_equal(through.voltage.values, outward.voltage.values), (
            "the two records have to differ in the current alone, or this says "
            "nothing about which wave is being read"
        )
        alone = share(record, record)
        assert share(outward, record) > 100.0 * alone
        assert share(through, record) == pytest.approx(alone, rel=0.01, abs=0.0), (
            "a component that only passes through has to leave the answer where "
            "it found it, to within what the cancellation rounds off"
        )

    def test_a_weaker_drive_makes_the_same_tail_worth_more(self):
        """S is a ratio to what drove the run, so halving the drive doubles
        what the same leakage costs. The port being read is the same one both
        times: what moves is the run's excitation."""
        record = ringing(cycles=40, decay=0.1, stagger=0.0)
        quiet = ringing(cycles=40, decay=0.1, drive=0.5, stagger=0.0)
        assert share(record, quiet) == pytest.approx(2.0 * share(record, record), rel=1e-9, abs=0.0)

    def test_the_worst_frequency_decides(self):
        """A mean over the band would let a bad notch average away against the
        passband either side of it - so the reported figure has to be the
        largest of the band's own, and the band has to be uneven enough here for
        the two to be different answers."""
        notched = ringing(cycles=40, decay=0.5, echo=0.9)
        samples = notched.voltage.times.size
        early = samples - int(samples * residual.TAIL)
        moved = np.abs(parameter(notched, notched, samples) - parameter(notched, notched, early))
        assert share(notched, notched) == pytest.approx(float(np.max(moved)), rel=0.0, abs=SETTLED)
        assert np.max(moved) > 3.0 * np.mean(moved), (
            "the band moved evenly enough here that its mean would have said the "
            "same, so this says nothing about which of the two is reported"
        )

    def test_the_transform_is_the_one_openems_takes(self):
        """An impulse of one at the origin, whose transform is ``2 dt`` at every
        frequency - the scaling ``DFT_time2freq`` applies to make the spectrum
        single sided. It cancels out of every ratio read here, so nothing else
        can say whether it was applied."""
        times = np.linspace(0.0, 1e-9, 101)
        values = np.zeros_like(times)
        values[0] = 1.0
        got = residual._transform(residual.Probe(times, values), BAND, 0, times.size)
        assert got == pytest.approx(
            np.full(BAND.size, 2 * (times[1] - times[0])), rel=1e-12, abs=0.0
        )

    def test_the_transform_reads_the_window_it_is_given_and_no_more(self):
        """The measure asks for a window out of the middle of a record, and the
        sum over a record is the sum over its parts. A transform reaching past
        the end it was given would carry the tail into the prefix and report a
        record as finished that is not."""
        times = np.linspace(0.0, 1e-9, 101)
        probe = residual.Probe(times, np.arange(101, dtype=float))
        parts = residual._transform(probe, BAND, 0, 37) + residual._transform(probe, BAND, 37, 101)
        assert residual._transform(probe, BAND, 0, 101) == pytest.approx(parts, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("samples", [0, 1, 2])
    def test_a_record_too_short_to_transform_accounts_for_nothing(self, samples):
        """A tenth less of it has to be a record in its own right, and the
        fewest samples with a step between them is two. Reporting a small share
        for anything shorter would certify a broken run."""
        assert share(*(2 * [_flat(samples)])) == 1.0

    def test_and_three_samples_is_the_fewest_that_reports_anything(self):
        """The boundary itself: a tenth off three leaves two, which has a step.
        Indexing that step is what a record of two would have to do to be
        answered, so the refusal above is where the arithmetic runs out rather
        than where somebody drew a line."""
        assert share(*(2 * [_flat(3)])) < 1.0


class TestWeighingAWholeDevice:
    """Every port of a run at once, which is how the driver asks.

    An S-parameter is a port's reflection over the *driven* port's incident
    wave, so one record answers every port in the device. Reading it again for
    each of them transforms it over for every other port there is.
    """

    def readings(self, records, driving=1):
        """Which probe each transform was taken on, and over which samples.

        The patch is undone before the next call, so a test asking twice does
        not have the second reading wrap the first and report both.
        """
        taken = []

        def watch(probe, frequencies, first, last):
            taken.append((id(probe), first, last))
            return real(probe, frequencies, first, last)

        with pytest.MonkeyPatch.context() as patch:
            real = residual._transform
            patch.setattr(residual, "_transform", watch)
            return residual.tail_shares(records, driving, BAND), taken

    def test_each_record_is_read_once_over_the_whole_and_once_over_the_window(self):
        """Including the one that drove, which is also a port being weighed.

        What a transform costs grows in the record, so anything asking a second
        time is transforming a record that is already in hand.
        """
        records = {number: ringing(cycles=40, decay=0.5) for number in range(1, 5)}
        _, taken = self.readings(records)
        samples = records[1].voltage.times.size
        early = samples - int(samples * residual.TAIL)
        wanted = {
            (id(probe), first, last)
            for record in records.values()
            for probe in (record.voltage, record.current)
            for first, last in ((0, samples), (early, samples))
        }
        assert sorted(taken) == sorted(wanted)

    def test_each_port_is_answered_with_its_own_reading_and_not_a_neighbour_s(self):
        """The sharing above changes what is computed and must not change what
        is answered.

        Against the measure written out by hand for each port, rather than
        against another call of the same function: ports of one device record
        alike, so a reading handed to the wrong port would be handed to it on
        both sides of a comparison and answer the same twice.
        """
        one = ringing(cycles=40, decay=0.5)
        two = ringing(cycles=40, decay=1e-2, reference=MEASURED_REFERENCE)
        three = ringing(cycles=40, decay=1e-3, echo=0.9)
        together = residual.tail_shares({1: one, 2: two, 3: three}, 1, BAND)
        samples = one.voltage.times.size
        early = samples - int(samples * residual.TAIL)
        for number, record in ((1, one), (2, two), (3, three)):
            want = np.max(np.abs(parameter(record, one, samples) - parameter(record, one, early)))
            assert together[number] == pytest.approx(want, rel=0.0, abs=SETTLED)

    def test_the_drive_is_the_port_that_was_excited_and_not_the_first_one(self):
        """A sweep excites each port in turn, so the driven port is not always
        the lowest numbered. Taking the first record the loop comes to would
        weigh every port against a wave that never drove it, and would agree
        with the truth on every device whose ports were driven alike."""
        loud = ringing(cycles=40, decay=0.5)
        quiet = ringing(cycles=40, decay=0.5, drive=0.5)
        samples = loud.voltage.times.size
        early = samples - int(samples * residual.TAIL)
        want = np.max(np.abs(parameter(loud, quiet, samples) - parameter(loud, quiet, early)))
        assert residual.tail_shares({1: loud, 2: quiet}, 2, BAND)[1] == pytest.approx(
            want, rel=0.0, abs=SETTLED
        )

    def test_and_the_drive_is_read_over_the_lengths_the_port_asks_for(self):
        """Each port is weighed over its own record's two lengths, and the wave
        that drove it has to be read over those same two - a truncation moves
        the drive as well as what came back. So a device whose ports stopped at
        different steps asks the drive for more than one reading, and the
        answer is checked against the measure written out by hand at the
        shorter port's own lengths."""
        driving = ringing(cycles=80, decay=0.5)
        shorter = ringing(cycles=40, decay=0.5)
        samples = shorter.voltage.times.size
        early = samples - int(samples * residual.TAIL)
        want = np.max(
            np.abs(parameter(shorter, driving, samples) - parameter(shorter, driving, early))
        )
        together = residual.tail_shares({1: driving, 2: shorter}, 1, BAND)
        assert together[2] == pytest.approx(want, rel=0.0, abs=SETTLED)
        assert together[2] != together[1], (
            "the two records are one mode over different lengths, so a measure "
            "reading one length for both would answer twice alike"
        )

    def test_what_it_keeps_does_not_grow_with_the_number_of_ports(self):
        """Only the record that drove the run is wanted again, so a port's own
        reading is let go where it was taken.

        Keeping every port's would hold four spectra a port for the length of
        the call, each as long as the answer is - which is the same shape as
        the array this whole measure was rewritten to stop building. What
        stands instead is the drive's reading, the reading of the port being
        weighed, and the one before it that has not been let go yet, and a
        device of four ports is already at that number.
        """
        taken = []
        standing = []
        real = residual._spectra

        def watch(probe, frequencies, samples, early):
            pair = real(probe, frequencies, samples, early)
            taken.extend(weakref.ref(spectrum) for spectrum in pair)
            standing.append(sum(1 for held in taken if held() is not None))
            return pair

        def most(ports, spread=0):
            standing.clear()
            taken.clear()
            records = {
                number: ringing(cycles=40 + spread * number, decay=0.5)
                for number in range(1, ports + 1)
            }
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(residual, "_spectra", watch)
                residual.tail_shares(records, 1, BAND)
            return max(standing)

        assert most(16) == most(4)
        # And where every port asks for a length the one before it did not, so
        # that the drive is read again for each of them. Holding on to the
        # reading each new length replaced is the same fault one step along.
        assert most(16, spread=1) == most(4, spread=1)


class TestWhatTheMeasurementHolds:
    """``driver._extract`` takes this measurement in the process that ran the
    engine, after the solve and before the results are written - so whatever it
    allocates is added to what that process is already holding, and running out
    there loses a finished run rather than costing a wait."""

    def kernels(self, records, frequencies):
        """The size of every kernel the transform builds, in elements.

        The patch is undone before the next call, so a test asking twice does
        not have the second reading wrap the first and report both.
        """
        sizes = []

        def counted(first, second):
            built = real(first, second)
            sizes.append(built.size)
            return built

        with pytest.MonkeyPatch.context() as patch:
            real = np.outer
            patch.setattr(np, "outer", counted)
            residual.tail_shares(records, 1, frequencies)
        return sizes

    def test_no_kernel_is_larger_than_the_block(self):
        """Frequency points and recorded steps are both dials the user turns,
        and the band here is wide enough that the two together are past the
        bound."""
        wide = np.linspace(4e9, 6e9, 2001)
        record = ringing(cycles=40, decay=0.5)
        assert record.voltage.times.size * wide.size > residual.BLOCK, (
            "the record and the band fit in one block here, so nothing below "
            "says anything about what bounds them"
        )
        assert max(self.kernels({1: record}, wide)) <= residual.BLOCK

    def test_and_the_bound_does_not_move_with_the_record(self):
        """What is held is the block rather than the product, so twice the
        record is the same peak taken twice as often."""
        wide = np.linspace(4e9, 6e9, 2001)
        short = self.kernels({1: ringing(cycles=40, decay=0.5)}, wide)
        longer = self.kernels({1: ringing(cycles=80, decay=0.5)}, wide)
        assert max(longer) == max(short)
        assert len(longer) > len(short)


class TestWhatItSays:
    def test_a_finished_run_says_nothing(self):
        assert residual.unfinished({1: 1e-4, 2: 1e-6}) is None

    def test_the_bar_itself_counts_as_finished(self):
        """WANTED is what a finished record reaches, not the first value that
        fails."""
        assert residual.unfinished({1: residual.WANTED}) is None

    def test_it_names_every_port_that_is_still_going_and_what_it_costs(self):
        message = residual.unfinished({1: 0.125, 2: 1e-6, 3: 0.03})
        assert "12.5% at port 1" in message
        assert "3% at port 3" in message
        assert "port 2" not in message

    def test_it_says_what_to_do_about_it(self):
        """A warning nobody can act on is noise, and this one has exactly one
        remedy."""
        assert "max_timesteps" in residual.unfinished({1: 1.0})


class TestTheBarFollowsWhatIsBeingRead:
    """Leakage is only harmless beside the term it lands on.

    A stopband of -40 dB *is* |S21| = 0.01, so a bar written against a response
    of one calls a leak negligible that is the whole of the quantity the device
    was built to deliver. What the study says it reads down to is what the bar
    is a share of.
    """

    @pytest.mark.parametrize("floor", [1.0, 0.1, 0.01, 1e-3])
    def test_the_bar_is_that_share_of_the_smallest_response_read(self, floor):
        """The bar itself still counts as finished, and the next float up does
        not - at every depth, so the scaling is the bar rather than a constant
        with a floor stirred in near it."""
        bar = residual.WANTED * floor
        assert residual.unfinished({1: bar}, floor) is None
        assert residual.unfinished({1: np.nextafter(bar, 1.0)}, floor) is not None

    def test_a_study_that_declares_nothing_reads_down_to_full_scale(self):
        """Full scale is what a study declaring nothing is held to, and there
        the share is an absolute error in S - which is the bar every matched
        structure in the acceptance gates is judged by."""
        for share in (residual.WANTED, 0.5):
            assert residual.unfinished({1: share}) == residual.unfinished({1: share}, 1.0)

    def test_leakage_a_full_scale_run_carries_can_be_the_whole_of_a_stopband(self):
        """The case the share exists for, and the one an absolute bar is blind
        to: a fifth of the stopband, and silent."""
        ringing = {1: 0.002}
        assert residual.unfinished(ringing) is None
        assert "port 1" in residual.unfinished(ringing, 0.01)

    def test_it_says_how_far_down_the_study_reads(self):
        """Otherwise the bar it quotes is a number with no argument behind it."""
        assert "-40 dB" in residual.unfinished({1: 1.0}, 0.01)

    def test_and_nothing_about_a_floor_when_there_is_none(self):
        assert "dB" not in residual.unfinished({1: 1.0})

    @pytest.mark.parametrize("declared", [-0.01, -33.5, -100.0])
    def test_the_floor_it_names_is_the_one_that_was_asked_for(self, declared):
        """Rounded to whole decibels, a fraction of one reads as "-0 dB" - the
        depth of a study that declared nothing, quoted beside a bar that is not
        the one such a study gets."""
        message = residual.unfinished({1: 1.0}, 10.0 ** (declared / 20.0))
        assert f"{declared:g} dB" in message

    @pytest.mark.parametrize("floor", [1.0, 0.1, 0.01, 1e-3])
    def test_the_bar_it_names_is_the_bar_it_kept(self, floor):
        """A message quoting the constant while the check used a scaled bar
        would send somebody after the wrong run length."""
        message = residual.unfinished({1: 1.0}, floor)
        quoted = re.search(r"against the ([0-9.]+)%", message)
        assert quoted, message
        assert float(quoted.group(1)) / 100 == pytest.approx(residual.WANTED * floor, rel=1e-6)
