# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

from Microwave.Objects.ports import (
    FIXED_IMPEDANCE,
    PORT_IMPEDANCE,
    EMPortLumped,
    EMPortMicrostrip,
    EMPortRectWaveguide,
    createEMPortLumped,
    createEMPortMicrostrip,
    createEMPortRectWaveguide,
)


class TestTheReferenceImpedanceIsShownOnlyWhenItIsRead:
    """A property that does nothing must not be offered.

    A port referenced to its own impedance never reads the number beside it, so
    the number goes away rather than sitting there greyed out: a stale ``50.00``
    answers the question the user is asking, and answers it wrongly.

    Hidden and not read-only for that reason, and the flag travels in the file
    - measured on FreeCAD 1.1.1, where a document saved with the port switched
    reopens with the property still hidden, so nothing has to restore it.
    """

    factories = pytest.mark.parametrize(
        "factory",
        [createEMPortLumped, createEMPortMicrostrip, createEMPortRectWaveguide],
        ids=lambda f: f.__name__,
    )

    @factories
    def test_it_is_visible_on_a_new_port(self, factory, doc):
        assert factory(doc=doc).getEditorMode("ReferenceImpedance") == []

    @factories
    def test_pointing_the_port_at_itself_hides_it(self, factory, doc):
        port = factory(doc=doc)
        port.ReferencedTo = PORT_IMPEDANCE
        port.Proxy.onChanged(port, "ReferencedTo")

        assert port.getEditorMode("ReferenceImpedance") == ["Hidden"]

    def test_pointing_it_back_at_a_number_shows_it_again(self, doc):
        port = createEMPortLumped(doc=doc)
        for value in (PORT_IMPEDANCE, FIXED_IMPEDANCE):
            port.ReferencedTo = value
            port.Proxy.onChanged(port, "ReferencedTo")

        assert port.getEditorMode("ReferenceImpedance") == []

    def test_no_other_property_moves_it(self, doc):
        """``onChanged`` fires for every property, including as ``__init__``
        creates them, so it has to be filtered on the name."""
        port = createEMPortLumped(doc=doc)
        port.ReferencedTo = PORT_IMPEDANCE
        port.Proxy.onChanged(port, "Number")

        assert port.getEditorMode("ReferenceImpedance") == []


class TestEveryPortArrivesNumbered:
    """A port with no number is refused by the translation.

    An ``assign_port_numbers`` that is exported and tested but called from
    nowhere leaves every port made from the toolbar unusable, with only a
    hand-typed ``Number`` making any gate pass - and its tests still green.
    """

    factories = pytest.mark.parametrize(
        "factory",
        [
            createEMPortLumped,
            createEMPortMicrostrip,
            createEMPortRectWaveguide,
        ],
        ids=lambda f: f.__name__,
    )

    @factories
    def test_a_port_of_any_kind_is_numbered_when_it_is_made(self, factory, doc):
        assert factory(doc=doc).Number >= 1

    def test_ports_are_numbered_in_the_order_they_are_made(self, doc):
        made = [createEMPortLumped(f"P{n}", doc=doc) for n in range(3)]
        assert [port.Number for port in made] == [1, 2, 3]

    def test_a_number_freed_by_a_deletion_is_reused(self, doc):
        first = createEMPortLumped("P1", doc=doc)
        createEMPortLumped("P2", doc=doc)
        doc.removeObject(first.Name)
        assert createEMPortLumped("P3", doc=doc).Number == 1

    def test_a_hand_edited_number_is_never_taken_twice(self, doc):
        createEMPortLumped("P1", doc=doc).Number = 7
        assert createEMPortLumped("P2", doc=doc).Number != 7

    def test_the_translation_accepts_what_the_factories_produce(self, doc):
        """The whole point: these numbers have to satisfy the layer that reads them."""
        from Microwave.Solvers.openems.document import _port_numbers

        made = [createEMPortLumped(f"P{n}", doc=doc) for n in range(2)]
        assert _port_numbers(made) == [1, 2]


class TestOneNameForOneThing:
    """Facts stored twice, which then diverged.

    A duplicated *fact* is worse than duplicated code: it drifts in silence.
    Both of these did, and neither had a test, because in both cases the two
    copies agreed about every model anyone had actually built.
    """

    def test_the_translation_spells_an_axis_the_way_the_document_does(self, doc):
        """``document`` imported the *envelope's* lowercase axis names and used
        them in nine messages about uppercase document properties. Eight said
        "x" for a property whose value the user sees as "X"; three other sites
        had grown a local ``.upper()`` patching the symptom. One assertion here
        covers every message, because they all read the same tuple."""
        from Microwave.Solvers.openems import document

        port = createEMPortMicrostrip("P", doc=doc)
        offered = set(port.getEnumerationsOfProperty("PropagationAxis"))

        assert offered, "the enumeration is empty; this test would prove nothing"
        assert set(document.AXIS_NAMES) <= offered

    def test_the_air_padding_default_is_one_number(self, doc):
        """``write.DEFAULT_PADDING`` and ``EMMeshPolicy.AirCells*`` are the same
        fact in two layers that may not import each other. They agree today; the
        point is that nothing made them, which is how FLATNESS diverged."""
        from Microwave.Objects.mesh import createEMMeshPolicy
        from Microwave.Solvers.openems.write import DEFAULT_PADDING

        settings = createEMMeshPolicy(doc=doc)
        declared = {
            int(getattr(settings, f"AirCells{axis}{side}"))
            for axis in ("X", "Y", "Z")
            for side in ("Min", "Max")
        }
        adapter = {int(n) for pair in DEFAULT_PADDING for n in pair}

        assert declared == adapter

    def test_the_run_length_default_is_one_number(self, doc):
        """``model.DEFAULT_TIMESTEPS`` against ``EMSolverOpenEMS.MaxTimesteps``.

        Same shape as the padding pair above. This one drifts cheaply - too
        high only costs runtime - which is exactly why nobody would notice.
        """
        from Microwave.Objects.solver import createEMSolverOpenEMS
        from Microwave.Solvers.openems.model import DEFAULT_TIMESTEPS

        solver = createEMSolverOpenEMS(doc=doc)
        assert int(solver.MaxTimesteps) == DEFAULT_TIMESTEPS

    def test_flatness_is_one_number(self):
        """Three definitions at two values (1e-6, 1e-6, 1e-7), each with its own
        justifying comment, in modules that already imported each other. No real
        geometry occupies the band between them, which is why nothing caught it
        and why the value is not what matters here - the agreement is.

        Both other copies are now assignments from ``portbox``, so what this pins
        is that they stay derived: write a literal back into either and it fails.
        """
        from Microwave import portbox
        from Microwave.Objects import port_setup
        from Microwave.Solvers.openems import document

        assert document.FLATNESS == portbox.FLATNESS == port_setup.FLATNESS


def test_every_waveguide_mode_the_gui_offers_can_actually_be_built(doc):
    """The dropdown and the adapter must agree about what a user can make.

    They did not: the enumeration offered TM01 and TM11, which upstream
    ``RectWGPort`` refuses outright (ports.py:434), so a port built from the
    toolbar was one no solver could run. The document layer may not import an
    adapter, so the two lists cannot be one constant - but nothing stops a
    test from reading the enumeration itself and putting every value through
    the envelope. Add a mode to the dropdown that openEMS cannot run and this
    fails, without anyone having to remember to update a copy.
    """
    from Microwave.Solvers.openems.model import Port

    obj = createEMPortRectWaveguide("WG", doc=doc)
    offered = obj.getEnumerationsOfProperty("Mode")
    assert offered, "the enumeration is empty; this test would prove nothing"

    for mode in offered:
        Port(
            number=1,
            kind="rect_waveguide",
            mode=mode,
            start=(0.0, 0.0, 0.0),
            stop=(10.7, 4.3, 2.0),
            propagation_axis=2,
        )


class TestEachKindDeclaresItsOwnSurface:
    """One table: what every port has, what each kind adds, and what it must not.

    One table, not a function per kind: that shape makes a new port kind six
    edits, and leaves the *isolation* claim to a hand-written test naming two
    kinds out of five. Here a kind that leaks a property into another is caught
    by that other kind's own row, and adding a kind is adding a row.

    ``must_not_have`` is not decoration. ``Length`` on the base class was a
    silent no-op of exactly that kind - a lumped port's box is the
    overlap of its two entities, entirely determined by geometry, so
    ``Length = 999`` moved nothing.
    """

    #: Common to every kind, from ``EMPortBase``. ``Number`` is 1 and not 0: a
    #: port that arrives unnumbered is refused by the translation, and this
    #: asserting 0 here would pin that defect as the contract.
    BASE_DEFAULTS = {
        "Number": 1,
        "Excitation": True,
        "ReferenceImpedance": 50.0,
        "ReferencedTo": FIXED_IMPEDANCE,
    }

    KINDS = [
        (
            createEMPortLumped,
            EMPortLumped,
            {"SourceEntity", "ReferenceEntity"},
            # A lumped port *is* a resistor across the gap, so its resistance is
            # part of the structure - a different quantity from the
            # microstrip's damping resistor, under a different name, and with a
            # different meaning for zero.
            {"Resistance": 50.0},
            {"ExcitationAxis": "X"},
            {"FeedResistance", "FeedOffset", "Length"},
        ),
        (
            createEMPortMicrostrip,
            EMPortMicrostrip,
            {"TraceEnd", "Length"},
            # Millimetres from the picked face, not fractions of the box: the
            # command fills MeasurementDistance in from the study's band, and
            # constructed bare it stays 0, which translation refuses by name.
            #
            # FeedResistance 0 is a bare voltage source, not a 50 ohm one. It
            # was 50, under the name ``Impedance`` and the description
            # "Reference impedance", so every port drawn in the GUI carried a
            # damping resistor the one measured configuration does not have.
            #
            # -Z because the default is the common case rather than a
            # placeholder: trace to ground runs downward on every board, and
            # leaving both axes at X would make every new port refuse itself.
            {
                "FeedOffset": 0.0,
                "MeasurementDistance": 0.0,
                "FeedResistance": 0.0,
                "ExcitationAxis": "-Z",
            },
            {"PropagationAxis": "X"},
            {"SourceEntity", "Resistance"},
        ),
        (
            createEMPortRectWaveguide,
            EMPortRectWaveguide,
            {"CrossSection"},
            {},
            {"Mode": "TE10"},
            {"SourceEntity", "TraceEnd"},
        ),
    ]

    kinds = pytest.mark.parametrize(
        "factory, proxy, must_have, defaults, choices, must_not_have",
        KINDS,
        ids=[factory.__name__ for factory, *_ in KINDS],
    )

    @kinds
    def test_the_factory_makes_the_kind_it_is_named_for(
        self, doc, factory, proxy, must_have, defaults, choices, must_not_have
    ):
        port = factory("P1", doc=doc)
        assert port.Label == "P1"
        assert isinstance(port.Proxy, proxy)

    @kinds
    def test_every_kind_carries_the_base_defaults(
        self, doc, factory, proxy, must_have, defaults, choices, must_not_have
    ):
        port = factory(doc=doc)
        for name, value in self.BASE_DEFAULTS.items():
            assert getattr(port, name) == value, name

    @kinds
    def test_a_kind_has_its_own_properties_and_only_its_own(
        self, doc, factory, proxy, must_have, defaults, choices, must_not_have
    ):
        port = factory(doc=doc)
        for name in must_have:
            assert hasattr(port, name), f"{proxy.__name__} is missing {name}"
        for name in must_not_have:
            assert not hasattr(port, name), (
                f"{proxy.__name__} carries {name}, which belongs to another kind"
            )

    @kinds
    def test_a_kind_arrives_at_the_values_it_was_argued_into(
        self, doc, factory, proxy, must_have, defaults, choices, must_not_have
    ):
        port = factory(doc=doc)
        for name, value in defaults.items():
            assert getattr(port, name) == value, name
        for name, expected in choices.items():
            assert expected in port.getEnumerationsOfProperty(name), name


class TestNoPortPropertyIsANoOp:
    """Every property a port offers must be one the adapter reads.

    An unsupported thing is a refusal that names it, never a silent no-op. A
    property in the editor that reaches no solver is one of those, and it is
    the failure mode this class exists to catch - it found them. A microstrip's
    two feed offsets on the base class give a waveguide port both, with
    ``_rect_waveguide`` dropping them on the floor; the envelope
    refuses a nonzero shift on a waveguide port, but the document layer never
    passed one on, so the refusal could not fire.
    ``Thickness`` and ``Conductivity`` sat on the microstrip port and were only
    ever read off the *material*.

    Every kind the document defines is swept, because no kind exists only to be
    refused.
    """

    @staticmethod
    def read_by_the_adapter():
        """Every ``something.Name`` the adapter reads, as a set of names.

        Parsed, not grepped. Asking whether the name appears
        anywhere in ``document.py`` as a *substring*, which passes on prose and
        on any identifier that merely contains it.

        **What this still cannot see** is *which kind* reaches a read. ``Length``
        is read as ``obj.Length`` - but only the microstrip and waveguide
        builders read it, so on the base class it was a no-op for lumped ports
        and this check would pass it either way. What found
        it by setting it to 999 and watching the box not move. The guard for that
        is structural, in ``test_port_subclass_property_isolation``: the property
        is not on the kind that ignores it. This one is the coarser net that
        catches a property no kind reads at all.
        """
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "Microwave" / "Solvers" / "openems" / "document.py").read_text())
        return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            # Exact string literals too, because the adapter reads the two
            # resistances through a helper - ``_resistance(obj, "Resistance")``
            # - so the name is data rather than an attribute. A whole literal,
            # not a substring: that distinction is the entire difference between
            # this and the version that passed a live no-op.
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

    @pytest.mark.parametrize("make", [EMPortMicrostrip, EMPortLumped, EMPortRectWaveguide])
    def test_the_adapter_reads_all_of_them(self, doc, make):
        obj = doc.addObject("App::FeaturePython", "Port")
        make(obj)
        read = self.read_by_the_adapter()
        # PropertiesList on the stub holds exactly what addProperty was given,
        # so there are no FreeCAD built-ins to filter out.
        unread = [name for name in obj.PropertiesList if name not in read]
        assert not unread, f"{make.__name__} offers, and openEMS never reads: {unread}"

    #: Zero ends the box at the measurement plane: the fixture feeds 20 mm in
    #: and measures 30 mm downstream of that, so 50. A stated length is taken
    #: as given, provided it still contains the measurement plane.
    @pytest.mark.parametrize("length,expected", [(0.0, 50.0), (60.0, 60.0)])
    def test_length_moves_the_microstrip_box_it_is_offered_on(self, length, expected):
        """The behavioural half, and the one that would have caught the no-op.

        Setting a property and watching the translated port is the only check
        that distinguishes "read somewhere" from "read for this kind". Length
        999 on a lumped port moved nothing, for a week, under a green suite.
        """
        from Microwave.Solvers.openems import document

        from .test_document_translation import ground, microstrip_port, model, trace

        port = microstrip_port(1, trace(), ground(), Length=length)
        built = document.problem(model(port=port).Objects[0]).ports[0]
        assert built.length == pytest.approx(expected, rel=1e-6, abs=0.0)

    def test_the_check_itself_notices_a_property_nobody_reads(self, doc):
        """The guard on the guard.

        A no-op test that has stopped detecting no-ops is invisible, and this
        one was: it passed a live fault for a week. So it is asserted to fail
        on a property invented for the purpose.
        """
        obj = doc.addObject("App::FeaturePython", "Port")
        EMPortLumped(obj)
        obj.addProperty("App::PropertyFloat", "ThoroughlyImaginary", "Lumped", "")
        unread = [name for name in obj.PropertiesList if name not in self.read_by_the_adapter()]
        assert unread == ["ThoroughlyImaginary"]


class TestTheClearanceIsWrittenIntoThePort:
    """The band decides how far the probes sit from the source. A property
    holds the answer; nothing computes it again later.

    That is the design, not an optimisation. The number depends on the analysis
    and on the materials, and FreeCAD has no dependency edge from either to a
    port - so computing it whenever the box is drawn would leave the picture
    stale the moment the band moved, and the picture is supposed to *be* the
    solve. Written down once, it is a number the engineer can see, change, and
    be refused on.
    """

    def study(self, doc, start="1.0 GHz", epsilon=4.4):
        from Microwave.Objects.analysis import EMAnalysis
        from Microwave.Objects.materials import EMMaterial

        analysis = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        EMAnalysis(analysis)
        analysis.FrequencyStart = start
        material = doc.addObject("App::FeaturePython", "FR4")
        EMMaterial(material)
        material.Permittivity = epsilon
        return analysis, material

    def test_a_new_port_arrives_with_it_filled_in(self, doc):
        from Microwave.Objects.ports import createEMPortMicrostrip

        self.study(doc)
        port = createEMPortMicrostrip("P", doc=doc)
        # 0.1 free-space wavelengths at 1 GHz is 29.98 mm, up to the next one.
        assert float(port.MeasurementDistance) == 30.0

    def test_it_comes_from_the_bottom_of_the_band(self, doc):
        """Where the wavelength is longest. Taking the top would ask for a
        tenth of the room across this band and read the impedance high."""
        from Microwave.Objects.ports import required_clearance

        self.study(doc, start="1.0 GHz")
        assert required_clearance(doc) == 30.0
        doc.Objects[0].FrequencyStart = "10.0 GHz"
        assert required_clearance(doc) == 3.0

    def test_no_material_in_the_document_can_change_it(self, doc):
        """The one that governs is the line's own eps_eff, and at this moment
        there is no line. A permittivity picked off whatever else is in the
        document is not that quantity, and it moved the answer by sqrt of an
        unrelated part - always shorter, which is the direction that returns
        a plausible impedance and no complaint."""
        from Microwave.Objects.ports import required_clearance

        _, material = self.study(doc, epsilon=1.0)
        assert required_clearance(doc) == 30.0
        material.Permittivity = 10.2
        assert required_clearance(doc) == 30.0

    def test_a_document_with_no_study_leaves_it_at_zero(self, doc):
        """Which translation then refuses by name - a port made before the
        analysis is an ordinary state, and a wrong number would be worse than
        none."""
        from Microwave.Objects.ports import createEMPortMicrostrip

        assert float(createEMPortMicrostrip("P", doc=doc).MeasurementDistance) == 0.0

    def test_two_studies_leave_it_at_zero_rather_than_guessing(self, doc):
        from Microwave.Objects.analysis import EMAnalysis
        from Microwave.Objects.ports import required_clearance

        self.study(doc)
        second = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        EMAnalysis(second)
        second.FrequencyStart = "5.0 GHz"
        assert required_clearance(doc) == 0.0
