# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The shipped examples, checked as files rather than as models.

Mostly zipfile and XML: no solver, and no FreeCAD beyond the stub the suite
already installs. These exist because an example is a *binary artefact* in the
repository, and binary artefacts drift silently. One was committed with its
solids saved hidden - swept up by `git add -A`, and visible only to a human who
opens it - and nothing in the suite could tell.

Every example, found by looking rather than by a list here. A list would be an
inventory to keep in step, and the failure it invites is the one this file is
about: a third example added, committed broken, and covered by nothing.
"""

import importlib
import math
import pkgutil
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path

import pytest

import Microwave.Objects
from tests import published
from tests.analytic import reference as analytic

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DOCUMENTS = sorted(EXAMPLES.glob("*.FCStd"))
#: Underscore-led files are shared helpers, not examples: they build no
#: document and are not meant to be run.
GENERATORS = sorted(p for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))


def test_there_are_examples_to_check():
    """The guard on the glob above.

    Without it, a rename or a move makes every test in this file pass by
    finding nothing - which is the loudest possible way for a suite to go
    quiet, and the quietest possible way for it to look fine.
    """
    assert DOCUMENTS, f"no .FCStd in {EXAMPLES}"
    assert GENERATORS, f"no generator scripts in {EXAMPLES}"


@pytest.mark.parametrize("generator", [path.name for path in GENERATORS])
def test_every_generator_has_a_committed_document(generator):
    """The other direction from :func:`test_it_has_a_script_that_rebuilds_it`.

    A document is only produced by running FreeCAD, which the suite does not
    do, so a script added without its output leaves an example that exists in
    the repository and cannot be opened. Nothing else here would notice: every
    test below is parametrised over the documents, so the missing one is simply
    never checked.
    """
    assert (EXAMPLES / f"{Path(generator).stem}.FCStd").is_file()


@pytest.fixture(params=[path.name for path in DOCUMENTS], scope="module")
def archive(request):
    with zipfile.ZipFile(EXAMPLES / request.param) as opened:
        return {name: opened.read(name) for name in opened.namelist()}


def document_xml(archive):
    return archive["Document.xml"].decode("utf-8", "replace")


def shapes(archive):
    """Every object in the document that carries geometry, by name."""
    root = ElementTree.fromstring(archive["Document.xml"])
    return [
        obj.get("name")
        for obj in root.iter("Object")
        if (obj.get("type") or "").startswith("Part::")
    ]


def workbench_objects(archive):
    """``{object name: (proxy class, the properties it was saved with)}``.

    The document records the proxy's module and class beside the object, which
    is the same thing ``Objects.kinds.kind_of`` reads at runtime and the only
    thing that tells one ``App::FeaturePython`` from another.
    """
    root = ElementTree.fromstring(archive["Document.xml"])
    found = {}
    for obj in root.find("ObjectData"):
        proxy = obj.find('.//Property[@name="Proxy"]/Python')
        if proxy is None or not (proxy.get("module") or "").startswith("Microwave."):
            continue
        stored = {p.get("name") for p in obj.iter("Property")}
        found[obj.get("name")] = (proxy.get("class"), stored)
    return found


def proxy_class(name):
    """The document class of that name, found by looking through the package.

    By search rather than by a table here, for the reason at the top of this
    file: a table is an inventory, and the drift it invites is a class added
    and silently left out of the check.
    """
    for module in pkgutil.iter_modules(Microwave.Objects.__path__):
        found = getattr(importlib.import_module(f"Microwave.Objects.{module.name}"), name, None)
        if isinstance(found, type) and found.__name__ == name:
            return found
    raise AssertionError(f"{name} is in a shipped document and in no Objects module")


def recorded_visibility(archive):
    seen = {}
    root = ElementTree.fromstring(archive["GuiDocument.xml"])
    for provider in root.iter("ViewProvider"):
        visible = provider.find('.//Property[@name="Visibility"]/Bool')
        if visible is not None:
            seen[provider.get("name")] = visible.get("value")
    return seen


@pytest.mark.parametrize("document", [path.name for path in DOCUMENTS])
def test_it_has_a_script_that_rebuilds_it(document):
    """A document nobody can regenerate is a fossil.

    Every parameter in these files was a decision, and the script is where the
    reason for it is written down. Losing the pair leaves a binary blob that
    can be opened and not understood.
    """
    assert (EXAMPLES / f"{Path(document).stem}.py").is_file()


def test_it_carries_gui_state(archive):
    """Without this the document opens with an empty 3D view.

    Measured on FreeCAD 1.1: a document restored with no GuiDocument.xml comes
    back with every Part::Feature hidden, whatever Document.xml says about
    Visibility. freecadcmd cannot write one, so the generator stamps it.
    """
    assert "GuiDocument.xml" in archive


def test_every_shape_is_recorded_visible(archive):
    """The failure this guards: an example committed with its solids hidden."""
    seen = recorded_visibility(archive)
    for name in shapes(archive):
        assert seen.get(name) == "true", f"{name} would open hidden: {seen}"


def test_every_object_is_recorded_visible(archive):
    """Not only the solids. Anything the stamp omits restores hidden.

    On FreeCAD 1.1 an object absent from ``GuiDocument.xml`` comes back with
    ``Visibility`` false whatever ``Document.xml`` says. The tree greys it out,
    so the entire markup reads as disabled - and a port is not cosmetic about
    it: it is the one document object with a display mode, so hidden means its
    arrow never draws.
    """
    document = ElementTree.fromstring(archive["Document.xml"])
    declared = {obj.get("name") for obj in document.iter("Object") if obj.get("name")}
    seen = recorded_visibility(archive)
    missing = sorted(name for name in declared if seen.get(name) != "true")
    assert not missing, f"these would open hidden: {missing}"


def test_the_gui_state_is_well_formed_xml(archive):
    """Hand-written XML in a format FreeCAD parses. It has to actually parse."""
    root = ElementTree.fromstring(archive["GuiDocument.xml"])
    assert root.tag == "Document"
    data = root.find("ViewProviderData")
    assert data is not None
    assert int(data.get("Count")) == len(list(data.iter("ViewProvider")))


def test_the_gui_state_ends_with_a_camera(archive):
    """Required, and found by bisection.

    Without a trailing ``<Camera/>`` FreeCAD 1.1.1 prints "Reading failed from
    embedded file: GuiDocument.xml" on every open. It still applies the
    visibility it read, so the file worked and nothing here could tell - but
    the example greeted anyone who opened it with a read error. Naming every
    object instead of only the shapes does not silence it.
    """
    root = ElementTree.fromstring(archive["GuiDocument.xml"])
    assert root.find("Camera") is not None


def test_the_geometry_is_still_there(archive):
    for name in shapes(archive):
        assert f"{name}.Shape.brp" in archive


def test_every_object_carries_what_its_class_declares(archive, doc):
    """A committed document is a snapshot of the classes as they stood.

    FreeCAD stores the properties an object *had* and reconciles nothing against
    the class on restore, so a property added to a document object leaves every
    committed example without it - and the translation, which reads properties
    directly, raises from inside the adapter rather than saying anything. The
    script beside each document is what rebuilds it; this is what says one has
    not been.

    Against the classes rather than a list of names, so a property added
    tomorrow is covered without anybody remembering to add it here. A subset
    check, because a restored object also carries what FreeCAD gives every
    object and a freshly built one does not.
    """
    for name, (class_name, stored) in workbench_objects(archive).items():
        probe = doc.addObject("App::FeaturePython", f"probe{class_name}")
        proxy_class(class_name)(probe)
        missing = sorted(set(probe.PropertiesList) - stored)
        assert not missing, f"{name} was saved without {missing}; rebuild it with its script"


def test_the_mesh_policy_is_the_current_one(archive):
    """A stale example is a document the adapter refuses, shipped as the thing
    to open first: it would have no ElementsPerWavelength at all."""
    assert "ElementsPerWavelength" in document_xml(archive)


#: What every port carries, whatever kind it is.
SHARED_PORT_PROPERTIES = ("Number", "Excitation", "ReferenceImpedance", "ReferencedTo")

#: And what each kind carries on top, keyed by the class the file names. A
#: document is checked against the kind it actually holds: the properties that
#: make a microstrip port one are absent from a lumped port by design, so a
#: single list would be a rule that only the first example shipped can pass.
#:
#: Not Thickness or Conductivity: those left the *port* and stayed on
#: EMMaterial, which is where they were being read from all along.
PORT_PROPERTIES = {
    "EMPortMicrostrip": ("FeedResistance", "TraceEnd", "FeedOffset", "MeasurementDistance"),
    "EMPortLumped": ("Resistance", "SourceEntity", "ReferenceEntity", "ExcitationAxis"),
}


def test_the_ports_are_the_current_ones(archive):
    """Same for the port properties.

    A saved document keeps whatever properties it was written with - nothing
    re-runs ``__init__`` on restore - so a stale example is a document the
    adapter refuses, shipped as the thing to open first.

    Only positive assertions: nothing can write a retired property name any
    more, and a stale example fails the assertions here.
    """
    ports = ports_of(archive)
    assert ports, "no port in this document"
    for name, kind in sorted(ports.items()):
        assert kind in PORT_PROPERTIES, f"{name} is an {kind}, which has no rule here"
        held = properties_of(archive, name)
        for prop in SHARED_PORT_PROPERTIES + PORT_PROPERTIES[kind]:
            assert prop in held, f"{name} carries no {prop}: {sorted(held)}"


def test_it_names_no_directory_on_the_machine_that_built_it(archive):
    """``SimDir`` must ship empty.

    It is an absolute path, and it is set on the first Run - so a document
    saved after one carries the builder's home directory into everybody else's
    copy, where the study then writes its envelope and its results somewhere
    that does not exist.
    """
    document = ElementTree.fromstring(archive["Document.xml"])
    for prop in document.iter("Property"):
        if prop.get("name") != "SimDir":
            continue
        path = prop.find("Path")
        assert path is not None and not path.get("value"), (
            f"the example ships a simulation directory: {path.get('value')!r}"
        )


def test_it_holds_no_mesh_preview(archive):
    """The preview is built on demand. One committed by accident would ship a
    grid that stops matching the document the moment anything changes."""
    assert "EMMeshPreview" not in document_xml(archive)


def test_it_holds_no_results(archive):
    """Same for an S-matrix.

    Results are the answer to one solve of one version of the model. Shipped
    inside the document they open as though they described what the user is
    looking at, and nothing on screen says they are a stranger's.
    """
    assert "EMSParameters" not in document_xml(archive)


def test_every_shape_in_the_archive_has_content(archive):
    """A zero-byte ``.brp`` is a shape that failed to build.

    FreeCAD writes the entry either way, so the file looks complete and the
    document opens - with the object marked Invalid and drawing nothing. An
    ``execute`` raising on a name that does not exist ships a port with a null
    shape while every other test here passes.
    """
    empty = sorted(name for name in archive if name.endswith(".brp") and not archive[name])
    assert not empty, f"shapes that failed to build: {empty}"


def properties_of(archive, name):
    """One object's properties, as ``{name: value}``.

    FreeCAD splits an object across two sections - ``Objects`` declares its
    type, ``ObjectData`` holds its values - and only the second is keyed by
    anything, so a reader that wants both has to join them by name.
    """
    root = ElementTree.fromstring(archive["Document.xml"])
    data = root.find("ObjectData")
    obj = next((o for o in data.iter("Object") if o.get("name") == name), None)
    assert obj is not None, f"no object named {name}"
    found = {}
    for prop in obj.iter("Property"):
        value = next(iter(prop), None)
        if value is not None:
            found[prop.get("name")] = value.get("value")
    return found


def ports_of(archive):
    """Every port in the document, as ``{object name: the class it restores as}``.

    A saved object names the class FreeCAD will instantiate for it, and that is
    the same fact ``kinds.kind_of`` reads off the proxy one step later. Selected
    on the class *name*, there being nothing else in the file to select on;
    :data:`PORT_PROPERTIES` then has to have a rule for whatever comes back, so
    a port kind renamed out from under this fails loudly rather than silently
    checking nothing.
    """
    root = ElementTree.fromstring(archive["Document.xml"])
    found = {}
    for obj in root.find("ObjectData").iter("Object"):
        for python in obj.iter("Python"):
            if (python.get("class") or "").startswith("EMPort"):
                found[obj.get("name")] = python.get("class")
    return found


def enum_of(archive, name, prop):
    """One enumeration property, as the word behind it.

    ``App::PropertyEnumeration`` stores the *index* of the chosen string - and
    FreeCAD writes the whole list beside it, so the document says what its own
    number means. Read from the file rather than from a copy of the list here,
    which is a copy that stops agreeing the day an enumeration grows a value in
    the middle: the test then quietly asserts a different word than it names.
    """
    root = ElementTree.fromstring(archive["Document.xml"])
    obj = next(o for o in root.find("ObjectData").iter("Object") if o.get("name") == name)
    found = next(p for p in obj.iter("Property") if p.get("name") == prop)
    choices = [enum.get("value") for enum in found.iter("Enum")]
    return choices[int(found.find("Integer").get("value"))]


def axis_of(archive, name, prop):
    """One port's axis property, as the word the user picked."""
    return enum_of(archive, name, prop)


def bound_to(archive, name):
    """What a material binding's ``References`` point at, by object name.

    Scoped to that one property. The object's own ``Material`` link is a
    ``<Link>`` too, and a search over the whole element picks it up as an
    unnamed extra.
    """
    root = ElementTree.fromstring(archive["Document.xml"])
    obj = next(o for o in root.find("ObjectData").iter("Object") if o.get("name") == name)
    references = next(p for p in obj.iter("Property") if p.get("name") == "References")
    return [link.get("obj") for link in references.iter("Link")]


def choices_of(archive, prop):
    """Every list of choices a named enumeration offers, one per object holding it.

    FreeCAD writes an enumeration's whole list into the document, so a saved
    file offers what it was written with rather than what its class offers now.
    """
    root = ElementTree.fromstring(archive["Document.xml"])
    return [
        [enum.get("value") for enum in found.iter("Enum")]
        for obj in root.find("ObjectData").iter("Object")
        for found in obj.iter("Property")
        if found.get("name") == prop
    ]


def test_the_excitation_offers_only_what_the_adapter_can_produce(archive):
    """The property editor of the first document a new user opens.

    An enumeration lives in the file, so narrowing one in the code reaches no
    document already written - and these are written, committed and shipped.
    Whatever the choices are, an example offering one the translation refuses
    is a silent no-op with an audience.
    """
    from Microwave.Solvers.openems.policy import GAUSSIAN

    offered = choices_of(archive, "Waveform")
    assert offered, "no analysis in this document"
    for choices in offered:
        assert choices == [GAUSSIAN]


def test_every_port_draws_its_box(archive):
    """A port is a Part::FeaturePython, so it has geometry of its own.

    Named rather than inferred: the check above catches a shape that broke, and
    this one catches a port that quietly stopped being drawable at all.
    """
    document = ElementTree.fromstring(archive["Document.xml"])
    ports = [
        obj.get("name")
        for obj in document.iter("Object")
        if obj.get("type") == "Part::FeaturePython"
    ]
    assert ports
    for name in ports:
        assert f"{name}.Shape.brp" in archive


class TestTheStubExampleIsStillATwoPort:
    """One example named here, because being named is the whole of its job.

    ``stub_notch.FCStd`` exists to show a transmission response with a feature
    in it. Everything that makes it that rather than a second copy of
    ``microstrip_50ohm.FCStd`` is a property of the document, and every one of
    them is an edit away from being lost without anything else noticing: drop a
    port and the run still succeeds, at half the time, plotting an S11 with
    nothing to compare against.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def document():
        with zipfile.ZipFile(EXAMPLES / "stub_notch.FCStd") as opened:
            return {name: opened.read(name) for name in opened.namelist()}

    def test_it_has_two_ports_and_drives_both(self, document):
        """Both excited, because a column of the S-matrix costs a solve.

        One driven port of a two-port measures S11 and S21 and leaves the other
        column unmeasured, which is honest and is not what this example is for.
        """
        numbers = set()
        for name in ("Port1", "Port2"):
            port = properties_of(document, name)
            numbers.add(port["Number"])
            assert port["Excitation"] == "true", f"{name} is not a source"
        assert numbers == {"1", "2"}

    def test_each_port_looks_inward_from_its_own_end(self, document):
        """Opposite ends of one line, each facing the stub between them.

        Both pointing the same way is a model that translates, meshes, solves
        and returns a plausible S21 measured through the absorber. So is
        swapping the two, which is why this pins each port to an end rather
        than checking that the pair of axes is the right pair.
        """
        for name, expected in (("Port1", "X"), ("Port2", "-X")):
            assert axis_of(document, name, "PropagationAxis") == expected
            # Down through the substrate at both ends: the field points from
            # trace to ground, and which end the wave came in at does not
            # change that. ``portbox.microstrip`` refuses the other spelling,
            # so this is the readable half of a guard rather than the guard.
            assert axis_of(document, name, "ExcitationAxis") == "-Z"

    def test_the_substrate_is_lossy(self, document):
        """Loss is most of what sets how deep the notch goes.

        Without it the resonance is limited only by radiation, and the two runs
        of the sweep then disagree about the ports' impedance by more than
        ``IMPEDANCE_TOLERANCE`` - so the assembled matrix comes back with the
        deepest point of the notch blanked. Measured on this geometry by taking
        the loss out. Which of the two runs to believe at a resonance is open.
        """
        fr4 = properties_of(document, "FR4")
        assert float(fr4["LossTangent"]) > 0
        # Fixed conductivity stands in for a loss tangent at one frequency, so
        # where that number was quoted is the only record of how good the
        # substitution is. Pre-flight warns when it is far from band centre.
        assert float(fr4["MeasuredAt"]) > 0

    def test_the_stub_is_a_conductor(self, document):
        """The feature is copper, not an unbound sheet.

        A binding that lost the stub leaves it out of the envelope entirely:
        the model translates, meshes and solves as a plain through line, and
        the notch simply is not there.
        """
        assert set(bound_to(document, "TraceBinding")) == {"Trace", "Stub"}


class TestTheSteppedLineStillHasAProfileToRead:
    """The other named example, for the same reason the stub is named.

    ``stepped_line.FCStd`` exists so the impedance chart has something to draw:
    a line whose impedance changes *along* it. Every property that makes it that
    is an edit away from being lost quietly - the run succeeds either way, and
    what comes back is a trace that looks perfectly ordinary and is flat, or
    scaled wrongly, or not takeable at all.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def document():
        with zipfile.ZipFile(EXAMPLES / "stepped_line.FCStd") as opened:
            return {name: opened.read(name) for name in opened.namelist()}

    def test_the_middle_section_is_the_narrow_one(self, document):
        """The step, in the direction the prose everywhere says it goes.

        Three sections of one width is a uniform line drawn three times, and its
        profile is the flat trace ``microstrip_50ohm.FCStd`` already gives.
        Narrow-wide-narrow is a line with a step in it too, and it is not the
        one the README, the chart's caption and this example's own docstring
        describe - a narrower trace is a *higher* impedance, so the middle
        section is what stands up out of the trace.
        """
        widths = [float(properties_of(document, f"Section{n}")["Width"]) for n in (1, 2, 3)]
        assert widths[0] == widths[2], "the line is meant to be its own mirror"
        assert widths[1] < widths[0]

    def test_the_sections_tile_the_board(self, document):
        """Butted end to end, and covering it.

        Shorten one and the line has a gap, which the trace reads as an open
        circuit where a step should be; the transitions also stop being far
        enough apart for the band to resolve them separately, which is the
        margin :data:`SECTION_LENGTH` was chosen for and the acceptance gate
        asserts.
        """
        lengths = [float(properties_of(document, f"Section{n}")["Length"]) for n in (1, 2, 3)]
        assert len(set(lengths)) == 1, f"the sections are not equal: {lengths}"
        assert sum(lengths) == float(properties_of(document, "Substrate")["Length"])

    def test_every_section_is_copper(self, document):
        """A binding that lost one leaves that section out of the envelope: the
        model translates and solves as a line with a gap in it, and the trace
        reads the gap as an open."""
        assert set(bound_to(document, "TraceBinding")) == {"Section1", "Section2", "Section3"}

    def test_both_ports_are_lumped_and_declare_the_same_impedance(self, document):
        """What makes the trace mean anything.

        A lumped port *is* its resistance, so the reflection carries the line's
        impedance and no port has an opinion about it. Swap one for a microstrip
        port and the reference becomes what that port extracted from this very
        line, so the section it sits on reads its own measurement back instead
        of being checked against a number: that plateau flattens towards the
        reference and stops being evidence. The steps past it stay genuine,
        which is why the trace is still worth taking - but the quantity this
        document is built to read is the one that goes.
        """
        ports = ports_of(document)
        assert set(ports.values()) == {"EMPortLumped"}
        for name in ports:
            held = properties_of(document, name)
            assert float(held["Resistance"]) == float(held["ReferenceImpedance"])

    def test_it_drives_port_1_and_sits_a_second_port_at_the_far_end(self, document):
        """Which port is driven, and that the other one is somewhere else.

        Both matter to the chart rather than to the solve. A step response can
        only be taken where a wave was launched, so driving port 2 instead
        leaves the README's instruction pointing at an entry the menu does not
        offer; driving neither is a document the Run button refuses. And the
        distance axis is a length divided by a delay, so two ports at the same
        end of the board measure a velocity across no distance - while the far
        end, unterminated, becomes an open circuit the trace reads as structure.
        """
        driving = {
            int(properties_of(document, name)["Number"])
            for name in ports_of(document)
            if properties_of(document, name)["Excitation"] == "true"
        }
        assert driving == {1}
        ends = {properties_of(document, name)["SourceEntity"] for name in ports_of(document)}
        assert ends == {"Section1", "Section3"}

    def test_the_sweep_starts_exactly_one_step_above_dc(self, document):
        """The setting that is not visible in the drawing.

        A step response is carried by its low frequencies, and everything below
        the first solved point is invented by the extrapolation to DC. A band
        starting anywhere else solves perfectly well and ``Results.tdr`` refuses
        it, so the example would ship unable to demonstrate the chart it is for.

        One step, not ``tdr.INVENTED_BINS`` of them. The bar is where a trace
        stops being answered at all, and a fixture sitting on it measures the
        guard instead of the line - ``test_acceptance_tdr`` holds its own sweep
        to one for the same reason.
        """
        analysis = properties_of(document, "EMAnalysis")
        step = float(analysis["FrequencyStop"]) / int(analysis["NumFrequencyPoints"])
        assert float(analysis["FrequencyStart"]) / step == pytest.approx(1.0, rel=1e-9, abs=0.0)

    def test_the_board_ends_inside_the_domain(self, document):
        """Air on every face, unlike the other two examples.

        The ports are resistances and terminate the line at every frequency,
        extrapolated DC included. Pull the domain in over an end instead and the
        absorber is in the signal path, where the trace reads it as structure.
        """
        for axis in "XYZ":
            for side in ("Min", "Max"):
                face = f"Padding{axis}{side}"
                assert enum_of(document, "EMMeshPolicy", face) == "Air", f"{face} is not Air"


class TestTheLowPassIsStillTheFilterItWasSynthesisedAs:
    """The third named example, and the only one that is a filter.

    ``stepped_lowpass_synthesised.FCStd`` exists to show what the design equations leave
    out, which only works while the board *is* what those equations produced.
    Every section's length was computed from a prototype value and an impedance,
    and a length nudged by hand detunes the filter into something the comparison
    no longer describes - silently, because a detuned filter is still a filter
    and its response still looks like one.

    So this re-derives the drawing rather than remembering it. The widths, the
    substrate and its permittivity are read out of the document; Hammerstad
    turns the widths into impedances; the prototype turns those into lengths.
    Only the two design inputs are written here - which prototype, and where the
    corner was put - and neither is a measurement.
    """

    #: Butterworth, order five, shunt capacitor first. A published constant.
    PROTOTYPE = ((0.6180, "C"), (1.6180, "L"), (2.0000, "C"), (1.6180, "L"), (0.6180, "C"))
    #: Where the corner was placed, in Hz, and the system impedance it was
    #: placed against. Both are design inputs rather than results.
    CORNER = 2.0e9
    SYSTEM = 50.0
    #: The drawing's own resolution. A length is drawn to the hundredth of a
    #: millimetre, so it may differ from the synthesis by half of one.
    RESOLUTION = 0.005

    @staticmethod
    @pytest.fixture(scope="class")
    def document():
        with zipfile.ZipFile(EXAMPLES / "stepped_lowpass_synthesised.FCStd") as opened:
            return {name: opened.read(name) for name in opened.namelist()}

    @staticmethod
    def section(document, number):
        found = properties_of(document, f"Section{number}")
        return float(found["Length"]), float(found["Width"])

    def test_the_line_is_two_leads_around_five_filter_sections(self, document):
        """Seven sheets, and which is which is decided by width alone.

        A filter section lost to an edit leaves a line that still solves, still
        returns an S21 with a corner in it, and is a different filter from the
        one the docstring describes.
        """
        shape_names = set(shapes(document))
        assert {f"Section{n}" for n in range(1, 8)} <= shape_names
        assert "Section8" not in shape_names

        widths = [self.section(document, n)[1] for n in range(1, 8)]
        assert widths[0] == widths[6], "the leads are meant to match"
        wide, narrow = widths[1], widths[2]
        assert narrow < widths[0] < wide, "the leads sit between the two section widths"
        # Capacitor, inductor, capacitor, inductor, capacitor.
        assert widths[1:6] == [wide, narrow, wide, narrow, wide]

    def test_it_is_its_own_mirror_image(self, document):
        """Which is what lets one driven port measure the whole matrix.

        Lose the symmetry and the second column is no longer the first one
        reflected, so the result the workbench assembles from a single solve
        stops describing the board - and nothing in the run says so.
        """
        for near, far in ((1, 7), (2, 6), (3, 5)):
            assert self.section(document, near) == self.section(document, far)

    def test_only_the_first_port_is_driven(self, document):
        """Unlike ``stub_notch.FCStd``, which drives both and says why.

        A symmetric two-port's second solve buys a column that repeats the
        first, at the price of the whole run again.
        """
        assert properties_of(document, "Port1")["Excitation"] == "true"
        assert properties_of(document, "Port2")["Excitation"] == "false"

    def test_every_section_is_the_length_the_synthesis_asked_for(self, document):
        """The assertion this class exists for.

        A short length of high-impedance line stands in for a series inductor
        and a short length of low-impedance line for a shunt capacitor, and what
        makes each stand in correctly is its *electrical* length: ``g Z0 / Zh``
        radians for an inductor, ``g Zl / Z0`` for a capacitor. Physical length
        follows from that and the guided wavelength at the corner.

        Nothing here is read off the drawing except the widths and the board, so
        a width changed without re-deriving the lengths fails - which is the
        edit that leaves the example looking entirely correct.
        """
        eps_r = float(properties_of(document, "FR4")["Permittivity"])
        height = float(properties_of(document, "Substrate")["Height"])

        for index, (g, kind) in enumerate(self.PROTOTYPE, start=2):
            drawn, width = self.section(document, index)
            impedance = analytic.characteristic_impedance(width, height, eps_r)
            ratio = self.SYSTEM / impedance if kind == "L" else impedance / self.SYSTEM
            electrical = g * ratio
            eps_eff = analytic.effective_permittivity(width, height, eps_r)
            wavelength = analytic.SPEED_OF_LIGHT / self.CORNER / eps_eff**0.5 * 1e3
            wanted = electrical / (2.0 * math.pi) * wavelength
            assert drawn == pytest.approx(wanted, abs=self.RESOLUTION), (
                f"Section{index} is drawn {drawn} mm and synthesises to {wanted:.4f} mm"
            )

    def test_the_leads_are_the_system_impedance(self, document):
        """The filter is specified in 50 ohm and has to be embedded in it.

        A lead at some other impedance is a sixth section of the filter that no
        prototype value asked for, and it moves the response without moving
        anything the drawing labels.
        """
        eps_r = float(properties_of(document, "FR4")["Permittivity"])
        height = float(properties_of(document, "Substrate")["Height"])
        width = self.section(document, 1)[1]
        impedance = analytic.characteristic_impedance(width, height, eps_r)
        # Hammerstad's own accuracy is about 1 %, and the width is drawn to the
        # hundredth of a millimetre; asking for closer would be asking the
        # reference for more than it has.
        assert impedance == pytest.approx(self.SYSTEM, rel=0.01)

    def test_the_sections_tile_the_board(self, document):
        """Butted end to end, and covering the substrate exactly.

        A gap is an open circuit where a step should be, and an overlap is two
        conductors in one place - which pre-flight refuses only when they carry
        different materials, and these carry one.
        """
        board = properties_of(document, "Substrate")
        assert sum(self.section(document, n)[0] for n in range(1, 8)) == pytest.approx(
            float(board["Length"]), abs=1e-9
        )


class TestTheMeasuredLowPassIsTheBoardThatWasBuilt:
    """The fourth named example, and the only one drawn from somebody else.

    ``stepped_lowpass_measured.FCStd`` is scored against a network analyser in
    ``tests/test_acceptance_lowpass.py``, and a measurement is only a reference
    for the board it was taken on. Every dimension in it was transcribed out of
    a paper, so what can go wrong here is not a detuned filter but a *different*
    one - and a different one still solves, still has a corner, and still looks
    entirely plausible next to the published curve.

    So this compares the committed document against the transcription itself,
    which lives in one place and is what the gate builds from. The example
    script has its own copy, because it runs under FreeCAD and cannot import
    from here; this is the assertion that keeps the two in step.
    """

    PAPER = published.CHEN_2025

    @staticmethod
    @pytest.fixture(scope="class")
    def document():
        with zipfile.ZipFile(EXAMPLES / "stepped_lowpass_measured.FCStd") as opened:
            return {name: opened.read(name) for name in opened.namelist()}

    @staticmethod
    def section(document, number):
        found = properties_of(document, f"Section{number}")
        return float(found["Length"]), float(found["Width"])

    def test_the_line_is_two_leads_around_five_filter_sections(self, document):
        """Seven sheets, and which is which is decided by width alone."""
        shape_names = set(shapes(document))
        assert {f"Section{n}" for n in range(1, 8)} <= shape_names
        assert "Section8" not in shape_names

    def test_every_filter_section_is_the_one_in_the_table(self, document):
        """The assertion this class exists for.

        Both numbers per section, because a width and a length are independent
        transcription mistakes and either alone makes a different filter.
        """
        for index, (length, kind) in enumerate(zip(self.PAPER.lengths, self.PAPER.kinds), start=2):
            wanted = self.PAPER.wide if kind == "C" else self.PAPER.narrow
            assert self.section(document, index) == (length, wanted), (
                f"Section{index} is drawn {self.section(document, index)} "
                f"and the table says {(length, wanted)}"
            )

    def test_the_substrate_is_the_laminate_the_paper_names(self, document):
        """Permittivity, loss and thickness together.

        The widths only realise the impedances the paper claims on the substrate
        it names, so a laminate edited here silently makes the whole comparison
        one between two different boards.
        """
        fr4 = properties_of(document, "FR4")
        assert float(fr4["Permittivity"]) == self.PAPER.eps_r
        assert float(fr4["LossTangent"]) == self.PAPER.loss_tangent
        assert float(properties_of(document, "Substrate")["Height"]) == self.PAPER.height

    def test_the_widths_realise_the_impedances_the_paper_designed_for(self, document):
        """Read off the document, not off the transcription.

        The gate makes this comparison against the table; this makes it against
        the file, so a width and a substrate that were *both* edited into
        agreement with each other still fail against Hammerstad.
        """
        eps_r = float(properties_of(document, "FR4")["Permittivity"])
        height = float(properties_of(document, "Substrate")["Height"])
        for index, kind in enumerate(self.PAPER.kinds, start=2):
            width = self.section(document, index)[1]
            declared = self.PAPER.low_impedance if kind == "C" else self.PAPER.high_impedance
            found = analytic.characteristic_impedance(width, height, eps_r)
            # Hammerstad's own accuracy is about 1 %, which is all that is asked.
            assert found == pytest.approx(declared, rel=0.01)

    def test_the_leads_are_the_system_the_filter_was_specified_in(self, document):
        """The leads are ours, not the paper's, so they get checked rather than
        compared: a lead at some other impedance is a sixth section nobody
        asked for, and it moves the corner without moving anything labelled."""
        eps_r = float(properties_of(document, "FR4")["Permittivity"])
        height = float(properties_of(document, "Substrate")["Height"])
        for number in (1, 7):
            width = self.section(document, number)[1]
            found = analytic.characteristic_impedance(width, height, eps_r)
            assert found == pytest.approx(self.PAPER.system, rel=0.01)

    def test_it_is_its_own_mirror_image(self, document):
        """Which is what lets one driven port measure the whole matrix."""
        for near, far in ((1, 7), (2, 6), (3, 5)):
            assert self.section(document, near) == self.section(document, far)

    def test_only_the_first_port_is_driven(self, document):
        """A symmetric two-port's second solve buys a column that repeats the
        first, at the price of the whole run again."""
        assert properties_of(document, "Port1")["Excitation"] == "true"
        assert properties_of(document, "Port2")["Excitation"] == "false"

    def test_the_sections_tile_the_board(self, document):
        """Butted end to end, and covering the substrate exactly."""
        board = properties_of(document, "Substrate")
        assert sum(self.section(document, n)[0] for n in range(1, 8)) == pytest.approx(
            float(board["Length"]), abs=1e-9
        )
