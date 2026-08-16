# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

import FreeCAD

from ._vp_hook import ViewProviderRestored

#: Values of ``EMPort.ReferencedTo``. The first is the default and is what a
#: "50-ohm system" means: every port's S-parameters reported against one number
#: the user typed.
FIXED_IMPEDANCE = "Fixed impedance"
#: Report this port against the impedance the port itself has - measured from
#: the field for a microstrip, analytic for a waveguide, the resistance for a
#: lumped element. It is what a bench does: TRL references a guide to the line
#: standard's own characteristic impedance, dispersive, with no number entered
#: anywhere. Nor is there one to enter - a guide's impedance is
#: convention-dependent by a factor of a quarter, and only the ratios a
#: reference cancels out of are convention-free.
PORT_IMPEDANCE = "Port impedance"


def is_port(obj):
    """True when this document object is one of this workbench's ports.

    By the proxy's class, not by its name - see ``kinds.py`` for why a prefix
    test is the wrong question.
    """
    return isinstance(getattr(obj, "Proxy", None), EMPortBase)


def next_port_number(doc):
    """The lowest port number this document is not already using.

    Every port is numbered when it is made, because a port without one is
    refused by the translation - ``ports._port_numbers`` needs it to index
    the S-matrix.

    Renumbering an existing port is deliberately not offered. The S-matrix a
    user reads back is indexed by these numbers, so moving one would silently
    re-label a result against the tree it came from.
    """
    used = {int(port.Number) for port in doc.Objects if is_port(port)}
    number = 1
    while number in used:
        number += 1
    return number


class EMPortBase(ViewProviderRestored):
    """What every port has, and nothing more.

    A property here reaches every kind that inherits it. Anything a particular
    solver ignores for a particular kind belongs on the kind instead -
    a silent no-op is never acceptable, and a property sitting in the editor
    doing nothing is one. A feed shift on a waveguide port is the
    trap: the adapter drops it on the floor, and the editor still offers it.
    """

    def __init__(self, obj):
        obj.addProperty("App::PropertyInteger", "Number", "Port", "Unique port number")
        obj.Number = 0  # 0 means auto-assign

        obj.addProperty(
            "App::PropertyBool", "Excitation", "Port", "Whether the port is an active source"
        )
        obj.Excitation = True

        # The impedance the S-parameters are reported against - the "50" in
        # "50-ohm system", not a property of the structure. A microstrip port
        # measures its own characteristic impedance and the result layer
        # renormalises to this; nothing about the model changes when it does.
        obj.addProperty(
            "App::PropertyFloat",
            "ReferenceImpedance",
            "Port",
            "Impedance the S-parameters are reported against (Ohms)",
        )
        obj.ReferenceImpedance = 50.0

        # Enumeration rather than a checkbox, for the reason ``EMAnalysis``
        # gives about ``Symmetry``: a guide has three impedances - wave,
        # power-voltage and power-current - and the day one of them has to be
        # named, a bool would have to be replaced rather than extended.
        obj.addProperty(
            "App::PropertyEnumeration",
            "ReferencedTo",
            "Port",
            "What this port's S-parameters are reported against",
        )
        obj.ReferencedTo = [FIXED_IMPEDANCE, PORT_IMPEDANCE]

        obj.Proxy = self

    def onChanged(self, obj, prop):
        """Hide ``ReferenceImpedance`` when nothing reads it.

        Hidden rather than read-only. A greyed-out ``50.00`` beside a result
        referenced to 475 ohm is a number that answers the question the user is
        asking and answers it wrongly. A property that looks like it does
        something and does not is a silent no-op, which a stale value shown in
        the editor is, more so than an absent one.

        ``onChanged`` fires for every property as ``__init__`` creates it and
        again for each one during restore, so it is filtered on the property
        name; ``ReferencedTo`` is created before ``Proxy`` is assigned and never
        reaches here during construction. Restore needs no guard either: the
        editor mode is a status bit on the property and travels in the file.
        """
        if prop == "ReferencedTo":
            obj.setEditorMode(
                "ReferenceImpedance", 2 if str(obj.ReferencedTo) == PORT_IMPEDANCE else 0
            )

    def execute(self, obj):
        """Draw the box the solver will build, if there is one yet.

        Live rather than manual: a port box is a dozen operations on bounding
        boxes with no dependence on the mesh or the band, so it recomputes like
        any other parametric feature. ``EMMeshPreview`` is the same class of
        object with a deliberately empty ``execute``, for the opposite reason.

        Never raises. A port is created before it is configured - the commands
        make one from whatever was selected and say what is missing - so an
        unfinished port has an empty shape rather than a traceback.
        """
        if not hasattr(obj, "Shape"):
            return
        from .port_shape import build

        obj.Shape = build(obj)

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class EMPortLumped(EMPortBase):
    """Lumped port across two faces or edges."""

    def __init__(self, obj):
        super().__init__(obj)
        obj.addProperty("App::PropertyLinkSub", "SourceEntity", "Lumped", "Source face/edge")
        obj.addProperty("App::PropertyLinkSub", "ReferenceEntity", "Lumped", "Reference face/edge")
        obj.addProperty(
            "App::PropertyEnumeration", "ExcitationAxis", "Lumped", "Axis of excitation"
        )
        obj.ExcitationAxis = ["X", "Y", "Z", "-X", "-Y", "-Z"]
        # A lumped port *is* its resistance - openEMS builds a resistive sheet
        # across the gap - so unlike the microstrip's damping resistor this one
        # is part of the structure. Not called FeedResistance for that reason:
        # the two are different quantities, and zero means opposite things about
        # them. Here it is a short, which openEMS models by laying metal across
        # the gap instead of a resistive sheet, and which is a legitimate thing
        # to ask for.
        obj.addProperty(
            "App::PropertyFloat",
            "Resistance",
            "Lumped",
            "Resistance of the lumped element; 0 lays metal across the gap (Ohms)",
        )
        obj.Resistance = 50.0
        obj.Proxy = self


class EMPortMicrostrip(EMPortBase):
    """Microstrip port for trace-on-dielectric."""

    def __init__(self, obj):
        super().__init__(obj)
        # Not TraceFace. A solid trace offers a Face here and a trace drawn as
        # a zero-thickness sheet offers an Edge, and both are correct - so a
        # name promising a topology is wrong half the time. What it means is the
        # trace's cross-section where the wave enters.
        obj.addProperty(
            "App::PropertyLinkSub",
            "TraceEnd",
            "Microstrip",
            "The trace's cross-section where the wave enters (a face, or an edge on a sheet)",
        )
        obj.addProperty(
            "App::PropertyLinkSub", "GroundReference", "Microstrip", "Reference face/edge of ground"
        )
        # Three absolute distances. FeedOffset and Length are measured inward
        # from TraceEnd; MeasurementDistance is measured from the *source*,
        # because that is what the physics is about - the probes must clear
        # the source's near field, and that requirement knows nothing about
        # where the picked face is. Together they are
        # the port's *requirement*: the box is drawn at this size whether or not
        # the trace under it is that long, so a feed line that is too short is
        # something visible. A box derived from the trace always looks like
        # it fits, which is exactly when it does not.
        #
        # Distances in millimetres, not fractions of the box length; see
        # ``portbox.PortBox`` for why, and ``portbox.CLEARANCE`` for the
        # measurement that sets the one that matters.
        obj.addProperty(
            "App::PropertyLength",
            "FeedOffset",
            "Microstrip",
            "How far in from TraceEnd the source sits (0 = on the picked face)",
        )
        obj.FeedOffset = 0.0
        obj.addProperty(
            "App::PropertyLength",
            "MeasurementDistance",
            "Microstrip",
            "How far downstream of the source the probes sit; filled in from the"
            " study's band when the port is made",
        )
        obj.MeasurementDistance = 0.0
        obj.addProperty(
            "App::PropertyLength",
            "Length",
            "Microstrip",
            "Extent along the propagation axis (0 = end at the measurement plane)",
        )
        obj.Length = 0.0
        # Zero is a bare voltage source, which is what the microstrip acceptance
        # gate runs. What the resistance does and how openEMS spells its absence
        # is in openems.ports._feed_resistance.
        obj.addProperty(
            "App::PropertyFloat",
            "FeedResistance",
            "Microstrip",
            "Series resistance damping the feed, 0 for a bare source (Ohms)",
        )
        obj.FeedResistance = 0.0
        obj.addProperty(
            "App::PropertyEnumeration",
            "PropagationAxis",
            "Microstrip",
            "Direction of wave propagation, into the structure",
        )
        obj.PropagationAxis = ["X", "Y", "Z", "-X", "-Y", "-Z"]
        obj.addProperty(
            "App::PropertyEnumeration",
            "ExcitationAxis",
            "Microstrip",
            "Direction of electric field, trace towards ground",
        )
        obj.ExcitationAxis = ["X", "Y", "Z", "-X", "-Y", "-Z"]
        # A board lies in XY with the trace above the ground plane, so the field
        # points down. The default is the common case rather than a placeholder:
        # leaving both axes at X would make every new port refuse itself.
        obj.ExcitationAxis = "-Z"
        obj.Proxy = self


class EMPortRectWaveguide(EMPortBase):
    """Rectangular waveguide port."""

    def __init__(self, obj):
        super().__init__(obj)
        obj.addProperty(
            "App::PropertyLinkSub",
            "CrossSection",
            "Waveguide",
            "The guide's cross-section (a face, or an edge on a sheet)",
        )
        obj.addProperty(
            "App::PropertyEnumeration",
            "PropagationAxis",
            "Waveguide",
            "Direction of wave propagation",
        )
        obj.PropagationAxis = ["X", "Y", "Z", "-X", "-Y", "-Z"]
        # AddRectWaveGuidePort puts the excitation on the near face of this box
        # and the probes on the far one, so this length *is* where the
        # measurement plane sits. Not a depth that happens to be invisible - a
        # reference plane.
        #
        # Zero means five mesh cells, which is what openEMS' own examples and
        # the WR-42 gate use. It is the one number in the whole port surface
        # that depends on the *mesh*, and so the one thing about a port that
        # cannot be drawn before meshing.
        obj.addProperty(
            "App::PropertyLength",
            "Length",
            "Waveguide",
            "Where the measurement plane sits along the propagation axis (0 = five mesh cells)",
        )
        obj.Length = 0.0
        obj.addProperty(
            "App::PropertyEnumeration",
            "Mode",
            "Waveguide",
            "Excitation mode; the first index counts half-waves across the broad wall",
        )
        # TE only: openEMS' RectWGPort refuses TM outright, so offering one in
        # the dropdown is a port the user can build that no solver can run.
        #
        # Named the textbook way round - TE10 is the dominant mode however the
        # guide is drawn - and not by axis, which is what openEMS wants. The
        # adapter renumbers it in `model.Port.waveguide_arguments`, because which
        # axis carries which index is openEMS' business and not the document's.
        obj.Mode = ["TE10", "TE01", "TE11", "TE20"]
        obj.Proxy = self


class EMPortCoaxial(EMPortBase):
    """Coaxial port on a line drawn as two concentric conductors."""

    def __init__(self, obj):
        super().__init__(obj)
        # One pick, and it carries everything: the ring between the inner
        # conductor and the shield's bore gives both radii and the axis they are
        # about. Neither radius is a property, because a radius that could be
        # typed is a radius that could disagree with the drawing - and the
        # impedance of a coaxial line is nothing but their ratio.
        obj.addProperty(
            "App::PropertyLinkSub",
            "Annulus",
            "Coaxial",
            "The ring between the inner conductor and the shield, at the line's end",
        )
        obj.addProperty(
            "App::PropertyEnumeration",
            "PropagationAxis",
            "Coaxial",
            "Direction of wave propagation, into the structure",
        )
        obj.PropagationAxis = ["X", "Y", "Z", "-X", "-Y", "-Z"]
        # The same three distances a microstrip port carries, measured the same
        # way and for the same reason: the source is a sheet across the annulus
        # and the probes have to be clear of its near field before the impedance
        # they difference is the line's.
        obj.addProperty(
            "App::PropertyLength",
            "FeedOffset",
            "Coaxial",
            "How far in from the picked ring the source sits (0 = on the ring)",
        )
        obj.FeedOffset = 0.0
        obj.addProperty(
            "App::PropertyLength",
            "MeasurementDistance",
            "Coaxial",
            "How far downstream of the source the probes sit; filled in from the"
            " study's band when the port is made",
        )
        obj.MeasurementDistance = 0.0
        obj.addProperty(
            "App::PropertyLength",
            "Length",
            "Coaxial",
            "Extent along the propagation axis (0 = end at the measurement plane)",
        )
        obj.Length = 0.0
        obj.Proxy = self


def _create(proxy, feature, name, doc):
    """Make one port: the object, its proxy, its number, its view provider.

    One function rather than five near-copies. What differs between the kinds
    is the proxy class and whether it carries a shape, and both are arguments.
    Duplicated instead, a step left out of one copy is a step left out of all
    five.
    """
    if doc is None:
        doc = FreeCAD.ActiveDocument
    obj = doc.addObject(feature, name)
    proxy(obj)
    obj.Number = next_port_number(doc)
    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, proxy.__name__)
    return obj


def createEMPortLumped(name="LumpedPort", doc=None):
    return _create(EMPortLumped, "Part::FeaturePython", name, doc)


def required_clearance(doc):
    """Feed-to-probe clearance for this document, in millimetres, or ``0.0``.

    The band is all it takes: a tenth of the free-space wavelength at the bottom
    of the sweep. See ``portbox.CLEARANCE`` for why no permittivity comes into
    it - in short, the one that governs is the line's own eps_eff, which does
    not exist yet when a port is created, and free space is the conservative
    bound on it.

    Written into the port when it is made rather than computed whenever the box
    is drawn, and that is the whole design. The number depends on the analysis,
    and FreeCAD has no dependency edge from an analysis to a port - so a live
    computation would leave the drawn box stale the moment the band moved, and
    the picture would quietly stop being the solve. As a property it is a number
    the engineer can see, change, and be refused on.
    """
    from .. import portbox
    from .analysis import analyses

    found = analyses(doc)
    if len(found) != 1:
        return 0.0
    start = float(getattr(getattr(found[0], "FrequencyStart", 0.0), "Value", 0.0))
    return portbox.clearance(start)


def createEMPortMicrostrip(name="MicrostripPort", doc=None):
    obj = _create(EMPortMicrostrip, "Part::FeaturePython", name, doc)
    obj.MeasurementDistance = required_clearance(obj.Document)
    return obj


def createEMPortRectWaveguide(name="WaveguidePort", doc=None):
    return _create(EMPortRectWaveguide, "Part::FeaturePython", name, doc)


def createEMPortCoaxial(name="CoaxialPort", doc=None):
    obj = _create(EMPortCoaxial, "Part::FeaturePython", name, doc)
    obj.MeasurementDistance = required_clearance(obj.Document)
    return obj
