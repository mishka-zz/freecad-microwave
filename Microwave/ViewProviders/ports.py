# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import Provider


class EMPortViewProvider(Provider):
    """A port draws its own box, so this class only states how it looks.

    A Coin arrow built by hand in ``attach`` shows direction and nothing else.
    It does not show how far the port reaches, where the source sits, or where
    the measurement plane is, and those are what decide whether a port is right.
    The object is a ``Part::FeaturePython``, and its shape is what the solver
    builds; see ``Objects/port_shape.py``.

    This class defines no ``attach``, no scene graph and no ``updateData``.
    FreeCAD's own Part view provider draws the shape, and the document's
    dependency graph recomputes it when a property or a linked solid moves.
    """

    #: Translucent, because the box sits inside the structure and an opaque one
    #: would hide the trace it is built on.
    TRANSPARENCY = 70

    #: Red, so that no material and no preview shares the colour. The conductors
    #: in the catalog are the coppers and golds, so an orange box on a trace
    #: reads as more trace, and the mesh preview draws its edges in blue.
    COLOUR = (0.85, 0.11, 0.20)

    def __init__(self, vobj):
        super().__init__(vobj)
        for name, value in (
            ("ShapeColor", self.COLOUR),
            ("Transparency", self.TRANSPARENCY),
            ("DisplayMode", "Shaded"),
        ):
            try:
                setattr(vobj, name, value)
            except Exception:
                # A headless document has no view object to set these on, and a
                # port that cannot be coloured must still be a port.
                pass

    # This class defines no attach, no getDisplayModes and no setDisplayMode. On
    # a Part::FeaturePython each of those takes over from Part's own view
    # provider and the geometry stops appearing. EMMeshPreviewViewProvider
    # states the mechanism.


class EMPortLumpedViewProvider(EMPortViewProvider):
    ICON = "PortLumped.svg"


class EMPortMicrostripViewProvider(EMPortViewProvider):
    ICON = "PortMicrostrip.svg"


class EMPortRectWaveguideViewProvider(EMPortViewProvider):
    ICON = "PortWaveguide.svg"


class EMPortCoaxialViewProvider(EMPortViewProvider):
    ICON = "PortCoaxial.svg"
