# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The S-matrix, as something the document owns.

``EMSParameters`` is a result object, and this is it: the N×N matrix a sweep
produced, sitting in the analysis group beside the ports it describes.

Stored **in the document**, as flat lists of floats, and not as a path to a
file beside it. A ``.s2p`` beside the document is a dangling reference the
moment the ``.FCStd`` is moved, renamed or emailed, and it fails silently by
showing the previous run's numbers. ``App::PropertyFloatList`` is what
FreeCAD's own FEM result objects use for bulk numbers, saves and restores with
everything else, and compresses down to tens of kilobytes.

Complex values are split into a real and an imaginary list rather than
interleaved: two lists of the same length are self-evident to anything that
opens the document by hand.

**No scikit-rf here.** Round-tripping a matrix through the document is pure
numpy, so a document full of results opens without the second the result
library costs to import. It is reached only for a
:class:`~..Results.sparameters.SParameters` ``network()`` or a Touchstone file.

Staleness is **not** tracked, unlike ``EMMeshPreview``. A result is stale for
too broad a set of reasons - any material, port or band change moves the answer
without necessarily moving a cell - for a cheap digest to cover honestly. What
is recorded instead is the envelope digest of every run that went into the
matrix, so the question can be answered exactly from what the object carries.
"""

import json

import FreeCAD
import numpy as np

from ..Results.sparameters import ResultError, SParameters
from ._vp_hook import ViewProviderRestored

#: Properties holding the matrix itself. Hidden in the property editor: a
#: 2004-element float list rendered as an editable list widget is noise, and
#: editing one by hand could only make the object lie about a solve that
#: happened. :func:`load` is the way to read them.
BULK = (
    "Frequency",
    "PortNumbers",
    "DrivenPorts",
    "DerivedPorts",
    "SelfReferencedPorts",
    "DiscardedPoints",
    "ScatteringReal",
    "ScatteringImag",
    "ReferenceReal",
    "ReferenceImag",
    "PortImpedanceReal",
    "PortImpedanceImag",
)

#: Read-only, and shown: what the property editor says about a result.
#: ``Provenance`` is here rather than hidden because it is the record that makes
#: the result falsifiable - which solver, which version, which envelope
#: digest. A user who cannot see it has to write a script to ask whether the
#: numbers in front of them came from the model in front of them.
SUMMARY = (
    "Ports",
    "Points",
    "FrequencyStart",
    "FrequencyStop",
    "Reference",
    "Provenance",
)


class EMSParameters(ViewProviderRestored):
    """One N×N S-matrix against frequency, referenced to a stated impedance.

    Solver-neutral. openEMS produced this one, but nothing about the object says
    so - a NEC2 or Palace adapter assembling the same
    :class:`~..Results.sparameters.SParameters` stores it through the same
    :func:`store`, and every consumer above works unchanged, which is what
    makes a result object solver-neutral.
    """

    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyFloatList", "Frequency", "Data", "Frequency of each sample, in Hz"
        )
        obj.addProperty(
            "App::PropertyIntegerList",
            "PortNumbers",
            "Data",
            "The document port number of each row and column, in order",
        )
        # Which columns were actually measured. FDTD drives one port per run, so
        # a study that drove port 1 only has column 1 and nothing else; the rest
        # are stored as nan and this is what says so.
        obj.addProperty(
            "App::PropertyIntegerList",
            "DrivenPorts",
            "Data",
            "The ports that were driven - the columns that were measured",
        )
        obj.addProperty(
            "App::PropertyIntegerList",
            "DerivedPorts",
            "Data",
            "Columns filled from a declared symmetry rather than solved",
        )
        # Which they are cannot be recovered from the numbers: a port asked for
        # 50 ohm that measured 50 ohm looks identical to one referenced to
        # itself, and a mirror's derived pair sits at neither port's own value.
        obj.addProperty(
            "App::PropertyIntegerList",
            "SelfReferencedPorts",
            "Data",
            "Ports reported against their own impedance rather than a number",
        )
        # Indices into Frequency, not frequencies. A float stored and read back
        # has to be matched against the axis to mean anything, and matching
        # floats is exactly the operation that goes wrong once.
        obj.addProperty(
            "App::PropertyIntegerList",
            "DiscardedPoints",
            "Data",
            "Frequency indices whose terms are not numbers: the solves disagreed"
            " about port impedance there",
        )
        # Row-major (frequency, receiving, driving), matching SParameters.s.
        obj.addProperty(
            "App::PropertyFloatList",
            "ScatteringReal",
            "Data",
            "Real part of the S-matrix, flattened (frequency, receiving, driving)",
        )
        obj.addProperty(
            "App::PropertyFloatList",
            "ScatteringImag",
            "Data",
            "Imaginary part of the S-matrix, flattened the same way",
        )
        # Complex, and split like every other complex array here. An undriven
        # port keeps its own measured impedance as its reference - moving it
        # needs the unmeasured column - and a microstrip's is genuinely
        # complex, so a float list would quietly discard the imaginary part.
        obj.addProperty(
            "App::PropertyFloatList",
            "ReferenceReal",
            "Data",
            "Real part of the impedance the matrix is referenced to, flattened (frequency, port)",
        )
        obj.addProperty(
            "App::PropertyFloatList", "ReferenceImag", "Data", "Imaginary part of the same"
        )
        # What the solver measured at each port, which is *not* the reference
        # the matrix is normalised to. Kept because it is the only record of
        # what a microstrip's Z0 actually did across the band, and it cannot be
        # recovered from a renormalised matrix.
        obj.addProperty(
            "App::PropertyFloatList",
            "PortImpedanceReal",
            "Data",
            "Real part of the impedance each port measured, flattened (frequency, port)",
        )
        obj.addProperty(
            "App::PropertyFloatList",
            "PortImpedanceImag",
            "Data",
            "Imaginary part of the impedance each port measured, flattened the same way",
        )

        obj.addProperty(
            "App::PropertyInteger", "Ports", "Result", "How many ports the matrix covers"
        )
        obj.addProperty(
            "App::PropertyInteger", "Points", "Result", "How many frequency samples it holds"
        )
        obj.addProperty(
            "App::PropertyFrequency", "FrequencyStart", "Result", "Lowest frequency in the result"
        )
        obj.addProperty(
            "App::PropertyFrequency", "FrequencyStop", "Result", "Highest frequency in the result"
        )
        obj.addProperty(
            "App::PropertyString",
            "Reference",
            "Result",
            "The impedance the matrix is referenced to",
        )
        obj.addProperty(
            "App::PropertyString",
            "Provenance",
            "Result",
            "Where this matrix came from: solver, version, envelope digest, as JSON",
        )

        for name in BULK:
            obj.setEditorMode(name, 2)  # hidden
        for name in SUMMARY:
            obj.setEditorMode(name, 1)  # read-only

        obj.Proxy = self

    def execute(self, obj):
        """Nothing. A result records a solve that already happened.

        Recomputing must never look like re-solving: the numbers here cost
        minutes of FDTD, and an ``execute`` that touched them would either
        silently discard a run or silently start one.
        """

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def createEMSParameters(doc=None):
    """An empty result object. Filling it is :func:`store`'s job."""
    doc = doc or FreeCAD.ActiveDocument
    obj = doc.addObject("App::FeaturePython", "EMSParameters")
    EMSParameters(obj)
    obj.Label = "S-Parameters"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMSParameters")
    return obj


def store(obj, result: SParameters):
    """Write an assembled matrix onto the document object. Returns ``obj``.

    Everything is converted element by element to plain Python floats and ints
    on the way in, and that is **load-bearing**, not tidiness. Measured on
    FreeCAD 1.1.1: ``App::PropertyFloatList`` handed a numpy array stores *N
    copies of its last element* - ``np.arange(5.0)`` reads back as
    ``[4.0, 4.0, 4.0, 4.0, 4.0]``. The corruption carries no signature: the
    length is right, the elements are plain floats, and the last sample is
    right, so a frequency axis reopens as a flat line at the top of the band and
    looks like a physics result. The stub in ``tests/conftest.py`` stores
    whatever it is handed, so nothing short of a real document catches it.
    """
    frequency = np.asarray(result.frequency, dtype=float)
    scattering = np.asarray(result.s, dtype=complex)
    reference = np.asarray(result.reference, dtype=complex)
    measured = np.asarray(result.measured_impedance, dtype=complex)

    obj.Frequency = [float(value) for value in frequency]
    obj.PortNumbers = [int(number) for number in result.port_numbers]
    obj.DrivenPorts = [int(number) for number in result.driven]
    obj.DerivedPorts = [int(number) for number in result.derived]
    obj.SelfReferencedPorts = [int(number) for number in result.self_referenced]
    obj.DiscardedPoints = [int(index) for index in result.discarded]
    obj.ScatteringReal = [float(value) for value in scattering.real.ravel()]
    obj.ScatteringImag = [float(value) for value in scattering.imag.ravel()]
    obj.ReferenceReal = [float(value) for value in reference.real.ravel()]
    obj.ReferenceImag = [float(value) for value in reference.imag.ravel()]
    obj.PortImpedanceReal = [float(value) for value in measured.real.ravel()]
    obj.PortImpedanceImag = [float(value) for value in measured.imag.ravel()]
    obj.Provenance = _provenance_json(result.provenance)

    obj.Ports = result.ports
    obj.Points = int(frequency.size)
    obj.FrequencyStart = float(frequency[0]) if frequency.size else 0.0
    obj.FrequencyStop = float(frequency[-1]) if frequency.size else 0.0
    obj.Reference = result.reference_description()
    return obj


def _provenance_json(provenance: dict) -> str:
    """Provenance as JSON, and never at the cost of the matrix.

    An adapter is free to record whatever it measured, and this is the one
    field where that freedom meets a serialiser. It goes wrong by degrading
    rather than raising:

    * a **value** JSON cannot express becomes its ``repr`` (``default=str``);
    * a **key** JSON cannot express is stringified before it reaches the
      serialiser at all. ``default=`` alone covers values and not keys, so one
      odd key raises ``TypeError`` out of ``store`` and loses a matrix that cost
      minutes of FDTD;
    * keys that cannot be *compared* - a ``str`` and an ``int`` in one dict -
      defeat ``sort_keys``. Stringifying settles that too, and the fallback
      catches whatever a nested value still manages.

    Integer keys come back from JSON as strings anyway. That is JSON's rule
    rather than a loss, and :func:`load` does not pretend otherwise; doing it
    here only makes the top level behave like every level below it.
    """
    safe = {str(key): value for key, value in provenance.items()}
    try:
        return json.dumps(safe, sort_keys=True, default=str, skipkeys=True)
    except (TypeError, ValueError):
        return json.dumps({"unserialisable": repr(provenance)})


def load(obj) -> SParameters:
    """Rebuild the neutral result object from what the document holds.

    Refuses loudly on a length that does not add up. A stored matrix is only
    ever written by :func:`store`, so a mismatch means the document was edited
    or truncated - and reshaping whatever is there would hand physics code an
    array of the right shape and the wrong contents, which is the one failure
    mode with no symptom.
    """
    frequency = np.asarray(list(obj.Frequency), dtype=float)
    numbers = tuple(int(number) for number in obj.PortNumbers)
    if not numbers:
        raise ResultError(
            f"{getattr(obj, 'Label', '?')!r} holds no S-matrix yet. Run the analysis to fill it"
        )

    points, ports = frequency.size, len(numbers)
    real = _list(obj, "ScatteringReal", points * ports * ports)
    imag = _list(obj, "ScatteringImag", points * ports * ports)
    scattering = (real + 1j * imag).reshape(points, ports, ports)

    reference = (
        _list(obj, "ReferenceReal", points * ports)
        + 1j * _list(obj, "ReferenceImag", points * ports)
    ).reshape(points, ports)
    measured = (
        _list(obj, "PortImpedanceReal", points * ports)
        + 1j * _list(obj, "PortImpedanceImag", points * ports)
    ).reshape(points, ports)

    try:
        provenance = json.loads(obj.Provenance) if obj.Provenance else {}
    except ValueError:
        provenance = {}

    return SParameters(
        frequency=frequency,
        s=scattering,
        port_numbers=numbers,
        reference=reference,
        measured_impedance=measured,
        # Empty means "every column", which is what an object filled by an
        # older ``store`` would mean - though it will not get this far: it
        # has no ``ReferenceReal`` either, and dies on that first. There is no
        # migration and none is wanted before v1.0; this is only so that a
        # hand-built object with nothing to say reads as complete rather than
        # as empty.
        driven=tuple(int(n) for n in obj.DrivenPorts) or None,
        derived=tuple(int(n) for n in obj.DerivedPorts),
        self_referenced=tuple(int(n) for n in obj.SelfReferencedPorts),
        discarded=tuple(int(i) for i in obj.DiscardedPoints),
        provenance=provenance,
    )


def _list(obj, name, expected) -> np.ndarray:
    values = np.asarray(list(getattr(obj, name)), dtype=float)
    if values.size != expected:
        raise ResultError(
            f"{getattr(obj, 'Label', '?')!r} holds {values.size} values in "
            f"{name}, but its {len(obj.PortNumbers)} ports over "
            f"{len(obj.Frequency)} frequencies need {expected}. The result was "
            "damaged; run the analysis again"
        )
    return values
