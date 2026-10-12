# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The S-matrix, as something the document owns.

``EMSParameters`` is the result object for it: the N×N matrix a sweep produced,
held in the run that solved it - see ``Objects/runs.py`` - or standing in a
study on its own.

The matrix is stored in the document, as flat lists of floats, rather than as a
path to a file beside it. A ``.s2p`` beside the document is a dangling reference
the moment the ``.FCStd`` is moved, renamed or emailed, and it fails silently by
showing the previous run's numbers. ``App::PropertyFloatList`` is what FreeCAD's
own FEM result objects use for bulk numbers, saves and restores with everything
else, and compresses inside the ``.FCStd``.

Complex values are split into a real and an imaginary list rather than
interleaved. Two lists of the same length are self-evident to anything that
opens the document by hand.

No scikit-rf here. Round-tripping a matrix through the document is pure numpy,
so a document full of results opens without paying for the result library's
import. The library is reached only for a
:class:`~..Results.sparameters.SParameters` ``network()`` or a Touchstone file.

Staleness is not tracked, unlike ``EMMeshPreview``. A result is stale for too
broad a set of reasons for a cheap digest to cover honestly: any material, port
or band change moves the answer without necessarily moving a cell. What is
recorded instead is the envelope digest of every run that went into the matrix,
so the question can be answered exactly from what the object carries.
"""

import base64
import json

import FreeCAD
import numpy as np

from ..Results import modelled, touchstone
from ..Results.sparameters import ResultError, SParameters
from ._vp_hook import ViewProviderRestored

#: Properties holding the matrix itself. Hidden in the property editor: a float
#: list thousands of entries long, rendered as an editable list widget, is
#: noise, and editing one by hand could only make the object misreport a solve
#: that happened. :func:`load` is the way to read them.
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

#: Read-only, and shown: what the property editor reports about a result.
#: ``Provenance`` is here rather than hidden because it is the record that makes
#: the result falsifiable - which solver, which version, which envelope digest.
#: A user who cannot see it has to write a script to find out whether the
#: numbers in front of them came from the model in front of them.
SUMMARY = (
    "Ports",
    "Points",
    "FrequencyStart",
    "FrequencyStop",
    "Reference",
    "Modelled",
    "Provenance",
)


#: Properties a table read from a Touchstone file carries, besides the matrix.
#: Read-only: they say what came in, and the file is never read again.
FILE = ("FileName", "FileDigest", "Imported", "ReadAs", "FileComments")

#: The file as it came in, base64 of its bytes. A string property holds text,
#: and saving one turns a control character into another character, so the
#: bytes are kept in a form that survives and checked against the digest.
EMBEDDED = "Embedded"

#: What an imported table's ports are referenced to: the user's declaration.
REFERENCED_TO = "ReferencedTo"


class EMSParameters(ViewProviderRestored):
    """One N×N S-matrix against frequency, referenced to a stated impedance.

    Solver-neutral. Every backend assembles the same
    :class:`~..Results.sparameters.SParameters` and stores it through the same
    :func:`store`, and every consumer above works unchanged. Which backend it
    was is in ``Provenance``, written by the run, and :func:`backend` reads it.
    A table read from a Touchstone file is one of these too, filled by
    :func:`hold_file`, and names no backend.
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
        # Which columns were measured. FDTD drives one port per run, so a study
        # that drove port 1 only has column 1 and nothing else. The rest are
        # stored as nan, and this property records which they are.
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
        # The numbers do not say which ports these are. A port asked for 50 ohm
        # that measured 50 ohm looks identical to one referenced to itself, and
        # a mirror's derived pair sits at neither port's own value.
        obj.addProperty(
            "App::PropertyIntegerList",
            "SelfReferencedPorts",
            "Data",
            "Ports reported against their own impedance rather than a number",
        )
        # Indices into Frequency rather than frequencies. A float stored and
        # read back has to be matched against the axis to mean anything, and
        # matching floats is the operation that eventually goes wrong.
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
        # port keeps its own measured impedance as its reference, because moving
        # it needs the unmeasured column, and a microstrip's impedance is
        # genuinely complex, so a float list would quietly discard the imaginary
        # part.
        obj.addProperty(
            "App::PropertyFloatList",
            "ReferenceReal",
            "Data",
            "Real part of the impedance the matrix is referenced to, flattened (frequency, port)",
        )
        obj.addProperty(
            "App::PropertyFloatList", "ReferenceImag", "Data", "Imaginary part of the same"
        )
        # What the solver measured at each port. This is not the reference the
        # matrix is normalised to. It is kept because it is the only record of
        # what a microstrip's Z0 did across the band, and it cannot be recovered
        # from a renormalised matrix.
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

        obj.addProperty("App::PropertyInteger", "Ports", "Result", "Number of ports in the matrix")
        obj.addProperty("App::PropertyInteger", "Points", "Result", "Number of frequency points")
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
        # Two backends solving one document solve two models of its loss, and
        # the matrix cannot show which. The words are read off the run's own
        # record rather than off the materials, which may have changed since.
        obj.addProperty(
            "App::PropertyStringList",
            "Modelled",
            "Result",
            "How the solver modelled each lossy material and each face open to free space",
        )
        obj.addProperty(
            "App::PropertyString",
            "Provenance",
            "Result",
            "The run that produced this matrix: solver, version and input checksum, as JSON",
        )

        for name in BULK:
            obj.setEditorMode(name, 2)  # hidden
        for name in SUMMARY:
            obj.setEditorMode(name, 1)  # read-only

        obj.Proxy = self

    def onChanged(self, obj, prop):
        """Restate an imported table's reference where the user redeclared it
        or moved its ports onto the drawing's.

        Restore is left alone: the summary travels in the file with what it
        summarises.
        """
        if prop not in (REFERENCED_TO, "PortNumbers") or not is_imported(obj):
            return
        document = getattr(obj, "Document", None)
        if document is None or getattr(document, "Restoring", False):
            return
        try:
            obj.Reference = load(obj).reference_description()
        except ResultError:
            obj.Reference = ""

    def execute(self, obj):
        """Nothing. A result records a solve that already happened, or a file
        read in.

        Recomputing must never look like re-solving. The numbers here cost
        minutes of FDTD, and an ``execute`` that touched them would silently
        discard a run or silently start one.
        """

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


#: The provenance key naming the mesh a matrix was solved on, where the backend
#: shows its mesh in the study. See ``Objects/fem_mesh.py``'s ``IDENTITY``.
SOLVED_ON = "mesh"


def createEMSParameters(doc=None):
    """An empty result object. :func:`store` fills it in."""
    doc = doc or FreeCAD.ActiveDocument
    obj = doc.addObject("App::FeaturePython", "EMSParameters")
    EMSParameters(obj)
    obj.Label = "S-Parameters"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMSParameters")
    return obj


def is_imported(obj) -> bool:
    """Whether ``obj`` holds a table read from a file rather than solved."""
    return hasattr(obj, EMBEDDED)


def hold_file(obj, name: str, data: bytes, read: touchstone.Read, when: str):
    """Fill ``obj`` with the table read from the file ``name``, whose bytes are
    ``data``, and keep the file in it. Returns ``obj``.

    The file's ports keep the file's numbers until the user moves them onto
    the drawing's in ``PortNumbers``, which every column then follows, so none
    is recorded as driven: every column of a file holds numbers.

    :param when: the local time the file came in, with its offset from UTC.
    """
    for prop, kind, doc in (
        ("FileName", "App::PropertyString", "The file this table was read from"),
        ("FileDigest", "App::PropertyString", "SHA-256 of the file, as it came in"),
        ("Imported", "App::PropertyString", "When the file was imported, in local time"),
        ("ReadAs", "App::PropertyStringList", "How the file was read"),
        ("FileComments", "App::PropertyStringList", "The comments at the head of the file"),
    ):
        obj.addProperty(kind, prop, "File", doc)
    obj.addProperty("App::PropertyString", EMBEDDED, "File", "The file, base64 of its bytes")
    obj.addProperty(
        "App::PropertyEnumeration",
        REFERENCED_TO,
        "File",
        "What the file's ports are referenced to. Choose Port impedance for a file"
        " calibrated to each line's own impedance, even where its option line states"
        " 50 ohm",
    )
    setattr(obj, REFERENCED_TO, list(touchstone.DECLARATIONS))

    store(obj, read.result)
    obj.DrivenPorts = []
    obj.SelfReferencedPorts = []
    obj.FileName = name
    obj.FileDigest = touchstone.digest(data)
    obj.Imported = when
    obj.ReadAs = list(read.said)
    obj.FileComments = list(read.comments)
    setattr(obj, EMBEDDED, base64.b64encode(data).decode("ascii"))
    setattr(obj, REFERENCED_TO, read.suggested)
    obj.Reference = load(obj).reference_description()

    for prop in FILE:
        obj.setEditorMode(prop, 1)
    obj.setEditorMode(EMBEDDED, 2)
    obj.setEditorMode("PortNumbers", 0)
    obj.setDocumentationOfProperty(
        "PortNumbers", "The port number in this document of each of the file's ports, in file order"
    )
    # A file records no impedance measured at a port, so these hold what it
    # states, which is what the matrix is referenced to.
    for prop in ("PortImpedanceReal", "PortImpedanceImag"):
        obj.setDocumentationOfProperty(
            prop, "The reference the file states, flattened (frequency, port)"
        )
    return obj


def file_bytes(obj) -> bytes:
    """The file ``obj`` was read from, as it came in.

    :raises ResultError: what the document holds is not that file.
    """
    data = base64.b64decode(getattr(obj, EMBEDDED))
    if touchstone.digest(data) != obj.FileDigest:
        raise ResultError(
            f"{getattr(obj, 'Label', '?')!r}: the copy of the imported file kept in the "
            "document is damaged. Import the file again"
        )
    return data


def backend(obj) -> str:
    """Which backend filled ``obj``, as its run's provenance names it, or ``""``.

    The name is read off what the run recorded rather than carried beside it:
    the adapter that ran wrote it, it is saved with the matrix, and it cannot
    disagree with the numbers it came with. An object
    whose provenance is missing or unreadable names no backend.
    """
    return str(provenance(obj).get("solver") or "")


def provenance(obj) -> dict:
    """What the run that filled ``obj`` recorded, or nothing where it is unreadable."""
    try:
        said = json.loads(obj.Provenance) if obj.Provenance else {}
    except ValueError:
        return {}
    return said if isinstance(said, dict) else {}


def store(obj, result: SParameters):
    """Write an assembled matrix onto the document object. Returns ``obj``.

    Everything is converted element by element to plain Python floats and ints
    on the way in, and the conversion is required rather than tidiness. Measured
    on FreeCAD 1.1.1: ``App::PropertyFloatList`` handed a numpy array stores N
    copies of its last element, so ``np.arange(5.0)`` reads back as
    ``[4.0, 4.0, 4.0, 4.0, 4.0]``. The corruption carries no signature: the
    length is right, the elements are plain floats, and the last sample is
    right, so a frequency axis reopens as a flat line at the top of the band and
    looks like a physics result. The stub in ``tests/conftest.py`` stores
    whatever it is handed, so nothing short of a real document catches it.

    An object restored from a document saved before one of its properties
    existed comes back without it, since FreeCAD does not reconcile a restored
    object against its class. That is refused by name before anything is
    written, so the matrix the object held stays whole.
    """
    for name in (*BULK, *SUMMARY):
        if not hasattr(obj, name):
            # Export reads the bulk and the provenance, so a result missing
            # anything else can still be written to a file before it goes.
            keep = (
                ""
                if name in (*BULK, "Provenance")
                else ". To keep it, export it with Export Touchstone first"
            )
            raise ResultError(
                f"{getattr(obj, 'Label', '?')!r} has no {name}, because an earlier version "
                f"of the workbench saved it. Delete it and run again{keep}"
            )
    frequency = np.asarray(result.frequency, dtype=float)
    scattering = np.asarray(result.s, dtype=complex)
    reference = np.asarray(result.reference, dtype=complex)
    measured = np.asarray(result.measured_impedance, dtype=complex)

    obj.Frequency = [float(value) for value in frequency]
    obj.PortNumbers = [int(number) for number in result.port_numbers]
    obj.DrivenPorts = [int(number) for number in result.drivers]
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
    # From the record in hand rather than from the JSON above, which drops
    # everything where it cannot be written.
    obj.Modelled = modelled.said(result.provenance.get("modelled") or ())
    return obj


def _provenance_json(provenance: dict) -> str:
    """Provenance as JSON, and never at the cost of the matrix.

    An adapter records whatever it measured, and this field is where that meets
    a serialiser. It degrades rather than raising:

    * a value JSON cannot express becomes its ``repr`` (``default=str``);
    * a key JSON cannot express is stringified before it reaches the serialiser
      at all. ``default=`` alone covers values and not keys, so one odd key
      raises ``TypeError`` out of ``store`` and loses a matrix that cost minutes
      of FDTD;
    * keys that cannot be compared - a ``str`` and an ``int`` in one dict -
      defeat ``sort_keys``. Stringifying settles that too, and the fallback
      catches whatever a nested value still manages.

    Integer keys come back from JSON as strings anyway. That is JSON's rule
    rather than a loss, and :func:`load` treats them as strings. Stringifying
    here only makes the top level behave like every level below it.
    """
    safe = {str(key): value for key, value in provenance.items()}
    try:
        return json.dumps(safe, sort_keys=True, default=str, skipkeys=True)
    except (TypeError, ValueError):
        # The backend's name survives, being what the result is filed under: a
        # result that lost it would be passed over by the next run of the same
        # backend, which would file another beside it.
        return json.dumps(
            {"unserialisable": repr(provenance), "solver": str(provenance.get("solver") or "")}
        )


def load(obj) -> SParameters:
    """Rebuild the neutral result object from what the document holds.

    Refuses loudly on a length that does not add up. Only :func:`store` writes a
    stored matrix, so a mismatch means the document was edited or truncated.
    Reshaping whatever is there would hand physics code an array of the right
    shape and the wrong contents, a failure with no symptom.
    """
    frequency = np.asarray(list(obj.Frequency), dtype=float)
    numbers = tuple(int(number) for number in obj.PortNumbers)
    if not numbers:
        raise ResultError(
            f"{getattr(obj, 'Label', '?')!r} holds no S-matrix yet. Run Simulation to fill it"
        )

    if len(set(numbers)) != len(numbers) or min(numbers) < 1:
        raise ResultError(
            f"{getattr(obj, 'Label', '?')!r} numbers its ports {list(numbers)}. Each port "
            "needs a different positive number"
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

    result = SParameters(
        frequency=frequency,
        s=scattering,
        port_numbers=numbers,
        reference=reference,
        measured_impedance=measured,
        # Empty means every column, which is what an object filled by an older
        # ``store`` would mean. Such an object will not get this far: it has no
        # ``ReferenceReal`` either, and fails on that first. There is no
        # migration and none is wanted before v1.0. This is only so that a
        # hand-built object with nothing to say reads as complete rather than as
        # empty.
        driven=tuple(int(n) for n in obj.DrivenPorts) or None,
        derived=tuple(int(n) for n in obj.DerivedPorts),
        self_referenced=tuple(int(n) for n in obj.SelfReferencedPorts),
        discarded=tuple(int(i) for i in obj.DiscardedPoints),
        provenance=provenance,
    )
    if is_imported(obj):
        return touchstone.declared(result, str(getattr(obj, REFERENCED_TO)))
    return result


def _list(obj, name, expected) -> np.ndarray:
    values = np.asarray(list(getattr(obj, name)), dtype=float)
    if values.size != expected:
        raise ResultError(
            f"{getattr(obj, 'Label', '?')!r} is damaged: {name} holds {values.size} "
            f"values, and {len(obj.PortNumbers)} ports at {len(obj.Frequency)} "
            f"frequencies need {expected}. Run Simulation again"
        )
    return values
