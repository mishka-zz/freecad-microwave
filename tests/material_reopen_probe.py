# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Pick materials from a catalog with a table, save, close and reopen them.

Run by ``test_material_reopen`` under ``freecadcmd``, with
``MATERIAL_REOPEN_OUT`` naming where to write. One material is left as picked,
one has its permittivity and its frequency typed over, and one loses its table
before the save, as a document written before the table existed would. What
each reads before the save and after the reopen, and what a study at each row's
band solves it with, goes into ``manifest.json``, written last - see
``tests/conftest.py``'s ``probe_manifest``.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave.Objects.materials import create_from_entry, createEMMaterialBinding  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.materials import solved  # noqa: E402
from Microwave.Solvers.properties import out_of_step  # noqa: E402
from tests.material_table import (  # noqa: E402
    CATALOG,
    ENTRY,
    FINE,
    HIGH_BAND,
    LOW_BAND,
    TYPED_FREQUENCY,
    TYPED_PERMITTIVITY,
    centre,
)


def read(material, reached_from):
    """What the material holds, and what each band solves it with.

    :param reached_from: What a refusal is asked to search, which reaches the
        material only through the binding naming it, as a study's members do.
        It is asked for the material without its table alone.
    """
    held = {
        "permittivity": float(material.Permittivity),
        "loss tangent": float(material.LossTangent),
        "measured at": float(material.MeasuredAt.Value),
        "source": material.Source,
        "digest": material.SourceDigest,
    }
    if reached_from:
        try:
            out_of_step(reached_from)
        except TranslationError as refusal:
            held["refused"] = str(refusal)
        return held
    try:
        held["table"] = [
            list(material.DispersionFrequency),
            list(material.DispersionPermittivity),
            list(material.DispersionLossTangent),
        ]
        for name, band in (("low", LOW_BAND), ("high", HIGH_BAND)):
            found = solved(material, centre(band))
            held[name] = [found.permittivity, found.loss_tangent, found.measured_at]
    except AttributeError:
        held["unread"] = True
    return held


def main(out):
    os.makedirs(out, exist_ok=True)
    doc = FreeCAD.newDocument("reopen")
    picked = ENTRY.at(centre(LOW_BAND))
    materials = {name: create_from_entry(doc, picked, CATALOG) for name in ("kept", "typed", "old")}
    materials["fine"] = create_from_entry(doc, FINE.at(centre(LOW_BAND)), CATALOG)
    materials["typed"].Permittivity = TYPED_PERMITTIVITY
    materials["typed"].MeasuredAt = TYPED_FREQUENCY
    materials["old"].removeProperty("DispersionFrequency")
    binding = createEMMaterialBinding("Binding")
    binding.Material = materials["old"]
    labels = {name: material.Label for name, material in materials.items()}
    answer = {"picked digest": picked.digest(), "before": {}, "after": {}}

    def everything(doc):
        [bound] = doc.getObjectsByLabel("Binding")
        for name, text in labels.items():
            [material] = doc.getObjectsByLabel(text)
            yield name, read(material, [bound] if name == "old" else [])

    answer["before"] = dict(everything(doc))
    path = os.path.join(out, "reopen.FCStd")
    doc.saveAs(path)
    FreeCAD.closeDocument(doc.Name)
    answer["after"] = dict(everything(FreeCAD.openDocument(path)))

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2)


main(os.environ["MATERIAL_REOPEN_OUT"])
