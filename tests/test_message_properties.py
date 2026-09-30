# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A message that names a property names one the user can find and edit.

A refusal that says "set ReferenceImpedance" is a claim that the property
editor shows ``ReferenceImpedance`` to the user reading it. It is false where
the name is no property, where it is the envelope's spelling of one
(``max_timesteps`` for ``MaxTimesteps``), and where the editor hides the
property or makes it read-only in the state that prints the message.

The population is every string in the workbench that is not a docstring and
holds a space, which is what a message, a log line and a tooltip have in
common. Three kinds of word in it are read:

* a word with two capitals and a lower-case letter, which is how a property is
  spelt - ``PMLCells`` as much as ``MaxTimesteps``;
* a capitalised word an f-string continues with a value, such as
  ``Padding{axis}{side}``, which has to be some property's name up to a
  capital;
* a snake-case word that is a property's name spelt the envelope's way.

A one-word property name reads as English, and nothing here tells it from prose.
Nor does anything here check that a name belongs to the object the message is
about: a name passes where any document class shows it.

The properties are the ones the document classes add, taken off constructed
objects rather than off the source, because some names are built at run time.
Each object is put through every choice of each of its enumerations, one at a
time, which is the state that hides a property today.
"""

from __future__ import annotations

import ast
import re

import pytest

from Microwave.Objects.analysis import createEMAnalysis
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding
from Microwave.Objects.mesh import (
    createEMGmshMesh,
    createEMMeshPolicy,
    createEMMeshRegion,
    createEMYeeGrid,
)
from Microwave.Objects.ports import (
    FIXED_IMPEDANCE,
    createEMPortCoaxial,
    createEMPortLumped,
    createEMPortMicrostrip,
    createEMPortRectWaveguide,
)
from Microwave.Objects.preview import createEMMeshPreview
from Microwave.Objects.results import createEMSParameters
from Microwave.Objects.solver import createEMSolverOpenEMS, createEMSolverPalace
from Microwave.Solvers.errors import TranslationError
from tests.repo import ROOT

from .test_openems_document_translation import model, solver_of

FACTORIES = (
    createEMAnalysis,
    createEMMaterial,
    createEMMaterialBinding,
    createEMMeshPolicy,
    createEMMeshRegion,
    createEMYeeGrid,
    createEMGmshMesh,
    createEMMeshPreview,
    createEMSolverOpenEMS,
    createEMSolverPalace,
    createEMPortLumped,
    createEMPortMicrostrip,
    createEMPortRectWaveguide,
    createEMPortCoaxial,
    createEMSParameters,
)

#: Words spelt like a property that are not one, and what each is.
NOT_PROPERTIES = {
    "BaseApp": "FreeCAD's exception module, named in a message about an exception",
    "ConductingSheet": "a value of a material's MaterialType",
    "EMAnalysis": "the study's class, named in a refusal about a document",
    "EMYeeGrid": "the Yee grid's class, named in a refusal about a study",
    "FreeCAD": "the application",
    "GHz": "a unit",
    "GiB": "a unit",
    "MHz": "a unit",
    "MeshParams": "the mesher's own parameters, named in an internal invariant",
    "NaN": "a value",
    "PETSc": "the library Palace solves port modes over, named as the user is told it",
    "RectWGPort": "openEMS' own class, which a message about openEMS names",
    "RuntimeWarning": "Python's warning class, in a pattern the bindings' line is read by",
    "SetupFDTD": "openEMS' own function, in a pattern its line is read by",
    "UnsupportedModel": "the kind an ERROR marker carries",
}

#: Capitalised words an f-string continues with a value that begin no property.
NOT_PREFIXES = {
    "TE": "a waveguide mode, numbered by the value",
}

#: Modules whose messages name a property the envelope's way, and why that is
#: the reader's spelling there. A document route reaching one of these names
#: states the property first, which the tests at the end hold.
SNAKE_SPOKEN = {
    "Microwave/Materials/catalog.py": "a catalog is TOML, and its keys are spelt so",
    "Microwave/Solvers/openems/model.py": "the envelope's validation names its own fields",
    "Microwave/Solvers/openems/regions.py": "the mesher's parameters name their own fields",
}

#: Properties the editor hides or makes read-only in some states and not others.
#: A message may name one of these only where the state printing it shows it,
#: and the tests below hold the messages that name one today.
SHOWN_WHERE_NAMED = ("ReferenceImpedance",)

WORD = re.compile(r"\b[A-Z][A-Za-z0-9]+\b")
SNAKE = re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b")
#: A value in an f-string, in the text the scan reads.
VALUE = "\0"


def _spelt_like_a_property(word):
    return sum(c.isupper() for c in word) >= 2 and any(c.islower() for c in word)


def _snake(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def _begins(word, name):
    """Whether ``name`` is ``word`` followed by whole capitalised parts."""
    rest = name[len(word) :]
    return name.startswith(word) and rest[:1].isupper()


def _docstrings(tree):
    kinds = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    return {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, kinds)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }


def _texts(tree):
    """Each string the module states, an f-string whole with its values marked."""
    skip = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            yield "".join(
                part.value if isinstance(part, ast.Constant) else VALUE for part in node.values
            )
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
        ):
            yield node.value


def _scan():
    """What the messages spell, by kind of word, with the modules that spell it."""
    found = {"word": {}, "prefix": {}, "snake": {}}
    for path in sorted((ROOT / "Microwave").rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        module = str(path.relative_to(ROOT))
        for text in _texts(ast.parse(path.read_text(encoding="utf-8"))):
            # A name built from values is often built on its own and put into a
            # message afterwards, so it is read whether its string holds a
            # space or not.
            for match in WORD.finditer(text):
                if text[match.end() : match.end() + 1] == VALUE:
                    found["prefix"].setdefault(match.group(), set()).add(module)
            if " " not in text:
                continue
            for match in WORD.finditer(text):
                word = match.group()
                if text[match.end() : match.end() + 1] != VALUE and _spelt_like_a_property(word):
                    found["word"].setdefault(word, set()).add(module)
            for word in SNAKE.findall(text):
                found["snake"].setdefault(word, set()).add(module)
    return found


def _choices(obj, name):
    """The values an enumeration takes, or none for any other property."""
    try:
        return list(obj.getEnumerationsOfProperty(name))
    except AttributeError:
        return []


def _modes():
    """Every property, with each editor mode each document class shows it in.

    Kept by class, because one name can belong to two: a result's
    ``FrequencyStart`` is read-only and a study's is not.
    """
    modes = {}
    for factory in FACTORIES:
        obj = factory()
        states = [
            (name, choice) for name in list(obj.PropertiesList) for choice in _choices(obj, name)
        ]
        for name, choice in [(None, None), *states]:
            if name is not None:
                setattr(obj, name, choice)
                # The stub calls no onChanged, and a class without one hides
                # nothing on an edit.
                changed = getattr(obj.Proxy, "onChanged", None)
                if changed is not None:
                    changed(obj, name)
            for prop in obj.PropertiesList:
                seen = modes.setdefault(prop, {}).setdefault(factory.__name__, set())
                seen.add(tuple(obj.getEditorMode(prop)))
    return modes


@pytest.fixture(scope="module")
def scan():
    return _scan()


@pytest.fixture
def modes(doc):
    """``doc`` makes the stub's active document the one the factories build in."""
    return _modes()


def test_every_word_a_message_spells_as_a_property_is_one(scan, modes):
    unknown = {
        word: sorted(modules)
        for word, modules in scan["word"].items()
        if word not in modes and word not in NOT_PROPERTIES
    }
    assert not unknown, unknown


def test_every_name_a_message_builds_begins_a_property(scan, modes):
    unknown = {
        word: sorted(modules)
        for word, modules in scan["prefix"].items()
        if word not in NOT_PREFIXES and not any(_begins(word, name) for name in modes)
    }
    assert not unknown, unknown


def test_no_message_spells_a_property_the_envelope_s_way(scan, modes):
    spellings = {_snake(name): name for name in modes if "_" in _snake(name)}
    spelt = {
        word: (spellings[word], sorted(set(modules) - set(SNAKE_SPOKEN)))
        for word, modules in scan["snake"].items()
        if word in spellings and set(modules) - set(SNAKE_SPOKEN)
    }
    assert not spelt, spelt


@pytest.mark.parametrize(
    "listed, kind",
    [(NOT_PROPERTIES, "word"), (NOT_PREFIXES, "prefix")],
    ids=["words", "prefixes"],
)
def test_every_listed_exception_is_still_in_a_message(scan, listed, kind):
    """A list of exceptions nothing uses any more excuses tomorrow's typo."""
    stale = sorted(word for word in listed if word not in scan[kind])
    assert not stale, stale


def test_every_module_allowed_the_envelope_s_spelling_still_uses_it(scan, modes):
    spellings = {_snake(name) for name in modes}
    using = {
        module for word, modules in scan["snake"].items() if word in spellings for module in modules
    }
    assert set(SNAKE_SPOKEN) <= using, set(SNAKE_SPOKEN) - using


def test_no_message_names_a_property_the_editor_never_offers(scan, modes):
    never = {
        word: sorted(modules)
        for word, modules in scan["word"].items()
        if word in modes and not any(() in seen for seen in modes[word].values())
    }
    assert not never, never


def test_a_property_hidden_in_some_states_is_one_the_tests_below_hold(scan, modes):
    sometimes = {
        word
        for word in scan["word"]
        if word in modes and any(() in seen and len(seen) > 1 for seen in modes[word].values())
    }
    assert sometimes == set(SHOWN_WHERE_NAMED)


class TestTheReferenceImpedanceIsShownWhereItIsNamed:
    """The openEMS refusal names ``ReferenceImpedance`` for a port referenced to
    a fixed impedance. The Touchstone refusal asks for ``ReferencedTo`` to be set
    to a fixed impedance, and that state shows it; what the refusal says is not
    read here."""

    ports = pytest.mark.parametrize(
        "factory",
        [
            createEMPortLumped,
            createEMPortMicrostrip,
            createEMPortRectWaveguide,
            createEMPortCoaxial,
        ],
        ids=lambda f: f.__name__,
    )

    @ports
    def test_the_refusal_of_a_non_positive_one_is_printed_where_it_is_shown(self, factory, doc):
        from Microwave.Solvers.openems.ports import _reference_impedance

        port = factory(doc=doc)
        port.ReferencedTo = FIXED_IMPEDANCE
        port.Proxy.onChanged(port, "ReferencedTo")
        port.ReferenceImpedance = 0.0

        with pytest.raises(TranslationError, match="ReferenceImpedance"):
            _reference_impedance(port)
        assert port.getEditorMode("ReferenceImpedance") == []

    @ports
    def test_a_fixed_impedance_shows_it(self, factory, doc):
        port = factory(doc=doc)
        port.ReferencedTo = FIXED_IMPEDANCE
        port.Proxy.onChanged(port, "ReferencedTo")

        assert port.getEditorMode("ReferenceImpedance") == []


class TestTheDocumentNamesWhatTheEnvelopeWouldSpellItsOwnWay:
    """The envelope refuses these by its own field names. From a document the
    property is refused first, by the name the editor shows.
    ``TimestepFactor`` is held the same way in
    ``tests/test_openems_document_translation.py``."""

    @pytest.mark.parametrize("name, value", [("MaxTimesteps", 0)])
    def test_a_solver_value_out_of_range_is_refused_by_its_property(self, name, value):
        from Microwave.Solvers.openems import document

        study = model().Objects[0]
        setattr(solver_of(study), name, value)
        with pytest.raises(TranslationError) as raised:
            document.problem(study)
        assert name in str(raised.value)
        assert _snake(name) not in str(raised.value)
