# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solver-neutral result objects.

S-parameters, far fields, field maps, eigenmodes and scalars mean the same thing
whichever backend produced them, so they are described once, here, and every
adapter reads its own output into these. This layer is the reason solver choice
is cheap for the user.

:mod:`.sparameters` *is* reached at workbench start - ``Objects.results``
imports it, because the document object that stores a matrix is described in
terms of one - and that costs numpy and the standard library, nothing more.
The expensive part stays deferred: :mod:`._skrf` is imported as a module but
does not touch scikit-rf, which is about a second (1.13.0 pulls in pandas and
scipy, and matplotlib wherever it is installed). That happens the first time a
matrix is
assembled, written to Touchstone or turned into a ``Network`` - never on the
way in or out of a document.
"""
