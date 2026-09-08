# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solver-neutral result objects.

S-parameters, far fields, field maps, eigenmodes and scalars mean the same thing
whichever backend produced them. They are described once, here, and every
adapter reads its own output into these. This layer keeps solver choice cheap
for the user.

``Objects.results`` imports :mod:`.sparameters` at workbench start, because the
document object that stores a matrix is described in terms of one. That import
costs numpy and the standard library, nothing more. :mod:`._skrf` is imported as
a module but does not touch scikit-rf, which pulls in pandas and scipy, and
matplotlib wherever it is installed. scikit-rf is imported the first time a
matrix is assembled, written to Touchstone or turned into a ``Network``, and
never on the way in or out of a document.
"""
