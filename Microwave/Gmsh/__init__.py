# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The tetrahedral mesher, which is Gmsh and is nobody's adapter.

Several backends read a Gmsh mesh, so the portfolio wants one mesher rather
than one per backend, and it cannot live under any of them: an adapter never
imports another adapter, and the second consumer would have to. It sits in the
bottom layer instead, and ``tests/test_import_graph.py`` holds it to reaching
nothing else of this workbench - not the document objects, not the shared box
vocabulary, not a solver. What it knows is shapes, sizes and labels.

A label is a string the caller chose. Nothing here reads one, and nothing here
has a word for a port, a material or a boundary: an adapter writes its own
input against the tags this package hands back and decides there what a label
means. A mesher that reads a label has taken on the vocabulary of whichever
backend it read it for.

The mechanism it is built on is one Gmsh return value. Each drawn shape is
imported on its own, everything is fragmented in one call, and
``occ.fragment`` hands back which output entities each input became - so a
label follows its shapes through being cut up, with nothing matched by
position or by measure.

Where two shapes overlap
------------------------

A piece of the drawing carries one label, because a backend gives an element
one attribute and a file naming two stops it while it is still reading the
mesh. Overlapping shapes are the ordinary case even so - a part stands inside
the domain that holds it, and a port face stands on the wall a wall label
covers - so the fragmenting cuts the shared region into pieces of its own and
every label drawn over it claims each of them.

Which label owns that region is the caller's to say, and :class:`.Piece` carries
a priority for it. The higher number takes the piece. Nothing here decides it:
a part meant to stand inside a domain and a body mistyped into another are the
same shape from the map's side, and a mesher choosing between them would return
a well formed mesh of a drawing nobody made.

Equal priority is a refusal, which is what a caller that stated none gets, and
the refusal says which of the regions stands inside which - free from the map,
and expensive to compute anywhere else - so that a caller learns from it what
to state. What the priority did settle comes back in the answer, one record per
piece.

Where a face ends the model
---------------------------

A caller can hand a label at the dimension below the filled one a direction:
the way from its faces into the model. What the fragmenting leaves on the other
side is left out of the mesh, and the faces are then where it ends. A drawing
made for a method that measures inside the model, a guide running on past its
port plane, is meshed this way for one that measures on the boundary.

Leaving a region out changes which problem is solved, so it is done only where
what stands behind the face is the region in front of it continued, and refused
otherwise. :mod:`.ends` says what that excludes, and each part left out comes
back in the answer with where it was.

Where a face stands between two problems
----------------------------------------

Whether what is filled holds together is asked of what the faces join, so a
region a caller's condition closes off is one region here. A caller whose
condition decouples the two sides of a face says so on the label, and what is
filled is asked a second time with those faces joining nothing. The parts come
back in the answer, each naming the labels on it and where it is.

No shape is taken apart for it, and no drawing is refused over the parts it
leaves. A part standing on its own is a cavity where the caller can reach it
and a mistake where it cannot, and which one it is depends on what the caller's
labels mean. What is refused is a label that says it divides and is declared
anywhere but at the dimension below the filled one.

A body the model does not hold
------------------------------

Some conditions lie on the skin of a body whose inside is not meshed. A caller
says so on the label, and the mesher takes out what the label keeps once the
priority has settled every contest over it. The label then holds the faces the
removal leaves behind, where the model now ends.

Who keeps a piece two labels were drawn over is still the caller's to state,
so a body leaves only where its label stands above what it was drawn inside.
A face among those it leaves behind that another label was drawn over is that
label's. What another label was drawn over inside the body leaves with it and
is reported, and a label left holding nothing is refused.

What comes back instead of a mesh
---------------------------------

A caller acts differently on each.

:class:`.Refused` is ours. The request, the drawing or its labels do not
describe a mesh, nothing has been meshed, and each complaint names something the
caller wrote or where in the drawing the fault is.

:class:`.Unmeshed` is Gmsh's. It was asked and there is no mesh of the drawing:
its kernel could not cut the drawn shapes against each other (:class:`.Uncut`),
or it stopped and said why, or it finished with no error and left a region of
the filled dimension empty, or it finished and returned elements turned inside
out.
The last of those is the one nothing else would notice: the elements are
well formed and the label map is whole, and what is wrong is inside them. No
file is written in any of these - the check runs before the write - so a caller
that finds a file has a mesh.

The failures below reach neither, and both belong to whatever started the
process.

A run that will not finish cannot be ended from here. Gmsh's Python interface
offers no way to interrupt a mesh in progress: the one hook it has is a callback
for the element size, and an exception raised in that is swallowed by the
foreign-function layer, leaving the callback to return zero and Gmsh to stop on
a message about the size. So a bound on how long a mesh may take is kept by the
caller, and ending one means ending the process.

A fault inside Gmsh ends the process outright, and nothing in Python catches it,
which is the whole reason the mesher is meant to run beside the document rather
than in it. The Python interface has Gmsh throw a C++ exception on every error,
and one thrown inside a parallel loop, where the high-order optimiser reports
its failure, ends the process on a signal. The mesher has Gmsh log an error
without throwing while that pass runs, so its failure arrives as
:class:`.Unmeshed`; a fault of any other kind still ends the process.

Import discipline
-----------------

This package is written to straddle a process boundary, and does not cross one
itself: it starts nothing, and a caller that wants the mesher off its own
thread or out of its own process puts it there. Gmsh is a C++ library, so a
fault inside it ends the process it is in and a mesh that will not finish holds
whatever is waiting for it, and that is what the split below is for.

* :mod:`.vocabulary` states a request and holds an answer, and imports Gmsh
  nowhere - so a caller can build one and read one where there is no mesher.
* :mod:`.coverage` asks what a label map says, and is the same: it takes
  numbers and returns complaints. So does :mod:`.ends`, which decides what is
  left out behind a face.
* :mod:`.labels` and :mod:`.mesh` are the Gmsh half, and are the half to put
  behind a process boundary.

``tests/test_gmsh_mesher.py`` imports the first two in a child interpreter and
asserts no Gmsh arrived.
"""
