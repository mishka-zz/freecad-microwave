# Where the domain ends, and where the absorber goes

An FDTD grid is finite and the world is not, so every run needs a box to solve
in and something at its faces to stop the wave coming back. This page is about
deciding where that box's walls are, which sounds like arithmetic and is not:
the answer depends on the mesh, and the mesh depends on the answer.

Read [Deciding where the grid lines go](sizing-field.md) first if you have not.
It settles how a cell size is chosen; this page is about a length that has to be
committed to before any cell exists.

## The absorber is grid, not empty space

A perfectly matched layer is a number of *cells* at the end of an axis, in which
the material is lossy. It is not free space around the model - it is part of the
grid and it eats lines from the end of every axis it is declared on. So its
thickness in millimetres is not a property of the absorber at all. It is the
number of cells times whatever pitch the mesher ends up laying there.

That is the whole difficulty. The depth has to be reserved before meshing, and
what it will actually measure is not known until after.

## Two ways to pad a face, and they are counted differently

Each face of the domain is padded either by a count of air cells or by
`THROUGH`.

**A cell count** leaves air between the structure and the absorber, and the
domain grows outward. The absorber sits beyond it, in air. This is what an
antenna wants: room for the near field to become a far field before anything
absorbs it.

**`THROUGH`** is not more air. The structure continues out through the absorber,
so the domain is pulled *inward* by the absorber's depth and the absorber lands
on the structure. That is what makes a transmission line infinite - substrate and
trace run into the PML, and the line never sees an end. Give a line air at its
ends instead and it radiates off an open circuit, contaminating every impedance
extracted from it with the reflection.

The two are counted in different cells, and the reason is what each one is
padding.

Outward padding is counted in the bulk size in *vacuum*, because what is being
padded is air. A clearance whose job is to let a wave in air decay must not
shrink as the substrate gets slower.

A `THROUGH` face is counted in the size of the material that will be **at the
wall**, because that is what the absorber is laid in. Counting it in a vacuum
cell over-reserves by the square root of the permittivity, and the error
compounds at low bands until the domain collapses on itself. Pulling in by the
smaller of the vacuum ceiling and the slowest material's size fails the other
way: it lets the absorber overrun the end of the structure wherever the slowest
material does not cross that wall, which puts the line inside its own PML.

## The circularity, and the way out of it

A `THROUGH` wall lands somewhere in a window, and its position depends on the
reservation, which depends on the material at the wall, which depends on where
the wall landed. Nothing can be evaluated in order.

The way out is to stop asking where the wall lands and take the **coarsest**
cell the mesher could lay anywhere in the window. Whatever material turns out to
be at the wall, its own size is no larger than that, and the pitch actually laid
is no larger again - because constraints only ever lower the sizing field, and
the cell count rounds up.

So the reservation is at or above what gets laid, and the grid can end short of
the structure but never past it. Short is waste, and is reported. Past would be a
reflector, and is refused.

### Why not the finest

Taking the finest value in the window instead would be tighter, and it is wrong.
A fine region anywhere in the window would shrink the reservation below what a
coarse region at the wall actually lays, which is exactly the overrun the
coarsest value exists to prevent.

Air counts as the ceiling, and so does a conductor: a conductor asks for no bulk
size, and the sizing field relaxes to the cap over one.

### It is a bound, not an estimate

Grading pulls the pitch down near any finer region further in, so the mesher can
lay considerably less than was reserved. The structure then overhangs the grid
wherever the wall material is slower than vacuum.

That is waste rather than error - openEMS clips to the grid - and it is reported
as a substitution rather than passed over, because a domain quietly smaller than
the drawing is the kind of thing that is never noticed afterwards.

## What counts as two faces

The mesher has a floor under its cell size, and that floor decides which faces
are two faces. Bands narrower than it are not places: the mesher would merge
their bounds, so a reservation that read a gap there would be predicting a grid
it is not going to build.
