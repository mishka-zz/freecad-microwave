# Internals

Written for somebody changing this workbench. The pages one directory up are
for somebody using it.

These assume no knowledge of the code: not its layout, not its vocabulary, and
not the reasoning that settled any of it. If a page here only makes sense to a
reader who has already read the module it describes, it has failed.

| Page | What it covers |
|---|---|
| [Spending a length across three axes](cell-allocation.md) | Given a length the grid must resolve, how large each axis's cell may be |
| [Deciding where the grid lines go](sizing-field.md) | Anchors, the sizing field, placement by arclength, seams, and symmetry |
| [Measuring what the grid has to resolve](feature-size.md) | Getting those lengths out of a CAD model, and which measurement finds which |
| [How much of a conductor's width the grid has to keep](conductor-width.md) | Where the bar came from, why the count of elements across is not it, and how sizing the face element holds it cheaply |
| [Where the domain ends, and where the absorber goes](domain-and-absorber.md) | Reserving a depth in cells before there are any cells |
| [Building one S-matrix out of several runs](s-matrix-from-runs.md) | Making columns from separate solves comparable, and spending a declared symmetry |
| [Getting a velocity out of a transmission phase](velocity-from-phase.md) | Turning phase against frequency into distance on an axis |

## What they are for

This workbench turns a device drawn in FreeCAD into an answer from an
electromagnetic solver. Most of that is ordinary programming and explains
itself from the code.

The mesher does not. openEMS solves Maxwell's equations on a grid of boxes, and
deciding where the lines of that grid go - given arbitrary geometry, a
frequency band, and a solver that will silently return a plausible wrong number
rather than complain - is most of the difficulty in the workbench. The rules
that decide it are short. What established them is not, and it is not
recoverable from the code afterwards.

That is the gap these pages fill. Code states what it does. It cannot state why
that is the correct thing to do, and a reader who cannot tell a derived choice
from an arbitrary one will eventually simplify straight through one.

## What belongs here, and what does not

Here:

- the working behind a rule the code states - the derivation, the alternatives
  that were tried, and why they fail;
- the shape of a problem that no single module owns.

Not here. Each of these stays in the code, where it cannot drift out of sight
of what it describes:

- what a function takes, returns and raises. That is its docstring.
- why one particular line is written the way it is. That is a comment beside
  the line.
- how openEMS, FreeCAD or the CAD kernel behaves. That is a comment beside the
  code that depends on it, carrying a file and a line in the other project so
  it can be checked.
- how the code came to be this way. Git holds that.

The rule stays where the code is; the working moves here.

## Why a directory of documents is allowed to exist

This project has thrown a document away for being unread. `tests/INDEX.md`
listed every test in the suite and `tests/test_index.py` checked the list was
complete. Both were deleted, because a document nobody reads does not stop
being stale - it stops being noticed.

The difference is that `INDEX.md` could be *derived*. Its whole content was
recoverable by asking the test runner, so maintaining it by hand was
duplication, and duplication rots. A derivation cannot be recovered from
anything. It is the one kind of writing that is genuinely lost if it is not
written down, which is exactly why it earns a file.

That is the test for a page here: if a program could regenerate it, it does not
belong.

They are kept from drifting anyway. Where the code cites a page, a test
checks that the page and the section still exist, so a document cannot quietly
become an orphan. And a page holds no figures that anything re-measures -
measured numbers live on the lines the test suite prints, in assertions, or
beside the code that produced them, never in prose.

## How they are written

Start to finish, by a reader who has not seen the code, in order: the problem,
why the obvious approach fails, then the approach taken. The paragraph
explaining why the obvious thing does not work is usually the most useful one
on the page.

Terms are defined the first time they appear. A page may assume its reader
knows engineering and mathematics; it may not assume they know what this
project means by an anchor, a preference, or a sizing field.
