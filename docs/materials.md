# Materials

A material is a document object holding *values*. Geometry gets a material
through a **binding**, which names one material and the solids made of it.

## Material types

| `MaterialType` | What it is | Properties read | openEMS |
|---|---|---|---|
| `Dielectric` | A substrate. Lossless, or lossy through `LossTangent` | `Permittivity`, `Permeability`, `LossTangent`, `MeasuredAt` | yes |
| `PEC` | A perfect conductor, no loss and no thickness to resolve | none | yes |
| `ConductingSheet` | A thin conductor with real loss, thickness as a property | `Conductivity`, `Thickness` | yes |
| `FrequencyDependentDielectric` | A substrate whose permittivity varies across the band | - | refused |

A dispersive material needs a fitted Debye or Lorentz pole set, which this
adapter does not write. It is refused by name rather than flattened to one
permittivity, because a flattened one is a wrong answer that looks like a right
one. Use a `Dielectric` quoted at the frequency of interest.

### Loss, and the frequency it was quoted at

openEMS carries dielectric loss as a conductivity. A loss tangent is converted
once, at band centre:

```
kappa = 2 * pi * f_centre * eps0 * eps_r * tan(delta)
```

That is openEMS' own convention, and it is an approximation with a name: a
fixed conductivity gives a loss tangent that falls as 1/f, so the model is
exact at band centre and drifts either side of it.

`MeasuredAt` is the frequency the permittivity and loss tangent were quoted at.
It is not decoration - nothing below the conversion can recover where the number
was true, and pre-flight compares it against the band being solved. FR-4
characterised at 1 GHz and solved to 20 GHz gets a warning saying so.

A loss tangent without a `MeasuredAt` is not a physical quantity, it is a number
somebody wrote down. Catalogs are required to state one wherever loss is
nonzero.

The band asks the same thing a second time, and `MeasuredAt` says nothing about
it: however well the loss tangent was quoted, one conductivity is one loss
tangent at one frequency. What the model carries is the declared figure scaled
by `f_centre / f`, so the bottom of a wide band is that many times lossier than
the material - a sweep from 50 MHz to 10 GHz is centred near 5 GHz and a
hundredfold too lossy at its own bottom end. Pre-flight warns once that factor
reaches four, which a band of 7:1 or wider does, and says what the factor is.

Both ends are wrong, and only one of them looks it. A fixed conductivity
attenuates the same at every frequency while a dielectric's loss rises with it,
so the two ends are out by the same amount of loss in opposite directions: the
bottom carries far more loss than it should, the top a little under half of
what it should. The ratio is alarming at the bottom and the missing decibels
are at the top.

Nothing you can set fixes it: narrow the band, split the sweep into studies, or
read the ends knowing which way each is wrong.

### Conducting sheets

A `ConductingSheet` is copper drawn as a zero-thickness sheet, carrying its
thickness as a property. openEMS models it as a surface impedance rather than
as metal to be meshed, which is what saves the cells.

Two separate questions are asked about one, and they are easy to read as one
question:

- **Is it thin against the cell that holds it?** A sheet thicker than the cell
  straddling it is not a sheet. This one is about the mesh, and refining fixes
  it.
- **Does openEMS have fitted surface-impedance coefficients for it at the top
  of the band?** This one is not about the mesh, and refining does not touch it.

A conducting sheet with `Thickness` left at zero has no surface impedance to
model and behaves as a **perfect conductor**. That is a warning rather than a
refusal - it is a legitimate thing to ask for - but it is rarely what was
meant when a conductivity has been typed in beside it.

## Catalogs

**Add Material from Catalog...** picks from every catalog installed on the
machine. A catalog is a TOML file - data, read with `tomllib`, never executed -
holding a `[catalog]` header and a list of `[[material]]` entries.

Catalogs are searched in this order, first mention winning:

1. the ones shipped with the workbench,
2. `<FreeCAD user data>/Microwave/materials/`,
3. anything listed in the parameter group `Mod/Microwave`, key
   `MaterialCatalogPaths`,
4. anything on `$MICROWAVE_MATERIAL_PATH`.

Several catalogs are live at once by design - a board house's laminates
beside the generic nominal ones. A material is identified as
`catalog:material`, so two catalogs both defining FR-4 give `generic:fr4` and
`myfab:fr4` with no collision to resolve.

One malformed catalog does not cost the others: it is reported and the rest
load. Inside a single file the rule is the opposite - one bad entry refuses the
whole file - because a silently missing *material* is indistinguishable from one
the vendor never shipped, while a missing *catalog* is obvious the moment
anybody looks for it.

### A catalog is a starting point, not a link

Picking a material **copies its values into the document**. Nothing is
consulted at solve time, which is what lets a `.FCStd` solve unchanged on a
machine with no catalogs installed at all.

Where the values came from is recorded beside them, read-only, in the
Provenance group:

| Property | |
|---|---|
| `Source` | `catalog:material` |
| `SourceCatalog` | The catalog's name and version when it was picked |
| `SourceDigest` | A fingerprint of the values as the catalog stated them |
| `Description` | What the catalog says the material is |

A measured laminate typed in as 4.15 over the catalog's 4.3 stays 4.15 for ever.
The digest then says the material has been edited since it was imported, which
is information rather than a fault.

Picking the same catalog entry twice makes a second material and says so. Both
readings are real - a mis-click, or a stackup wanting FR-4 on both sides of a
core with their own numbers - and nothing but the engineer can tell them apart.

### Writing a catalog

Copy the shipped catalog and change its `[catalog] id`, or two copies refuse to
load beside each other. Units are millimetres, hertz and S/m, matching the
document.

Nothing that changes physics has a default: `epsilon_r` is required on a
dielectric even when it is 1.0. A key that means nothing for that kind - a loss
tangent on a PEC - is a refusal rather than an ignored field. A typo comes back
as a line number and a suggestion, not as a permittivity of 1.

## Binding a material to geometry

Select the material and the solids it is made of, in any order, and press
**Bind Material to Shape**. Order says nothing here and is not read: one of the
picks is a material and the rest are geometry, so the pick tells its own halves
apart. (A port is the opposite case - see [Ports](ports.md).)

One binding carries one material and any number of solids or faces. Use one
binding per material.

**A binding is how geometry gets into the simulation at all.** Anything nothing
points at is absent from the run - see [Drawing the device](geometry.md).

**Air is not modelled.** Everything the geometry does not fill is vacuum, so
there is no background material to set and nothing to draw for the space around
the board. The catalog carries an air entry for the case where a volume of it
has to exist as an object - a cavity waiting to be filled with something else -
and an ordinary study never needs it.

A binding with a link left empty is still created, and translation refuses it by
name later: half a binding with the other half to fill in is a better place to
stand than no binding and an error message.

Bound solids take the material's `Color` in the 3D view, so the model is
readable at a glance. Where a solid is bound twice, the last binding in document
order is the one that shows.
