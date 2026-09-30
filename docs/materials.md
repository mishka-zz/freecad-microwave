# Materials

An `EMMaterial` object defines electromagnetic properties (permittivity,
permeability, loss tangent, conductivity, thickness). Geometry is associated
with materials using `EMMaterialBinding` objects, which link one material to
one or more shapes or solid faces.

## Material types

| `MaterialType` | Description | Active properties | openEMS support | Palace support |
|---|---|---|---|---|
| `Dielectric` | Substrates and isolators. Lossless, or lossy via `LossTangent` or `Conductivity` | `Permittivity`, `Permeability`, `LossTangent`, `Conductivity`, `MeasuredAt` | Supported | Supported |
| `PEC` | Perfect Electric Conductor. Zero loss and zero thickness | None | Supported | Sheets and bodies. A body leaves the region, and its faces carry the conductor |
| `ConductingSheet` | Thin lossy conductor with thickness modeled as a property | `Conductivity`, `Thickness`, and on Palace `Permeability` | Supported | Sheets only. A body is refused |
| `FrequencyDependentDielectric` | Substrate with frequency-dispersive permittivity | - | Rejected | Rejected |

For conductors (`PEC` and `ConductingSheet`), keep `Permittivity` and
`Permeability` at 1.0 and `LossTangent` at 0.0, and keep a `PEC`'s
`Conductivity` at 0.0. A perfect conductor has no loss, and both backends refuse
a `PEC` carrying a conductivity. A `ConductingSheet` switched to `PEC` keeps its
conductivity. A `ConductingSheet` of zero `Thickness` is refused on both. OpenEMS uses only electrical
conductivity for conductors. Non-unity permittivity or permeability values
on conductors alter grid cell sizing calculations and are refused during
translation, before the pre-flight checks run. Palace refuses the same values,
because a conductor there is a condition on a face and nothing reads them. The
one exception is the `Permeability` of a `ConductingSheet`, which Palace uses for
the skin depth.

Dispersive dielectrics requiring multi-pole Debye or Lorentz models are not
currently supported by either adapter. Dispersive material definitions
are rejected during validation rather than approximated with a single value.
Use a constant `Dielectric` specified at the center frequency of interest.

### Dielectric loss and characterization frequency on openEMS

Palace uses `LossTangent` as declared at every frequency, and does not compare
`MeasuredAt` with the band: see [Loss on Palace](#loss-on-palace). The rest of
this section describes openEMS.

OpenEMS models dielectric loss using an equivalent constant conductivity
calculated at the sweep center frequency `f_centre`:

```
kappa = 2 * pi * f_centre * eps0 * eps_r * tan(delta)
```

Because equivalent conductivity is held constant throughout the time-domain
run, the effective loss tangent scales as `1/f`. The loss model is exact at
`f_centre` and deviates toward the sweep band edges.

The `MeasuredAt` property specifies the characterization frequency of the
loss tangent. Pre-flight compares it with the simulation center frequency and
generates a warning if the difference is significant (for example, FR-4
characterized at 1 GHz but simulated up to 20 GHz). For a material that follows
a measurement table (see below), the frequency compared is that of the row the
study solves at, which the warning names.
Omitting `MeasuredAt` on a lossy dielectric emits a warning prompting you to
record the characterization frequency.

For wideband sweeps (frequency ratio of 7:1 or greater), the fixed
conductivity model overestimates loss at the low end and underestimates loss
at the high end. For high-ratio sweeps, consider splitting the simulation
into narrower sub-band studies.

### Conducting sheets

A `ConductingSheet` represents thin metal (such as copper foil) modeled as a
2D surface impedance rather than a volumetric 3D mesh, avoiding the need
for fine grid cells across the foil thickness.

Palace's own rules for a `ConductingSheet` are under [Loss on
Palace](#loss-on-palace).

Validation requirements for `ConductingSheet` on openEMS:
1. **Planar 2D surface**: The bound geometry must be a 2D planar face or
   sheet. Volumetric 3D solids cannot use `ConductingSheet`; openEMS applies
   surface impedance only to 2D elements.
2. **Sheet thickness vs cell size**: The specified `Thickness` must be
   smaller than the grid cell holding the sheet. If foil thickness exceeds
   the cell dimension, model the conductor as a 3D solid (`Part::Box`) with
   `PEC`.
3. **Valid surface impedance fit**: OpenEMS must be able to fit surface
   impedance coefficients across the simulation band.
4. **Clear of the domain boundary**: A sheet must not lie in the plane of a
   grid line the boundary holds - the outermost line behind a wall, the outer
   two behind `Mur`, anywhere in a `PML`. OpenEMS solves a
   sheet there as a perfect conductor and does not say so. Leave air between
   the sheet and that side of the domain, or bind it to a `PEC`. A sheet that
   only runs into the boundary, as a trace runs into an absorber, is solved.

On openEMS, curved lossy foils are not supported because 3D curved surfaces are
automatically extruded into volumetric shells by the adapter. Model curved
conductors as `PEC`.

A `ConductingSheet` with `Thickness` set to 0 is refused, because openEMS
would build it as a lossless `PEC`.

### Loss on Palace

Palace and openEMS model some materials differently. Where they do, a result
from each answers a different question, and the difference does not shrink with
a finer mesh. Each result states how its backend modelled each lossy material,
in its `Modelled` property and under its chart - see
[Results](results.md#stored-properties).

**Dielectric loss.** Palace uses the `LossTangent` as declared at every
frequency in the sweep. OpenEMS converts it into one conductivity at the band
centre, as described above. On a wide sweep, Palace's loss is the declared
number and openEMS' falls as `1/f`. A `Dielectric`'s `Conductivity` is held
fixed across the band on both backends, which makes it the one loss both model
alike. OpenEMS adds it to the conductivity a `LossTangent` becomes; Palace takes
one or the other, and refuses a material carrying both.

**Conducting sheets.** Palace applies the surface impedance of the metal to the
faces the sheet is bound to, from the skin depth that the `Conductivity` and
`Permeability` give at each frequency.

- A sheet inside the region carries the impedance on each of its two faces, and
  no field passes through it. This models metal that stops the field. OpenEMS
  models a sheet as a single plane driven by the net current through it. The two
  answers differ where the magnetic field is the same on both faces of the sheet:
  Palace then reports loss on each face and openEMS reports none.
- A sheet thin enough to let a thousandth of the field through is refused. A
  sheet of sheet resistance `R`, which is `1 / (Conductivity * Thickness)`, lets
  through up to `2R / (2R + Z)` of a wave, where `Z` is the wave impedance of
  anything filling the model. Palace would let none of it through.
  A resistive film is refused for this reason. Copper at 5.8e7 S/m is solved
  from a tenth of a micrometre up, in a model filled with air. The bound is for
  a wave meeting the sheet squarely, and a guide's TE modes let less through.
- A sheet whose `Conductivity` carries less than a thousand times its
  displacement current at the top of the band is refused. It is a lossy
  dielectric rather than a conductor. Draw it as a body and bind that to a
  `Dielectric` carrying the conductivity.
- `Thickness` is used where the metal is thin compared with the skin depth at
  the bottom of the band. It is omitted where the metal is 40 skin depths or
  more, because it has no effect there.
- A sweep so wide that a thin sheet is thin at the bottom and very thick at the
  top is refused. Palace's thickness model gives no number there. Split the
  sweep into narrower studies.
- A sheet with a thickness is refused if one binding covers faces both inside
  the region and where it ends. Palace doubles the thickness of a sheet where the
  region ends. Bind the two sets of faces separately.
- The rest of the boundary, which nothing was drawn on, remains a perfect
  conductor. The run log says so.

**Metal that closes off part of the region.** No field crosses a sheet of metal
on Palace, so a skin of it closed round a space cuts that space out of the
problem the rest of the region poses. The part behind the skin is driven by
nothing: the run meshes and solves it and reports no number measured on it, and
at a resonance of its own the problem there has no single answer. Palace
therefore refuses a part of the region that no port stands on, naming what
closes it and where it is. A `PEC` and a `ConductingSheet` both close a part
this way; a cavity with a port in it is solved as before.

A post drawn as the surface of a body is the usual way to reach the refusal.
Bind the body itself to `PEC` instead: Palace then takes it out of the region,
and the faces it leaves carry the conductor. This works in a study open to free
space too, where the air round the body is the reserved air. A body of a
`ConductingSheet` is refused, so metal with loss is drawn as a sheet.

Palace 0.18.1 can hang on more than one process where a wave port
meets a sheet of finite conductivity and no material on the port's face is
lossy. In a run with a `ConductingSheet` and a wave port, the adapter therefore
gives each lossless material a loss tangent of `RANK_LOSS_TANGENT` in
`Microwave/Solvers/palace/config.py`, and the log states the value. The loss is
far below the precision of the numbers Palace solves with. The answer agrees
with the one solved without it to within the solver's tolerance. The run uses
the processes `Processes` asks for.

## Catalogs

The **Add Material from Catalog...** command provides access to material
libraries defined in TOML format. Material catalogs are read using standard
data parsers and are not executed as code.

Catalogs are searched in the following priority order:
1. Built-in catalogs shipped with the workbench.
2. `<FreeCAD user data>/Microwave/materials/`.
3. Directories listed in parameter `Mod/Microwave/MaterialCatalogPaths`.
4. Directories listed in environment variable `$MICROWAVE_MATERIAL_PATH`.

Paths in `MaterialCatalogPaths` and `$MICROWAVE_MATERIAL_PATH` use the
system path separator (`:` on POSIX, `;` on Windows) and can point to
directories or individual `.toml` files.

Materials are identified as `catalog_id:material_id` (e.g. `generic:fr4`
and `vendor:fr4`), preventing naming collisions across libraries.

### Material assignment and provenance

Selecting a material from a catalog copies its property values directly
into the FreeCAD document. Simulations do not require external catalog
files at runtime, allowing `.FCStd` files to be shared portably.

If a catalog entry includes a measurement table, the material carries the
whole table in the read-only properties `DispersionFrequency`,
`DispersionPermittivity` and `DispersionLossTangent`. The entry's headline
values are a row of it where the catalog quotes them at a frequency the table
does not hold. `Permittivity`, `LossTangent` and `MeasuredAt` show the row
nearest the centre of the first study's band as it was when the material was
picked, or the headline values where the document had no study yet. Each study
solves at the row nearest its own band centre, so one material serves a
low-band and a high-band study over the same board. The row is taken as
measured and never interpolated.

Material provenance is stored in read-only properties:
- `Source`: Catalog identifier (`catalog:material`).
- `SourceCatalog`: Catalog name and version at the time of selection.
- `SourceDigest`: SHA-256 hash of the catalog values when imported.
- `Description`: Description text from the catalog.

A material follows its table while its values are the ones `SourceDigest`
records. Type a different value over them in the Property Editor, and every
study solves at the values shown; `SourceDigest` stays as it was, as the record
of what the catalog stated. The comparison is of the values themselves, so a
value typed back to the catalog's own returns the material to its table, and a
dielectric's `Thickness`, which nothing reads, is not compared. A catalog dielectric in a document saved before
materials carried their table is refused by name when the study is run.
Re-create it from its catalog and bind it again to give it the table.

### Authoring custom catalogs

Custom catalogs are created by copying an existing `.toml` file and assigning
a unique `[catalog] id`. Units in catalog files are millimetres (mm),
hertz (Hz), and Siemens per metre (S/m).

Key catalog formatting rules:
- Include the `schema` line at the top of the file.
- Required physical keys per kind: `epsilon_r` for `dielectric` (even if
  1.0); `conductivity` and `thickness` for `conducting_sheet`. Optional keys
  like `mu_r` (defaults to 1.0) and `loss_tangent` (defaults to 0.0) may be
  omitted.
- Unrecognized or incompatible properties (such as `loss_tangent` on a `PEC`
  entry) are rejected during parsing.
- Measurement tables are specified using `[[material.dispersion]]` entries,
  ordered by increasing frequency.

## Manual material creation

The **Create Material** command creates a new empty `EMMaterial` object
defaulting to vacuum properties (`Dielectric`, $\varepsilon_r=1$, $\mu_r=1$).

To configure a custom material:
1. Set `MaterialType` (`Dielectric`, `PEC`, or `ConductingSheet`).
2. Enter the relevant physical properties (`Permittivity`, `LossTangent`,
   `Conductivity`, `Thickness`).
3. For lossy dielectrics, set `MeasuredAt` to the characterization
   frequency.

Materials reside at the document root, allowing multiple simulation studies
within the same document to share material definitions.

## Binding materials to geometry

To assign a material to CAD geometry:
1. Select the `EMMaterial` object and the target 3D solids or 2D faces in the
   tree or 3D viewport.
2. Click **Bind Material to Shape**.

An `EMMaterialBinding` links one material to the selected shapes or faces.
Geometry without an active material binding is ignored during simulation.

Background space is vacuum unless the mesh policy's `Medium` links a
dielectric, which then fills it (see [Meshing](meshing.md#medium)). Explicit air
volumes are not required around planar boards or open structures.

Assigned solids adopt the material's `Color` in the 3D view. When a solid is
referenced by multiple bindings, the last binding in document order sets the
viewport display color.

### Two materials in one space

Two dielectric bodies may not share a volume, and a translation that finds such
an overlap refuses it by name. Cut one out of the other. A metal body fills the
space it is drawn in. See [Geometry](geometry.md#overlaps).

Two boxes of the same material in exactly the same place draw a warning on
openEMS, which discretises one of them and reports the other as an "Unused
primitive" naming only the material. Delete the spare.

