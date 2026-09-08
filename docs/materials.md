# Materials

An `EMMaterial` object defines electromagnetic properties (permittivity,
permeability, loss tangent, conductivity, thickness). Geometry is associated
with materials using `EMMaterialBinding` objects, which link one material to
one or more shapes or solid faces.

## Material types

| `MaterialType` | Description | Active properties | openEMS support |
|---|---|---|---|
| `Dielectric` | Substrates and isolators. Lossless or lossy via `LossTangent` | `Permittivity`, `Permeability`, `LossTangent`, `MeasuredAt` | Supported |
| `PEC` | Perfect Electric Conductor. Zero loss and zero thickness | None | Supported |
| `ConductingSheet` | Thin lossy conductor with thickness modeled as a property | `Conductivity`, `Thickness` | Supported |
| `FrequencyDependentDielectric` | Substrate with frequency-dispersive permittivity | - | Rejected |

For conductors (`PEC` and `ConductingSheet`), keep `Permittivity` and
`Permeability` at 1.0 and `LossTangent` at 0.0. OpenEMS uses only electrical
conductivity for conductors. Non-unity permittivity or permeability values
on conductors alter grid cell sizing calculations and are refused during
translation, before the pre-flight checks run.

Dispersive dielectrics requiring multi-pole Debye or Lorentz models are not
currently supported by the openEMS adapter. Dispersive material definitions
are rejected during validation rather than approximated with a single value.
Use a constant `Dielectric` specified at the center frequency of interest.

### Dielectric loss and characterization frequency

OpenEMS models dielectric loss using an equivalent constant conductivity
calculated at the sweep center frequency `f_centre`:

```
kappa = 2 * pi * f_centre * eps0 * eps_r * tan(delta)
```

Because equivalent conductivity is held constant throughout the time-domain
run, the effective loss tangent scales as `1/f`. The loss model is exact at
`f_centre` and deviates toward the sweep band edges.

The `MeasuredAt` property specifies the characterization frequency of the
loss tangent. Pre-flight checks compare `MeasuredAt` with the simulation
center frequency and generate a warning if the difference is significant
(for example, FR-4 characterized at 1 GHz but simulated up to 20 GHz).
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

Validation requirements for `ConductingSheet`:
1. **Planar 2D surface**: The bound geometry must be a 2D planar face or
   sheet. Volumetric 3D solids cannot use `ConductingSheet`; openEMS applies
   surface impedance only to 2D elements.
2. **Sheet thickness vs cell size**: The specified `Thickness` must be
   smaller than the grid cell holding the sheet. If foil thickness exceeds
   the cell dimension, model the conductor as a 3D solid (`Part::Box`) with
   `PEC`.
3. **Valid surface impedance fit**: OpenEMS must be able to fit surface
   impedance coefficients across the simulation band.

Curved lossy foils are not supported because 3D curved surfaces are
automatically extruded into volumetric shells by the adapter. Model curved
conductors as `PEC`.

A `ConductingSheet` with `Thickness` set to 0 generates a warning and is
treated as lossless `PEC`.

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

If a catalog entry includes frequency-dependent measurement tables, the
workbench automatically copies the row closest to the active study center
frequency. Create the `EMAnalysis` container before selecting materials
so the center frequency is known.

Material provenance is stored in read-only properties:
- `Source`: Catalog identifier (`catalog:material`).
- `SourceCatalog`: Catalog name and version at the time of selection.
- `SourceDigest`: SHA-256 hash of the catalog values when imported.
- `Description`: Description text from the catalog.

Modifying material properties in the FreeCAD Property Editor updates the
simulation model. The original `SourceDigest` remains unchanged for
auditing.

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

Background space is treated as vacuum by default. Explicit air volumes are
not required around planar boards or open structures.

Assigned solids adopt the material's `Color` in the 3D view. When a solid is
referenced by multiple bindings, the last binding in document order sets the
viewport display color.

### Coincident solids validation

Pre-flight validation detects coincident solid bodies occupying identical
spatial coordinates:
- **Same material**: Generates a warning. OpenEMS discretizes one solid and
  drops coincident duplicate bodies with an "Unused primitive" notification.
  Delete redundant duplicate bodies in CAD.
- **Different materials**: Generates an error and halts simulation. OpenEMS
  resolves overlapping cells by material hierarchy (conductors take
  precedence over dielectrics), which can silently alter intended geometry.
  Assign each physical volume to a single material.

