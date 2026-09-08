# Extracting phase velocity from transmission phase

Plotting time-domain reflectometry (TDR) or spatial reflection profiles against
physical distance requires the propagation velocity. S-parameter results store
transmission phase across frequency, but do not retain CAD geometry. This document
describes how propagation velocity is derived from transmission phase, and explains
why common alternative estimators introduce systematic errors.

The physical distance between port reference planes must be provided by the caller.
The extracted delay scales across the full path between reference planes, including
launch transitions. This yields the correct scaling for distance-axis plots,
though it differs from the intrinsic effective permittivity of the uniform
transmission line alone.

## Broadband phase accumulation versus local group delay

Group delay represents the local frequency derivative of transmission phase
($\tau_g = -\frac{d\phi}{d\omega}$). However, averaging local group delays across
a frequency sweep can yield inaccurate velocity estimates.

Structures with internal discontinuities or impedance steps store reactive
electromagnetic energy. This reactive storage produces frequency-dependent delays
independent of physical propagation length. Over a sufficiently wide bandwidth,
these reactive phase variations complete full ripple cycles and cancel when
evaluating total accumulated phase:

$$\Delta \phi = \phi(f_{\text{max}}) - \phi(f_{\text{min}})$$

Consequently, effective phase velocity is derived from total accumulated phase
across the entire sweep band rather than by averaging local group delay derivatives.

## Bandwidth limits and reactive energy storage

A potential strategy is to detect and warn when reactive energy storage
distorts velocity estimates. However, statistical indicators based on group delay
variation fail on practical microwave networks:

- Peak-to-peak group delay variation reacts strongly to sharp transmission nulls,
  scaling with frequency step size rather than physical line properties.
- Percentile filtering discards localized peaks, but fails to distinguish between
  physical propagation delay and resonant group delay peaks.
- Phase non-linearity tests fail below filter cutoff frequencies, where phase
  remains linear but exhibits an altered slope.
- Return loss magnitude cannot identify internal reactive reflections in matched
  multi-section structures.

A lumped element network can synthesize the same transmission phase as a distributed
transmission line without possessing physical length. Therefore, transmission phase
alone cannot separate physical propagation delay from localized reactive storage
without a differential two-length measurement.

The plot basis displays the velocity derived from total band delay, documenting that
the distance axis reflects total phase delay including any localized reactive
storage.

## Phase unwrapping conventions

Phase unwrapping reconstructs continuous phase by adding integer multiples of
$2\pi$ at phase discontinuities. Unwrapping algorithms that center phase jumps
around the expected band slope can fail near transmission zeros.

Near transmission notches, transmission phase can rotate rapidly, producing phase
steps exceeding $\pi$ radians. Centering the unwrap threshold around the band mean
can introduce an erroneous $2\pi$ shift across resonant transitions, altering the
derived line length without triggering an explicit error. The workbench uses
standard consecutive unwrapping to avoid assuming a predetermined line delay.

## Frequency resolution requirements

When frequency steps are too coarse ($\Delta f > 1 / (2 \tau)$), phase unwrapping
aliases. Aliased phase remains linear but assumes an artificially lower slope,
yielding an unphysically high velocity.

The workbench validates the derived velocity against the speed of light in vacuum
($c_0$). While this check catches severe phase aliasing, selecting sufficient
frequency points to resolve phase progression remains the user's responsibility.
