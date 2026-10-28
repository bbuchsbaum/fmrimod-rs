# Turn events into a design

The target is the effect of a unit-amplitude task event relative to baseline.
All participants use the same HRF, coding, sampling grid and region identities.
Those choices make the intended effect comparable; matching labels alone would
not establish equivalence.

The example samples 180 scans at TR 1.5 seconds. Onsets and durations are in
seconds relative to their run. The event table borrows its input arrays, while
HRF evaluation produces the sampled task column:

```rust,ignore
{{#include ../../crates/fmrimod/examples/guide_workflow.rs:design}}
```

`x` has rows for scans and columns `[intercept, task]`. No extra normalization
is applied. A coefficient therefore measures response signal per unit of this
particular convolved predictor. Changing the HRF scale changes that coefficient's
meaning and must be reflected in the estimand definition.

Observation IDs bind each scan to the corresponding design row. The feature
axis names four regions in one declared space. When using images, preserve the
actual spatial identity and feature ordering through loading, fitting and export.
An arbitrary new space label does not prove registration or anatomical alignment.

The complete
[example source](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/crates/fmrimod/examples/guide_workflow.rs)
also generates a deterministic response for each participant: a task effect,
a participant-specific random offset in that effect, and AR(1) noise with
coefficient 0.4 and innovation standard deviation 0.15. These generating values
are simulation inputs; the fitting code estimates its own noise parameters.

Next, [prepare and fit the subjects](fit.md). For other HRFs and sampling methods,
use the [HRF API](api/fmri_hrf/index.html).
