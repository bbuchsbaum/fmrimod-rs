# License map

This workspace retains the license declared by each component. There is no
blanket license that replaces the individual declarations.

| Packages | Declared license | Text |
| --- | --- | --- |
| fmri-hrf, fmri-design, fmri-noise, fmri-reg, fmri-splines | MIT OR Apache-2.0 | Each crate contains `LICENSE-MIT` and `LICENSE-APACHE` |
| fmri-data, fmri-group, fmrimod | GPL-3.0-or-later | Each crate contains `LICENSE` |

Dependency licenses remain their own. In particular, the `fmri-reg` data adapter
and the `fmrimod` facade depend on GPL-declared components. Renaming and
consolidation have not changed these declarations. Imported source provenance is
recorded in [the original manifest](evidence/source-manifest.json).
