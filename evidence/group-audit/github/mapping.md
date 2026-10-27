Reproduced in a fresh R 4.5.1 process against a fresh native build of fmrigds 0.1.0.9000 at commit `273b029bbe9c6c9024bfc12fe4ff6cdb98e18639`. Relevant current GitHub default-branch implementation was also inspected; this report does not claim a full default-branch binary rebuild. Existing open/closed repository issues were searched before filing.

## Reproduction

```r
library(fmrigds)
g <- new_gds(
  list(beta = array(c(2, NA_real_), c(2,1,1)), var = array(1, c(2,1,1))),
  space_sample_labels(c("kept","unused")), "s1", "c1")
out <- compute(map_to(g, space_sample_labels("target"), matrix(c(1,0),1)))
print(as.numeric(assay(out, "beta"))) # NA, expected 2
stopifnot(is.na(assay(out, "beta")[1]))
```

## Expected behavior and scope

A zero map coefficient means that source feature is outside the target's support. Its missing value should not invalidate an otherwise observed target. The independent-variance path uses dense matrix multiplication, so `0 * NA` contaminates the result. This is distinct from missing subjects in a reducer (issue #7). Suggested priority: P2. Iterate nonzero support; preserve missingness for genuinely contributing sources. Do not replace all missing values with zero. A narrow workaround is to subset to the map's actual support before mapping. Regression tests should cover dense/sparse maps, missing effects and variances outside support, and missing values inside support.

