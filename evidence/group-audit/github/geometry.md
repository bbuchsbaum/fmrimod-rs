Reproduced in a fresh R 4.5.1 process against a fresh native build of fmrigds 0.1.0.9000 at commit `273b029bbe9c6c9024bfc12fe4ff6cdb98e18639`. Relevant current GitHub default-branch implementation was also inspected; this report does not claim a full default-branch binary rebuild. Existing open/closed repository issues were searched before filing.

## Reproduction

```r
library(fmrigds)
a <- diag(4)
b <- a
b[1,4] <- 10
s1 <- space_voxel(c(2,2,2), a, storage="dense")
s2 <- space_voxel(c(2,2,2), b, storage="dense")
print(assert_compatible_spaces(s1,s2)) # TRUE despite different coordinates
stopifnot(assert_compatible_spaces(s1,s2))
```

## Expected behavior and scope

The default validator checks dimensions and optional identity metadata but omits the voxel affine. Thus equal indices can denote different physical locations while compatibility succeeds. Expected: default voxel-grid compatibility rejects this translation, or an explicitly weaker metadata-only API is clearly distinguished from geometric compatibility. Suggested priority: P2. This is a validator gap, not the NIfTI export affine defects in #6/#8. Workaround: explicitly compare affine and ordered support (or request `fields=c('type','dim','affine')` for this dense example). Passing geometric checks alone still does not establish anatomical registration.

