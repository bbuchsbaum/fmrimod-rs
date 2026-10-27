Reproduced in a fresh R 4.5.1 process against a fresh native build of fmrigds 0.1.0.9000 at commit `273b029bbe9c6c9024bfc12fe4ff6cdb98e18639`. Relevant current GitHub default-branch implementation was also inspected; this report does not claim a full default-branch binary rebuild. Existing open/closed repository issues were searched before filing.

## Reproduction

```r
library(fmrigds)
fit <- function(scale=1) {
 g <- new_gds(
  list(beta=array(c(1,2,3)*scale,c(1,3,1)),
       var=array(scale^2,c(1,3,1))),
  space_sample_labels("f1"), c("s1","s2","s3"), "c1")
 compute(reduce(g, method="fixed"))
}
a <- fit()
b <- fit(1e-8)
print(c(original=assay(a,"z_g")[1], rescaled=assay(b,"z_g")[1]))
# 3.464101615, 0.034641016
print(c(original=assay(a,"p_g")[1], rescaled=assay(b,"p_g")[1]))
stopifnot(abs(assay(a,"z_g")[1]-assay(b,"z_g")[1]) > 1)
```

## Expected behavior and scope

Multiplying effects by a positive unit conversion and variances by its square should preserve z and p. The default `eps=1e-12` clamps positive variances below an absolute, unit-dependent floor. Here the original variances are 1 and rescaled variances are 1e-16, all positive and representable; z shrinks by 100. Expected: scale-stable inference, or an explicit regularization policy with units and a visible warning when applied. Suggested priority: P2 (narrow small-variance trigger, but silently changes inference). This is not a request to accept zero/negative variances. A workaround is to express data in units keeping positive variances above the floor; choosing a smaller eps requires care. Tests should cover scale invariance, heterogeneous weights, and invalid variances for both R and compiled paths.

