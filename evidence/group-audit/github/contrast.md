Reproduced in a fresh R 4.5.1 process against a fresh native build of fmrigds 0.1.0.9000 at commit `273b029bbe9c6c9024bfc12fe4ff6cdb98e18639`. Relevant current GitHub default-branch implementation was also inspected; this report does not claim a full default-branch binary rebuild. Existing open/closed repository issues were searched before filing.

## Reproduction

```r
library(fmrigds)
g <- new_gds(
 list(beta=array(c(1,2),c(1,1,2)), var=array(1,c(1,1,2))),
 space_sample_labels("f1"), "s1", c("a","b"))
h <- harmonise_contrasts(g,c(a="b"))
print(contrasts(h)) # "b" "b"
print(as.numeric(assay(h,"beta"))) # 1 2: distinct estimates still present
stopifnot(anyDuplicated(contrasts(h)) > 0)
```

## Expected behavior and scope

A partial rename colliding with an unmapped ID produces duplicate contrast identifiers without aggregation or an error. Distinct estimates then have the same lookup key. Expected: validate the entire resulting axis and reject collisions before changing the object. Many-to-one aggregation needs a separate explicit operation and uncertainty rule. Suggested priority: P2. Workaround: validate uniqueness of the full proposed names before calling. Test both mapped-to-mapped and mapped-to-unmapped collisions, and ensure valid renames preserve contrast metadata alignment.

