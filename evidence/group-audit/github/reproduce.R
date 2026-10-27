.libPaths(c("evidence/group-audit/r-library", .libPaths()))
cat("\nmapping\n")
library(fmrigds)
g <- new_gds(
  list(beta = array(c(2, NA_real_), c(2,1,1)), var = array(1, c(2,1,1))),
  space_sample_labels(c("kept","unused")), "s1", "c1")
out <- compute(map_to(g, space_sample_labels("target"), matrix(c(1,0),1)))
print(as.numeric(assay(out, "beta"))) # NA, expected 2
stopifnot(is.na(assay(out, "beta")[1]))
cat("\ngeometry\n")
library(fmrigds)
a <- diag(4)
b <- a
b[1,4] <- 10
s1 <- space_voxel(c(2,2,2), a, storage="dense")
s2 <- space_voxel(c(2,2,2), b, storage="dense")
print(assert_compatible_spaces(s1,s2)) # TRUE despite different coordinates
stopifnot(assert_compatible_spaces(s1,s2))
cat("\ncontrast\n")
library(fmrigds)
g <- new_gds(
 list(beta=array(c(1,2),c(1,1,2)), var=array(1,c(1,1,2))),
 space_sample_labels("f1"), "s1", c("a","b"))
h <- harmonise_contrasts(g,c(a="b"))
print(contrasts(h)) # "b" "b"
print(as.numeric(assay(h,"beta"))) # 1 2: distinct estimates still present
stopifnot(anyDuplicated(contrasts(h)) > 0)
cat("\nvariance-floor\n")
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
writeLines(capture.output(sessionInfo()),"evidence/group-audit/github/report-session.txt")

