# Run from fmrimod-rs; fresh source installation required (see audit report).
args <- commandArgs(trailingOnly = TRUE)
lib <- if (length(args)) args[[1]] else "evidence/group-audit/r-library"
library(fmrigds, lib.loc = lib)
ns <- asNamespace("fmrigds")
f <- function(name) get(name, envir = ns)
rows <- list()
record <- function(name, claim, value) {
  rows[[name]] <<- list(claim = claim, observed = value)
  cat(name, ": ", claim, "\n", sep = "")
}
near <- function(x,y) stopifnot(isTRUE(all.equal(as.numeric(x),as.numeric(y), tolerance=1e-10)))
g <- function(b,v=1) {
  if (length(dim(b)) == 2) b <- array(t(b), c(ncol(b),nrow(b),1))
  a <- list(beta=b)
  if (!is.null(v)) a$var <- array(v,dim(b))
  new_gds(a, space_sample_labels(paste0("f",seq_len(dim(b)[1]))),
          paste0("s",seq_len(dim(b)[2])), paste0("c",seq_len(dim(b)[3])))
}
# Analytic independent-subject mean versus known-variance common effect.
y <- matrix(c(0,2,4,6),4,1); v <- matrix(c(1,1,4,4),4,1)
fe <- f("core_meta_fe_kernel")(y,v)
near(fe$beta_g,1.8); near(fe$var_g,.4)
ols <- compute(reduce(g(y), method="ols:voxelwise",formula=~1))
near(assay(ols,"coef:(Intercept)"),3)
near(assay(ols,"se_coef:(Intercept)"),sd(y)/2)
record("different_estimands", "OLS mean 3; common-effect estimate 1.8, variance 0.4",c(ols=3,fe=fe$beta_g,var=fe$var_g))
re <- f("core_meta_re_dl_kernel")(matrix(c(-1,1),2),matrix(1,2))
near(re$tau2,1); near(re$var_g,1)
record("dl_analytic", "Two studies -1,+1 with variance 1 give tau2=1 and group variance=1",re[c("beta_g","var_g","tau2")])
# Source-matched native and R implementations; agreement is not an independent oracle.
set.seed(2701)
y <- matrix(rnorm(60),12,5); v <- matrix(runif(60,.1,2),12,5)
y[2,1] <- NA; v[3,2] <- 0
native <- f("core_meta_fe_kernel")(y,v)
r <- f(".colwise_fe")(y,v,min_subj=2L)
for (nm in names(r)) near(native[[nm]],r[[nm]])
X <- cbind(1,seq(-1,1,length.out=12))
a <- f("core_meta_re_reg_dl_kernel")(y,v,X)
b <- f(".meta_re_reg_dl_r")(y,v,X)
for (nm in names(b)) near(a[[nm]],b[[nm]])
record("native_reference", "Fresh compiled FE and DL meta-regression agree with R paths on missing/invalid-variance fixtures",TRUE)
# Public OLS against base R QR, with feature-specific missingness.
y <- matrix(c(1,2,4,3,7,8, 2,NA,3,6,4,9),6,2)
fit <- suppressWarnings(compute(reduce(g(y),method="ols:voxelwise",formula=~1)))
for (i in 1:2) {
  ref <- summary(lm(y[,i]~1))
  near(assay(fit,"coef:(Intercept)")[i],ref$coefficients[1,1])
  near(assay(fit,"se_coef:(Intercept)")[i],ref$coefficients[1,2])
  near(assay(fit,"df_res")[i],ref$df[2])
}
record("ols_missing", "Public OLS matches independent base-R QR including per-feature degrees of freedom",as.numeric(assay(fit,"df_res")))
# Absolute variance floor changes inference under a change of measurement units.
y <- matrix(c(1,2,3),3); v <- matrix(1,3)
a <- f("core_meta_fe_kernel")(y,v)
b <- f("core_meta_fe_kernel")(y*1e-8,v*1e-16)
stopifnot(abs(a$z_g-b$z_g)>1)
record("absolute_floor", "Default 1e-12 variance floor breaks unit-scale invariance below that floor",c(z_original=a$z_g,z_rescaled=b$z_g))
# Public mapping: zero coefficient should not depend on an absent unrelated feature.
x <- g(array(c(2,NA_real_),c(2,1,1)),1)
mapped <- compute(map_to(x,space_sample_labels("target"),matrix(c(1,0),1)))
stopifnot(is.na(assay(mapped,"beta")[1]))
record("map_zero_times_na", "Public map [1,0] of [2,NA] yields NA; support-only result is 2",as.numeric(assay(mapped,"beta")))
M <- matrix(c(.5,.5),1); beta <- array(c(1,3),c(2,1,1)); variance <- array(1,dim(beta))
a <- f("propagate_variance_independent")(M,beta,variance)
b <- f("propagate_variance_covariance")(M,beta,variance,function(idx) matrix(c(1,.8,.8,1),2))
near(a$var,.5); near(b$var,.9)
record("mapping_covariance", "Equal-weight average variance is 0.5 under independence, 0.9 at correlation 0.8",c(independent=a$var,correlated=b$var))
beta <- array(1,c(2,2,1)); variance <- array(c(1,1,9,9),dim(beta))
b <- f("propagate_variance_covariance")(M,beta,variance,function(idx) diag(length(idx)))
near(b$var,c(.5,.5))
record("provider_scope", "Covariance provider receives only feature indices; same covariance used for both subjects despite distinct marginal input variances",as.numeric(b$var))
# Coordinate and contrast identity are not implied by shape or a label rename.
s1 <- space_voxel(c(2,2,2),diag(4),storage="dense")
aff <- diag(4); aff[1,4] <- 10
s2 <- space_voxel(c(2,2,2),aff,storage="dense")
stopifnot(assert_compatible_spaces(s1,s2))
record("affine_compatibility", "Default compatibility accepts same dimensions with 10-unit affine translation",TRUE)
x <- g(array(1,c(1,2,2)))
x <- harmonise_contrasts(x,c(c1="same",c2="same"))
stopifnot(anyDuplicated(contrasts(x))>0)
record("contrast_collision", "Public contrast renaming accepts duplicate resulting IDs",contrasts(x))
x <- g(matrix(c(0,2,4,6),4,1),1)
keep <- compute(mask(x,MaskPolicy(rule="union")))
drop <- compute(mask(x,MaskPolicy(rule="union",zero_is_missing=TRUE)))
stopifnot(is.finite(assay(keep,"beta")[1]),is.na(assay(drop,"beta")[1]),is.na(assay(drop,"var")[1]))
record("valid_zero", "Zeros retained by default; explicit zero policy masks effect and variance together",TRUE)
# Pooled shape is a global fit, not a feature-local kernel.
set.seed(2702)
Y <- cbind(rep(rnorm(10,sd=5),each=3)+rnorm(30,sd=.1),rnorm(30))
cube <- function(Y) aperm(array(Y,c(3,10,ncol(Y))),c(3,2,1))
x <- g(cube(Y))
pooled <- function(z) compute(reduce(z,method="lmm:ri",formula=~1,options=list(theta_mode="pooled")))
a <- pooled(x)
b1 <- pooled(g(cube(Y[,1,drop=FALSE])))
b2 <- pooled(g(cube(Y[,2,drop=FALSE])))
values <- c(joint=assay(a,"lambda")[1],alone1=assay(b1,"lambda")[1],alone2=assay(b2,"lambda")[1])
stopifnot(diff(range(values))>1)
record("pooled_scope", "Separately fitting feature blocks changes pooled covariance; streaming must preserve declared fit scope",values)
# One missing repeated observation rejects the whole feature in current LMM.
x$assays$beta[1,1,1] <- NA_real_
a <- pooled(x)
stopifnot(is.na(assay(a,"coef:(Intercept)")[1]),is.finite(assay(a,"coef:(Intercept)")[2]))
record("lmm_missing", "Current repeated LMM rejects a feature with any missing observation",TRUE)
# Max-T family cannot be split independently across feature blocks.
set.seed(2703); Y <- matrix(rnorm(48),8,6)
signs <- as.matrix(expand.grid(rep(list(c(-1,1)),8)))
a <- f("core_perm_onesample_kernel")(Y,opts=list(signs=signs))
b <- sapply(seq_len(ncol(Y)),function(i) f("core_perm_onesample_kernel")(Y[,i,drop=FALSE],opts=list(signs=signs))$p_fwer)
stopifnot(any(a$p_fwer>b+1e-10))
record("max_t_family", "Familywise p-values differ if each feature block is treated as a separate family",list(full=a$p_fwer,singleton=b))
# Correlated first-level contrasts: variance of difference needs off-diagonal.
C <- matrix(c(1,.8,.8,1),2)
near(drop(t(c(1,-1))%*%C%*%c(1,-1)),.4)
record("contrast_covariance", "Marginal variances alone give difference variance 2; true covariance gives 0.4",c(diagonal=2,full=.4))
# Independent dense covariance oracle for the compiled low-rank known-variance path.
X <- cbind(1, rep(c(-1,0,1),4)); y <- sin(seq_len(12)) + .3*X[,2]
v <- seq(.2,1.3,length.out=12); U <- cbind(rep(.7,3),c(-.2,0,.2))
V <- diag(v+.15) + kronecker(diag(4),tcrossprod(U))
Vi <- solve(V); A <- solve(t(X)%*%Vi%*%X)
beta <- A%*%t(X)%*%Vi%*%y; residual <- y-X%*%beta
fit <- f("lmm_knownvar_fit_cpp")(y,v,X,U,.15)
near(fit$coef,beta); near(fit$se_coef,sqrt(diag(A)))
objective <- as.numeric(determinant(V,logarithm=TRUE)$modulus) +
  as.numeric(determinant(t(X)%*%Vi%*%X,logarithm=TRUE)$modulus) +
  drop(t(residual)%*%Vi%*%residual)
near(f("lmm_knownvar_objective_cpp")(y,v,X,U,.15),objective)
record("knownvar_dense_oracle", "Compiled low-rank known-variance coefficients, SE and REML objective match independent dense covariance solve",TRUE)
# Constructor contract should not force meaningless variance into effect-only OLS.
rejected <- try(g(matrix(1:4,4),v=NULL),silent=TRUE)
stopifnot(inherits(rejected,"try-error"))
record("effect_only_constructor", "new_gds rejects beta without var/se, although OLS and unweighted LMM do not use sampling variance",as.character(rejected))
jsonlite::write_json(list(package_path=find.package("fmrigds"),R=R.version.string,
  tolerances=list(relative=1e-10),seed=c(2701,2702,2703),witnesses=rows),
  "evidence/group-audit/results.json",pretty=TRUE,auto_unbox=TRUE,na="null",digits=17)
writeLines(capture.output(sessionInfo()),"evidence/group-audit/session.txt")
cat("PASS: ",length(rows)," witnesses\n",sep="")
