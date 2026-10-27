# Run with R_LIBS pointing to evidence/r-library built from recorded checkouts.
suppressPackageStartupMessages(library(fmriAR))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(normalizePath(find.package('fmriAR')) == normalizePath('evidence/r-library/fmriAR'))
t <- seq_len(24)
X <- cbind(1, sin(t*.37), cos(t*.19))
Y <- cbind(2 + X[,2]*.8 - X[,3]*.3 + sin(t*1.7)*.2,
           -.7 + X[,2]*.2 + X[,3]*1.2 + cos(t*1.3)*.4)
C <- rbind(c(0,1,0),c(0,0,1))
reference <- function(X,Y) {
  fit <- lm.fit(X,Y)
  df <- nrow(X)-fit$rank
  variance <- colSums(fit$residuals^2)/df
  V <- solve(crossprod(X))
  effect <- drop(C[1,,drop=FALSE] %*% fit$coefficients)
  se <- sqrt(drop(C[1,,drop=FALSE] %*% V %*% C[1,])*variance)
  F <- colSums((solve(C%*%V%*%t(C), C%*%fit$coefficients))*(C%*%fit$coefficients))/nrow(C)/variance
  list(beta=unname(fit$coefficients),variance=variance,df=df,effect=effect,se=se,t=effect/se,
       p=2*pt(-abs(effect/se),df),F=F,Fp=pf(F,nrow(C),df,lower.tail=FALSE))
}
runs <- rep(1:2,each=12)
phi <- c(.4,-.25)
plan <- fmriAR::compat$plan_from_phi(phi=lapply(phi,c),runs=runs,pooling='run',exact_first=TRUE)
w <- fmriAR::whiten_apply(plan,X,Y,parallel=FALSE)
# Independent dense covariance solve, not another invocation of whitening.
Sigma <- matrix(0,24,24)
for (r in 1:2) { idx <- which(runs==r); Sigma[idx,idx] <- outer(seq_along(idx),seq_along(idx),function(i,j) phi[r]^abs(i-j))/(1-phi[r]^2) }
precision <- solve(Sigma)
V <- solve(crossprod(X,precision%*%X))
beta <- V%*%crossprod(X,precision%*%Y)
resid <- Y-X%*%beta
variance <- colSums(resid*(precision%*%resid))/(nrow(X)-ncol(X))
stopifnot(max(abs(reference(w$X,w$Y)$beta-beta))<1e-12)
source <- path.expand('~/code/fmriAR')
files <- c(list.files(file.path(source,'R'),full.names=TRUE),list.files(file.path(source,'src'),pattern='\\.(cpp|h)$',full.names=TRUE))
hashes <- as.list(tools::md5sum(files)); names(hashes)<-substring(names(hashes),nchar(source)+2L)
out <- list(provenance=list(R=R.version.string,source=source,commit=system2('git',c('-C',source,'rev-parse','HEAD'),stdout=TRUE),source_md5=hashes),
 X=unname(X),Y=unname(Y),C=unname(C),ols=reference(X,Y),phi=phi,runs=runs,
 whitened_X=unname(w$X),whitened_Y=unname(w$Y),gls=reference(w$X,w$Y),dense_gls_beta=unname(beta),dense_gls_variance=variance)
dir.create('crates/fmrireg/tests/fixtures',recursive=TRUE,showWarnings=FALSE)
write_json(out,'crates/fmrireg/tests/fixtures/estimation.json',pretty=TRUE,auto_unbox=TRUE,digits=17)
cat('R lm, source-built fmriAR and independent dense GLS references generated\n')
# Estimation oracle: run centering, censor-separated pairs, global/run pooling.
noise <- cbind(sin(t*.31)+.25*cos(t*1.7), cos(t*.43)+.2*sin(t*1.1))
noise[13:24,] <- noise[13:24,]+17
censor <- c(3L,8L,15L)
noise_cases <- lapply(c('run','global'),function(pool) {
  plan <- fmriAR::fit_noise(resid=noise,runs=runs,censor=censor,method='ar',p=1L,p_max=1L,pooling=pool)
  list(pooling=pool,phi=unname(unlist(plan$phi)))
})
noise_fixture <- list(provenance=out$provenance,noise=unname(noise),run_lengths=c(12L,12L),censor=censor,cases=noise_cases)
dir.create('crates/fmri-noise/tests/fixtures',recursive=TRUE,showWarnings=FALSE)
write_json(noise_fixture,'crates/fmri-noise/tests/fixtures/estimation.json',pretty=TRUE,auto_unbox=TRUE,digits=17)
# Design-corrected AR(1), with intercept/drift/task, run offsets and censor gaps.
tc <- seq_len(120)
Xc <- cbind(1,tc/120,sin(tc*.11),cos(tc*.13))
Zc <- cbind(sin(tc*.31)+.2*cos(tc*1.71),cos(tc*.29)+.3*sin(tc*1.3))
Rc <- lm.fit(Xc,Zc)$residuals
runs_c <- rep(1:2,each=60); censor_c <- c(8L,19L,72L,91L)
A <- fmriAR::acvf_bias_matrix(Xc,runs=runs_c,censor=censor_c,max_lag=8L)
pc <- fmriAR::fit_noise(resid=Rc,runs=runs_c,censor=censor_c,p=1L,p_max=1L,pooling='run',design=Xc,correction_max_lag=8L)
write_json(list(provenance=out$provenance,design=unname(Xc),residuals=unname(Rc),run_lengths=c(60L,60L),
 censor=censor_c,max_lag=8L,bias=lapply(A,unname),phi=unname(unlist(pc$phi))),
 'crates/fmri-noise/tests/fixtures/correction.json',pretty=TRUE,auto_unbox=TRUE,digits=17)
