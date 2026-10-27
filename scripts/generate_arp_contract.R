suppressPackageStartupMessages(library(fmriAR))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(normalizePath(find.package('fmriAR')) == normalizePath('evidence/r-library/fmriAR'))
source <- path.expand('~/code/fmriAR')
files <- c(list.files(file.path(source,'R'),full.names=TRUE),list.files(file.path(source,'src'),pattern='\\.(cpp|h)$',full.names=TRUE))
hashes <- as.list(tools::md5sum(files));names(hashes)<-substring(names(hashes),nchar(source)+2L)
provenance <- list(source=source,commit=system2('git',c('-C',source,'rev-parse','HEAD'),stdout=TRUE),source_md5=hashes,R=R.version.string,
 binary_md5=as.list(tools::md5sum(list.files(system.file('libs',package='fmriAR'),full.names=TRUE,recursive=TRUE))))
t <- seq_len(160); runs <- rep(1:2,each=80); censor <- c(7L,8L,19L,84L,99L,128L)
X <- cbind(1,t/160,sin(t*.11),cos(t*.13))
noise <- cbind(sin(t*.31)+.2*cos(t*1.71),cos(t*.29)+.3*sin(t*1.3))
noise[81:160,] <- noise[81:160,]+7
residuals <- lm.fit(X,noise)$residuals
cases <- list()
for(corrected in c(FALSE,TRUE)) for(pooling in c('run','global')) for(p in c(1L,2L,3L,4L)) {
 R <- if(corrected) residuals else noise
 plan <- fmriAR::fit_noise(resid=R,runs=runs,censor=censor,p=p,p_max=p,pooling=pooling,
  design=if(corrected) X else NULL,correction_max_lag=10L,exact_first='none')
 w <- fmriAR::whiten_apply(plan,X,noise,runs=runs,censor=censor,parallel=FALSE)
 q <- qr(w$X); beta<-qr.coef(q,w$Y); variance<-colSums((w$Y-w$X%*%beta)^2)/(nrow(X)-q$rank)
 cases[[length(cases)+1L]] <- list(corrected=corrected,pooling=pooling,order=p,
  phi=unname(lapply(plan$phi,unname)),Xw=unname(w$X),Yw=unname(w$Y),beta=unname(beta),variance=variance)
}
known <- list()
for(phi in list(c(.4),c(.6,-.2),c(.4,.15,-.1),c(.5,-.25,.1,.05),c(1.8,-.81))) {
 n<-16L; cuts<-c(1L,3L,10L); ends<-c(cuts[-1]-1L,n)
 Sigma<-matrix(0,n,n)
 for(j in seq_along(cuts)) {
  idx<-cuts[j]:ends[j];rho<-ARMAacf(ar=phi,lag.max=max(length(phi),length(idx)-1L))
  gamma0<-1/(1-sum(phi*rho[seq_along(phi)+1L]))
  Sigma[idx,idx]<-toeplitz(gamma0*rho[seq_along(idx)])
 }
 W<-solve(t(chol(Sigma)))
 plan<-fmriAR::compat$plan_from_phi(phi=list(phi),runs=rep(1L,n),exact_first=FALSE)
 z<-fmriAR::whiten_apply(plan,diag(n),diag(n),censor=cuts[-1]-1L,parallel=FALSE)$X
 Xk<-cbind(1,seq_len(n)/n,sin(seq_len(n)*.3))
 Yk<-cbind(cos(seq_len(n)*.7),sin(seq_len(n)*.4))
 precision<-solve(Sigma); V<-solve(crossprod(Xk,precision%*%Xk))
 bk<-V%*%crossprod(Xk,precision%*%Yk);rk<-Yk-Xk%*%bk
 vk<-colSums(rk*(precision%*%rk))/(n-ncol(Xk))
 known[[length(known)+1L]]<-list(X=unname(Xk),Y=unname(Yk),beta=unname(bk),variance=vk,phi=phi,n=n,resets=cuts-1L,covariance=unname(Sigma),stationary=unname(W),zero=unname(z))
}
# Explicit non-PD pair-count ACVF and near-boundary coefficients test repair.
repair <- lapply(list(c(1,1.2,.1),c(1,.95,-.6,.2)),function(g){
 fixed<-fmriAR:::`.shrink_to_pd`(g)
 yw<-fmriAR:::yw_from_acvf_fast(fixed,length(g)-1L)
 list(gamma=g,phi=unname(fmriAR:::enforce_stationary_ar(yw$phi,.99)),scale=fixed[2]/g[2])
})
bias <- fmriAR::acvf_bias_matrix(X,runs=runs,censor=censor,max_lag=10L)
corrected_gamma <- lapply(1:2,function(r){
 idx<-which(runs==r);keep<-setdiff(idx,censor);seg<-cumsum(c(1L,as.integer(diff(keep)>1L)))
 pooled<-fmriAR:::`.pooled_acvf_segments`(residuals[keep,,drop=FALSE],seg,10L)
 drop(solve(bias[[r]],pooled$num/pooled$pairs))
})
write_json(list(bias=unname(lapply(bias,unname)),corrected_gamma=corrected_gamma,provenance=provenance,X=unname(X),noise=unname(noise),residuals=unname(residuals),
 run_lengths=c(80L,80L),censor=censor,correction_max_lag=10L,cases=cases,known=known,repair=repair),
 'crates/fmri-noise/tests/fixtures/arp.json',auto_unbox=TRUE,pretty=TRUE,digits=17)
cat(length(cases),'estimation cases and',length(known),'independent stationary/zero-history transforms generated\n')
