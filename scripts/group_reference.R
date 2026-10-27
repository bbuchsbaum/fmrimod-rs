# Independent numerical fixtures and pinned fmrigds comparisons.
reference_lib <- normalizePath("evidence/group-audit/r-library", mustWork=TRUE)
library(fmrigds, lib.loc=reference_lib)
stopifnot(requireNamespace("metafor",quietly=TRUE))
n <- 12L; nf <- 8L
X <- cbind(intercept=1, age=seq(-1,1,length.out=n))
Y <- outer(seq_len(n),seq_len(nf),function(i,j) .3*j+.2*i+sin(i*j*.71))
V <- outer(seq_len(n),seq_len(nf),function(i,j) .2+(i+j)/17)
Y[2,2] <- NA; Y[c(1,3),3] <- NA; V[4,4] <- 0
cvec <- c(1,-.4)
ols <- lapply(seq_len(nf),function(j){
 ok <- is.finite(Y[,j]);x<-X[ok,,drop=FALSE];y<-Y[ok,j]
 fit <- lm.fit(x,y);df<-length(y)-fit$rank;s2<-sum(fit$residuals^2)/df
 # QR gives coefficient covariance independently of Rust SVD.
 R <- qr.R(fit$qr);cov<-chol2inv(R)*s2
 effect<-sum(cvec*fit$coefficients);se<-sqrt(drop(t(cvec)%*%cov%*%cvec));t<-effect/se
 list(effect=effect,se=se,statistic=t,p=2*pt(-abs(t),df),df=df,n=sum(ok))
})
meta <- function(method) lapply(seq_len(nf),function(j){
 ok<-is.finite(Y[,j]) & is.finite(V[,j]) & V[,j]>0
 fit<-metafor::rma.uni(yi=Y[ok,j],vi=V[ok,j],method=method,test="z")
 list(effect=as.numeric(fit$b),se=fit$se,statistic=fit$zval,p=fit$pval,
      tau2=fit$tau2,Q=fit$QE,n=sum(ok))
})
fe<-meta("FE");dl<-meta("DL")
# Fresh compiled fmrigds agrees on supported inputs, but is not the sole oracle.
for (method in c("core_meta_fe_kernel","core_meta_re_dl_kernel")) {
 out<-get(method,asNamespace("fmrigds"))(Y,V)
 ref<-if(method=="core_meta_fe_kernel") fe else dl
 stopifnot(max(abs(out$beta_g-vapply(ref,`[[`,numeric(1),"effect")))<1e-10)
 stopifnot(max(abs(out$se_g-vapply(ref,`[[`,numeric(1),"se")))<1e-10)
}
pv<-c(.001,.04,.03,NA,.8,.04,1,0)
ok<-is.finite(pv)
adj<-function(method){out<-rep(NA_real_,length(pv));out[ok]<-p.adjust(pv[ok],method);out}
jsonlite::write_json(list(n=n,features=nf,X=as.numeric(t(X)),effects=as.numeric(t(Y)),
 variance=as.numeric(t(V)),contrast=cvec,ols=ols,fe=fe,dl=dl,
 p_values=pv,bh=adj("BH"),by=adj("BY")),"crates/fmri-group/tests/reference.json",
 pretty=TRUE,auto_unbox=TRUE,na="null",digits=17)
writeLines(capture.output(sessionInfo()),"evidence/group-tranche1/reference-session.txt")
cat("PASS: base-R QR, metafor FE/DL, native fmrigds FE/DL and global BH/BY fixture\n")
