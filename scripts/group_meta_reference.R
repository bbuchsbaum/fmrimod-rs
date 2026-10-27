# Fresh installed metafor plus a newly source-built, isolated fmrigds.
root <- 'evidence/group-meta-regression'
library(fmrigds, lib.loc=file.path(root,'r-library'))
library(metafor)
stopifnot(requireNamespace('jsonlite',quietly=TRUE))
flatten <- function(x) as.numeric(t(x))
normalised_error <- function(a,b) max(abs(a-b)/(1+abs(b)))
qr_oracle <- function(x,y,v,method) {
  solve_at <- function(t) {
    w<-1/(v+t); z<-sqrt(w)*x
    qrfit<-qr(z); inverse<-qr.solve(qrfit,diag(length(y)))
    b<-drop(inverse%*%(sqrt(w)*y)); g<-tcrossprod(inverse)
    r<-y-drop(x%*%b)
    list(b=b,g=g,q=sum(w*r*r),d=-sum((w*r)^2),c=sum(w)-sum(w*w*rowSums((x%*%g)*x)))
  }
  fixed<-solve_at(0); df<-length(y)-ncol(x); tau<-0
  if(method=='DL' && fixed$q>df) tau<-(fixed$q-df)/fixed$c
  if(method=='PM' && fixed$q>df) {
    hi<-max((fixed$q-df)/fixed$c,min(v))
    while(solve_at(hi)$q>df) hi<-2*hi
    tau<-uniroot(function(t)solve_at(t)$q-df,c(0,hi),tol=1e-12*median(v),maxiter=1000)$root
  }
  list(fit=solve_at(tau),q0=fixed$q,tau=tau,df=df)
}
# The moderate fixtures match all three independent implementations. Difficult
# extreme-weight/offset cases have separate analytic tests, not a normal-equation oracle.
n<-14L; p<-3L; nf<-7L
X<-cbind(intercept=1, age=seq(-1,1,length.out=n), group=rep(c(0,1),length.out=n))
C<-rbind(c(1,0,0),c(0,1,0),c(0,-.5,1))
Y<-outer(seq_len(n),seq_len(nf),function(i,j) .2*j+.15*i+1.4*sin(i*j*.71))
V<-outer(seq_len(n),seq_len(nf),function(i,j) .2+(i+2*j)/25)
Y[,4]<-drop(X%*%c(.2,-.4,.5))+.01*sin(seq_len(n))
Y[,5]<-drop(X%*%c(.3,-.4,.2))
Y[c(2,5),2]<-NA;V[3,3]<-0;Y[4,6]<-NA
cases<-list()
for(scale in c(1,1e-8,1e8)) {
 y<-Y*scale;v<-V*scale^2; results<-list()
 for(method in c('FE','DL','PM')) for(inference in c('z','adhoc')) {
  rows<-lapply(seq_len(nf),function(j) {
   ok<-is.finite(y[,j]) & is.finite(v[,j]) & v[,j]>0
   fit<-suppressWarnings(rma.uni(y[ok,j],v[ok,j],mods=X[ok,,drop=FALSE],intercept=FALSE,method=method,test=inference,
       control=list(tol=1e-12*scale^2,tau2.max=100*scale^2,maxiter=1000)))
   independent<-qr_oracle(X[ok,,drop=FALSE],y[ok,j],v[ok,j],method)
   stopifnot(normalised_error(fit$b,independent$fit$b)<1e-8,
             normalised_error(fit$tau2,independent$tau)<1e-8)
   effects<-drop(C%*%fit$b); cov<-C%*%fit$vb%*%t(C);se<-sqrt(diag(cov)); statistic<-effects/se
   df<-sum(ok)-p
   q_final<-sum((y[ok,j]-drop(X[ok,,drop=FALSE]%*%fit$b))^2/(v[ok,j]+fit$tau2))
   critical<-if(inference=='z')qnorm(.975) else qt(.975,df)
   prob<-if(inference=='z')2*pnorm(-abs(statistic)) else 2*pt(-abs(statistic),df)
   list(n=sum(ok),effects=effects,se=se,covariance=flatten(cov),statistic=statistic,p=prob,
        intervals=flatten(cbind(effects-critical*se,effects+critical*se)),q_fixed=fit$QE,q_final=q_final,
        tau=fit$tau2,df=df,scale=if(inference=='z')1 else max(1,q_final/df))
  })
  results[[paste(method,inference,sep='-')]]<-rows
 }
 if(scale==1) {
  for(method in c('FE','DL')) {
   fun<-get(if(method=='FE')'meta_fe_reg_cpp' else 'meta_re_reg_dl_cpp',asNamespace('fmrigds'))
   native<-fun(y,v,X,eps=0)
   for(j in seq_len(nf)) {
    ok<-is.finite(y[,j]) & is.finite(v[,j]) & v[,j]>0
    fit<-rma.uni(y[ok,j],v[ok,j],mods=X[ok,,drop=FALSE],intercept=FALSE,method=method)
    stopifnot(normalised_error(native$coef[,j],fit$b)<1e-9,
              normalised_error(native$se_coef[,j],fit$se)<1e-9)
   }
  }
 }
 cases[[length(cases)+1]]<-list(scale=scale,n=n,p=p,features=nf,X=flatten(X),contrasts=flatten(C),effects=flatten(y),variance=flatten(v),results=results)
}
jsonlite::write_json(list(oracle='fresh metafor + independent base-R QR/uniroot; fmrigds compiled FE/DL',metafor=as.character(packageVersion('metafor')),cases=cases),
    'crates/fmri-group/tests/meta_reference.json',pretty=TRUE,auto_unbox=TRUE,digits=17,na='null')
writeLines(capture.output(sessionInfo()),file.path(root,'reference-session.txt'))
cat('PASS: fresh metafor, dense QR/uniroot and compiled fmrigds references\n')

args <- commandArgs(trailingOnly=TRUE)
if (any(c('--experiments','--benchmarks-only') %in% args)) {
 save_matrix<-function(x,path)writeBin(as.double(t(x)),path,size=8,endian='little')
 cases<-list(); times<-list()
 for(config in list(c(20,200,3),c(20,1000,3),c(100,200,5),c(100,1000,5),c(100,10000,5))) {
  n<-config[1];nf<-config[2];p<-config[3];id<-paste(n,nf,p,sep='-')
  X<-outer(seq_len(n),0:(p-1),function(i,j)cos(pi*(i-.5)*j/n))
  Y<-outer(seq_len(n),seq_len(nf),function(i,j).25+sin(i*j*.017)+.3*cos(i*.23))
  V<-outer(seq_len(n),seq_len(nf),function(i,j).2+((i+3*j)%%17)*.03)
  save_matrix(Y,file.path(root,paste0('input-',id,'-y.bin')));save_matrix(V,file.path(root,paste0('input-',id,'-v.bin')))
  cases[[length(cases)+1]]<-list(id=id,n=n,features=nf,p=p,X=flatten(X))
  for(method in c('FE','DL',if(nf<=200)'PM')) {
   native<-method!='PM'
   run<-if(native) {
    f<-get(if(method=='FE')'meta_fe_reg_cpp' else 'meta_re_reg_dl_cpp',asNamespace('fmrigds'))
    function(){a<-f(Y,V,X,eps=0);cbind(t(a$coef),t(a$se_coef),a$Q,if(method=='FE')rep(0,nf) else a$tau2,rep(n,nf),a$df_res)}
   } else {
    function()t(vapply(seq_len(nf),function(j){a<-rma.uni(Y[,j],V[,j],mods=X,intercept=FALSE,method=method,control=list(tol=1e-12,maxiter=1000));c(as.numeric(a$b),a$se,a$QE,a$tau2,n,n-p)},numeric(2*p+4)))
   }
   t0<-proc.time()[['elapsed']];reference<-run();first<-proc.time()[['elapsed']]-t0
   save_matrix(reference,file.path(root,paste0('reference-',id,'-',method,'.bin')))
   # Amortize the millisecond R elapsed timer. Preserve the unbatched first call
   # and raw batch times; report per-fit times without claiming cold-cache timing.
   iterations<-if(native) max(1L,min(200L,ceiling(.15/max(first,.001)))) else 1L
   batch_elapsed<-replicate(5,{gc(FALSE);t0<-proc.time()[['elapsed']];for(k in seq_len(iterations))a<-run();proc.time()[['elapsed']]-t0})
   elapsed<-batch_elapsed/iterations
   times[[length(times)+1]]<-list(id=id,method=method,provider=if(native)'fmrigds compiled C++' else 'metafor rma.uni',seconds=elapsed,median_seconds=median(elapsed),first_seconds=first,iterations_per_trial=iterations,batch_seconds=batch_elapsed)
   cat('BENCH',id,method,median(elapsed),'s\n')
  }
 }
 jsonlite::write_json(cases,file.path(root,'benchmark-inputs.json'),auto_unbox=TRUE,digits=17,pretty=TRUE)
 jsonlite::write_json(times,file.path(root,'r-benchmark.json'),auto_unbox=TRUE,digits=17,pretty=TRUE)
 if ('--benchmarks-only' %in% args) quit(status=0L)
 # Fixed, prespecified calibration: independent normal subjects, known sampling
 # variances, linear covariates. FE with true heterogeneity is a negative control.
 set.seed(271027);reps<-600L;panels<-list();calibration<-list()
 for(n in c(8L,20L)) for(tau in c(0,.5)) {
  p<-if(n==8L)2L else 3L;id<-paste(n,p,tau,sep='-')
  X<-outer(seq_len(n),0:(p-1),function(i,j)cos(pi*(i-.5)*j/n));truth<-rep(.2,p);truth[p]<-0
  vi<-.2+(seq_len(n)%%5)*.06;V<-matrix(vi,n,reps);Y<-X%*%truth+matrix(0,n,1)
  Y<-matrix(drop(X%*%truth),n,reps)+matrix(rnorm(n*reps),n,reps)*sqrt(V+tau)
  save_matrix(Y,file.path(root,paste0('cal-',id,'-y.bin')));save_matrix(V,file.path(root,paste0('cal-',id,'-v.bin')))
  panels[[length(panels)+1]]<-list(id=id,n=n,p=p,features=reps,X=flatten(X),tau_true=tau,truth=truth)
  for(method in c('FE','DL','PM')) for(inference in if(method=='FE')'z' else c('z','adhoc')) {
   refs<-t(vapply(seq_len(reps),function(j){a<-rma.uni(Y[,j],vi,mods=X,intercept=FALSE,method=method,test=inference,control=list(tol=1e-12,maxiter=1000));c(a$b[p],a$se[p],a$pval[p],a$ci.lb[p],a$ci.ub[p],a$tau2)},numeric(6)))
   save_matrix(refs,file.path(root,paste0('cal-reference-',id,'-',method,'-',inference,'.bin')))
   rejected<-sum(refs[,3]<.05); covered<-sum(refs[,4]<=truth[p] & refs[,5]>=truth[p])
   ci<-binom.test(covered,reps)$conf.int
   calibration[[length(calibration)+1]]<-list(id=id,method=method,inference=inference,seed=271027,replicates=reps,rejections=rejected,covered=covered,coverage=covered/reps,coverage_mcse=sqrt(covered/reps*(1-covered/reps)/reps),coverage_binomial_interval=ci,negative_control=method=='FE' && tau>0)
  }
  cat('CAL',id,'complete\n')
 }
 jsonlite::write_json(panels,file.path(root,'calibration-inputs.json'),auto_unbox=TRUE,digits=17,pretty=TRUE)
 jsonlite::write_json(calibration,file.path(root,'r-calibration.json'),auto_unbox=TRUE,digits=17,pretty=TRUE)
 writeLines(capture.output(sessionInfo()),file.path(root,'experiments-session.txt'))
}
