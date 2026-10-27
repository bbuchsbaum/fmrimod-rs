stopifnot(requireNamespace('jsonlite',quietly=TRUE))
input<-'evidence/completion/repeated-calibration.jsonl'
cases<-lapply(readLines(input),jsonlite::fromJSON,simplifyVector=FALSE)
max_error<-0;checked<-0L;analytic_checked<-0L;max_parameter_error<-0
for(case in cases){
 n<-case$observations;p<-case$p;ns<-case$subjects
 t<-rep(c(-1,0,1),ns);X<-if(p==1)matrix(1,n,1)else cbind(1,t)
 Z<-if(p==1)matrix(1,3,1)else cbind(1,c(-1,0,1))
 S<-matrix(.05,3,3);diag(S)<-.3
 for(sample in case$samples){
  g<-matrix(unlist(sample$g),p,p,byrow=TRUE);V<-S+sample$eta*diag(3)+Z%*%g%*%t(Z)
  L<-t(chol(V));y<-unlist(sample$y)
  xw<-do.call(rbind,lapply(seq_len(ns),function(i)forwardsolve(L,X[(3*i-2):(3*i),,drop=FALSE])))
  yw<-unlist(lapply(seq_len(ns),function(i)forwardsolve(L,y[(3*i-2):(3*i)])))
  qrfit<-qr(xw);beta<-qr.coef(qrfit,yw);resid<-qr.resid(qrfit,yw);R<-qr.R(qrfit)
  inv<-qr.solve(qrfit,diag(n));covariance<-tcrossprod(inv)
  objective<-.5*(2*ns*sum(log(diag(L)))+sum(resid^2)+2*sum(log(abs(diag(R)))))
  actual<-c(unlist(sample$beta),unlist(sample$covariance),sample$objective)
  expected<-c(beta,as.numeric(t(covariance)),objective)
  error<-max(abs(actual-expected)/(1+abs(expected)));max_error<-max(max_error,error)
  if(error>1e-9)stop(sprintf('repeated dense oracle mismatch %s N%d error%g',case$scenario,ns,error))
  checked<-checked+1L
  # Independent balanced-design eigenspace solution. This does not reuse the
  # production optimizer/gradient. When the unconstrained upper covariance is
  # PSD, it attains the global REML minimum with the independent lower component.
  subject_y<-matrix(y,ns,3,byrow=TRUE)
  if(p==1){
   exact<-max(0,var(rowMeans(subject_y))-.4/3)
   error<-abs(g[1,1]-exact)/(1+exact)
   analytic_checked<-analytic_checked+1L
   max_parameter_error<-max(max_parameter_error,error)
   stopifnot(error<2e-5)
  }else{
   U<-cbind(rep(1/sqrt(3),3),c(-1,0,1)/sqrt(2),c(1,-2,1)/sqrt(6))
   scores<-subject_y%*%U;exact_eta<-max(0,mean(scores[,3]^2)-.25)
   D<-diag(1/c(sqrt(3),sqrt(2)))
   exact_g<-D%*%(cov(scores[,1:2])-diag(c(.4+exact_eta,.25+exact_eta)))%*%D
   if(min(eigen(exact_g,symmetric=TRUE,only.values=TRUE)$values)>0){
    error<-max(abs(c(g, sample$eta)-c(exact_g,exact_eta))/(1+abs(c(exact_g,exact_eta))))
    analytic_checked<-analytic_checked+1L;max_parameter_error<-max(max_parameter_error,error)
    stopifnot(error<2e-5)
   }
  }
 }
}
jsonlite::write_json(list(oracle='independent dense Cholesky/QR plus balanced eigenspace REML solutions; first 10 replicates per case',comparisons=checked,max_normalized_error=max_error,analytic_parameter_comparisons=analytic_checked,max_parameter_error=max_parameter_error),
 'evidence/completion/repeated-calibration-oracle.json',pretty=TRUE,auto_unbox=TRUE,digits=17)
cat(sprintf('PASS: %d repeated dense checks and %d analytic REML parameter checks; errors %.3g / %.3g\n',checked,analytic_checked,max_error,max_parameter_error))
