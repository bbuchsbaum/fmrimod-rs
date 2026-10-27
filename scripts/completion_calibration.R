# Independent oracle audit of source-bound Rust simulation inputs and outputs.
library(metafor)
args<-commandArgs(TRUE)
input<-if(length(args))args[[1]] else 'evidence/completion/group-calibration.jsonl'
out<-if(length(args)>1)args[[2]] else 'evidence/completion/group-calibration-oracle.json'
cases<-lapply(readLines(input),jsonlite::fromJSON,simplifyVector=FALSE)
method_map<-c(CommonEffect='FE',DerSimonianLaird='DL',PauleMandel='PM')
checked<-0L;max_error<-0;reports<-list()
wilson<-function(k,n) {if(n==0)return(c(NA_real_,NA_real_));z<-qnorm(.975);p<-k/n;den<-1+z*z/n;mid<-(p+z*z/(2*n))/den;half<-z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den;c(mid-half,mid+half)}
for(case in cases) {
 n<-case$n;p<-case$p;b<-case$replicates
 X<-matrix(unlist(case$x),n,p,byrow=TRUE)
 Y<-matrix(vapply(case$y,function(v)if(is.null(v))NA_real_ else v,0.),n,b,byrow=TRUE)
 V<-matrix(unlist(case$v),n,b,byrow=TRUE)
 for(method in case$methods) {
  covered<-0L;success<-0L
  for(j in seq_len(b)) {
   got<-method$estimates[[j]]
   if(is.null(got))next
   success<-success+1L;ok<-is.finite(Y[,j])&is.finite(V[,j])&V[,j]>0
   critical<-if(method$inference=='Normal')qnorm(.975)else qt(.975,sum(ok)-p)
   covered<-covered+as.integer(abs(got$effect)<=critical*got$se)
   if(j<=25L) {
    fit<-suppressWarnings(rma.uni(Y[ok,j],V[ok,j],mods=X[ok,,drop=FALSE],intercept=FALSE,
      method=unname(method_map[method$method]),test=if(method$inference=='Normal')'z' else 'adhoc',
      control=list(tol=1e-12,threshold=1e-10,maxiter=1000,tau2.max=100)))
    expected<-c(as.numeric(fit$b[p]),fit$se[p],fit$tau2)
    observed<-c(got$effect,got$se,got$tau)
    err<-max(abs(observed-expected)/(1+abs(expected)))
    if(err>2e-8)stop(sprintf('oracle mismatch %s n=%d method=%s j=%d err=%g',case$scenario,n,method$method,j,err))
    max_error<-max(max_error,err);checked<-checked+1L
   }
  }
  stopifnot(covered==method$covered,success==method$successes)
  reports[[length(reports)+1L]]<-list(scenario=case$scenario,n=n,method=method$method,inference=method$inference,
    attempted=b,successes=success,covered=covered,conditional_coverage=if(success>0)covered/success else NULL,
    wilson95=wilson(covered,success),failures=method$failures)
 }
}
jsonlite::write_json(list(oracle='metafor first 25 replicates per method/scenario; independent R interval counting for all',metafor=as.character(packageVersion('metafor')),comparisons=checked,max_normalized_error=max_error,results=reports),out,pretty=TRUE,auto_unbox=TRUE,digits=17)
cat(sprintf('PASS: %d fresh metafor fits; all interval counts match; max error %.3g\n',checked,max_error))
