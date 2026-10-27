suppressPackageStartupMessages(library(fmriAR))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(normalizePath(find.package('fmriAR'))==normalizePath('evidence/r-library/fmriAR'))
a<-commandArgs(TRUE); z<-as.integer(a[1:7]);n<-z[1];p<-z[2];v<-z[3];order<-z[4];block<-z[5];workers<-z[6];reps<-z[7]
X<-outer(seq_len(n),seq_len(p),function(i,j) sin(i*j*.013));X[,1]<-1
runs<-rep(1:2,each=n/2);censor<-c(n/4,n/4+1,3*n/4+1)
data<-function(start,b) outer(seq_len(n),start+seq_len(b),function(t,f) sin(t*.071+f*.03)+.3*cos(t*.317+f*.13)+.1*sin(t*1.3+f*.017))
tick<-function() proc.time()[[3]]
s<-tick();q0<-qr(X);initial_prepare<-tick()-s
noise<-data(0,32);s<-tick();residual<-qr.resid(q0,noise);residual_seconds<-tick()-s
s<-tick();bias<-acvf_bias_matrix(X,runs=runs,censor=censor,max_lag=10L);correction_prepare<-tick()-s
# Passing precomputed corrections separates setup from estimation.
s<-tick();plan<-fit_noise(residual,runs=runs,censor=censor,p=order,p_max=order,pooling='run',acvf_correction=bias,exact_first='none');noise_fit<-tick()-s
s<-tick();Xw<-whiten_apply(plan,X,matrix(numeric(0),n,0),runs=runs,censor=censor,parallel=FALSE)$X;design_whiten<-tick()-s
s<-tick();q<-qr(Xw);cov<-chol2inv(qr.R(q));gls_prepare<-tick()-s
fit<-function(Yw) {beta<-qr.coef(q,Yw);r<-Yw-Xw%*%beta;list(beta=beta,variance=colSums(r*r)/(n-q$rank))}
contrast<-function(f) {effect<-f$beta[2,];se<-sqrt(f$variance*cov[2,2]);stat<-effect/se;list(effect=effect,se=se,t=stat,p=2*pt(-abs(stat),n-q$rank))}
run<-function(output=NULL) {
 totals<-c(generate=0,whiten=0,fit=0,contrast=0);begin<-tick()
 for(start in seq.int(0,v-1,by=block)) {
  s<-tick();Y<-data(start,min(block,v-start));totals[1]<-totals[1]+tick()-s
  s<-tick();Yw<-whiten_apply(plan,matrix(numeric(0),n,0),Y,runs=runs,censor=censor,parallel=FALSE)$Y;totals[2]<-totals[2]+tick()-s
  s<-tick();f<-fit(Yw);totals[3]<-totals[3]+tick()-s
  s<-tick();c<-contrast(f);totals[4]<-totals[4]+tick()-s
  if(!is.null(output))writeBin(as.double(rbind(f$beta,f$variance,c$effect,c$se,c$t,c$p)),output,size=8,endian='little')
 }
 c(list(wall=tick()-begin),as.list(totals))
}
cold<-run();warm<-lapply(seq_len(reps),function(i)run())
if(a[8]!='-') {out<-file(a[8],'wb');invisible(run(out));close(out)}
cat(toJSON(list(n=n,p=p,features=v,order=order,block=block,workers=workers,rank=q$rank,df=n-q$rank,phi=unname(lapply(plan$phi,unname)),setup=list(initial_prepare=initial_prepare,residual=residual_seconds,correction_prepare=correction_prepare,noise_fit=noise_fit,design_whiten=design_whiten,gls_prepare=gls_prepare),cold=cold,warm=warm,R=R.version.string,BLAS=extSoftVersion()[['BLAS']]),auto_unbox=TRUE,digits=17))
