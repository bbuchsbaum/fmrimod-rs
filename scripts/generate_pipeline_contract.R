suppressPackageStartupMessages(library(jsonlite))
f<-fromJSON('crates/fmri-design/tests/fixtures/r_core_contract.json',simplifyVector=FALSE)
d<-f$designs[[3]];task<-do.call(rbind,lapply(d$expected,unlist));lens<-unlist(f$frame$blocklens);n<-sum(lens)
runs<-rep(seq_along(lens),lens);X<-cbind(model.matrix(~0+factor(runs)),task);p<-ncol(X)
Y<-outer(seq_len(n),1:7,function(t,v)sin(t*.39+v*.2)+.1*cos(t*1.7+v)) + X%*%outer(seq_len(p),1:7,function(i,j)i*.1+j*.03)
phi<-list(c(.5,-.15),c(.3,.1),c(.6,-.2));Sigma<-matrix(0,n,n)
for(r in seq_along(lens)){idx<-which(runs==r);rho<-ARMAacf(ar=phi[[r]],lag.max=lens[r]-1);g0<-1/(1-sum(phi[[r]]*rho[2:3]));Sigma[idx,idx]<-toeplitz(g0*rho)}
P<-solve(Sigma);V<-solve(crossprod(X,P%*%X));beta<-V%*%crossprod(X,P%*%Y);resid<-Y-X%*%beta;variance<-colSums(resid*(P%*%resid))/(n-p)
c<-c(rep(0,3),1,-1,rep(0,4));effect<-drop(c%*%beta);se<-sqrt(drop(c%*%V%*%c)*variance);stat<-effect/se
write_json(list(input_fixture='crates/fmri-design/tests/fixtures/r_core_contract.json',input_md5=unname(tools::md5sum('crates/fmri-design/tests/fixtures/r_core_contract.json')),R=R.version.string,X=unname(X),Y=unname(Y),phi=phi,beta=unname(beta),variance=variance,effect=effect,se=se,statistic=stat,p_value=2*pt(-abs(stat),n-p)), 'crates/fmrimod/tests/fixtures/pipeline.json',auto_unbox=TRUE,pretty=TRUE,digits=17)
