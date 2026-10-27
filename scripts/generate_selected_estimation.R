suppressPackageStartupMessages(library(jsonlite))
set.seed(60271)
lengths<-c(36L,31L);n<-sum(lengths);keep<-setdiff(0:(n-1),c(0,3,4,10,18,19,35,36,39,44,45,56,66));ids<-rep(1:2,lengths)[keep+1L]
segments<-cumsum(c(TRUE,diff(keep)!=1|diff(ids)!=0));x<-cbind(1,(0:(n-1))/n,sin((0:(n-1))*.2));xs<-x[keep+1L,,drop=FALSE]
Q<-qr.Q(qr(xs));M<-diag(length(keep))-tcrossprod(Q);C<-diag(length(keep))
for(r in 1:2){ii<-which(ids==r);C[ii,ii]<-C[ii,ii]-1/length(ii)}
R<-C%*%M;lag<-5L;cases<-list()
for(policy in c('preserve','restart'))for(pool in c('global','run')){
 groups<-if(pool=='global')rep(1L,length(ids))else ids;ng<-max(groups)
 pairs<-list();S<-list()
 for(g in 1:ng)for(h in 0:lag){
  ii<-which(outer(keep,keep,'-')==h & outer(ids,ids,'==') & outer(groups,groups,function(a,b)a==g&b==g) & (policy=='preserve'|outer(segments,segments,'==')),arr.ind=TRUE)
  pairs[[length(pairs)+1L]]<-ii;s<-matrix(0,length(keep),length(keep));s[ii]<-1;if(h>0)s<-s+t(s);S[[length(S)+1L]]<-s
 }
 A<-matrix(0,length(S),length(S))
 for(k in seq_along(S)){z<-R%*%S[[k]]%*%t(R);for(h in seq_along(S))A[h,k]<-mean(z[pairs[[h]]])}
 phi<-if(pool=='global')c(.45,.45)else c(.5,-.25)
 sigma<-matrix(0,length(keep),length(keep))
 for(i in seq_along(keep))for(j in seq_along(keep))if(ids[i]==ids[j]&&(policy=='preserve'||segments[i]==segments[j]))sigma[i,j]<-phi[ids[i]]^abs(keep[i]-keep[j])/(1-phi[ids[i]]^2)
 error<-t(chol(sigma))%*%matrix(rnorm(length(keep)*64),nrow=length(keep));ys<-xs%*%matrix(rep(c(2,-.7,1.1),64),3)+error
 residual<-M%*%ys;centered<-C%*%residual
 raw<-vapply(pairs,function(ii)mean(vapply(seq_len(ncol(centered)),function(j)mean(centered[ii[,1],j]*centered[ii[,2],j]),numeric(1))),numeric(1))
 gamma<-as.vector(solve(A,raw));estimated<-vapply(0:(ng-1),function(g)gamma[g*(lag+1)+2]/gamma[g*(lag+1)+1],numeric(1));if(pool=='global')estimated<-rep(estimated,2)
 stopifnot(all(abs(estimated)<1))
 estimated_sigma<-sigma
 for(i in seq_along(keep))for(j in seq_along(keep))if(ids[i]==ids[j]&&(policy=='preserve'||segments[i]==segments[j]))estimated_sigma[i,j]<-estimated[ids[i]]^abs(keep[i]-keep[j])/(1-estimated[ids[i]]^2)
 P<-solve(estimated_sigma);v<-solve(crossprod(xs,P%*%xs));beta<-v%*%crossprod(xs,P%*%ys);res<-ys-xs%*%beta;variance<-colSums(res*(P%*%res))/(nrow(xs)-ncol(xs));tstat<-beta[3,]/sqrt(v[3,3]*variance)
 original_y<-matrix(0,n,64);original_y[keep+1L,]<-ys
 cases[[length(cases)+1L]]<-list(policy=policy,pool=pool,y=unname(original_y),residual=unname(residual),bias=unname(A),pair_counts=vapply(pairs,nrow,integer(1)),gamma=gamma,coefficients=estimated,beta=unname(beta),variance=variance,t=tstat)
}
# Adversarial smooth residual: correction yields a nonstationary estimate.
adverse_residual<-M%*%sin((keep+1L)*0.14907407407407408)
z<-C%*%adverse_residual
adverse_pairs<-lapply(0:lag,function(h)which(outer(keep,keep,'-')==h & outer(ids,ids,'=='),arr.ind=TRUE))
adverse_raw<-vapply(adverse_pairs,function(ii)mean(z[ii[,1]]*z[ii[,2]]),numeric(1))
adverse_gamma<-solve(cases[[1]]$bias,adverse_raw)
stopifnot(adverse_gamma[1]>0,adverse_gamma[2]/adverse_gamma[1]>1)
write_json(list(provenance=list(R=R.version.string,seed=60271,method='Explicit dense C(I-QQt) S_k (I-QQt)Ct moments and dense stationary GLS; not the full-grid fmriAR correction'),lengths=lengths,keep=keep,x=unname(x),lag=lag,cases=cases,adverse=list(residual=unname(adverse_residual),coefficient=adverse_gamma[2]/adverse_gamma[1])),'crates/fmri-reg/tests/fixtures/selected_estimation.json',auto_unbox=TRUE,pretty=TRUE,digits=17)
