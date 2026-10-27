suppressPackageStartupMessages(library(jsonlite))
set.seed(60272)
lengths<-c(36L,31L);n<-sum(lengths);keep<-setdiff(0:(n-1),c(0,3,4,10,18,19,35,36,39,44,45,56,66));ids<-rep(1:2,lengths)[keep+1L]
segments<-cumsum(c(TRUE,diff(keep)!=1|diff(ids)!=0));x<-cbind(1,(0:(n-1))/n,sin((0:(n-1))*.2));xs<-x[keep+1L,,drop=FALSE]
Q<-qr.Q(qr(xs));M<-diag(length(keep))-tcrossprod(Q);C<-diag(length(keep))
for(r in 1:2){ii<-which(ids==r);C[ii,ii]<-C[ii,ii]-1/length(ii)}
R<-C%*%M;lag<-5L;cases<-list()
for(order in 2:4)for(policy in c('preserve','restart'))for(pool in c('global','run')){
 groups<-if(pool=='global')rep(1L,length(ids))else ids;ng<-max(groups)
 pairs<-list();S<-list()
 for(g in 1:ng)for(h in 0:lag){
  ii<-which(outer(keep,keep,'-')==h & outer(ids,ids,'==') & outer(groups,groups,function(a,b)a==g&b==g) & (policy=='preserve'|outer(segments,segments,'==')),arr.ind=TRUE)
  pairs[[length(pairs)+1L]]<-ii;s<-matrix(0,length(keep),length(keep));s[ii]<-1;if(h>0)s<-s+t(s);S[[length(S)+1L]]<-s
 }
 A<-matrix(0,length(S),length(S))
 for(k in seq_along(S)){z<-R%*%S[[k]]%*%t(R);for(h in seq_along(S))A[h,k]<-mean(z[pairs[[h]]])}
 phi0<-list(c(.45,-.18),c(.4,-.16,.08),c(.4,-.15,.08,-.04))[[order-1L]]
 phi<-list(phi0,if(pool=='global')phi0 else -phi0*.7)
 acvf<-function(phi){a<-as.numeric(ARMAacf(ar=phi,lag.max=n));a/(1-sum(phi*a[seq_along(phi)+1L]))}
 gtrue<-lapply(phi,acvf)
 sigma<-matrix(0,length(keep),length(keep))
 for(i in seq_along(keep))for(j in seq_along(keep))if(ids[i]==ids[j]&&(policy=='preserve'||segments[i]==segments[j]))sigma[i,j]<-gtrue[[ids[i]]][abs(keep[i]-keep[j])+1L]
 error<-t(chol(sigma))%*%matrix(rnorm(length(keep)*64),nrow=length(keep));ys<-xs%*%matrix(rep(c(2,-.7,1.1),64),3)+error
 residual<-M%*%ys;centered<-C%*%residual
 raw<-vapply(pairs,function(ii)mean(vapply(seq_len(ncol(centered)),function(j)mean(centered[ii[,1],j]*centered[ii[,2],j]),numeric(1))),numeric(1))
 gamma<-as.vector(solve(A,raw));estimated<-list();conditions<-numeric();fractions<-numeric()
 for(g in 0:(ng-1)){
  gam<-gamma[g*(lag+1)+seq_len(lag+1)];T<-toeplitz(gam[seq_len(order)]/gam[1]);rhs<-gam[2:(order+1)]/gam[1]
  chol(T);coef<-as.vector(solve(T,rhs));estimated[[g+1L]]<-coef;sd<-svd(T)$d;conditions[g+1L]<-max(sd)/min(sd);fractions[g+1L]<-1-sum(coef*rhs)
  stopifnot(fractions[g+1L]>0,all(Mod(polyroot(c(1,-coef)))>1))
 }
 if(pool=='global'){estimated<-rep(estimated,2);conditions<-rep(conditions,2);fractions<-rep(fractions,2)}
 gest<-lapply(estimated,acvf)
 estimated_sigma<-sigma
 for(i in seq_along(keep))for(j in seq_along(keep))if(ids[i]==ids[j]&&(policy=='preserve'||segments[i]==segments[j]))estimated_sigma[i,j]<-gest[[ids[i]]][abs(keep[i]-keep[j])+1L]
 P<-solve(estimated_sigma);v<-solve(crossprod(xs,P%*%xs));beta<-v%*%crossprod(xs,P%*%ys);res<-ys-xs%*%beta;variance<-colSums(res*(P%*%res))/(nrow(xs)-ncol(xs));tstat<-beta[3,]/sqrt(v[3,3]*variance)
 original_y<-matrix(0,n,64);original_y[keep+1L,]<-ys
 cases[[length(cases)+1L]]<-list(order=order,policy=policy,pool=pool,conditions=conditions,innovation_fraction=fractions,y=unname(original_y),residual=unname(residual),bias=unname(A),pair_counts=vapply(pairs,nrow,integer(1)),gamma=gamma,coefficients=estimated,beta=unname(beta),variance=variance,t=tstat)
}
write_json(list(provenance=list(R=R.version.string,seed=60272,method='Dense retained residual projection and lag covariance operators; direct Yule-Walker solve, ARMAacf covariance and dense GLS'),lengths=lengths,keep=keep,x=unname(x),lag=lag,cases=cases),'crates/fmrireg/tests/fixtures/selected_arp_estimation.json',auto_unbox=TRUE,pretty=TRUE,digits=17)
