suppressPackageStartupMessages(library(jsonlite))
lengths<-c(14L,11L,7L);n<-sum(lengths)
# Zero-based original rows: leading/interior/trailing gaps and a fully excluded run.
keep<-setdiff(0:(n-1),c(0L,2L,3L,10L,13L,14L,18L,19L,24L,25:31))
x<-cbind(1,(0:(n-1))/n,sin((0:(n-1))*.6))
y<-outer(0:(n-1),1:5,function(i,j)2+.07*i+.3*j+sin(i*.7+j*.4)+.2*cos(i*.13*j))
coeffs<-list(numeric(),.96,c(.5,-.2),c(.45,-.16,.07),c(.4,-.15,.08,-.03))
covariance<-function(n,phi,start){
 if(length(phi)==0)return(diag(n))
 if(start=='stationary'){
  a<-as.numeric(ARMAacf(ar=phi,lag.max=max(n-1,length(phi))))
  g<-a/(1-sum(phi*a[seq_along(phi)+1L]))
  return(toeplitz(g[seq_len(n)]))
 }
 a<-diag(n)
 for(i in seq_len(n))for(k in seq_along(phi))if(i>k)a[i,i-k]<- -phi[k]
 h<-solve(a);tcrossprod(h)
}
cases<-list()
for(order in 0:5)for(start in c('stationary','zero'))for(policy in c('preserve','restart')){
 phi<-if(order==5) -.85 else coeffs[[order+1L]];phis<-list(phi,phi*.7,phi*.5)
 sigma<-matrix(0,length(keep),length(keep));offset<-0L
 for(run in seq_along(lengths)){
  idx<-which(keep>=offset & keep<offset+lengths[run]);local<-keep[idx]-offset
  if(length(idx)) {
   if(policy=='preserve')sigma[idx,idx]<-covariance(lengths[run],phis[[run]],start)[local+1L,local+1L,drop=FALSE]
   else {
    groups<-cumsum(c(TRUE,diff(local)!=1L))
    for(g in unique(groups)){ ii<-idx[groups==g];sigma[ii,ii]<-covariance(length(ii),phis[[run]],start) }
   }
  }
  offset<-offset+lengths[run]
 }
 xs<-x[keep+1L,,drop=FALSE];ys<-y[keep+1L,,drop=FALSE];precision<-solve(sigma)
 v<-solve(crossprod(xs,precision%*%xs));beta<-v%*%crossprod(xs,precision%*%ys)
 residual<-ys-xs%*%beta;df<-nrow(xs)-ncol(xs);variance<-colSums(residual*(precision%*%residual))/df
 c<-c(0,1,-1);effect<-as.vector(crossprod(c,beta));se<-sqrt(as.numeric(t(c)%*%v%*%c)*variance);t<-effect/se
 C<-rbind(c(0,1,0),c(0,0,1));b<-C%*%beta;fv<-solve(C%*%v%*%t(C));F<-colSums(b*(fv%*%b))/2/variance
 cases[[length(cases)+1L]]<-list(order=length(phi),start=start,policy=policy,coefficients=phis,covariance=unname(sigma),beta=unname(beta),unit_covariance=unname(v),variance=variance,df=df,t=t,t_p=2*pt(-abs(t),df),effect=effect,se=se,F=F,F_p=pf(F,2,df,lower.tail=FALSE))
}
write_json(list(provenance=list(R=R.version.string,stats=as.character(packageVersion('stats')),method='Dense original AR covariance from stats::ARMAacf or inverse finite zero-history filter, followed by covariance subsetting or independent segments; explicit GLS normal equations'),lengths=lengths,keep=keep,x=unname(x),y=unname(y),cases=cases),'crates/fmrireg/tests/fixtures/selected_gls.json',auto_unbox=TRUE,pretty=TRUE,digits=17)
