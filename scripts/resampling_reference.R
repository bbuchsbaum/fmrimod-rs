# Independent base-R QR and complete enumeration for the Rust max-T API.
dir.create('evidence/completion/resampling',recursive=TRUE,showWarnings=FALSE)
dir.create('crates/fmri-group/tests/fixtures',recursive=TRUE,showWarnings=FALSE)
flat <- function(x) as.numeric(t(x))
n <- 6L
x <- cbind(1,c(-1.2,-.5,.1,.4,.8,1.5))
y <- outer(seq_len(n),seq_len(3),function(i,j) .3*j+.2*i+sin(.71*i*j))
tstat <- function(x,y,term) {
  fit<-lm.fit(x,y); q<-qr(x); inv<-qr.solve(q,diag(nrow(x)))
  se<-sqrt(sum(fit$residuals^2)/(nrow(x)-ncol(x))*sum(inv[term,]^2))
  fit$coefficients[term]/se
}
signs<-as.matrix(expand.grid(rep(list(c(1,-1)),n)))
observed<-apply(y,2,function(y)tstat(x[,1,drop=FALSE],y,1))
null<-t(apply(signs,1,function(s)apply(y*s,2,function(y)tstat(x[,1,drop=FALSE],y,1))))
summarise<-function(obs,null) {
  maxima<-apply(abs(null),1,max)
  list(observed=obs,p=vapply(seq_along(obs),function(j)mean(abs(null[,j])>=abs(obs[j])-1e-12),0.),
       p_max=vapply(abs(obs),function(t)mean(maxima>=t-1e-12),0.),maxima=maxima)
}
perms<-function(v) {if(length(v)==1L)return(matrix(v,1,1));do.call(rbind,lapply(seq_along(v),function(i)cbind(v[i],perms(v[-i]))))}
a<-perms(1:3);b<-perms(4:6)
maps<-do.call(rbind,lapply(seq_len(nrow(a)),function(i)cbind(a[rep(i,nrow(b)),,drop=FALSE],b)))
reduced<-vapply(seq_len(ncol(y)),function(j)fitted(lm(y[,j]~1)),numeric(n))
residual<-y-reduced
observed_fl<-apply(y,2,function(y)tstat(x,y,2))
null_fl<-t(apply(maps,1,function(ix)apply(reduced+residual[ix,,drop=FALSE],2,function(y)tstat(x,y,2))))
c<-c(1,.5);z<-x%*%c(-.5,1)
contrast_t<-function(y) {fit<-lm.fit(x,y);inv<-qr.solve(qr(x),diag(n));sum(c*fit$coefficients)/sqrt(sum(fit$residuals^2)/(n-ncol(x))*sum(drop(c%*%inv)^2))}
reduced_general<-apply(y,2,function(y)drop(z%*%qr.solve(z,y)))
residual_general<-y-reduced_general
observed_general<-apply(y,2,contrast_t)
null_general<-t(apply(maps,1,function(ix)apply(reduced_general+residual_general[ix,,drop=FALSE],2,contrast_t)))
jsonlite::write_json(list(n=n,features=ncol(y),x=flat(x),y=flat(y),
  sign=summarise(observed,null),freedman_lane=summarise(observed_fl,null_fl),freedman_lane_general=summarise(observed_general,null_general)),
  'crates/fmri-group/tests/fixtures/resampling_reference.json',pretty=TRUE,auto_unbox=TRUE,digits=17)
writeLines(capture.output(sessionInfo()),'evidence/completion/resampling/r-session.txt')
cat('PASS: base-R QR, 64 sign flips and 36 restricted residual permutations\n')
