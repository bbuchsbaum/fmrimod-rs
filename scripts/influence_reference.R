library(metafor)
dir.create('evidence/completion/influence',recursive=TRUE,showWarnings=FALSE)
dir.create('crates/fmri-group/tests/fixtures',recursive=TRUE,showWarnings=FALSE)
flat<-function(x)as.numeric(t(x))
n<-10L;x<-cbind(1,seq(-1,1,length.out=n));y<-3*sin(seq_len(n)*1.3)+.4*x[,2]
v<-.2+seq_len(n)/20
result<-list()
for(method in c('FE','DL','PM')) {
 full<-rma.uni(y,v,mods=x,intercept=FALSE,method=method,control=list(tol=1e-12,threshold=1e-10,maxiter=1000))
 extract<-function(f)list(effect=as.numeric(f$b),se=as.numeric(f$se),tau=f$tau2)
 fixed<-refit<-fixed_mkh<-vector('list',n)
 full_mkh<-rma.uni(y,v,mods=x,intercept=FALSE,method=method,test="adhoc",control=list(tol=1e-12,threshold=1e-10,maxiter=1000))
 for(i in seq_len(n)) {
   refit[[i]]<-extract(rma.uni(y[-i],v[-i],mods=x[-i,,drop=FALSE],intercept=FALSE,method=method,control=list(tol=1e-12,threshold=1e-10,maxiter=1000)))
   fixed[[i]]<-extract(rma.uni(y[-i],v[-i],mods=x[-i,,drop=FALSE],intercept=FALSE,method=method,tau2=full$tau2))
   fixed_mkh[[i]]<-extract(rma.uni(y[-i],v[-i],mods=x[-i,,drop=FALSE],intercept=FALSE,method=method,tau2=full$tau2,test="adhoc"))
 }
 result[[method]]<-list(full=extract(full),fixed=fixed,refit=refit,full_mkh=extract(full_mkh),fixed_mkh=fixed_mkh)
}
jsonlite::write_json(list(n=n,x=flat(x),y=y,v=v,results=result,metafor=as.character(packageVersion('metafor'))),
 'crates/fmri-group/tests/fixtures/influence_reference.json',pretty=TRUE,auto_unbox=TRUE,digits=17)
writeLines(capture.output(sessionInfo()),'evidence/completion/influence/r-session.txt')
cat('PASS: metafor FE/DL/PM full and exact fixed/refitted tau deletion references\n')
