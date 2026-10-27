reference_lib <- normalizePath("evidence/group-audit/r-library", mustWork=TRUE)
library(fmrigds, lib.loc=reference_lib)
set.seed(2710)
root<-"evidence/group-tranche1"
workloads<-list(c(20,1000,0,1),c(100,10000,0,1),c(100,10000,1,1),c(100,100000,0,1),c(100,10000,0,5),c(100,10000,0,20))
records<-list()
for (w in workloads) {
 n<-w[1];nf<-w[2];missing<-w[3];p<-w[4]
 id<-paste(format(n,scientific=FALSE),format(nf,scientific=FALSE),missing,sep="-")
 if(p>1) id<-paste0(id,"-p",p)
 Y<-outer(seq_len(n),seq_len(nf),function(i,j) .25+sin(i*j*.017)+cos(i*.23)*.3)
 V<-outer(seq_len(n),seq_len(nf),function(i,j) .2+((i+3*j)%%17)*.03)
 if(missing==1) {Y[2,seq.int(7,nf,7)]<-NA;Y[5,seq.int(11,nf,11)]<-NA}
 covariates<-if(p>1) as.data.frame(outer(seq_len(n),seq_len(p-1),function(i,j) cos(pi*(i-.5)*j/n))) else data.frame(row.names=seq_len(n))
 rownames(covariates)<-paste0("s",seq_len(n))
 start<-proc.time()[3]
 g<-new_gds(list(beta=array(t(Y),c(nf,n,1)),var=array(t(V),c(nf,n,1))),
  space_sample_labels(paste0("f",seq_len(nf)-1)),paste0("s",seq_len(n)),"effect",col_data=covariates)
 setup<-unname(proc.time()[3]-start)
 for (method in if(p==1) c("ols","fe","dl") else "ols") {
  plan<-if(method=="ols") reduce(g,method="ols:voxelwise",formula=if(p==1) ~1 else reformulate(names(covariates))) else reduce(g,method=if(method=="fe") "fixed" else "random")
  term<-if(p==1) "(Intercept)" else tail(names(covariates),1)
  run<-function(){
   r<-suppressWarnings(compute(plan))
   a<-function(name) as.numeric(assay(r,name))
   if(method=="ols") cbind(a(paste0("coef:",term)),a(paste0("se_coef:",term)),a(paste0("t_coef:",term)),a(paste0("p_coef:",term)),0,0,a("n_obs"))
   else cbind(a("beta_g"),a("se_g"),a("z_g"),a("p_g"),a("Q"),if(method=="dl") a("tau2") else 0,a("n_eff"))
  }
  start<-proc.time()[3];out<-run();first<-unname(proc.time()[3]-start)
  times<-numeric(5)
  for (rep in 1:5) {gc();start<-proc.time()[3];out<-run();times[rep]<-proc.time()[3]-start}
  con<-file(file.path(root,paste0("reference-",id,"-",method,".bin")),"wb")
  writeBin(as.numeric(t(out)),con,size=8,endian="little");close(con)
  records[[length(records)+1]]<-list(id=id,n=n,features=nf,missing=missing,method=method,
    columns=p,setup_seconds=setup,first_seconds=first,seconds=times,median_seconds=median(times))
  cat(id,method,"median",median(times),"seconds\n");flush.console()
 }
}
jsonlite::write_json(records,file.path(root,"r-benchmark.json"),pretty=TRUE,auto_unbox=TRUE,digits=17)
# Explicit independent normal-null calibration inputs, not timing workloads.
n<-20L;nf<-2000L;Y<-matrix(rnorm(n*nf),n,nf)
jsonlite::write_json(list(n=n,features=nf,effects=as.numeric(t(Y))),file.path(root,"null-input.json"),auto_unbox=TRUE,digits=17)
writeLines(capture.output(sessionInfo()),file.path(root,"benchmark-r-session.txt"))
