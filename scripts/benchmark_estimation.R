suppressPackageStartupMessages(library(jsonlite))
cases <- list()
for (size in list(c(300,8,1000),c(300,8,10000),c(1200,32,1000))) {
 n<-size[1]; p<-size[2]; v<-size[3]
 X<-outer(seq_len(n),seq_len(p),function(i,j) sin(i*j*.013)); X[,1]<-1
 Y<-outer(seq_len(n),seq_len(v),function(i,j) sin(i*.071+j*.03)+.1*cos(i*1.3))
 start<-proc.time()[[3]]; q<-qr(X); setup<-proc.time()[[3]]-start
 fit<-function() {beta<-qr.coef(q,Y); residual<-Y-X%*%beta; list(beta=beta,variance=colSums(residual^2)/(n-q$rank))}
 warm<-fit(); times<-replicate(5,system.time(invisible(fit()))[['elapsed']])
 cases[[length(cases)+1L]]<-list(n=n,p=p,features=v,prepare_seconds=setup,median_fit_seconds=median(times),beta_sum=sum(warm$beta),variance_sum=sum(warm$variance),repetitions=5)
}
write_json(list(R=R.version.string,BLAS=extSoftVersion()[['BLAS']],cases=cases),'evidence/r-throughput.json',auto_unbox=TRUE,pretty=TRUE,digits=17)
