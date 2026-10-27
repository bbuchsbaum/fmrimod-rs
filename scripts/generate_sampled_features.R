suppressPackageStartupMessages(library(fmrihrf))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(normalizePath(find.package('fmrihrf'))==normalizePath('evidence/r-library/fmrihrf'))
source<-path.expand('~/code/fmrihrf');files<-list.files(file.path(source,'R'),full.names=TRUE)
hashes<-as.list(tools::md5sum(files));names(hashes)<-substring(names(hashes),nchar(source)+2L)
inputs<-list(list(values=cbind(c(2,3,0,5,8,1),c(4,-1,3,2,0,7)),mask=c(TRUE,TRUE,FALSE,TRUE,FALSE,TRUE)),list(values=cbind(c(10,11,15,12,9),c(-2,1,4,3,0)),mask=c(TRUE,FALSE,TRUE,TRUE,TRUE)))
cases<-list();lens<-c(16L,13L);tr<-c(.7,1.1);start<-c(.15,.2);precision<-.025
for(irregular in c(FALSE,TRUE))for(center in c(FALSE,TRUE))for(scale in c('none','sd'))for(masked in c(FALSE,TRUE)){
 expected<-list();amps<-list();onsets<-list();widths<-list()
 for(r in 1:2){y<-inputs[[r]]$values;n<-nrow(y);tm<-if(irregular)cumsum(c(.1,rep(c(.15,.3,.25),length.out=n-1)))else .1+(0:(n-1))*.2
  regs<-lapply(1:2,function(j)if(irregular)feature_regressor(y[,j],HRF_SPMG2,times=tm,center=center,scale=scale,mask=if(masked)inputs[[r]]$mask else NULL,span=24)else feature_regressor(y[,j],HRF_SPMG2,dt=.2,start=.1,center=center,scale=scale,mask=if(masked)inputs[[r]]$mask else NULL,span=24))
  grid<-start[r]+(0:(lens[r]-1))*tr[r];val<-lapply(regs,function(reg)evaluate(reg,grid,precision=precision))
  expected[[r]]<-cbind(val[[1]][,1],val[[2]][,1],val[[1]][,2],val[[2]][,2])
  amps[[r]]<-cbind(regs[[1]]$amplitude,regs[[2]]$amplitude);onsets[[r]]<-regs[[1]]$onsets;widths[[r]]<-regs[[1]]$duration
 }
 cases[[length(cases)+1L]]<-list(irregular=irregular,center=center,scale=scale,masked=masked,onsets=onsets,widths=widths,amplitudes=unname(do.call(rbind,amps)),expected=unname(do.call(rbind,expected)))
}
write_json(list(provenance=list(source=source,commit=system2('git',c('-C',source,'rev-parse','HEAD'),stdout=TRUE),source_md5=hashes,R=R.version.string),inputs=inputs,lens=lens,tr=tr,start=start,precision=precision,cases=cases),'crates/fmri-hrf/tests/fixtures/sampled_features.json',auto_unbox=TRUE,pretty=TRUE,digits=17)
