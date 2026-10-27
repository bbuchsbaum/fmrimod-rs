suppressPackageStartupMessages(library(fmrihrf))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(normalizePath(find.package('fmrihrf')) == normalizePath('evidence/r-library/fmrihrf'))
time<-c(-1,0,.001,.5,2.7,6,12,18,23.999,24,25)
cases<-list()
for(degree in 1:4) for(n in unique(c(max(1,degree-1),5,8))) {
 cases[[length(cases)+1]]<-list(degree=degree,n=n,time=time,expected=unname(fmrihrf::hrf_bspline(time,span=24,N=n,degree=degree)))
}
manifest<-fromJSON('evidence/source-manifest.json',simplifyVector=FALSE)
write_json(list(provenance=manifest[['R:fmrihrf']],cases=cases),'crates/fmri-hrf/tests/fixtures/current_splines.json',pretty=TRUE,auto_unbox=TRUE,digits=17)
