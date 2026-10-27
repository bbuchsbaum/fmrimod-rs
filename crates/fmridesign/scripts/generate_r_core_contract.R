#!/usr/bin/env Rscript
# Use R_LIBS pointing to a freshly built fmrihrf package; load R fmridesign
# source without editing or rebuilding that source tree.
source <- path.expand(Sys.getenv("FMRIDESIGN_R_SOURCE", "~/code/fmridesign"))
suppressPackageStartupMessages(pkgload::load_all(source,quiet=TRUE,compile=FALSE))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "tests/fixtures/r_core_contract.json"
frame <- sampling_frame(c(24,17,21),TR=c(1.5,1.1,2),start_time=c(.4,.2,.6),precision=.05)
events <- data.frame(onset=c(.13,27.9,2.41,.27,37.11),duration=c(0,1.13,.65,.15,2.71),
                     condition=factor(c("A","B","A","B","A")),block=factor(c(1,1,2,3,3),levels=1:3))
designs <- lapply(c("spmg1","spmg2","spmg3"),function(basis) {
  formula <- as.formula(paste0('onset ~ hrf(condition, basis="',basis,'", precision=.05)'))
  model <- event_model(formula,data=events,block=~block,sampling_frame=frame,
                       durations=events$duration,precision=.05)
  matrix <- design_matrix(model)
  nb <- switch(basis,spmg1=1L,spmg2=2L,spmg3=3L)
  # Current R aligns values AND names in condition-major order. Rust retains
  # basis-major order, so permute both together and verify each condition.
  perm <- as.vector(t(matrix(seq_len(2L*nb),nrow=nb)))
  reordered <- as.matrix(matrix)[,perm,drop=FALSE]
  for (j in seq_len(2L)) {
    single <- events; single$amplitude <- as.numeric(single$condition == levels(single$condition)[j])
    single_formula <- as.formula(paste0('onset ~ hrf(amplitude, basis="',basis,'", precision=.05)'))
    single_model <- event_model(single_formula,data=single,block=~block,sampling_frame=frame,
                                durations=single$duration,precision=.05)
    stopifnot(max(abs(as.matrix(design_matrix(single_model))-reordered[,seq(j,2L*nb,by=2L),drop=FALSE])) < 1e-12)
  }
  list(basis=basis,expected=unname(reordered),names=colnames(matrix)[perm],
       r_raw_names=colnames(matrix),
       r_raw_expected=unname(as.matrix(matrix)),r_columns_for_basis_major=perm)
})
modulated_events <- events
modulated_events$rating <- c(-1.3,2.2,0,.45,-.8)
modulations <- lapply(c("rating","condition:rating"),function(term) {
  # Build each interaction amplitude explicitly; this also gives an independent
  # oracle for the Rust formula interaction parser.
  if (term == "rating") amps <- matrix(modulated_events$rating,ncol=1L)
  else amps <- model.matrix(~condition-1,events)*modulated_events$rating
  pieces <- lapply(seq_len(ncol(amps)),function(j) {
    data <- modulated_events; data$amplitude <- amps[,j]
    model <- event_model(onset~hrf(amplitude,basis="spmg1",precision=.05),data=data,
                         block=~block,sampling_frame=frame,durations=events$duration,precision=.05)
    as.matrix(design_matrix(model))
  })
  list(term=term,expected=unname(do.call(cbind,pieces)))
})
baselines <- lapply(c(1L,3L),function(degree) {
  model <- baseline_model(basis="poly",degree=degree,sframe=frame,intercept="runwise")
  matrix <- design_matrix(model)
  list(basis="poly",degree=degree,expected=unname(as.matrix(matrix)),names=colnames(matrix))
})
for (basis in c("bs","ns")) for (df in c(3L,5L)) {
  model <- baseline_model(basis=basis,degree=df,sframe=frame,intercept="runwise")
  matrix <- as.matrix(design_matrix(model))
  baselines[[length(baselines)+1L]] <- list(basis=basis,degree=df,expected=unname(matrix),names=colnames(matrix))
}
spline_cases <- list()
x <- c(0,.03,.2,.7,1.1,1.9,2.8,4.2,5)
xnew <- c(-1,-.5,0,.13,2.9,5,5.5,6)
for (kind in c("bs","ns")) for (intercept in c(FALSE,TRUE)) for (df in c(4L,6L)) {
  basis <- if(kind=="bs") splines::bs(x,df=df,intercept=intercept) else splines::ns(x,df=df,intercept=intercept)
  spline_cases[[length(spline_cases)+1L]] <- list(kind=kind,intercept=intercept,df=df,x=x,xnew=xnew,
    expected=unname(basis),predicted=unname(suppressWarnings(predict(basis,xnew))))
}
files <- list.files(file.path(source,"R"),full.names=TRUE)
hashes <- as.list(tools::md5sum(files)); names(hashes) <- substring(names(hashes),nchar(source)+2L)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
jsonlite::write_json(list(provenance=list(source=source,
  commit=system2("git",c("-C",shQuote(source),"rev-parse","HEAD"),stdout=TRUE),
  source_md5=hashes,hrf_version=as.character(packageVersion("fmrihrf")),R=R.version.string),
  frame=list(blocklens=frame$blocklens,tr=frame$TR,start_time=frame$start_time,precision=.05,
             samples=fmrihrf::samples(frame,global=TRUE)),
  events=events,designs=designs,baselines=baselines,modulated_events=modulated_events,modulations=modulations,spline_cases=spline_cases),output,pretty=TRUE,digits=17,auto_unbox=TRUE)
cat("R event designs and baselines written to",output,"\n")
