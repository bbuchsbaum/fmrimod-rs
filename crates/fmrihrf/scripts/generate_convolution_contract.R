#!/usr/bin/env Rscript
# Install a fresh copy of ~/code/fmrihrf into an isolated R library first, then
# set R_LIBS to that library. This script never installs into or edits a source.
suppressPackageStartupMessages(library(fmrihrf))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "tests/fixtures/convolution_contract.json"
source <- path.expand(Sys.getenv("FMRIHRF_R_SOURCE", "~/code/fmrihrf"))
sha <- system2("git", c("-C", shQuote(source), "rev-parse", "HEAD"), stdout = TRUE)
files <- c(list.files(file.path(source, "R"), full.names=TRUE),
           list.files(file.path(source, "src"), pattern="\\.(cpp|h)$", full.names=TRUE))
hashes <- as.list(tools::md5sum(files))
names(hashes) <- substring(names(hashes), nchar(source)+2L)

cases <- list(
  list(name="mixed_off_grid", grid=seq(0, 65, by=.7), onsets=c(.13, 5.27, 11.91, 23.17),
       durations=c(0, .15, 2.71, 14.13), amplitudes=c(1, -.7, 2, .4)),
  list(name="single_point", grid=7.13, onsets=c(.21, 2.77),
       durations=c(0, .07), amplitudes=c(1, -2)),
  list(name="irregular_duplicates", grid=c(0, .1, 1.13, 2.2, 2.2, 5.71, 13.11, 34.2, 60),
       onsets=c(.17, 4.31), durations=c(1.07, 13.73), amplitudes=c(1.1, -.3)),
  list(name="late_block_tail", grid=c(80.13, 100.1, 103.7, 110.3, 129.9, 140),
       onsets=c(0, 50.71), durations=c(100, 70.31), amplitudes=c(1, -.3))
)
results <- list()
for (case in cases) for (basis in c("spmg1", "spmg3")) for (summate in c(TRUE,FALSE)) {
  h <- if (basis == "spmg1") HRF_SPMG1 else HRF_SPMG3
  reg <- regressor(case$onsets, hrf=h, duration=case$durations,
                   amplitude=case$amplitudes, span=40, summate=summate)
  for (precision in c(.33,.05)) for (method in c("conv","loop")) {
    values <- evaluate(reg, case$grid, precision=precision, method=method)
    values <- matrix(values,nrow=length(case$grid))
    results[[length(results)+1L]] <- c(case,list(basis=basis,span=reg$span,summate=summate,
      precision=precision,method=method,ncols=ncol(values),expected=unname(values)))
  }
}
normalization_cases <- lapply(c("none","spm","unit_peak","unit_integral","unit_peak_per_basis"),function(mode) {
  time <- c(-1,0,.137,2.19,5.31,9.77,24,27)
  list(mode=mode,time=time,expected=unname(as.matrix(evaluate(normalize_hrf(HRF_SPMG3,mode),time))))
})
dir.create(dirname(output), recursive=TRUE, showWarnings=FALSE)
write_json(list(provenance=list(source=source, commit=sha, source_md5=hashes,
  package_version=as.character(packageVersion("fmrihrf")), R=R.version.string,
  methods=c("freshly compiled conv", "independent R loop")),cases=results,normalization_cases=normalization_cases),
  output, auto_unbox=TRUE, pretty=TRUE,digits=17)
cat(length(results),"cases written to",output,"\n")

# Persist the formerly /tmp-only reference needed by the ordinary test suite.
time <- seq(0,30,by=.1)
write.table(data.frame(time=time,r_original=hrf_spmg1(time),r_alternate=hrf_spmg1(time,A1=1/gamma(6))),
  file=file.path(dirname(output),"spm_canonical_r_reference.csv"),sep=",",quote=FALSE,row.names=FALSE)
