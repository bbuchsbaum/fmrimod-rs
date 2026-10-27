#!/usr/bin/env Rscript
# Independent V1 reader/writer; requires jsonlite and digest.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L || !args[[1]] %in% c("write", "validate", "rewrite", "oracle")) stop("usage: group_portable_reference.R write OUT | validate BUNDLE | rewrite IN OUT | oracle IN OUT_JSON P")
if (!requireNamespace("jsonlite", quietly=TRUE) || !requireNamespace("digest", quietly=TRUE)) stop("requires jsonlite and digest")
hash <- function(p) digest::digest(file=p, algo="sha256", serialize=FALSE)
validate <- function(d) {
 m <- jsonlite::fromJSON(file.path(d,"manifest.json"), simplifyVector=FALSE); stopifnot(identical(m$format,"fmri-group-estimate"), identical(m$version,1L)); n<-m$shape[[1]]; f<-m$shape[[2]]
 check <- function(name,x) { p<-file.path(d,name); stopifnot(identical(x$file,name), file.info(p)$size==n*f*8, identical(hash(p),x$sha256)); matrix(readBin(p,"double",n=n*f,size=8L,endian="little"),nrow=n,ncol=f) }
 invisible(list(manifest=m,effects=check("effects.f64",m$effects),variances=if(is.null(m$variances) || length(m$variances)==0L) NULL else check("variances.f64",m$variances)))
}
if (args[[1]] == "validate") { validate(args[[2]]); cat("valid V1 bundle\n") } else if (args[[1]] == "oracle") {
 if (length(args) != 4L) stop("usage: group_portable_reference.R oracle IN OUT_JSON P")
 if (!requireNamespace("metafor", quietly=TRUE)) stop("oracle requires metafor")
 x <- validate(args[[2]]); if (is.null(x$variances)) stop("oracle requires variances.f64")
 n <- nrow(x$effects); nf <- ncol(x$effects); p <- as.integer(args[[4]])
 if (is.na(p) || p < 1L || p >= n) stop("P must satisfy 1 <= P < subjects")
 X <- outer(seq_len(n), 0:(p - 1L), function(i,j) cos(pi * (i - .5) * j / n))
 coefficients <- vector("list", nf); ses <- vector("list", nf); q_fixed <- rep(NA_real_, nf); used_n <- integer(nf); df <- integer(nf)
 for (j in seq_len(nf)) {
   ok <- is.finite(x$effects[,j]) & is.finite(x$variances[,j]) & x$variances[,j] > 0
   used_n[[j]] <- sum(ok); df[[j]] <- used_n[[j]] - p
   if (used_n[[j]] <= p) { coefficients[[j]] <- rep(NA_real_, p); ses[[j]] <- rep(NA_real_, p); next }
   fit <- tryCatch(suppressWarnings(metafor::rma.uni(x$effects[ok,j], x$variances[ok,j], mods=X[ok,,drop=FALSE], intercept=FALSE, method="FE", test="z")), error=function(e) NULL)
   if (is.null(fit)) { coefficients[[j]] <- rep(NA_real_, p); ses[[j]] <- rep(NA_real_, p); next }
   coefficients[[j]] <- as.numeric(fit$b); ses[[j]] <- as.numeric(fit$se); q_fixed[[j]] <- as.numeric(fit$QE)
 }
 out <- list(oracle="metafor::rma.uni", method="FE", test="z", design="X[i,j]=cos(pi*(i-.5)*(j-1)/n), one-based i,j", shape=list(subjects=n,features=nf,parameters=p), coefficients=lapply(coefficients,I), ses=lapply(ses,I), q_fixed=I(as.numeric(q_fixed)), n=I(as.integer(used_n)), df=I(as.integer(df)), receipt=list(R=R.version.string, metafor=as.character(utils::packageVersion("metafor")), session=capture.output(sessionInfo())))
 jsonlite::write_json(out,args[[3]],auto_unbox=TRUE,pretty=TRUE,digits=17,na="null"); cat("wrote metafor oracle\n")
} else if (args[[1]] == "rewrite") {
 if (length(args) != 3L) stop("usage: group_portable_reference.R rewrite IN OUT"); x <- validate(args[[2]]); d <- args[[3]]; if (file.exists(d)) stop("output must not exist"); dir.create(d)
 writeBin(as.double(x$effects), file.path(d,"effects.f64"), size=8L, endian="little"); x$manifest$effects <- list(file="effects.f64",sha256=hash(file.path(d,"effects.f64")))
 if (!is.null(x$variances)) { writeBin(as.double(x$variances), file.path(d,"variances.f64"), size=8L, endian="little"); x$manifest$variances <- list(file="variances.f64",sha256=hash(file.path(d,"variances.f64"))) }
 jsonlite::write_json(x$manifest,file.path(d,"manifest.json"),auto_unbox=TRUE,pretty=TRUE,digits=17); validate(d); cat("rewrote V1 bundle\n")
} else {
 d<-args[[2]]; if(file.exists(d)) stop("output must not exist"); dir.create(d); y<-matrix(c(1,2,0,NaN,5,6),nrow=3L); writeBin(as.double(y),file.path(d,"effects.f64"),size=8L,endian="little")
 m<-list(format="fmri-group-estimate",version=1L,shape=c(3L,2L),metadata=list(subjects=lapply(1:3,function(i) list(observation=paste0("o",i),dataset="R",participant=paste0("p",i),measurement="run-1")),estimand=list(id="synthetic",definition="R fixture",units="a.u."),feature_space=list(kind="index",namespace="R",ids=c("f0","f1")),variance_origins=NULL,fit_receipts=replicate(3,list(model="synthetic",noise="iid",selection="all"),simplify=FALSE),admission=list(target="fixture",evidence="synthetic")),effects=list(file="effects.f64",sha256=hash(file.path(d,"effects.f64"))),variances=NULL)
 jsonlite::write_json(m,file.path(d,"manifest.json"),auto_unbox=TRUE,pretty=TRUE); validate(d); cat("wrote V1 fixture\n")
}
