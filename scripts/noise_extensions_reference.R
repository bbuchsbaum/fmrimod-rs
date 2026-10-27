#!/usr/bin/env Rscript
# Independent base-R witnesses for the extension contract.  This script does
# not call fmri-noise; it records QR conditional scores and scalar ARMA filters.
conditional_scores <- function(y, max_order, criterion = c("AIC", "BIC")) {
  criterion <- match.arg(criterion); y <- y - mean(y); n <- length(y) - max_order
  out <- lapply(0:max_order, function(p) {
    yy <- y[(max_order + 1):length(y)]
    if (p == 0) e <- yy else {
      z <- sapply(1:p, function(k) y[(max_order + 1 - k):(length(y) - k)])
      fit <- tryCatch(qr.solve(as.matrix(z), yy), error = function(e) NULL)
      if (is.null(fit)) return(c(order = p, observations = n, variance = NA, score = Inf))
      e <- yy - as.matrix(z) %*% fit
    }
    s2 <- mean(e^2); penalty <- if (criterion == "AIC") 2 * (p + 1) else log(n) * (p + 1)
    c(order = p, observations = n, variance = s2, score = n * log(s2) + penalty)
  }); do.call(rbind, out)
}
arma_zero_history <- function(x, phi, theta, starts = 1L) {
  e <- numeric(length(x)); starts <- sort(unique(starts))
  for (t in seq_along(x)) {
    first <- t %in% starts; lag <- if (first) 0L else min(t - max(starts[starts <= t]), max(length(phi), length(theta)))
    e[t] <- x[t] - sum(phi[seq_len(min(length(phi), lag))] * rev(x[(t-min(length(phi),lag)):(t-1)])) - sum(theta[seq_len(min(length(theta), lag))] * rev(e[(t-min(length(theta),lag)):(t-1)]))
  }; e
}
pooled_feature_acvf <- function(resid, parcels, order) {
  ids <- sort(unique(parcels)); out <- vector("list", length(ids)); names(out) <- ids
  all_g <- numeric(order + 1); total <- ncol(resid)
  for (id in ids) {
    z <- resid[, parcels == id, drop = FALSE]; z <- sweep(z, 2, colMeans(z))
    g <- sapply(0:order, function(k) mean(colSums(z[(k+1):nrow(z),,drop=FALSE] * z[1:(nrow(z)-k),,drop=FALSE]) / (nrow(z)-k)))
    out[[as.character(id)]] <- g; all_g <- all_g + ncol(z) / total * g
  }
  list(parcel = out, global = all_g)
}
hr_arma_reference <- function(y, p, q, high_order, iterations = 0L) {
  y <- as.numeric(y) - mean(y); n <- length(y)
  if (n <= high_order + p + q) stop("insufficient data")
  z0 <- sapply(1:high_order, function(k) y[(high_order + 1 - k):(n-k)])
  a0 <- qr.solve(as.matrix(z0), y[(high_order+1):n]); e <- numeric(n)
  e[(high_order+1):n] <- y[(high_order+1):n] - as.matrix(z0) %*% a0
  m <- max(high_order, p, q); phi <- numeric(p); theta <- numeric(q)
  for (ii in 0:iterations) {
    idx <- (m+1):n
    z <- cbind(if (p) sapply(1:p, function(k) y[idx-k]) else NULL,
               if (q) sapply(1:q, function(k) e[idx-k]) else NULL)
    b <- qr.solve(as.matrix(z), y[idx]); if (p) phi <- b[seq_len(p)]; if (q) theta <- b[p + seq_len(q)]
    e <- arma_zero_history(y, phi, theta, 1L)
  }
  list(phi = phi, theta = theta, sigma2 = mean(e[(m+1):n]^2), observations = n-m)
}
cat("Reference helpers loaded; choose explicit inputs and record them as fixtures.\n")

# Executable, full-precision fixture generation (including all explicit inputs).
set.seed(811); innovations<-rnorm(120); series<-numeric(120)
for(i in seq_along(series))series[i]<-innovations[i]+if(i>1).4*series[i-1]+.3*innovations[i-1] else 0
hr<-hr_arma_reference(series,1L,1L,8L,1L)
resid<-cbind(sin(seq_len(24)*1.3),-sin(seq_len(24)*1.3),sin(seq_len(24)*2.2))
pooled<-pooled_feature_acvf(resid,c(1L,1L,2L),1L)
source_root<-path.expand('~/code/fmriAR')
source_paths<-c('R/hr_arma.R','R/noise_acvf.R','R/fit_and_apply.R')
hashes<-setNames(lapply(source_paths,function(p)strsplit(system2('shasum',c('-a','256',shQuote(file.path(source_root,p))),stdout=TRUE),' +')[[1]][1]),source_paths)
commit<-system2('git',c('-C',shQuote(source_root),'rev-parse','HEAD'),stdout=TRUE)
fixture<-list(reference=list(repository=source_root,commit=commit,sha256=hashes),
 criterion='common-sample conditional Gaussian criterion; sample centering, common mean penalty omitted',
 series=series,
 r_scores_aic=conditional_scores(series,5L,'AIC'),r_scores_bic=conditional_scores(series,5L,'BIC'),
 r_arma_zero_history=arma_zero_history(c(1,2,4,8,3,6,12,24),.4,.3,c(1L,3L,5L)),
 r_hr=list(p=1L,q=1L,high_order=8L,iterations=1L,phi=hr$phi,theta=hr$theta,sigma2=hr$sigma2,observations=hr$observations),
 r_pooled_feature_acvf=list(parcel_1=pooled$parcel[['1']],parcel_2=pooled$parcel[['2']],global_feature_weighted=pooled$global))
jsonlite::write_json(fixture,'crates/fmri-noise/tests/fixtures/noise_extensions.json',auto_unbox=TRUE,pretty=TRUE,digits=17)
dir.create('evidence/completion/noise',recursive=TRUE,showWarnings=FALSE)
writeLines(capture.output(sessionInfo()),'evidence/completion/noise/r-session.txt')
cat('PASS: regenerated full-precision QR order, HR estimation, pooled ACVF and ARMA recursion fixtures\n')
