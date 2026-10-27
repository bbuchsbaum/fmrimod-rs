# Independent dense oracle for fmri-group repeated-measures witnesses.
#
# This script intentionally does not call fmrigds or metafor.  It constructs
# each V directly, uses Cholesky whitening plus QR for GLS, and uses `optim`
# only to provide an independent optimizer witness for a correlated RI/slope
# synthetic problem.  Run from the repository root:
#   Rscript scripts/repeated_reference.R

stopifnot(requireNamespace("jsonlite", quietly = TRUE))

dense_gls <- function(y, X, blocks, fit = c("ML", "REML")) {
  fit <- match.arg(fit)
  L <- lapply(blocks, chol) # R chol is upper triangular: t(U) U = V
  ends <- cumsum(vapply(blocks, nrow, integer(1))); begins <- c(1L, head(ends, -1L) + 1L)
  yw <- unlist(Map(function(lo, hi, u) forwardsolve(t(u), y[lo:hi]), begins, ends, L), use.names = FALSE)
  Xw <- do.call(rbind, Map(function(lo, hi, u) forwardsolve(t(u), X[lo:hi, , drop = FALSE]), begins, ends, L))
  qr_x <- qr(Xw, tol = 1e-12)
  stopifnot(qr_x$rank == ncol(X))
  r <- qr.resid(qr_x, yw)
  beta <- qr.coef(qr_x, yw)
  R <- qr.R(qr_x)[seq_len(ncol(X)), seq_len(ncol(X)), drop = FALSE]
  cov_beta <- chol2inv(R)
  logdet_v <- 2 * sum(vapply(L, function(u) sum(log(diag(u))), numeric(1)))
  # qrx uses a column-pivoted QR; the determinant is invariant to pivots.
  logdet_a <- 2 * sum(log(abs(diag(R))))
  n <- length(y); p <- ncol(X)
  objective <- .5 * (logdet_v + sum(r * r) + if (fit == "REML") logdet_a else 0)
  list(beta = as.numeric(beta), covariance = unname(cov_beta), objective = objective,
       sse = sum(r * r), logdet_v = logdet_v, logdet_a = logdet_a,
       normalizing_constant = (if (fit == "ML") n else n - p) * log(2 * pi) / 2)
}

ri_blocks <- function(tau2, eta, s = .3) {
  replicate(4, diag(s + eta, 3) + matrix(tau2, 3, 3), simplify = FALSE)
}
y <- rep(c(-1, 0, 1, 2), each = 3)
X <- matrix(1, length(y), 1)
fixed_ml <- dense_gls(y, X, ri_blocks(1.15, 0), "ML")
fixed_reml <- dense_gls(y, X, ri_blocks(47 / 30, 0), "REML")
stopifnot(all.equal(fixed_ml$beta, .5, tolerance = 1e-12),
          all.equal(fixed_ml$covariance[1, 1], .3125, tolerance = 1e-12),
          all.equal(fixed_reml$covariance[1, 1], 5 / 12, tolerance = 1e-12))

# A correlated RI/slope example with non-diagonal sampling covariance.  The objective is
# profiled over fixed effects by dense QR and is optimized in PSD Cholesky
# coordinates, independently from the Rust projected-gradient implementation.
t <- rep(c(-1, 0, 1), 5)
xi <- rep(seq(-1, 1, length.out = 5), each = 3)
Xi <- matrix(1, length(t), 1)
yi <- c(-2.3, -0.7, 1.1, -1.0, -0.4, 0.5, -0.8, 0.8, 2.5,
         1.8, 1.6, 1.5, 1.1, 3.6, 5.0)
Si <- replicate(5, matrix(c(.25,.04,.01,.04,.30,.03,.01,.03,.28), 3, 3), simplify = FALSE)
interior_objective <- function(theta) {
  if (any(!is.finite(theta)) || any(abs(theta) > 20)) return(Inf)
  l <- matrix(c(exp(theta[1]), 0, theta[2], exp(theta[3])), 2, 2)
  g <- l %*% t(l); eta <- exp(theta[4])
  z <- cbind(1, c(-1,0,1))
  b <- lapply(Si, function(s) s + eta * diag(3) + z %*% g %*% t(z))
  tryCatch(dense_gls(yi, Xi, b, "ML")$objective, error = function(e) Inf)
}
starts <- rbind(c(log(.1),0,log(.1),log(.05)), c(log(.7),.1,log(.5),log(.2)), c(log(1),-.4,log(.8),log(.5)))
fits <- lapply(seq_len(nrow(starts)), function(i) optim(starts[i, ], interior_objective, method = "BFGS", control = list(reltol = 1e-12, maxit = 2000)))
best <- fits[[which.min(vapply(fits, `[[`, numeric(1), "value"))]]

# Unequal visit counts and individual sampling blocks, with a one-dimensional
# independent REML optimizer witness in addition to fixed-covariance GLS.
sizes <- c(2L,3L,4L,2L)
ragged_t <- unlist(lapply(sizes, function(n) seq(-1, 1, length.out=n)))
ragged_y <- c(-1.8,-.1, -.4,.6,1.8, .1,.4,1.3,2.1, 1.4,2.9)
ragged_x <- cbind(1,ragged_t)
ragged_s <- lapply(seq_along(sizes), function(i) diag(.2+.03*i,sizes[i])+matrix(.02,sizes[i],sizes[i]))
ragged_z <- split(ragged_t,rep(seq_along(sizes),sizes))
ragged_g <- matrix(c(.7,.12,.12,.25),2)
ragged_v <- Map(function(s,t) {z<-cbind(1,t);s+.08*diag(length(t))+z%*%ragged_g%*%t(z)},ragged_s,ragged_z)
ragged_fixed <- dense_gls(ragged_y,ragged_x,ragged_v,"REML")
ragged_objective <- function(tau) dense_gls(ragged_y,ragged_x,lapply(ragged_s,function(s) s+matrix(tau,nrow(s),ncol(s))),"REML")$objective
ragged_opt <- optimize(ragged_objective,c(0,10),tol=1e-11)
stopifnot(ragged_opt$minimum>0,ragged_opt$minimum<10)

out <- list(
  oracle = "independent dense base-R Cholesky whitening and QR; not compiled fmrigds parity",
  ragged = list(sizes=sizes,time=ragged_t,y=ragged_y,sampling_blocks=ragged_s,
                g=ragged_g,eta=.08,fixed=ragged_fixed,ri_tau=ragged_opt$minimum,ri_objective=ragged_opt$objective),
  ri = list(y = y, sampling_variance = .3, tau_ml = 1.15, eta_ml = 0,
            ml = fixed_ml, tau_reml = 47 / 30, eta_reml = 0, reml = fixed_reml),
  interior_slope = list(time = t, x = xi, y = yi, sampling_blocks = Si,
                        start_objectives = vapply(fits, `[[`, numeric(1), "value"),
                        optimum = best$value, par = best$par, convergence = best$convergence),
  generated_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
dir.create("crates/fmri-group/tests/fixtures", recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(out, "crates/fmri-group/tests/fixtures/repeated_reference.json", auto_unbox = TRUE, digits = 17, pretty = TRUE)
