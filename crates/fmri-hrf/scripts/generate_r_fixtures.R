suppressPackageStartupMessages(library(fmrihrf))
# Use an installed authoritative source build; see PORT_PARITY_AUDIT.md.

# Determine output directory based on where the script is run.
# Canonical location is the Rust repo (`fmrihrf-rs`), but we keep
# backwards compatibility with running from the R repo root.
if (nzchar(Sys.getenv("FMRIHRF_FIXTURE_OUT"))) {
  out_dir <- Sys.getenv("FMRIHRF_FIXTURE_OUT")
} else if (file.exists("Cargo.toml") && dir.exists("tests")) {
  # Running from Rust repo root
  out_dir <- "tests/fixtures"
} else if (file.exists("DESCRIPTION") && dir.exists("../fmrihrf-rs/tests")) {
  # Running from R repo root
  out_dir <- "../fmrihrf-rs/tests/fixtures"
} else {
  # Fallback: create local tests/fixtures
  out_dir <- "tests/fixtures"
}

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

message("Generating HRF fixtures into: ", normalizePath(out_dir, mustWork = FALSE))

## 1. Core scalar HRFs -------------------------------------------------------

times_30 <- seq(0, 30, by = 0.5)

# Gamma
gamma_vals <- fmrihrf:::hrf_gamma(times_30, shape = 6, rate = 1)
write.csv(
  data.frame(time = times_30, value = gamma_vals),
  file = file.path(out_dir, "gamma_hrf_shape6_rate1.csv"),
  row.names = FALSE
)

# Gaussian
gauss_vals <- fmrihrf:::hrf_gaussian(times_30, mean = 6, sd = 2)
write.csv(
  data.frame(time = times_30, value = gauss_vals),
  file = file.path(out_dir, "gaussian_hrf_mean6_sd2.csv"),
  row.names = FALSE
)

# Mexican hat
mex_vals <- fmrihrf:::hrf_mexhat(times_30, mean = 6, sd = 2)
write.csv(
  data.frame(time = times_30, value = mex_vals),
  file = file.path(out_dir, "mexhat_hrf_mean6_sd2.csv"),
  row.names = FALSE
)

# LWU default
lwu_vals <- fmrihrf:::hrf_lwu(times_30)
write.csv(
  data.frame(time = times_30, value = lwu_vals),
  file = file.path(out_dir, "lwu_hrf_default.csv"),
  row.names = FALSE
)

## 1b. Additional analytic HRFs and decorators -------------------------------

# Half-cosine HRF (6-parameter default)
halfcos_vals <- fmrihrf::hrf_half_cosine(times_30)
write.csv(
  data.frame(time = times_30, value = halfcos_vals),
  file = file.path(out_dir, "half_cosine_hrf_default.csv"),
  row.names = FALSE
)

# Lagged SPMG1 (HRF_SPMG1 lagged by 2s)
spmg1_obj <- fmrihrf:::HRF_SPMG1
lagged_spmg1 <- fmrihrf:::lag_hrf(spmg1_obj, lag = 2)
lagged_vals <- fmrihrf:::evaluate(lagged_spmg1, times_30)
write.csv(
  data.frame(time = times_30, value = as.numeric(lagged_vals)),
  file = file.path(out_dir, "spmg1_lag2_hrf.csv"),
  row.names = FALSE
)

# Blocked SPMG1 (width=5s, default precision/half_life/summate, no normalize)
blocked_spmg1 <- fmrihrf:::block_hrf(spmg1_obj, width = 5)
blocked_vals <- fmrihrf:::evaluate(blocked_spmg1, times_30)
write.csv(
  data.frame(time = times_30, value = as.numeric(blocked_vals)),
  file = file.path(out_dir, "spmg1_block_width5_hrf.csv"),
  row.names = FALSE
)

# Multi-lag SPMG1 basis (manual equivalent of gen_hrf_lagged with normalize=FALSE)
lags <- c(0, 2, 4)
lagged_mat <- sapply(lags, function(l) {
  # gen_hrf_lagged semantics: evaluate base HRF at (t - lag)
  fmrihrf:::HRF_SPMG1(times_30 - l)
})
lagged_out <- cbind(time = times_30, as.matrix(lagged_mat))
colnames(lagged_out) <- c("time", paste0("lag", lags))
write.csv(
  lagged_out,
  file = file.path(out_dir, "spmg1_gen_hrf_lagged_lags0_2_4.csv"),
  row.names = FALSE
)

# Multi-width SPMG1 basis (manual equivalent of gen_hrf_blocked with default settings)
widths <- c(1, 2, 4)
blocked_mat_set <- sapply(widths, function(w) {
  blocked <- fmrihrf:::block_hrf(spmg1_obj, width = w)
  fmrihrf:::evaluate(blocked, times_30)
})
blocked_out <- cbind(time = times_30, as.matrix(blocked_mat_set))
colnames(blocked_out) <- c("time", paste0("w", widths))
write.csv(
  blocked_out,
  file = file.path(out_dir, "spmg1_gen_hrf_blocked_widths1_2_4.csv"),
  row.names = FALSE
)

## 2. Tent basis HRF ---------------------------------------------------------

times_24 <- seq(0, 24, by = 0.5)
tent_obj <- fmrihrf:::hrf_tent_generator(nbasis = 5, span = 24)
tent_mat <- fmrihrf:::evaluate(tent_obj, times_24)
tent_out <- cbind(time = times_24, as.matrix(tent_mat))
colnames(tent_out) <- c("time", paste0("b", seq_len(ncol(tent_mat))))
write.csv(
  tent_out,
  file = file.path(out_dir, "tent_hrf_n5_span24.csv"),
  row.names = FALSE
)

## 3. SPMG2 / SPMG3 HRFs -----------------------------------------------------

spm_times <- seq(0, 30, by = 0.5)
h2 <- fmrihrf:::HRF_SPMG2
m2 <- fmrihrf:::evaluate(h2, spm_times)
out2 <- cbind(time = spm_times, as.matrix(m2))
colnames(out2) <- c("time", "c1", "c2")
write.csv(
  out2,
  file = file.path(out_dir, "spmg2_hrf_default.csv"),
  row.names = FALSE
)

h3 <- fmrihrf:::HRF_SPMG3
m3 <- fmrihrf:::evaluate(h3, spm_times)
out3 <- cbind(time = spm_times, as.matrix(m3))
colnames(out3) <- c("time", "c1", "c2", "c3")
write.csv(
  out3,
  file = file.path(out_dir, "spmg3_hrf_default.csv"),
  row.names = FALSE
)

## 4. Block-wise regressor_design examples -----------------------------------

T <- fmrihrf:::sampling_frame(
  blocklens = c(100L, 100L),
  TR = c(2, 2),
  start_time = c(0, 0),
  precision = 2 / 100  # 1/100 of TR, matches Rust example
)
onsets <- c(10, 30, 50, 20, 40, 60)
fac <- c("A", "B", "A", "B", "A", "B")
block <- c(1L, 1L, 1L, 2L, 2L, 2L)
dm <- fmrihrf:::regressor_design(onsets, fac, block, T, precision = T$precision)

times <- fmrihrf:::samples(T, global = TRUE)
if (length(times) == nrow(dm)) {
  design_out <- data.frame(
    time = times,
    A = dm[, 1],
    B = dm[, 2]
  )
  write.csv(
    design_out,
    file = file.path(out_dir, "regressor_design_block_example.csv"),
    row.names = FALSE
  )
} else {
  warning("SamplingFrame samples length does not match regressor_design rows; skipping design fixture.")
}

# Multi-basis design using HRF_SPMG3 (3 basis functions per condition).
dm_spmg3 <- fmrihrf:::regressor_design(onsets, fac, block, T, hrf = fmrihrf:::HRF_SPMG3, precision = T$precision)
if (length(times) == nrow(dm_spmg3)) {
  design_spmg3_out <- data.frame(
    time = times,
    A_c1 = dm_spmg3[, 1],
    A_c2 = dm_spmg3[, 2],
    A_c3 = dm_spmg3[, 3],
    B_c1 = dm_spmg3[, 4],
    B_c2 = dm_spmg3[, 5],
    B_c3 = dm_spmg3[, 6]
  )
  write.csv(
    design_spmg3_out,
    file = file.path(out_dir, "regressor_design_spmg3_block_example.csv"),
    row.names = FALSE
  )
} else {
  warning("SamplingFrame samples length does not match regressor_design rows for SPMG3; skipping SPMG3 design fixture.")
}

## 5. Non-SPM multi-basis design (tent basis) --------------------------------

# Use tent basis generator (5 basis functions, span=24) as the HRF, with the
# same block-wise design as above but in a shorter frame.
T_tent <- fmrihrf:::sampling_frame(
  blocklens = c(60L, 60L),
  TR = c(1.0, 1.0),
  start_time = c(0, 0),
  precision = 1 / 10
)
onsets_tent <- c(5, 15, 25, 10, 20, 30)
fac_tent <- c("A", "B", "A", "B", "A", "B")
block_tent <- c(1L, 1L, 1L, 2L, 2L, 2L)
tent_hrf_obj <- fmrihrf:::hrf_tent_generator(nbasis = 5, span = 24)
dm_tent <- fmrihrf:::regressor_design(onsets_tent, fac_tent, block_tent, T_tent, hrf = tent_hrf_obj, precision = T_tent$precision)

times_tent <- fmrihrf:::samples(T_tent, global = TRUE)
if (length(times_tent) == nrow(dm_tent)) {
  design_tent_out <- data.frame(
    time = times_tent,
    A_b1 = dm_tent[, 1],
    A_b2 = dm_tent[, 2],
    A_b3 = dm_tent[, 3],
    A_b4 = dm_tent[, 4],
    A_b5 = dm_tent[, 5],
    B_b1 = dm_tent[, 6],
    B_b2 = dm_tent[, 7],
    B_b3 = dm_tent[, 8],
    B_b4 = dm_tent[, 9],
    B_b5 = dm_tent[, 10]
  )
  write.csv(
    design_tent_out,
    file = file.path(out_dir, "regressor_design_tent_block_example.csv"),
    row.names = FALSE
  )
} else {
  warning("SamplingFrame samples length does not match regressor_design rows for tent; skipping tent design fixture.")
}


message("Fixture generation complete.")
