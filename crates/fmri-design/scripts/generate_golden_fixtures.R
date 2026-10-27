#!/usr/bin/env Rscript
#
# Generate Golden Test Fixtures for Rust fmridesign
#
# This script generates JSON test fixtures from the R fmridesign package
# for use in validating the Rust implementation. Fixtures are saved to
# golden_tests/fixtures/ and should be committed to version control.
#
# Usage:
#   Rscript scripts/generate_golden_fixtures.R [--all | --module MODULE]
#
# Options:
#   --all         Generate all fixtures (default)
#   --module      Generate fixtures for specific module (basis, events, contrast, etc.)
#   --output-dir  Output directory (default: golden_tests/fixtures)
#

library(fmridesign)
library(jsonlite)
library(fmrihrf)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
output_dir <- "golden_tests/fixtures"
generate_all <- TRUE
specific_module <- NULL

if (length(args) > 0) {
  for (i in seq_along(args)) {
    if (args[i] == "--output-dir" && i < length(args)) {
      output_dir <- args[i + 1]
    } else if (args[i] == "--module" && i < length(args)) {
      generate_all <- FALSE
      specific_module <- args[i + 1]
    } else if (args[i] == "--all") {
      generate_all <- TRUE
    }
  }
}

# Create output directory
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "basis"), showWarnings = FALSE)
dir.create(file.path(output_dir, "events"), showWarnings = FALSE)
dir.create(file.path(output_dir, "contrast"), showWarnings = FALSE)
dir.create(file.path(output_dir, "baseline"), showWarnings = FALSE)
dir.create(file.path(output_dir, "design"), showWarnings = FALSE)

cat("Generating golden test fixtures...\n")
cat(sprintf("Output directory: %s\n", output_dir))

# Helper function to save fixture with metadata
save_fixture <- function(fixture, name, module) {
  output_path <- file.path(output_dir, module, paste0(name, ".json"))

  # Add metadata
  fixture$_metadata <- list(
    generated_at = Sys.time(),
    r_version = R.version.string,
    fmridesign_version = as.character(packageVersion("fmridesign")),
    fmrihrf_version = as.character(packageVersion("fmrihrf")),
    generator_script = "scripts/generate_golden_fixtures.R"
  )

  write_json(fixture, output_path, digits = 15, auto_unbox = TRUE, pretty = TRUE)
  cat(sprintf("  Generated: %s\n", output_path))
}

# ============================================================================
# BASIS FUNCTION FIXTURES
# ============================================================================

generate_basis_fixtures <- function() {
  cat("\n=== Generating Basis Function Fixtures ===\n")

  # 1. Polynomial Basis
  cat("  Generating polynomial basis fixtures...\n")
  x <- seq(0, 10, length.out = 50)

  # Poly degree 3
  poly3 <- Poly(x, degree = 3)
  fixture_poly3 <- list(
    input = list(
      x = as.vector(x),
      degree = 3,
      orthogonal = TRUE
    ),
    output = list(
      matrix = as.matrix(poly3$y),
      nbasis = nbasis(poly3),
      levels = levels(poly3),
      columns = columns(poly3),
      name = "Polynomial"
    )
  )
  save_fixture(fixture_poly3, "polynomial_degree3", "basis")

  # 2. B-Spline Basis
  cat("  Generating B-spline basis fixtures...\n")
  bs_deg3 <- BSpline(x, degree = 3, nknots = 5)
  fixture_bs <- list(
    input = list(
      x = as.vector(x),
      degree = 3,
      nknots = 5
    ),
    output = list(
      matrix = as.matrix(bs_deg3$y),
      nbasis = nbasis(bs_deg3),
      levels = levels(bs_deg3),
      columns = columns(bs_deg3),
      name = "BSpline"
    )
  )
  save_fixture(fixture_bs, "bspline_degree3_nknots5", "basis")

  # 3. Identity Basis
  cat("  Generating identity basis fixtures...\n")
  varnames <- c("x1", "x2", "x3")
  x_mat <- matrix(rnorm(50 * 3), ncol = 3)
  colnames(x_mat) <- varnames
  ident <- Ident(x_mat, varnames = varnames)

  fixture_ident <- list(
    input = list(
      x = x_mat,
      varnames = varnames
    ),
    output = list(
      matrix = as.matrix(ident$y),
      nbasis = nbasis(ident),
      levels = levels(ident),
      columns = columns(ident),
      name = "Identity"
    )
  )
  save_fixture(fixture_ident, "identity_3vars", "basis")

  # 4. Scale Transform
  cat("  Generating scale transform fixtures...\n")
  x_scaled <- Scale(x, argname = "time")
  fixture_scale <- list(
    input = list(
      x = as.vector(x),
      argname = "time"
    ),
    output = list(
      matrix = as.matrix(x_scaled$y),
      nbasis = nbasis(x_scaled),
      levels = levels(x_scaled),
      columns = columns(x_scaled),
      name = "Scale",
      center = attr(x_scaled, "center"),
      scale = attr(x_scaled, "scale")
    )
  )
  save_fixture(fixture_scale, "scale_transform", "basis")
}

# ============================================================================
# EVENT FIXTURES
# ============================================================================

generate_event_fixtures <- function() {
  cat("\n=== Generating Event Fixtures ===\n")

  # 1. EventFactor
  cat("  Generating event factor fixtures...\n")
  onsets <- c(0, 10, 20, 30, 40, 0, 10, 20, 30, 40)
  durations <- rep(2, 10)
  condition <- factor(rep(c("A", "B"), each = 5))

  ef <- event_factor(
    onsets = onsets,
    durations = durations,
    varname = "condition",
    levels = condition
  )

  fixture_ef <- list(
    input = list(
      onsets = onsets,
      durations = durations,
      varname = "condition",
      levels = as.character(condition)
    ),
    output = list(
      name = ef$varname,
      event_type = "Categorical",
      n_events = length(onsets),
      levels_out = levels(ef),
      cells = cells(ef)
    )
  )
  save_fixture(fixture_ef, "event_factor_basic", "events")

  # 2. EventVariable
  cat("  Generating event variable fixtures...\n")
  values <- rnorm(10, mean = 50, sd = 10)
  ev <- event_variable(
    onsets = onsets,
    durations = durations,
    varname = "response_time",
    values = values
  )

  fixture_ev <- list(
    input = list(
      onsets = onsets,
      durations = durations,
      varname = "response_time",
      values = values
    ),
    output = list(
      name = ev$varname,
      event_type = "Continuous",
      n_events = length(onsets),
      values_out = as.vector(ev$values)
    )
  )
  save_fixture(fixture_ev, "event_variable_continuous", "events")
}

# ============================================================================
# CONTRAST FIXTURES
# ============================================================================

generate_contrast_fixtures <- function() {
  cat("\n=== Generating Contrast Fixtures ===\n")

  # Setup: Create a simple design matrix
  onsets <- rep(c(0, 10, 20, 30), 3)
  condition <- factor(rep(c("A", "B", "C"), each = 4))
  durations <- rep(2, 12)

  sframe <- sampling_frame(TR = 2, n_scans = 30)

  ef <- event_factor(onsets, durations, "condition", condition)
  eterm <- event_term(hrf(condition, onsets = onsets, durations = durations), hrf = hrf_spmg1)
  model <- event_model(eterm, sampling_frame = sframe, blockids = rep(1, 12))
  dmat <- design_matrix(model)
  colnames_dmat <- colnames(dmat)

  # 1. Pair Contrast
  cat("  Generating pair contrast fixtures...\n")
  pc <- pair_contrast("A", "B", "condition")
  weights_pc <- contrast_weights(pc, model)

  fixture_pc <- list(
    input = list(
      level1 = "A",
      level2 = "B",
      condition = "condition",
      column_names = colnames_dmat
    ),
    output = list(
      weights = as.vector(weights_pc),
      name = pc$name,
      contrast_type = "pair"
    )
  )
  save_fixture(fixture_pc, "pair_contrast_A_vs_B", "contrast")

  # 2. Oneway Contrast
  cat("  Generating oneway contrast fixtures...\n")
  oc <- oneway_contrast("condition")
  weights_oc <- contrast_weights(oc, model)

  fixture_oc <- list(
    input = list(
      condition = "condition",
      column_names = colnames_dmat
    ),
    output = list(
      weight_matrix = as.matrix(weights_oc),
      name = oc$name,
      contrast_type = "oneway"
    )
  )
  save_fixture(fixture_oc, "oneway_contrast_condition", "contrast")
}

# ============================================================================
# BASELINE FIXTURES
# ============================================================================

generate_baseline_fixtures <- function() {
  cat("\n=== Generating Baseline Fixtures ===\n")

  # 1. Polynomial drift
  cat("  Generating polynomial drift fixtures...\n")
  sframe <- sampling_frame(TR = 2, n_scans = 100)
  bmod <- baseline_model(baseline(degree = 3, sframe = sframe))
  bmat <- design_matrix(bmod)

  fixture_drift <- list(
    input = list(
      degree = 3,
      TR = 2,
      n_scans = 100
    ),
    output = list(
      matrix = as.matrix(bmat),
      column_names = colnames(bmat),
      nterms = length(terms(bmod))
    )
  )
  save_fixture(fixture_drift, "polynomial_drift_degree3", "baseline")
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if (generate_all || specific_module == "basis") {
  generate_basis_fixtures()
}

if (generate_all || specific_module == "events") {
  generate_event_fixtures()
}

if (generate_all || specific_module == "contrast") {
  generate_contrast_fixtures()
}

if (generate_all || specific_module == "baseline") {
  generate_baseline_fixtures()
}

cat("\n=== Golden fixture generation complete! ===\n")
cat(sprintf("Total fixtures in: %s\n", output_dir))
cat("\nTo run Rust golden tests:\n")
cat("  cargo test --test golden_tests\n")
