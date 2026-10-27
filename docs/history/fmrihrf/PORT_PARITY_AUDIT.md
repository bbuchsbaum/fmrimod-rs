> Imported historical document. Current workspace scope and evidence are in
> [the workspace README](../../README.md) and [plan](../../docs/PLAN.md).
> Current R SPMG3 uses a dispersion basis; old parity statements below are historical.

# Core port audit — 2026-09-05

The complete cross-crate report is [fmridesign-rs/PORT_PARITY_AUDIT.md](../fmridesign-rs/PORT_PARITY_AUDIT.md).
It identifies the source revisions and dirty-source hashes, tested contracts,
upstream R discrepancies, compatibility changes, performance configuration,
and remaining work. The companion fixture generators and contract tests live
in this crate under `scripts/` and `tests/`.

The numerical core now preserves fractional events, block integrals, multiple
HRF bases, run isolation, custom HRF identity, and fixed-reference normalization.
Direct and FFT share one projected-input contract; `conv`/`auto` selects between
them based on input density. The required linear-algebra backend is pure Rust.

[Local convolution measurements](port-evidence/convolution-benchmark.json) compare
600 scans and 240 events with three SPMG bases. They are not R speedup claims.
The complete audit includes the final validation receipt and remaining limitations.
