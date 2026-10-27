#!/usr/bin/env python3
"""Run and summarize the weighted-group qualification from the workspace root.

The R step uses the isolated, freshly built fmrigds library recorded in
evidence/group-meta-regression/fmrigds-source.json, plus installed metafor.
No downloads or source-repository changes are performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/group-meta-regression"
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           VECLIB_MAXIMUM_THREADS="1")


def read(name):
    return json.loads((OUT / name).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", Path(__file__),
             ROOT / "scripts/group_meta_reference.R"]
    for crate in (ROOT / "crates").iterdir():
        if (crate / "Cargo.toml").exists():
            paths.append(crate / "Cargo.toml")
        for folder in ("src", "tests", "examples"):
            paths.extend(p for p in (crate / folder).rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def command(argv, log, records):
    start = time.monotonic()
    with (OUT / log).open("w") as stream:
        result = subprocess.run(argv, cwd=ROOT, env=ENV, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    records.append(dict(argv=argv, log=log, exit_code=result.returncode,
                        elapsed_seconds=time.monotonic() - start))
    write("commands.json", records)
    print(f"{result.returncode}: {' '.join(argv)} ({log})", flush=True)
    if result.returncode:
        raise SystemExit(f"Failed; inspect {OUT / log}")


def summarize():
    checks = read("commands.json")
    assert checks and all(c["exit_code"] == 0 for c in checks)
    source = read("rust-source.json")
    assert source == sources(), "Source changed since qualification; rerun --run"
    rust, reference = read("rust-benchmark.json"), read("r-benchmark.json")
    r_map = {(r["id"], r["method"]): r for r in reference}
    assert len(rust) == 3 * len(r_map) == 36
    comparisons = []
    for r in rust:
        if r["profile"] == "marginal":
            oracle = r_map[(r["id"], r["method"])]
            comparisons.append(dict(
                id=r["id"], method=r["method"], oracle=oracle["provider"],
                rust_seconds=r["median_seconds"], r_seconds=oracle["median_seconds"],
                r_over_rust=oracle["median_seconds"] / r["median_seconds"],
                max_scaled_error=r["max_scaled_error"],
                allocation_calls=r["allocation_calls"],
                allocated_bytes=r["allocated_bytes"],
                numeric_working_bytes=r["numeric_working_bytes"]))
    calibration, r_cal = read("rust-calibration.json"), read("r-calibration.json")
    key = lambda r: (r["id"], r["method"], r["inference"])
    cal_map = {key(r): r for r in r_cal}
    assert set(map(key, calibration)) == set(cal_map) and len(calibration) == 20
    for r in calibration:
        expected = cal_map[key(r)]
        for field in ("replicates", "covered", "rejections"):
            assert r[field] == expected[field], (key(r), field)
        assert r["max_scaled_error"] < 2e-8
    artifact_paths = [p for p in OUT.iterdir() if p.is_file()
                      and p.suffix in (".bin", ".json", ".log", ".txt", ".R")
                      and p.name != "verification.json"]
    artifact_paths += list((OUT / "r-library/fmrigds/libs").glob("*"))
    artifact_paths.append(ROOT / "target/release/examples/meta_regression_benchmark")
    results = dict(
        status="passed for the documented workload and numerical contract",
        platform=platform.platform(), machine=platform.machine(),
        rustc=subprocess.check_output(["rustc", "-Vv"], text=True).strip(),
        threads={k: ENV[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                     "VECLIB_MAXIMUM_THREADS")},
        source_manifest="rust-source.json", fmrigds_source="fmrigds-source.json",
        commands=checks, benchmark_comparisons=comparisons,
        calibration_panels=len(calibration), simulation_replicates=2400,
        calibrated_fits=sum(r["replicates"] for r in calibration),
        all_rejection_and_coverage_counts_match=True,
        max_calibration_scaled_error=max(r["max_scaled_error"] for r in calibration),
        calibration_tolerance=2e-8, fixture_test_tolerance=2e-9,
        limitations=[
            "Synthetic independent Gaussian subjects with known variances; no general coverage guarantee.",
            "FE under true heterogeneity is an explicit negative control.",
            "Rust includes cohort reads and collection; fmrigds calls compiled regression directly.",
            "PM compares against per-feature metafor calls, including R model-construction overhead.",
            "First-fit time is measured after preparation, not cold process or cold filesystem time.",
            "Allocation bytes are cumulative requests, not peak RSS; budget excludes caller-owned input/output.",
            "Subnormal normalized weights are rejected; stress fixtures are not a universal accuracy certificate.",
        ],
        artifact_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(artifact_paths) if p.is_file()})
    write("verification.json", results)
    for r in comparisons:
        print(f"{r['id']} {r['method']}: Rust {r['rust_seconds']:.6f}s, "
              f"R {r['r_seconds']:.6f}s, ratio {r['r_over_rust']:.2f}")
    print(f"PASS: {results['calibrated_fits']} fits; every coverage/rejection count matches")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run Rust checks and release experiments")
    parser.add_argument("--r", choices=("reference", "benchmarks", "experiments"),
                        help="refresh R evidence first using the recorded isolated library")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    if args.r:
        flags = {"reference": [], "benchmarks": ["--benchmarks-only"],
                 "experiments": ["--experiments"]}[args.r]
        # Keep prior Rust qualification receipts when refreshing R alone.
        records = read("commands.json") if (OUT / "commands.json").exists() else []
        command(["Rscript", "scripts/group_meta_reference.R", *flags], "r-refresh.log", records)
    if args.run:
        before = sources()
        write("rust-source.json", before)
        command(["cargo", "test", "-p", "fmri-group", "-p", "fmrimod",
                 "--all-features", "--all-targets"], "final-tests.log", records)
        command(["cargo", "test", "-p", "fmri-group", "-p", "fmrimod",
                 "--all-features", "--doc"], "doctests.log", records)
        command(["cargo", "clippy", "-p", "fmri-group", "--all-targets", "--no-deps",
                 "--", "-D", "warnings"], "clippy.log", records)
        rust_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "crates/fmri-group").rglob("*.rs"))
        command(["rustfmt", "--edition", "2021", "--check", *rust_files], "format.log", records)
        command(["cargo", "run", "-p", "fmri-group", "--example", "weighted_group"],
                "example.log", records)
        command(["cargo", "build", "--release", "-p", "fmri-group", "--example",
                 "meta_regression_benchmark"], "release-build.log", records)
        command(["target/release/examples/meta_regression_benchmark"], "rust-experiments.log", records)
        assert before == sources(), "Source changed during qualification; rerun"
    summarize()


if __name__ == "__main__":
    main()
