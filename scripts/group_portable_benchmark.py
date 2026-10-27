#!/usr/bin/env python3
"""Qualify portable group execution; retain raw trials and exact command receipts.

Run from anywhere with --run. The pre-change binary and independent R fixtures
must already exist in evidence/group-portable-execution. No original R checkout
is modified. A new timestamped results directory is created for each execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/group-portable-execution"
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           VECLIB_MAXIMUM_THREADS="1")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sources():
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", Path(__file__),
             ROOT / "scripts/group_portable_reference.R"]
    for crate in (ROOT / "crates").iterdir():
        if (crate / "Cargo.toml").exists():
            paths.append(crate / "Cargo.toml")
        for directory in ("src", "tests", "examples"):
            paths.extend(p for p in (crate / directory).rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print((BASE / "latest.json").read_text())
        return
    out = BASE / time.strftime("run-%Y%m%dT%H%M%SZ", time.gmtime())
    out.mkdir()
    commands = []

    def save(name, data):
        (out / name).write_text(json.dumps(data, indent=2) + "\n")

    def run(argv, label, cwd=ROOT):
        argv = list(map(str, argv))
        begin = time.monotonic()
        with (out / (label + ".log")).open("w") as stream:
            result = subprocess.run(argv, cwd=cwd, env=ENV, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        record = dict(argv=argv, cwd=str(cwd), exit_code=result.returncode,
                      seconds=time.monotonic() - begin, log=label + ".log")
        commands.append(record)
        save("commands.json", commands)
        print(f"{label}: exit {result.returncode}, {record['seconds']:.3f}s", flush=True)
        if result.returncode:
            raise SystemExit(f"Inspect {out / (label + '.log')}")

    initial = sources()
    save("source.json", initial)
    run(["cargo", "test", "-p", "fmri-data", "-p", "fmri-group", "-p", "fmrimod",
         "--all-features", "--all-targets"], "tests")
    run(["cargo", "test", "-p", "fmri-data", "-p", "fmri-group", "-p", "fmrimod",
         "--all-features", "--doc"], "doctests")
    run(["cargo", "clippy", "-p", "fmri-group", "-p", "fmri-data", "--all-targets",
         "--all-features", "--no-deps", "--", "-D", "warnings"], "clippy")
    rust_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "crates/fmri-group").rglob("*.rs"))
    rust_files += ["crates/fmri-data/src/array.rs", "crates/fmri-data/tests/file_source.rs"]
    run(["rustfmt", "--edition", "2021", "--check", *rust_files], "format")
    run(["cargo", "run", "-p", "fmri-group", "--example", "portable_first_level",
         "--", out / "first-level-bundle"], "first-level-example")
    run(["cargo", "build", "--release", "-p", "fmri-group", "--example",
         "meta_regression_benchmark", "--example", "portable_execution_benchmark"], "release-build")
    executable = ROOT / "target/release/examples/portable_execution_benchmark"
    candidate = ROOT / "target/release/examples/meta_regression_benchmark"
    save("binary.json", {"portable": sha(executable), "candidate": sha(candidate),
                         "baseline": sha(BASE / "baseline_benchmark")})
    small = out / "roundtrip-rust"
    rewritten = out / "roundtrip-r"
    oracle = out / "metafor-oracle.json"
    run([executable, "generate", small, 20, 33, 3, 7], "generate-small")
    run(["Rscript", "scripts/group_portable_reference.R", "rewrite", small, rewritten], "r-rewrite")
    run([executable, "compare", small, rewritten, 3], "roundtrip")
    run(["Rscript", "scripts/group_portable_reference.R", "oracle", small, oracle, 3], "r-oracle")
    run([executable, "oracle", small, oracle, 3], "oracle-compare")
    run(["Rscript", "scripts/group_portable_reference.R", "oracle",
         out / "first-level-bundle", out / "first-level-oracle.json", 1], "first-level-r-oracle")
    run([executable, "oracle", out / "first-level-bundle", out / "first-level-oracle.json", 1], "first-level-oracle-compare")
    # Nontrivial affine/df values must survive JSON round trips exactly, not
    # merely on the convenient integer-valued geometry of a basic fixture.
    precise = out / "precision-input"
    precise_r = out / "precision-r"
    shutil.copytree(small, precise)
    manifest = json.loads((precise / "manifest.json").read_text())
    manifest["metadata"]["feature_space"] = dict(
        kind="volume", dimensions=[33, 1, 1], support=list(reversed(range(33))),
        affine=[[1.2345678901234567, 0, 0, 11.23456789012345],
                [0, 2.345678901234567, 0, -3.141592653589793],
                [0, 0, 3.456789012345678, 0], [0, 0, 0, 1]], template="precision-fixture")
    manifest["metadata"]["variance_origins"][0] = dict(
        kind="estimated", df=dict(kind="scalar", value=12.345678901234567),
        method="precision-fixture", approximate=True)
    (precise / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run(["Rscript", "scripts/group_portable_reference.R", "rewrite", precise, precise_r], "precision-r")
    run([executable, "compare", precise, precise_r, 3], "precision-compare")

    # Paired ABBA runs retain adverse timing fluctuations. Each executable
    # independently checks all requested outputs and simulation fits before
    # returning success; their old fixture directories are isolated.
    paired = []
    for index, which in enumerate(("baseline", "candidate", "candidate", "baseline")):
        binary = BASE / "baseline_benchmark" if which == "baseline" else candidate
        cwd = BASE / (which + "-run")
        run([binary], f"weighted-{index}-{which}", cwd=cwd)
        data = json.loads((cwd / "evidence/group-meta-regression/rust-benchmark.json").read_text())
        save(f"weighted-{index}-{which}.json", data)
        shutil.copy2(cwd / "evidence/group-meta-regression/rust-calibration.json",
                     out / f"calibration-{index}-{which}.json")
        paired.append((which, data))
    comparisons = []
    for row in paired[0][1]:
        key = (row["id"], row["method"], row["profile"])
        versions = {}
        for version in ("baseline", "candidate"):
            matches = [next(r for r in data if (r["id"], r["method"], r["profile"]) == key)
                       for name, data in paired if name == version]
            versions[version] = dict(
                median_seconds=statistics.median(t for r in matches for t in r["seconds"]),
                allocation_calls=matches[0]["allocation_calls"],
                allocated_bytes=matches[0]["allocated_bytes"],
                max_scaled_error=max(r["max_scaled_error"] for r in matches))
        comparisons.append(dict(id=key[0], method=key[1], profile=key[2], **versions,
                                speedup=versions["baseline"]["median_seconds"] /
                                versions["candidate"]["median_seconds"]))
    save("workspace-comparison.json", comparisons)

    # Each fresh executable measures its own getrusage(RUSAGE_SELF) high-water
    # RSS; this is separate from numeric budgets and cumulative allocations.
    assert platform.system() == "Darwin", "RSS units currently qualified on macOS"
    process_results = []
    for n, nf, p in ((20, 1000, 3), (100, 100000, 5)):
        bundle = out / f"input-{n}-{nf}-{p}"
        run([executable, "generate", bundle, n, nf, p, 256], f"generate-{n}-{nf}-{p}")
        run([executable, "compare", bundle, bundle, p], f"validate-{n}-{nf}-{p}")
        for storage, strategy in (("disk", "general"), ("disk", "shared"),
                                  ("memory", "general"), ("memory", "shared")):
            label = f"bench-{n}-{nf}-{p}-{storage}-{strategy}"
            result_path = out / (label + ".json")
            run([executable, "bench", bundle, result_path,
                 storage, strategy, p, 256], label)
            record = json.loads(result_path.read_text())
            assert record["peak_rss_bytes"] > 0, "Missing fresh-process RSS measurement"
            process_results.append(dict(id=f"{n}-{nf}-{p}", storage=storage, strategy=strategy,
                                        peak_rss_bytes=record["peak_rss_bytes"], result=record))
    save("process-results.json", process_results)
    assert initial == sources(), "Source changed during qualification; retain run and refresh"
    test_text = (out / "tests.log").read_text()
    passed = sum(int(n) for n in re.findall(r"test result: ok\. (\d+) passed", test_text))
    summary = dict(status="passed for stated contract and synthetic workloads",
                   results=str(out.relative_to(ROOT)), tests_passed=passed,
                   rustc=subprocess.check_output(["rustc", "-Vv"], text=True).strip(),
                   platform=platform.platform(), threads=1,
                   limitations=["Shared workstation; timing variability retained.",
                                "Shared path is common-effect only, full rows and exactly matched normalized weights.",
                                "Joint first-level estimands and repeated participants are outside the portable profile.",
                                "Source metadata is retained in memory; numeric block bounds are not a process RSS bound.",
                                "Inputs are synthetic; no new population inference claim."])
    save("verification.json", summary)
    # Hash completed artifacts, including manifests/payloads, after logs close.
    save("artifacts.json", {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                            if p.is_file() and p.name != "artifacts.json"})
    (BASE / "latest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
