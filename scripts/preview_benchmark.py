#!/usr/bin/env python3
"""Bounded synthetic, portable GLS timings with explicit setup and warm scope."""
from pathlib import Path
import datetime
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/preview"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-provenance", type=Path, default=ROOT / "crates/fmri-noise/tests/fixtures/arp.json")
    parser.add_argument("--r-source", type=Path, help="Relocated source checkout with the recorded hashes")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CARGO_BUILD_JOBS="2", FMRIREG_BACKEND="portable",
               VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    started = time.monotonic()
    command = ["cargo", "build", "--release", "-p", "fmri-reg", "--example", "gls_benchmark", "--locked"]
    with (OUT / "benchmark-build.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    build_seconds = time.monotonic()-started
    binary = ROOT / "target/release/examples/gls_benchmark"
    records = []
    for n, p, features in [(300, 8, 1000), (1200, 32, 5000)]:
        output = OUT / f"gls-{n}-{p}-{features}.f64"
        cmd = [str(binary), str(n), str(p), str(features), "2", "250", "1", "5", str(output), "reuse"]
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        record = json.loads(result.stdout)
        record.update(command=cmd, output_bytes=output.stat().st_size,
                      output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        records.append(record)
    # Require the exact source-built R provider that qualified the AR fixtures.
    manifest = json.loads(args.reference_provenance.read_text())
    provenance = manifest.get("provenance", manifest)
    reference_source = args.r_source or Path(provenance["source"])
    for rel, digest in provenance["source_md5"].items():
        assert hashlib.md5((reference_source / rel).read_bytes()).hexdigest() == digest, rel
    for rel, digest in provenance["binary_md5"].items():
        assert hashlib.md5((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    env["R_LIBS_USER"] = str(ROOT / "evidence/r-library")
    references = []
    for record in records:
        n, p, features = record["n"], record["p"], record["features"]
        reference = OUT / f"gls-{n}-{p}-{features}-r.f64"
        command = ["Rscript", "--vanilla", "scripts/benchmark_gls.R", str(n), str(p), str(features),
                   "2", "250", "1", "5", str(reference)]
        run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        expected = np.fromfile(reference, dtype="<f8")
        actual = np.fromfile(OUT / f"gls-{n}-{p}-{features}.f64", dtype="<f8")
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        error = float(np.max(np.abs(actual-expected)/(1+np.abs(expected))))
        assert error < 2e-8, error
        references.append(dict(command=command, result=json.loads(run.stdout), max_scaled_error=error,
                               values=len(actual), output_sha256=hashlib.sha256(reference.read_bytes()).hexdigest()))
    receipt = dict(version="0.1.0-alpha.1", generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(), platform=platform.platform(),
                   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), build_seconds=build_seconds,
                   scope="synthetic known workload, design-corrected estimated AR(2); shared workstation; matched-output R reference; timing includes generation/whitening/fitting/contrasts; no universal speedup claim",
                   records=records, R_references=references, R_provenance=provenance, reference_source=str(reference_source.resolve()),
                   reference_manifest_sha256=hashlib.sha256(args.reference_provenance.read_bytes()).hexdigest())
    sources = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", Path(__file__), ROOT / "crates/fmri-reg/examples/gls_benchmark.rs", ROOT / "scripts/benchmark_gls.R"]
    sources += [p for name in ["fmri-reg", "fmri-noise"] for p in (ROOT / "crates" / name / "src").rglob("*.rs")]
    receipt["source_sha256"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    (OUT / "benchmark.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"build_seconds":build_seconds,"workloads":len(records)}))


if __name__ == "__main__":
    main()
