#!/usr/bin/env python3
"""Build separate projects against an immutable public Git revision."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/bbuchsbaum/fmrimod-rs.git"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rev", required=True, help="Full immutable commit SHA already pushed to GitHub")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/git-consumer")
    args = parser.parse_args()
    if len(args.rev) != 40 or any(c not in "0123456789abcdef" for c in args.rev):
        parser.error("--rev must be a full hexadecimal commit SHA")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CARGO_BUILD_JOBS="2", CARGO_NET_GIT_FETCH_WITH_CLI="true",
               CARGO_TARGET_DIR=str(ROOT / "target"), VECLIB_MAXIMUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    receipt = dict(revision=args.rev, repository=URL,
                   started=datetime.datetime.now(datetime.timezone.utc).isoformat(), commands=[],
                   harness_source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(__file__), *sorted((ROOT / "examples/git-consumer").glob("*.rs")),
                             ROOT / "crates/fmri-group/examples/group_pipeline.rs",
                             ROOT / "crates/fmri-group/examples/repeated_group.rs",
                             ROOT / "crates/fmrimod/examples/complete_first_level.rs"]})

    def save():
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def run(label, command, cwd):
        start = time.monotonic()
        log = output / (label + ".log")
        with log.open("w", encoding="utf-8") as f:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=f, text=True, encoding="utf-8")
            f.write(result.stdout)
        receipt["commands"].append(dict(label=label, command=command, exit_code=result.returncode,
                                        seconds=time.monotonic()-start,
                                        log=log.name, sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        save()
        print(label, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(f"{label}: inspect {log}")
        return result.stdout

    try:
        with tempfile.TemporaryDirectory(prefix="fmrimod-git-consumer-") as temp:
            for case in ["facade", "components"]:
                directory = Path(temp) / case
                (directory / "src/bin").mkdir(parents=True)
                dependencies = ['ndarray = "0.16"']
                if case == "components":
                    dependencies += ['serde_json = "1"']
                if case == "facade":
                    dependencies += [f'fmrimod = {{ git = "{URL}", rev = "{args.rev}" }}']
                else:
                    for name in ["fmrimod", "fmri-data", "fmri-hrf", "fmri-design", "fmri-noise", "fmri-reg", "fmri-group", "fmri-splines"]:
                        features = ', features = ["formula", "nifti", "accelerate"]' if name == "fmrimod" else ''
                        dependencies += [f'{name} = {{ git = "{URL}", rev = "{args.rev}"{features} }}']
                manifest = '[package]\nname = "external-'+case+'"\nversion = "0.0.0"\nedition = "2021"\n\n[workspace]\n\n[dependencies]\n'+'\n'.join(dependencies)+'\n\n[profile.dev]\ndebug = 0\nincremental = false\n'
                (directory / "Cargo.toml").write_text(manifest)
                shutil.copyfile(ROOT / "Cargo.lock", directory / "Cargo.lock")
                shutil.copyfile(ROOT / "examples/git-consumer" / (case+".rs"), directory / "src/bin" / (case+".rs"))
                bins = [case]
                if case == "components":
                    for name, crate in [("group_pipeline", "fmri-group"), ("repeated_group", "fmri-group"), ("complete_first_level", "fmrimod")]:
                        shutil.copyfile(ROOT / "crates" / crate / "examples" / (name+".rs"), directory / "src/bin" / (name+".rs"))
                        bins.append(name)
                raw = run(case+"-metadata", ["cargo", "metadata", "--format-version", "1"], directory)
                metadata = json.loads(raw)
                git_packages = [p for p in metadata["packages"] if p.get("source", "") and p["source"].startswith("git+")]
                assert len(git_packages) == (7 if case == "facade" else 8)
                assert all(p["source"].endswith("#"+args.rev) for p in git_packages)
                if case == "facade":
                    assert not any(p["name"].startswith("polars") or p["name"] == "nifti" for p in metadata["packages"])
                (output / (case+"-Cargo.toml")).write_text(manifest)
                shutil.copyfile(directory / "Cargo.lock", output / (case+"-Cargo.lock"))
                receipt[case] = dict(git_packages={p["name"]:p["source"] for p in git_packages}, manifests_contain_no_path_dependencies=True)
                for name in bins:
                    text = run(case+"-"+name, ["cargo", "run", "--locked", "--bin", name], directory)
                    if name == "repeated_group":
                        assert "status=Estimated" in text and "mean=0.500000" in text
                    if name == "group_pipeline":
                        assert "6 tests in BH family" in text
        receipt["failure"] = None
    except Exception as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"
    receipt["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    if receipt["failure"]:
        raise SystemExit(receipt["failure"])


if __name__ == "__main__":
    main()
