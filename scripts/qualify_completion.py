#!/usr/bin/env python3
"""Source-bound local qualification. Hosted CI is a separate claim."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/completion"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    candidates = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    for prefix in ["crates", "scripts"]:
        candidates.extend(p for p in (ROOT / prefix).rglob("*")
                          if p.is_file() and "evidence" not in p.parts
                          and p.suffix in {".rs", ".toml", ".json", ".R", ".py", ".cjs"})
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(candidates))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--feature-set", choices=["both", "default", "all", "post-only"], default="both",
                        help="Select check coverage; both is complete, post-only runs format/lint/public example.")
    args = parser.parse_args()
    if not args.checks and not args.calibration:
        args.checks = args.calibration = True
    run = OUT / ("run-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run.mkdir(parents=True)
    env = os.environ.copy()
    env.update(LC_ALL="C", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", CARGO_BUILD_JOBS="2")
    receipt = dict(started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   platform=platform.platform(), machine=platform.machine(),
                   git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   rustc=subprocess.check_output(["rustc", "--version"], cwd=ROOT, text=True).strip(),
                   commands=[], sources_before=sources(), threads={k: env[k] for k in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]})

    def save():
        (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        (OUT / "latest.json").write_text(json.dumps({"run": str(run.relative_to(ROOT)), "receipt": str((run / "receipt.json").relative_to(ROOT))}, indent=2) + "\n")

    def command(name, cmd, stdout=None, clean_executables=False):
        free = shutil.disk_usage(ROOT).free
        if free < 600_000_000:
            raise RuntimeError("less than 600 MB free; preserve shared files and stop building")
        print("RUN", name, flush=True)
        start = time.monotonic()
        log = run / (name + ".log")
        with log.open("wb") as errors:
            if stdout:
                with (OUT / stdout).open("wb") as output:
                    process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=output, stderr=errors)
            else:
                process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=errors, stderr=subprocess.STDOUT)
        receipt["commands"].append(dict(name=name, command=cmd, exit_code=process.returncode,
                                       seconds=time.monotonic()-start, log=str(log.relative_to(ROOT)),
                                       free_bytes_before=free, free_bytes_after=shutil.disk_usage(ROOT).free))
        entry = receipt["commands"][-1]
        entry["log_sha256"] = digest(log)
        output_text = log.read_text(errors="replace")
        summaries = re.findall(r"test result: .*? (\d+) passed; (\d+) failed; (\d+) ignored", output_text)
        entry["tests"] = {key: sum(int(row[i]) for row in summaries)
                          for i, key in enumerate(["passed", "failed", "ignored"])}
        if clean_executables:
            # Cargo identifies this command's newly produced executables. Remove
            # only those and same-inode aliases; shared libraries/cache stay intact.
            entry["removed_executables"] = []
            for line in output_text.splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if item.get("reason") != "compiler-artifact" or item.get("fresh", True):
                    continue
                executable = item.get("executable")
                if not executable:
                    continue
                path = Path(executable).resolve()
                if not path.is_relative_to(ROOT / "target") or not path.is_file():
                    continue
                stat = path.stat()
                aliases = [p for p in path.parent.iterdir() if p.is_file()
                           and p.stat().st_ino == stat.st_ino and p.stat().st_dev == stat.st_dev]
                entry["removed_executables"].append(dict(paths=[str(p.relative_to(ROOT)) for p in aliases],
                                                        sha256=digest(path), bytes=stat.st_size))
                save()  # Preserve ownership and regeneration evidence before cleanup.
                for alias in aliases:
                    alias.unlink()
        save()
        print("EXIT", name, process.returncode, flush=True)
        if process.returncode:
            raise RuntimeError(f"{name} failed; inspect {log}")

    failure = None
    save()
    try:
        if args.checks:
            metadata = json.loads(subprocess.check_output(
                ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"], cwd=ROOT))
            targets = [t for p in metadata["packages"] if p["id"] in metadata["workspace_members"]
                       for t in p["targets"]]
            receipt["target_inventory"] = targets
            sets = (["default", "all"] if args.feature_set == "both" else
                    [] if args.feature_set == "post-only" else [args.feature_set])
            for features in sets:
                flags = ["--all-features"] if features == "all" else []
                common = ["--workspace", "--locked", "-j", "1", *flags]
                command(features+"-libraries", ["cargo", "test", *common, "--lib", "--message-format=json"],
                        clean_executables=True)
                for kind, action, flag in [("test", "test", "--test"), ("example", "build", "--example")]:
                    for name in sorted({t["name"] for t in targets if kind in t["kind"]}):
                        command(features+"-"+kind+"-"+name,
                                ["cargo", action, *common, flag, name, "--message-format=json"],
                                clean_executables=True)
                command(features+"-doc", ["cargo", "test", *common, "--doc"])
            command("format", ["cargo", "fmt", "--all", "--check"])
            command("numerical-clippy", ["cargo", "clippy", "-p", "fmri-group", "-p", "fmri-noise", "-p", "fmrireg", "--lib", "--all-features", "--locked", "--", "-D", "warnings"])
            command("repeated-example", ["cargo", "run", "-p", "fmri-group", "--example", "repeated_group", "--locked"])
        if args.calibration:
            for name, package, example, replicates in [
                ("ar", "fmrireg", "completion_ar_calibration", "500"),
                ("group", "fmri-group", "completion_group_calibration", "500"),
                ("repeated", "fmri-group", "repeated_calibration", "1000"),
                ("permutation", "fmri-group", "permutation_calibration", "200"),
            ]:
                command(name+"-calibration", ["cargo", "run", "--release", "-p", package, "--example", example, "--locked", "--", replicates], stdout=name+"-calibration.jsonl")
            command("group-metafor-oracle", ["Rscript", "--vanilla", "scripts/completion_calibration.R"])
            command("repeated-dense-oracle", ["Rscript", "--vanilla", "scripts/repeated_calibration.R"])
    except Exception as error:
        failure = str(error)
    receipt["sources_after"] = sources()
    receipt["source_changes_during_run"] = [p for p in sorted(set(receipt["sources_before"]) | set(receipt["sources_after"]))
                                             if receipt["sources_before"].get(p) != receipt["sources_after"].get(p)]
    receipt["failure"] = failure
    receipt["artifacts"] = {}
    if args.calibration:
        for artifact in sorted(OUT.glob("*calibration*.json*")):
            try:
                shutil.copy2(artifact, run / artifact.name)
                receipt["artifacts"][artifact.name] = digest(artifact)
            except OSError as error:
                receipt.setdefault("artifact_errors", []).append(dict(path=str(artifact), error=str(error)))
                failure = receipt["failure"] = failure or str(error)
    receipt["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(str(run), flush=True)
    if failure:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
