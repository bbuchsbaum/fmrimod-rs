#!/usr/bin/env python3
"""Build the pinned mdBook guide and Rust API into a verified Pages artifact."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

from check_site import check, Links
from install_mdbook import install, VERSION

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-example", action="store_true", help="Only when the guide example was already executed by the calling gate")
    args = parser.parse_args()
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target"))).resolve()
    site = ROOT / "target/site"
    output = ROOT / "evidence/docs"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, command in [
        ("guide", ["cargo", "run", "-p", "fmrimod", "--example", "guide_workflow", "--locked"]),
        ("rustdoc", ["cargo", "doc", "--workspace", "--all-features", "--no-deps", "--locked"]),
        ("book", [str(install()), "build", "docs", "--dest-dir", str(site)]),
    ]:
        if name == "guide" and args.skip_example:
            continue
        log = output / (name + ".log")
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        records.append(dict(name=name, command=command, exit_code=result.returncode,
                            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        if result.returncode:
            raise RuntimeError(f"{name} failed: {log}")
    shutil.copytree(target / "doc", site / "api", dirs_exist_ok=True)
    # Workspace Rustdoc has no root index. Its help/settings links need one.
    (site / "api/index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=fmrimod/index.html">'
        '<title>fmrimod Rust API</title><a href="fmrimod/index.html">Open the fmrimod API</a>', encoding="utf-8")
    # Rustdoc emits async implementation-index URLs even for unused public traits
    # with no implementation index. Supply empty indexes instead of publishing 404s.
    empty_indexes = []
    for page in (site / "api").rglob("*.html"):
        links = Links()
        links.feed(page.read_text(encoding="utf-8"))
        for link in links.links:
            if "trait.impl/" not in link or not link.endswith(".js"):
                continue
            path = (page.parent / link).resolve()
            if path.is_relative_to((site / "api/trait.impl").resolve()) and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('(function(){var i={};if(window.register_implementors){window.register_implementors(i);}else{window.pending_implementors=i;}})();', encoding="utf-8")
                empty_indexes.append(str(path.relative_to(site)))
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    info = dict(version=version, source_revision=source, mdbook=VERSION,
                built_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                rustc=subprocess.check_output(["rustc", "--version"], text=True).strip())
    (site / "build-info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (site / ".nojekyll").touch()
    stats = check(site)
    receipt = dict(build=info, commands=records, site_check=stats, empty_implementor_indexes=empty_indexes,
                   companion_sha256=hashlib.sha256((ROOT / "crates/fmrimod/examples/guide_workflow.rs").read_bytes()).hexdigest())
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
