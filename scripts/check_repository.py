#!/usr/bin/env python3
"""Check the public workspace's structure, package contents and guide links."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from sync_reference_fixtures import PAIRS

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"fmrimod", "fmri-data", "fmri-hrf", "fmri-design", "fmri-noise",
            "fmri-reg", "fmri-group", "fmri-splines"}


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metadata = json.loads(run("cargo", "metadata", "--no-deps", "--locked", "--format-version", "1"))
    packages = metadata["packages"]
    assert {p["name"] for p in packages} == EXPECTED
    tracked = set(run("git", "ls-files").splitlines())
    assert not any(p.startswith((".mote/authority/", ".mote/local/", ".dashboard/", "target/"))
                   for p in tracked), "Private/runtime state in Git index"
    assert not any(p.endswith(".bundle") for p in tracked), "Nested Git bundle in source tree"
    assert not list((ROOT / "crates").glob("*/Cargo.lock")), "Redundant member lockfile"
    for source, destination, name in PAIRS:
        a = ROOT / "crates" / source / "tests/fixtures" / name
        b = ROOT / "crates" / destination / "tests/fixtures" / name
        assert a.read_bytes() == b.read_bytes(), "Run scripts/sync_reference_fixtures.py"
    reports = []
    for package in packages:
        name = package["name"]
        directory = Path(package["manifest_path"]).parent
        assert directory == ROOT / "crates" / name
        assert package["publish"] == [], (name, "registry publication must be explicit")
        assert package["repository"] == "https://github.com/bbuchsbaum/fmrimod-rs"
        assert package["rust_version"] == "1.91.1"
        library = next(t for t in package["targets"] if "lib" in t["kind"] or "rlib" in t["kind"])
        assert library["name"] == name.replace("-", "_")
        assert (directory / "README.md").is_file()
        licenses = (["LICENSE"] if package["license"] == "GPL-3.0-or-later"
                    else ["LICENSE-MIT", "LICENSE-APACHE"])
        assert package["license"] in {"GPL-3.0-or-later", "MIT OR Apache-2.0"}
        for license_file in licenses:
            assert (directory / license_file).is_file(), (name, license_file)
        internal = []
        for dependency in package["dependencies"]:
            if dependency.get("path"):
                assert Path(dependency["path"]).resolve().is_relative_to(ROOT)
                assert dependency["name"] in EXPECTED
                internal.append(dependency["name"])
        files = run("cargo", "package", "-p", name, "--list", "--allow-dirty", "--offline").splitlines()
        assert all(f in files for f in ["README.md", *licenses])
        assert not any(f.startswith(("evidence/", "target/")) or f.endswith(".bundle") for f in files)
        reports.append(dict(name=name, import_name=library["name"], license=package["license"],
                            internal_dependencies=sorted(internal), package_files=files))

    # Current guides must resolve in a fresh checkout, not only against ignored local files.
    guides = [ROOT / p for p in tracked if p.endswith(".md") and
              ("/" not in p or p.startswith(("docs/", "crates/"))) and
              not p.startswith("docs/history/")]
    broken = []
    for guide in guides:
        for target in re.findall(r"\]\(([^)]+)\)", guide.read_text(errors="replace")):
            target = target.split("#", 1)[0].strip("<>")
            if not target or ":" in target:
                continue
            # Generated guide targets are checked against actual HTML by check_site.py.
            if guide.is_relative_to(ROOT / "docs/guide") and (target.startswith("api/") or target == "build-info.json"):
                continue
            path = (guide.parent / target).resolve()
            if not path.is_relative_to(ROOT):
                broken.append((str(guide.relative_to(ROOT)), target))
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative not in tracked and not any(p.startswith(relative + "/") for p in tracked):
                broken.append((str(guide.relative_to(ROOT)), target))
    assert not broken, broken
    report = dict(packages=reports, current_guides_checked=len(guides),
                  tracked_files=len(tracked), public_guide_links_valid=True,
                  no_external_path_dependencies=True, registry_publication_enabled=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(packages)} packages; {len(guides)} guides; package contents and workspace policy")


if __name__ == "__main__":
    main()
