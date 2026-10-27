#!/usr/bin/env python3
"""Copy shared independent references into self-contained package test suites."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("fmri-hrf", "fmri-design", "sampled_features.json"),
    ("fmri-design", "fmrimod", "r_core_contract.json"),
    ("fmri-noise", "fmri-reg", "arp.json"),
]


def main():
    for source, destination, name in PAIRS:
        a = ROOT / "crates" / source / "tests/fixtures" / name
        b = ROOT / "crates" / destination / "tests/fixtures" / name
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a, b)
        print(f"{source} -> {destination}: {name}")


if __name__ == "__main__":
    main()
