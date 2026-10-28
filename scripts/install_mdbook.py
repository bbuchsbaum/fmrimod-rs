#!/usr/bin/env python3
"""Install an official, checksum-pinned mdBook binary in the local build directory."""
import hashlib
import io
import os
from pathlib import Path
import platform
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.5.4"
ASSETS = {
    ("Darwin", "arm64"): ("aarch64-apple-darwin.tar.gz", "03e8a6d8b13a2971e0b3280affd03b388373c1485e26f73407c3a76b0b1838df"),
    ("Darwin", "x86_64"): ("x86_64-apple-darwin.tar.gz", "a47d7bf0d5d670cff9ee6cce95537cbeb62dc10704d9e7131ffbd13e2b59a5de"),
    ("Linux", "x86_64"): ("x86_64-unknown-linux-musl.tar.gz", "5222beabd3e37dc5be0d18ff99b79058469354db5c220153a1b92db5ba12be89"),
    ("Linux", "aarch64"): ("aarch64-unknown-linux-musl.tar.gz", "753e5c5c363ee8a56972344dcf91466f005a51db84a7aeffe427ae3ef83d6d44"),
    ("Windows", "AMD64"): ("x86_64-pc-windows-msvc.zip", "8a6b2421aa522de06d746871d2b8fff9c8d71773467e310a4516d2566e7f2de4"),
}


def install():
    suffix, digest = ASSETS[(platform.system(), platform.machine())]
    output = ROOT / "target/docs-tools" / ("mdbook.exe" if os.name == "nt" else "mdbook")
    marker = output.with_suffix(".sha256")
    if output.exists() and marker.exists() and marker.read_text().strip() == digest:
        return output
    url = f"https://github.com/rust-lang/mdBook/releases/download/v{VERSION}/mdbook-v{VERSION}-{suffix}"
    request = urllib.request.Request(url, headers={"User-Agent": "fmrimod-docs-build"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("mdBook release archive checksum mismatch")
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            name = next(n for n in archive.namelist() if Path(n).name == "mdbook.exe")
            binary = archive.read(name)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            member = next(m for m in archive if m.isfile() and Path(m.name).name == "mdbook")
            binary = archive.extractfile(member).read()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary)
    output.chmod(0o755)
    marker.write_text(digest + "\n")
    return output


if __name__ == "__main__":
    print(install())
