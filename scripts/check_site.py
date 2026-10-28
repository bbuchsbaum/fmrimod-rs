#!/usr/bin/env python3
"""Validate generated guide/API pages and local assets without network access."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        for key in ["href", "src"]:
            if attrs.get(key):
                self.links.append(attrs[key])


def check(root):
    root = root.resolve()
    pages = {}
    for page in root.rglob("*.html"):
        parser = Links()
        parser.feed(page.read_text(encoding="utf-8"))
        pages[page] = parser
    broken = []
    checked = 0
    for page, parser in pages.items():
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            path = unquote(url.path)
            if path.startswith("/fmrimod-rs/"):
                target = root / path.removeprefix("/fmrimod-rs/")
            elif path.startswith("/"):
                target = root / path.lstrip("/")
            else:
                target = page.parent / path if path else page
            target = target.resolve()
            if target.is_dir():
                target /= "index.html"
            checked += 1
            if not target.is_relative_to(root) or not target.is_file():
                broken.append((str(page.relative_to(root)), link))
            elif url.fragment and page.parent == root and target in pages and target.parent == root:
                if unquote(url.fragment) not in pages[target].ids:
                    broken.append((str(page.relative_to(root)), link))
    assert not broken, f"Broken local site links: {broken[:30]} ({len(broken)} total)"
    assert (root / "index.html").is_file()
    for crate in ["fmrimod", "fmri_data", "fmri_hrf", "fmri_design", "fmri_noise", "fmri_reg", "fmri_group", "fmri_splines"]:
        assert (root / "api" / crate / "index.html").is_file(), crate
    return dict(html_pages=len(pages), local_links_checked=checked, broken_links=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.directory), indent=2))
