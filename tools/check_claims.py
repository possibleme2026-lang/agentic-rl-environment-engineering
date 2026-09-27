#!/usr/bin/env python3
"""Validate the claims this repository makes about its own contents.

The README asserts three things that are easy to get wrong silently:

  1. every relative link resolves to a file that exists on disk;
  2. the offsets quoted in ``offsets.json`` are internally consistent with the
     per-family counts and with the stated total;
  3. every file listed in the repository manifest is actually present.

None of these are checked by any compiler, and a broken one would look exactly
like a working one. Staying within the standard library means CI needs no
dependency install, so this gate is cheap enough to run on every push.

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def root() -> Path:
    """Resolve the repository root at call time so tests can retarget it."""
    return ROOT

# Files that the README links to or names and which must therefore exist.
REQUIRED = (
    "README.md",
    "LICENSE",
    "blog/mimo-v2.6-rl.html",
    "evidence/VERIFICATION.md",
    "evidence/offsets.json",
    "tools/check_claims.py",
    ".gitattributes",
    ".github/workflows/ci.yml",
)

# Matches markdown inline links, capturing the target. Reference-style links and
# pure anchors are skipped by the scheme/content filters below.
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

failures: list[str] = []
checks = 0


def fail(message: str) -> None:
    failures.append(message)


def check_required_files() -> None:
    global checks
    for rel in REQUIRED:
        checks += 1
        if not (root() / rel).is_file():
            fail(f"missing required file: {rel}")


def check_relative_links() -> None:
    """Every relative link in every markdown file must resolve on disk."""
    global checks
    for md in sorted(root().rglob("*.md")):
        if ".git" in md.parts:
            continue
        text = md.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            # Strip any anchor fragment before resolving.
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            checks += 1
            resolved = (md.parent / path_part).resolve()
            if not resolved.exists():
                fail(f"{md.relative_to(root())}: dead relative link -> {target}")


def check_offsets() -> None:
    """offsets.json must agree with itself and with the stated total."""
    global checks
    path = root() / "evidence/offsets.json"
    if not path.is_file():
        return  # already reported by check_required_files
    data = json.loads(path.read_text(encoding="utf-8"))

    families = data["image_families"]
    total = data["image_mapping_total"]

    checks += 1
    summed = sum(entry["count"] for entry in families)
    if summed != total:
        fail(f"offsets.json: family counts sum to {summed}, total claims {total}")

    checks += 1
    if total != data["dockerhub_tag_count"]:
        fail(
            "offsets.json: image_mapping_total "
            f"({total}) != dockerhub_tag_count ({data['dockerhub_tag_count']})"
        )

    checks += 1
    rows = data["parquet_rows"]
    ledger = data["code_cyber_ledger"]
    for domain, parquet_key in (("code", "code_rows"), ("cyber", "cyber_rows")):
        checks += 1
        if ledger[parquet_key] != rows[domain]:
            fail(
                f"offsets.json: {domain} parquet rows ({rows[domain]}) "
                f"!= ledger {parquet_key} ({ledger[parquet_key]})"
            )

    # The two domains that have exactly one image per task must match exactly.
    for entry in families:
        domain = entry["domain"]
        if domain in ("code", "cyber"):
            checks += 1
            if entry["count"] != rows[domain]:
                fail(
                    f"offsets.json: image count for {domain} ({entry['count']}) "
                    f"!= parquet rows ({rows[domain]})"
                )

    # The shared-image domains must have fewer images than tasks. Domain names
    # do not all match parquet filenames (visual -> webdev), so map explicitly.
    domain_to_parquet = {"visual": "webdev", "general": "general"}
    for entry in families:
        domain = entry["domain"]
        parquet_key = domain_to_parquet.get(domain)
        if parquet_key is None:
            continue
        checks += 1
        task_count = rows[parquet_key]
        if entry["count"] >= task_count:
            fail(
                f"offsets.json: {domain} should share images "
                f"({entry['count']} images for {task_count} tasks); "
                "if this is now one-image-per-task, update the README prose"
            )

    # Unexplained domains are a silent hole: a new family could appear and the
    # ledger would still balance only because nothing referenced it.
    checks += 1
    known = {"code", "cyber", "general", "visual"}
    seen = {entry["domain"] for entry in families}
    if seen != known:
        fail(f"offsets.json: domain set changed, expected {sorted(known)}, got {sorted(seen)}")


def check_blog_selfcontained() -> None:
    """The blog must load with no network access and no build step."""
    global checks
    path = root() / "blog/mimo-v2.6-rl.html"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")

    checks += 1
    for needle, label in (
        ("http://cdn", "a CDN over http"),
        ("https://cdn", "a CDN over https"),
        ("https://unpkg.com", "unpkg"),
        ("https://cdn.jsdelivr.net", "jsDelivr"),
    ):
        if needle in text:
            fail(f"blog/mimo-v2.6-rl.html: references {label} ({needle}); must be self-contained")

    checks += 1
    # External <a href> links are fine (they are citations the reader may click);
    # <script src> / <link rel=stylesheet> are not.
    for tag in ("<script src=", "<link rel=\"stylesheet\""):
        if tag in text:
            fail(f"blog/mimo-v2.6-rl.html: contains external resource tag {tag!r}")

    checks += 1
    if "<style>" not in text:
        fail("blog/mimo-v2.6-rl.html: expected inline <style> block")


def check_readme_markers() -> None:
    """Guard the README's own structure so edits cannot quietly drop sections."""
    global checks
    path = root() / "README.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")

    checks += 1
    if text.count("```") % 2 != 0:
        fail("README.md: unbalanced code fences")

    checks += 1
    for section in ("## Findings", "## Reproduce", "## License", "## References"):
        if section not in text:
            fail(f"README.md: missing required section {section!r}")

    checks += 1
    if "blog/mimo-v2.6-rl.html" not in text:
        fail("README.md: does not link the blog post")


def main() -> int:
    check_required_files()
    check_relative_links()
    check_offsets()
    check_blog_selfcontained()
    check_readme_markers()

    if failures:
        print(f"FAIL — {len(failures)} problem(s) across {checks} checks:\n")
        for item in failures:
            print(f"  - {item}")
        return 1

    print(f"OK — {checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
