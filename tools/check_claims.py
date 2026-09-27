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
import posixpath
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
    "README.zh-CN.md",
    "LICENSE",
    "index.html",
    ".nojekyll",
    "blog/environment-engineering.html",
    "blog/environment-engineering.md",
    "blog/mimo-v2.6-rl.html",
    "blog/mimo-v2.6-rl.md",
    "evidence/VERIFICATION.md",
    "evidence/offsets.json",
    "tools/check_claims.py",
    ".gitattributes",
    ".github/workflows/ci.yml",
)

# Pages that must render with no network access and no build step. The landing
# page and both instalments are served straight off the filesystem (and by
# Pages), so an off-site asset in any of them is a silent breakage.
SELF_CONTAINED = (
    "index.html",
    "blog/environment-engineering.html",
    "blog/mimo-v2.6-rl.html",
)

# Each post exists twice: as the HTML that renders on Pages, and as Markdown for
# readers who want plain text, diffs, or an offline copy. Two representations of
# one document drift silently, so the pairs are checked for structural parity.
#
# ``toc_heading`` is the label of an <h2> that carries no content (the sidebar
# "Contents" nav), which the converter drops on purpose.
POST_PAIRS = (
    ("blog/environment-engineering.html", "blog/environment-engineering.md", "Contents"),
    ("blog/mimo-v2.6-rl.html", "blog/mimo-v2.6-rl.md", None),
)

# The series is only navigable if each instalment links the other and the
# landing page reaches both. A rename would otherwise leave one-directional
# links that still render fine and still 404.
SERIES = {
    "blog/environment-engineering.html": "blog/mimo-v2.6-rl.html",
    "blog/mimo-v2.6-rl.html": "blog/environment-engineering.html",
}

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
    """The post and the landing page must load with no network and no build step."""
    global checks
    for rel in SELF_CONTAINED:
        path = root() / rel
        if not path.is_file():
            continue  # already reported by check_required_files
        text = path.read_text(encoding="utf-8")

        checks += 1
        for needle, label in (
            ("http://cdn", "a CDN over http"),
            ("https://cdn", "a CDN over https"),
            ("https://unpkg.com", "unpkg"),
            ("https://cdn.jsdelivr.net", "jsDelivr"),
        ):
            if needle in text:
                fail(f"{rel}: references {label} ({needle}); must be self-contained")

        checks += 1
        # External <a href> links are fine (they are citations the reader may
        # click); <script src> / <link rel=stylesheet> are not.
        for tag in ("<script src=", '<link rel="stylesheet"'):
            if tag in text:
                fail(f"{rel}: contains external resource tag {tag!r}")

        checks += 1
        if "<style>" not in text:
            fail(f"{rel}: expected inline <style> block")


def check_pages_entry() -> None:
    """The Pages root must exist and reach every instalment in the series."""
    global checks
    path = root() / "index.html"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")

    for rel in SELF_CONTAINED:
        if rel == "index.html":
            continue
        checks += 1
        if rel not in text:
            fail(f"index.html: does not link {rel}; the Pages root would hide it")

    checks += 1
    # GitHub Pages disables Jekyll when this exists; without it, a directory or
    # file name starting with an underscore silently vanishes from the build.
    if not (root() / ".nojekyll").is_file():
        fail(".nojekyll: missing; GitHub Pages will run Jekyll over this site")


def check_series_links() -> None:
    """Each instalment must link the other, in the direction it claims.

    Links are checked as they are written on the page (relative to the file's own
    directory), not as repository-relative paths: the two instalments are
    siblings under ``blog/``, so each links the other by bare filename.
    """
    global checks
    for rel, target in sorted(SERIES.items()):
        checks += 1
        path = root() / rel
        if not path.is_file():
            continue  # already reported by check_required_files
        text = path.read_text(encoding="utf-8")
        href = posixpath.relpath(target, posixpath.dirname(rel))
        if href not in text:
            other = "Part 1" if "environment-engineering" in target else "Part 2"
            fail(f"{rel}: does not link {target} (as {href!r}); {other} becomes a dead end")

    # The landing page must reach both, or the series has an orphan. It sits at
    # the repository root, so its hrefs are repository-relative.
    checks += 1
    index = root() / "index.html"
    if index.is_file():
        text = index.read_text(encoding="utf-8")
        missing = [rel for rel in SERIES if posixpath.relpath(rel, ".") not in text]
        if missing:
            fail(f"index.html: does not link {', '.join(sorted(missing))}")


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


def check_markdown_parity() -> None:
    """Each post's HTML and Markdown must describe the same document.

    The Markdown is generated from the HTML by ``tools/html_to_markdown.py``, and
    nothing forces a regeneration when the HTML changes. A stale pair still looks
    like a complete post in both formats, so the counts are compared directly:
    section headings, display equations, and table rows must line up. Fenced code
    blocks are counted from the HTML source too, because the converter turns both
    ``<pre>`` listings and ``.eq`` equations into fences.
    """
    global checks
    for html_rel, md_rel, toc_heading in POST_PAIRS:
        html_path = root() / html_rel
        md_path = root() / md_rel
        if not html_path.is_file() or not md_path.is_file():
            continue  # already reported by check_required_files

        html_text = html_path.read_text(encoding="utf-8")
        md_text = md_path.read_text(encoding="utf-8")

        # 1. Section headings: every <h2> except a pure-nav one must appear as "##".
        html_h2 = re.findall(r"<h2[^>]*>(.*?)</h2>", html_text, flags=re.DOTALL)
        html_h2 = [re.sub(r"<[^>]+>", "", h).strip() for h in html_h2]
        if toc_heading:
            html_h2 = [h for h in html_h2 if h != toc_heading]
        html_sections = len(html_h2)
        md_sections = len(re.findall(r"^## ", md_text, flags=re.MULTILINE))
        checks += 1
        if html_sections != md_sections:
            fail(
                f"{md_rel}: has {md_sections} sections but {html_rel} has "
                f"{html_sections}; regenerate with tools/html_to_markdown.py"
            )

        # 2. Fenced blocks: <pre> listings plus .eq equations become fences.
        html_pre = len(re.findall(r"<pre[^>]*>", html_text))
        html_eq = len(re.findall(r'class="eq"', html_text))
        expected_fences = html_pre + html_eq
        # Fences inside a blockquote are written "> ```", so strip quote markers.
        md_fences = sum(
            1
            for line in md_text.splitlines()
            if re.sub(r"^(\s*>\s*)+", "", line).lstrip().startswith("```")
        ) // 2
        checks += 1
        if md_fences != expected_fences:
            fail(
                f"{md_rel}: has {md_fences} fenced blocks but {html_rel} has "
                f"{expected_fences} ({html_pre} listings + {html_eq} equations); "
                "regenerate with tools/html_to_markdown.py"
            )

        # 3. Table rows: one Markdown row per <tr>, plus a separator per table,
        #    plus a synthetic blank header for each source table that had none
        #    (GFM requires a header row, and Part 1's paper-index tables are
        #    tbody-only). Counting by classification rather than by one pattern,
        #    because ``|  |  |  |`` is pipes and spaces and matches a naive
        #    separator test -- that mistake reports 8 phantom lost rows.
        html_rows = len(re.findall(r"<tr[^>]*>", html_text))
        html_tables = len(re.findall(r"<table[^>]*>", html_text))
        html_thead = len(re.findall(r"<thead[^>]*>", html_text))
        html_headerless = max(html_tables - html_thead, 0)
        sep_re = re.compile(r"^\|[\s|:\-]*-[\s|:\-]*\|$")
        md_table_lines = 0
        md_separators = 0
        for line in md_text.splitlines():
            stripped = re.sub(r"^(\s*>\s*)+", "", line).lstrip()
            if stripped.startswith("|"):
                md_table_lines += 1
                if sep_re.match(stripped):
                    md_separators += 1
        expected_lines = html_rows + html_tables + html_headerless
        checks += 1
        if md_table_lines != expected_lines:
            fail(
                f"{md_rel}: has {md_table_lines} table lines but {html_rel} implies "
                f"{expected_lines} ({html_rows} rows + {html_tables} separators + "
                f"{html_headerless} synthetic headers); regenerate with "
                "tools/html_to_markdown.py"
            )

        # 4. Presentation must not survive into the Markdown.
        checks += 1
        for leak in ("<style", "@media", "box-sizing", "<div"):
            if leak in md_text:
                fail(f"{md_rel}: presentation leaked into the Markdown ({leak!r})")

        # 5. Every table in the source must have produced a Markdown table.
        checks += 1
        if html_tables and md_separators != html_tables:
            fail(
                f"{md_rel}: {html_tables} tables in {html_rel} but {md_separators} "
                "Markdown separators; regenerate with tools/html_to_markdown.py"
            )


def main() -> int:
    check_required_files()
    check_relative_links()
    check_offsets()
    check_blog_selfcontained()
    check_pages_entry()
    check_series_links()
    check_markdown_parity()
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
