"""Assert the series navigation actually renders and actually resolves.

Reading the source HTML only proves the markup was written. This drives a real
headless browser, dumps the post-script DOM, and checks:

  1. every page parses and the series elements survive into the DOM;
  2. the nav text says what it should (Part 1 vs Part 2, correct direction);
  3. every relative href in the rendered DOM resolves to a file on disk.

Anything less would be checking my copy of the page rather than the page.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

PAGES = ("index.html", "blog/environment-engineering.html", "blog/mimo-v2.6-rl.html")
HREF_RE = re.compile(r'href="([^"#][^"]*)"')
TAG_RE = re.compile(r"<[^>]+>")


def dump_dom(rel: str) -> str:
    url = ROOT.joinpath(rel).as_uri()
    proc = subprocess.run(
        [
            str(CHROME),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--virtual-time-budget=5000",
            "--dump-dom",
            url,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return proc.stdout


def text_of(dom: str, class_name: str) -> list[str]:
    """Extract the visible text of every element carrying ``class_name``."""
    out = []
    for block in re.findall(
        rf'<div class="{class_name}"[^>]*>(.*?)</div>\s*(?=<div|</main>|</body>)',
        dom,
        flags=re.DOTALL,
    ):
        out.append(html.unescape(TAG_RE.sub(" ", block)).strip())
    return out


def main() -> int:
    if not CHROME.is_file():
        print(f"SKIP — chrome not found at {CHROME}")
        return 0

    failures: list[str] = []
    checks = 0

    for rel in PAGES:
        dom = dump_dom(rel)
        checks += 1
        if len(dom) < 1000:
            failures.append(f"{rel}: DOM came back nearly empty ({len(dom)} bytes)")
            continue

        # 1. the page must have rendered its own content, not just a shell
        checks += 1
        body_text = html.unescape(TAG_RE.sub(" ", dom))
        if len(body_text.split()) < 200:
            failures.append(f"{rel}: rendered text is only {len(body_text.split())} words")

        # 2. every relative href must resolve on disk
        for href in HREF_RE.findall(dom):
            if href.startswith(("http://", "https://", "mailto:", "//")):
                continue
            path_part = href.split("#", 1)[0]
            if not path_part:
                continue
            checks += 1
            if not (ROOT / rel).parent.joinpath(path_part).resolve().exists():
                failures.append(f"{rel}: href {href!r} does not resolve on disk")

    # 3. Part 1 and Part 2 must each carry a series bar naming their position
    dom1 = dump_dom("blog/environment-engineering.html")
    dom2 = dump_dom("blog/mimo-v2.6-rl.html")

    checks += 1
    bars1 = text_of(dom1, "series-bar")
    if not bars1 or "Part 1 of 2" not in bars1[0]:
        failures.append(f"Part 1: series bar missing or mislabelled -> {bars1}")

    checks += 1
    bars2 = text_of(dom2, "series-bar")
    if not bars2 or "Part 2 of 2" not in bars2[0]:
        failures.append(f"Part 2: series bar missing or mislabelled -> {bars2}")

    # 4. each must offer the onward link, and the direction must be right
    checks += 1
    if "mimo-v2.6-rl.html" not in dom1:
        failures.append("Part 1: no rendered link to Part 2")
    checks += 1
    if "environment-engineering.html" not in dom2:
        failures.append("Part 2: no rendered link to Part 1")

    # 5. the landing page must present both instalments
    dom0 = dump_dom("index.html")
    checks += 1
    for target in ("blog/environment-engineering.html", "blog/mimo-v2.6-rl.html"):
        if target not in dom0:
            failures.append(f"index.html: no rendered link to {target}")

    if failures:
        print(f"FAIL — {len(failures)} problem(s) across {checks} rendered checks:\n")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"OK — {checks} rendered checks passed")
    for bar in bars1 + bars2:
        print(f"     nav: {' '.join(bar.split())[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
