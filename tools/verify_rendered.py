"""Assert the series navigation actually renders and actually resolves.

Reading the source HTML only proves the markup was written. This drives a real
headless browser, dumps the post-script DOM, and checks:

  1. every page parses and the series elements survive into the DOM;
  2. the nav text says what it should (Part 1 vs Part 2, correct direction);
  3. every relative href in the rendered DOM resolves to a file on disk;
  4. every diagram survives rendering with real geometry and a caption.

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

# The post with the diagrams, and the floor each one must clear to count as a
# drawing rather than an empty frame. Deliberately not a shape-count floor:
# the diagrams here are not all box-and-arrow. One is a nested hierarchy whose
# structure is in the widths, one is a table laid out in text, one is a row of
# table columns. A shape floor tuned for the first kind failed the other two
# and would have "caught" correct output -- the failure mode being guarded
# against is an <svg> that renders empty, which is a floor of zero shapes and
# no labels, not a floor of fewer than six boxes.
FIGURES_PAGE = "blog/mimo-v2.6-rl.html"
MIN_SHAPES_PER_FIGURE = 1
MIN_ELEMENTS_PER_FIGURE = 8
MIN_LABELS_PER_FIGURE = 3


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


def figures_in(dom: str) -> list[tuple[str, int, int, int, int]]:
    """Return one tuple per rendered figure: (aria-label, elements, shapes, labels, caption).

    Counting what the SVG actually contains rather than trusting the <figure>
    tag, because a diagram that lost its contents still renders as a bordered
    empty box, and that is indistinguishable from "fine" without looking at the
    pixels. ``elements`` covers everything that would paint: geometry, labels
    and their spans. A caption is reported as negative when the <svg> has no
    viewBox, which means it has no coordinate system and will not scale.
    """
    out = []
    for block in re.findall(r'<figure class="fig".*?</figure>', dom, flags=re.DOTALL):
        label = re.search(r'aria-label="([^"]*)"', block)
        cap = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", block, flags=re.DOTALL)
        svg = re.search(r"<svg[^>]*>", block)
        body = block[svg.end() :] if svg else ""
        shapes = len(
            re.findall(r"<(?:rect|line|path|circle|ellipse|polygon|polyline)\b", body)
        )
        labels = len(re.findall(r"<text\b", body))
        elements = shapes + labels + len(re.findall(r"<tspan\b", body))
        caption_len = len(html.unescape(TAG_RE.sub("", cap.group(1))).strip()) if cap else 0
        has_viewbox = bool(svg and "viewBox" in svg.group(0))
        out.append(
            (
                (label.group(1)[:60] if label else ""),
                elements,
                shapes,
                labels,
                caption_len if has_viewbox else -caption_len,
            )
        )
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

    # 6. every diagram must survive rendering: geometry, labels, a viewBox and a
    #    caption. An empty <figure> is the failure this is here to catch.
    figs = figures_in(dump_dom(FIGURES_PAGE))
    checks += 1
    if not figs:
        failures.append(f"{FIGURES_PAGE}: no rendered figures found")
    for aria, elements, shapes, labels, caption in figs:
        checks += 1
        name = aria or "(no aria-label)"
        if caption < 0:
            failures.append(f"{FIGURES_PAGE}: figure {name!r} rendered without a viewBox")
        elif caption < 80:
            failures.append(
                f"{FIGURES_PAGE}: figure {name!r} caption is only {caption} chars; "
                "it has to stand alone for a reader who cannot see the drawing"
            )
        if shapes < MIN_SHAPES_PER_FIGURE:
            failures.append(
                f"{FIGURES_PAGE}: figure {name!r} rendered {shapes} shapes; "
                "nothing in it would paint"
            )
        if elements < MIN_ELEMENTS_PER_FIGURE:
            failures.append(
                f"{FIGURES_PAGE}: figure {name!r} rendered only {elements} elements "
                f"(< {MIN_ELEMENTS_PER_FIGURE}); looks like an empty frame"
            )
        if labels < MIN_LABELS_PER_FIGURE:
            failures.append(
                f"{FIGURES_PAGE}: figure {name!r} rendered {labels} labels; "
                "a diagram with no readable labels explains nothing"
            )
        if not aria:
            failures.append(
                f"{FIGURES_PAGE}: figure has no aria-label, so it is invisible to "
                "a screen reader"
            )

    if failures:
        print(f"FAIL — {len(failures)} problem(s) across {checks} rendered checks:\n")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"OK — {checks} rendered checks passed")
    for bar in bars1 + bars2:
        print(f"     nav: {' '.join(bar.split())[:110]}")
    for aria, elements, shapes, labels, caption in figs:
        print(
            f"     fig: {elements:>3} elements ({shapes} shapes, {labels} labels) "
            f"{caption:>4} chars — {' '.join(aria.split())[:60]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
