#!/usr/bin/env python3
"""Convert the two series HTML posts into Markdown, and verify nothing was lost.

Why not just run pandoc: pandoc maps ``<pre>`` to *indented* code blocks, which
are fragile (a list continuation or a wrapped line silently breaks them), and it
leaves ``<div>``/``<span>`` scaffolding behind whenever a class carries meaning.
Part 2 alone has 19 code blocks and 20 tables; a converter that quietly drops
one of them produces a Markdown file that still looks plausible.

So the mapping is explicit here, and the conversion is self-checking: the source
DOM is counted before writing and the Markdown is counted after, and the script
exits non-zero if any table row or code block failed to survive. A converter
that can fail loudly is worth more than one that is merely short.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parent.parent

# Blocks whose entire subtree is presentation, not content.
SKIP_TAGS = {"style", "script", "nav"}

# Where a class carries meaning, map the container to a Markdown construct
# rather than letting its children flatten into loose paragraphs.
PULLQUOTE_CLASSES = {"pull"}
CALLOUT_CLASSES = {"note", "note key", "note warn", "note good"}
EQUATION_CLASSES = {"eq"}


def collapse(text: str) -> str:
    return re.sub(r"[ \t\r\n]+", " ", text)


def text_with_breaks(node: Tag) -> str:
    """Flatten an element's text, turning ``<br>`` into a real space.

    ``get_text()`` ignores ``<br>`` entirely, so a title written as
    ``...Trains In.<br>Nobody Is Scaling the World.`` would come out as
    ``In.Nobody`` with the two sentences fused.
    """
    parts = []
    for child in node.descendants:
        if isinstance(child, NavigableString):
            if type(child).__name__ == "Comment":
                continue
            parts.append(str(child))
        elif getattr(child, "name", None) == "br":
            parts.append(" ")
    return collapse("".join(parts))


def md_inline(node: Tag | NavigableString) -> str:
    """Render an inline subtree as Markdown, collapsing insignificant whitespace."""
    if isinstance(node, NavigableString):
        if type(node).__name__ == "Comment":
            return ""
        return collapse(str(node))

    name = node.name
    if name in SKIP_TAGS:
        return ""

    parts = [md_inline(child) for child in node.children]
    inner = "".join(parts)

    if name in ("strong", "b"):
        stripped = inner.strip()
        return f"**{stripped}**" if stripped else ""
    if name in ("em", "i"):
        stripped = inner.strip()
        return f"*{stripped}*" if stripped else ""
    if name == "code":
        body = collapse(node.get_text()).strip()
        if not body:
            return ""
        # A backtick inside a span needs a longer fence around it.
        fence = "`" if "`" not in body else "``"
        pad = " " if body.startswith("`") or body.endswith("`") else ""
        return f"{fence}{pad}{body}{pad}{fence}"
    if name == "a":
        href = node.get("href", "")
        label = collapse(text_with_breaks(node)).strip()
        if not href:
            return label
        return f"[{label}]({href})"
    if name == "br":
        return "\n"
    if name in ("sup", "sub"):
        return f"<{name}>{inner}</{name}>"
    if name == "mark":
        stripped = inner.strip()
        return f"**{stripped}**" if stripped else ""
    if name == "span":
        # Spans are pure presentation: .num, .num-t, .by, .lbl, .sec-num, .tag.
        # Their text is the content; the styling is not reproducible in Markdown.
        # The one exception is the hero metadata row, where the spans are
        # separate facts that would otherwise fuse into one run-on string.
        parent_classes = set(getattr(node.parent, "get", lambda *_: [])("class", []) or [])
        if "meta-row" in parent_classes:
            body = collapse(inner).strip()
            return f" · {body}" if body else ""
        return inner
    return inner


def fenced(raw: str, info: str = "") -> str:
    """Wrap verbatim text in a fence long enough not to collide with its content."""
    body = raw.strip("\n")
    fence = "```"
    while fence in body:
        fence += "`"
    return f"{fence}{info}\n{body}\n{fence}"


def table_md(table: Tag) -> str:
    """Render a table as GFM, synthesising a header when the source has none.

    GFM requires a header row and separator. The paper-index tables in Part 1 are
    ``<tbody>``-only, so without this they would emit as invalid Markdown and
    GitHub would show the pipes as literal text.
    """
    rows: list[list[str]] = []
    thead = table.find("thead")
    if thead is not None:
        for tr in thead.find_all("tr"):
            rows.append([md_inline(c).strip() for c in tr.find_all(["th", "td"])])

    body_rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        if thead is not None and tr.find_parent("thead") is not None:
            continue
        cells = tr.find_all(["th", "td"])
        if cells:
            body_rows.append([md_inline(c).strip() for c in cells])

    if not rows and body_rows:
        # No header in the source: emit a blank one so the table is valid GFM.
        width = max(len(r) for r in body_rows)
        rows = [[""] * width]
        body_rows = [r + [""] * (width - len(r)) for r in body_rows]
    if not rows:
        return ""

    width = len(rows[0])
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    for row in body_rows:
        row = row + [""] * (width - len(row))
        out.append("| " + " | ".join(row[:width]) + " |")
    return "\n".join(out)


def block_md(node: Tag) -> list[str]:
    """Render a block subtree into a list of Markdown blocks."""
    if isinstance(node, NavigableString):
        # HTML comments carry section markers like ``<!-- ==== 1 ==== -->``.
        # They are not content and must not become paragraphs.
        if type(node).__name__ == "Comment":
            return []
        text = collapse(str(node)).strip()
        return [text] if text else []

    name = node.name
    if name in SKIP_TAGS:
        return []

    classes = set(node.get("class", []) or [])

    if name in ("section", "main", "article", "header", "footer", "body", "html"):
        out: list[str] = []
        for child in node.children:
            out.extend(block_md(child))
        return out

    if name == "div" or name == "dl":
        if name == "dl":
            return glossary_md(node)
        if classes & PULLQUOTE_CLASSES:
            return pullquote_md(node)
        if classes & CALLOUT_CLASSES:
            return callout_md(node)
        if classes & EQUATION_CLASSES:
            return [fenced(node.get_text(), "")]
        if "series-bar" in classes:
            return series_bar_md(node)
        if "read-next" in classes or "read-prev" in classes:
            return read_next_md(node)
        if "toc" in classes:
            return []  # the heading structure is the table of contents
        if "filepath" in classes:
            body = collapse(node.get_text()).strip()
            return [f"`{body}`"] if body else []
        if "hero" in classes:
            return hero_md(node)
        if "meta-row" in classes or "meta" in classes:
            # The metadata spans are separate facts; md_inline keeps them apart.
            body = collapse(md_inline(node).replace("·  ", "· ")).strip(" ·").strip()
            return [f"*{body}*"] if body else []
        if "kicker-h" in classes or "kicker" in classes:
            body = collapse(node.get_text()).strip()
            return [f"*{body}*"] if body else []
        out = []
        for child in node.children:
            out.extend(block_md(child))
        return out

    if name == "h1" or name == "h2":
        # h1 is the post title and h2 the sections; the DOM nests them under a
        # wrapper, so levels are assigned here rather than inherited.
        level = 1 if name == "h1" else 2
        text = text_with_breaks(node).strip()
        return [f"{'#' * level} {text}"] if text else []
    if name in ("h3", "h4", "h5", "h6"):
        level = int(name[1])
        text = text_with_breaks(node).strip()
        return [f"{'#' * level} {text}"] if text else []

    if name == "p":
        # A .sec-sub is the italic deck under a section heading; a .cap is a
        # table caption; a .lede is the opening paragraph.
        text = md_inline(node).strip()
        if not text:
            return []
        if "sec-sub" in classes or "sub" in classes or "cap" in classes:
            return [f"*{text}*"]
        return [text]

    if name == "table":
        rendered = table_md(node)
        caption = node.find_next_sibling("p")
        if caption is not None and "cap" in set(caption.get("class", []) or []):
            rendered = f"{rendered}\n\n*{md_inline(caption).strip()}*"
        return [rendered] if rendered else []

    if name == "pre":
        code = node.find("code")
        raw = (code or node).get_text()
        return [fenced(raw)]

    if name in ("ul", "ol"):
        return list_md(node)

    if name == "blockquote":
        blocks: list[str] = []
        for child in node.children:
            blocks.extend(block_md(child))
        body = "\n\n".join(b for b in blocks if b.strip())
        quoted = "\n".join(f"> {line}" if line else ">" for line in body.splitlines())
        return [quoted] if quoted.strip() else []

    if name == "hr":
        return ["---"]

    if name in ("strong", "b", "em", "i", "code", "a", "span", "mark", "sup", "sub"):
        text = md_inline(node).strip()
        return [text] if text else []

    # Unknown block: recurse rather than drop, so nothing vanishes silently.
    out = []
    for child in node.children:
        out.extend(block_md(child))
    return out


def list_md(node: Tag, depth: int = 0) -> list[str]:
    """Render ul/ol, nesting one level per indent unit."""
    ordered = node.name == "ol"
    lines: list[str] = []
    for index, li in enumerate(node.find_all("li", recursive=False), start=1):
        marker = f"{index}." if ordered else "-"

        # Separate the item's own text from any nested list.
        pieces: list[str] = []
        nested: list[Tag] = []
        for child in li.children:
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                nested.append(child)
            elif isinstance(child, Tag) and child.name in ("p", "div"):
                pieces.append(md_inline(child).strip())
            else:
                pieces.append(md_inline(child))
        text = collapse("".join(pieces)).strip()
        lines.append(f"{'  ' * depth}{marker} {text}")
        for sub in nested:
            lines.extend(list_md(sub, depth + 1))
    return ["\n".join(lines)] if lines else []


def pullquote_md(node: Tag) -> list[str]:
    body = node.find("span", class_="by")
    attribution = collapse(body.get_text()).strip() if body else ""
    if body is not None:
        body.extract()
    text = collapse(node.get_text()).strip()
    out = [f"> {text}"] if text else []
    if attribution:
        out.append(f">\n> {attribution}")
    return ["\n".join(out)] if out else []


def callout_md(node: Tag) -> list[str]:
    """Render a .note callout as a blockquote without flattening its blocks.

    Joining the child blocks into one collapsed string would fuse a nested
    ``<pre>`` into the surrounding prose and destroy its fence -- two of Part 2's
    code listings live inside callouts. Each block is kept intact and quoted
    line by line instead, which preserves fences and paragraph breaks.
    """
    label = node.find("span", class_="lbl")
    heading = collapse(label.get_text()).strip() if label else ""
    if label is not None:
        label.extract()

    blocks: list[str] = []
    for child in node.children:
        for block in block_md(child):
            if block and block.strip():
                blocks.append(block.strip())

    lines: list[str] = []
    if heading:
        lines.append(f"> **{heading}**")
        lines.append(">")
    for index, block in enumerate(blocks):
        if index:
            lines.append(">")
        for line in block.splitlines():
            lines.append(f"> {line}" if line.strip() else ">")
    return ["\n".join(lines)] if lines else []


def glossary_md(node: Tag) -> list[str]:
    lines: list[str] = []
    for dt in node.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        term = collapse(dt.get_text()).strip()
        definition = collapse(dd.get_text()).strip() if dd else ""
        lines.append(f"**{term}** — {definition}")
    return ["\n\n".join(lines)] if lines else []


def series_bar_md(node: Tag) -> list[str]:
    label = node.find("span", class_="series-label")
    here = node.find("span", class_="series-here")
    link = node.find("a")
    bits = []
    if label is not None:
        bits.append(f"**{collapse(label.get_text()).strip()}**")
    if here is not None:
        bits.append(collapse(here.get_text()).strip())
    if link is not None:
        bits.append(f"[{collapse(link.get_text()).strip()}]({link.get('href', '')})")
    joined = " · ".join(b for b in bits if b)
    return [f"> {joined}"] if joined else []


def read_next_md(node: Tag) -> list[str]:
    label = node.find("div", class_=True)
    heading = node.find(["h3", "h2"])
    para = node.find("p")
    lines = []
    if label is not None and "label" in " ".join(label.get("class", []) or []):
        lines.append(f"**{collapse(label.get_text()).strip()}**")
    if heading is not None:
        link = heading.find("a")
        if link is not None:
            lines.append(f"### [{collapse(link.get_text()).strip()}]({link.get('href', '')})")
        else:
            lines.append(f"### {collapse(heading.get_text()).strip()}")
    if para is not None:
        lines.append(md_inline(para).strip())
    return ["\n\n".join(lines)] if lines else []


def hero_md(node: Tag) -> list[str]:
    """The hero holds the title, deck and metadata; emit the title as the H1."""
    out: list[str] = []
    for child in node.children:
        if isinstance(child, Tag) and child.name == "h1":
            text = text_with_breaks(child).strip()
            if text:
                out.append(f"# {text}")
        elif isinstance(child, Tag) and child.name in ("div", "p", "span"):
            out.extend(block_md(child))
    return out


def convert(src: Path) -> str:
    soup = BeautifulSoup(src.read_text(encoding="utf-8"), "html.parser")
    body = soup.find("body") or soup

    blocks: list[str] = []
    for child in body.children:
        if isinstance(child, Tag):
            blocks.extend(block_md(child))

    text = "\n\n".join(b.strip() for b in blocks if b and b.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def count_md(md: str) -> dict[str, int]:
    lines = md.splitlines()

    # Fences inside a blockquote are written ``> ``` ``, so a bare
    # ``startswith("```")`` misses them and reports a lost listing that is
    # actually present. Strip the quote marker before looking for the fence.
    def unfence(line: str) -> str:
        return re.sub(r"^(\s*>\s*)+", "", line).lstrip()

    fences = sum(1 for line in lines if unfence(line).startswith("```"))

    # Counted by classification, not by one regex with a stated meaning. A
    # headerless source table is given a blank header row, and that row is
    # ``|  |  |  |`` -- pipes and spaces only, which matches any naive
    # "separator" pattern and inflated the table count by 8. A real separator
    # must contain a dash; a synthetic blank header has none. Getting this wrong
    # reports lost rows that were never lost.
    sep_re = re.compile(r"^\|[\s|:\-]*-[\s|:\-]*\|$")
    blank_header_re = re.compile(r"^\|[\s|]*\|$")

    separators = 0
    blank_headers = 0
    content_rows = 0
    for raw in lines:
        line = unfence(raw)
        if not line.startswith("|"):
            continue
        if sep_re.match(line):
            separators += 1
        elif blank_header_re.match(line):
            blank_headers += 1
        else:
            content_rows += 1

    return {
        "code_blocks": fences // 2,
        "tables": separators,
        "content_rows": content_rows,
        "blank_headers": blank_headers,
        "table_rows": content_rows + separators,
    }


def main() -> int:
    targets = [
        ("blog/environment-engineering.html", "blog/environment-engineering.md"),
        ("blog/mimo-v2.6-rl.html", "blog/mimo-v2.6-rl.md"),
    ]
    failures: list[str] = []
    report: list[str] = []

    for src_rel, dst_rel in targets:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        md = convert(src)

        soup = BeautifulSoup(src.read_text(encoding="utf-8"), "html.parser")
        body = soup.find("body") or soup
        src_tables = len(body.find_all("table"))
        src_rows = len(body.find_all("tr"))
        # Both <pre> code listings and .eq display equations become fenced
        # blocks, so the source count has to include the equations or a dropped
        # listing can hide behind a preserved equation.
        src_pre = len(body.find_all("pre"))
        src_eq = len(body.find_all("div", class_="eq"))
        src_fenced = src_pre + src_eq

        got = count_md(md)
        report.append(
            f"  {dst_rel}: {len(md)} B, {len(md.splitlines())} lines\n"
            f"    code blocks  src {src_pre:>3} pre + {src_eq} eq = {src_fenced:>3}"
            f" -> md {got['code_blocks']:>3}\n"
            f"    tables       src {src_tables:>3} -> md {got['tables']:>3}\n"
            f"    table rows   src {src_rows:>3} -> md {got['table_rows']:>3}"
        )

        # Every source code block must appear; a converter that drops one is
        # worse than no converter, because the result still looks complete.
        if got["code_blocks"] != src_fenced:
            failures.append(
                f"{dst_rel}: fenced blocks {src_fenced} expected ({src_pre} pre + "
                f"{src_eq} equations), got {got['code_blocks']} "
                f"(lost {src_fenced - got['code_blocks']})"
            )
        # Tables gain one separator line each, so rows should be src + tables.
        expected_rows = src_rows + src_tables
        if got["table_rows"] != expected_rows:
            failures.append(
                f"{dst_rel}: table rows {expected_rows} expected (src {src_rows} + "
                f"{src_tables} separators), got {got['table_rows']}"
            )
        if got["tables"] != src_tables:
            failures.append(f"{dst_rel}: tables {src_tables} -> {got['tables']}")

        # Nothing from the style block or the table-of-contents nav should leak.
        for leak in ("--ink:", "@media", "box-sizing", "class=\"series-bar\""):
            if leak in md:
                failures.append(f"{dst_rel}: presentation leaked into Markdown ({leak!r})")

        # newline="\n" is load-bearing: .gitattributes pins *.md to eol=lf and
        # promises byte-stability, but Path.write_text translates \n to os.linesep,
        # so on Windows this would emit CRLF and the CI job (which regenerates and
        # diffs on Linux) would see every line as changed.
        with open(dst, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(md.rstrip() + "\n")

    print("Conversion ledger:")
    print("\n".join(report))
    if failures:
        print(f"\nFAIL — {len(failures)} problem(s):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nOK — all code blocks and table rows preserved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
