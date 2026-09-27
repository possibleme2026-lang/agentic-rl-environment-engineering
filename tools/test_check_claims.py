#!/usr/bin/env python3
"""Negative tests for tools/check_claims.py.

A gate that only ever passes proves nothing. Each test below builds a throwaway
repository whose only difference from a valid one is a single deliberate defect,
then asserts the gate detects it. If the gate ever stops failing on a mutation,
these tests fail — so the gate cannot silently degrade into always-green.

Standard library only; needs pytest.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "tools" / "check_claims.py"

# Minimal fixture: enough files for the gate's required-file check to pass, so a
# test failure points at the mutation rather than at missing scaffolding.
MINIMAL_README = """# Title

## Findings
## Reproduce
## License
## References

See [blog/mimo-v2.6-rl.html](blog/mimo-v2.6-rl.html).
"""

MINIMAL_OFFSETS = {
    "parquet_rows": {"code": 2698, "cyber": 1000, "general": 989, "webdev": 2093, "music": 1000, "total": 7780},
    "parquet_breakdown": {"general_agent": 925, "terminal_bench": 64},
    "image_mapping_total": 3764,
    "dockerhub_tag_count": 3764,
    "image_families": [
        {"family": "format-code-task-*", "domain": "code", "count": 2698},
        {"family": "arvo-rl:v1-arvo-*", "domain": "cyber", "count": 1000},
        {"family": "general-agent-env-*", "domain": "general", "count": 65},
        {"family": "webdev-rl-opensource:v2", "domain": "visual", "count": 1},
    ],
    "code_cyber_ledger": {"code_rows": 2698, "code_images": 2698, "cyber_rows": 1000, "cyber_images": 1000},
}

MINIMAL_BLOG = """<html><head><style>body{}</style></head>
<body><p>self-contained</p><a href="environment-engineering.html">part 1</a></body></html>
"""

MINIMAL_PART1 = """<html><head><style>body{}</style></head>
<body><p>self-contained</p><a href="mimo-v2.6-rl.html">part 2</a></body></html>
"""

# The Markdown twins must mirror the HTML's headings, fences and tables. Each
# fixture below is the valid twin of the HTML above: one h1, one h2, one fenced
# block, one two-column table with one body row.
MINIMAL_MD = """# Title

## Section

```bash
echo hi
```

| a | b |
|---|---|
| 1 | 2 |
"""

MINIMAL_INDEX = """<html><head><style>body{}</style></head>
<body>
<a href="blog/mimo-v2.6-rl.html">post</a>
<a href="blog/environment-engineering.html">part 1</a>
</body></html>
"""

# HTML twins matching MINIMAL_MD, used only for the parity checks.
MINIMAL_HTML_POST = """<html><head><style>body{}</style></head>
<body>
<header class="hero"><h1>Title</h1></header>
<h2>Section</h2>
<pre><code>echo hi</code></pre>
<table><thead><tr><th>a</th><th>b</th></tr></thead>
<tbody><tr><td>1</td><td>2</td></tr></tbody></table>
<a href="PLACEHOLDER">other</a>
</body></html>
"""


def build_fixture(dest: Path) -> None:
    """Materialise a valid repository at ``dest``."""
    (dest / "blog").mkdir(parents=True, exist_ok=True)
    (dest / "evidence").mkdir(parents=True, exist_ok=True)
    (dest / "tools").mkdir(parents=True, exist_ok=True)
    (dest / ".github" / "workflows").mkdir(parents=True, exist_ok=True)

    (dest / "README.md").write_text(MINIMAL_README, encoding="utf-8")
    (dest / "README.zh-CN.md").write_text(MINIMAL_README, encoding="utf-8")
    (dest / "index.html").write_text(MINIMAL_INDEX, encoding="utf-8")
    (dest / ".nojekyll").write_text("", encoding="utf-8")
    (dest / "LICENSE").write_text("Apache License\n", encoding="utf-8")
    (dest / "blog" / "mimo-v2.6-rl.html").write_text(
        MINIMAL_HTML_POST.replace("PLACEHOLDER", "environment-engineering.html"),
        encoding="utf-8",
    )
    (dest / "blog" / "mimo-v2.6-rl.md").write_text(MINIMAL_MD, encoding="utf-8")
    (dest / "blog" / "environment-engineering.html").write_text(
        MINIMAL_HTML_POST.replace("PLACEHOLDER", "mimo-v2.6-rl.html"), encoding="utf-8"
    )
    (dest / "blog" / "environment-engineering.md").write_text(MINIMAL_MD, encoding="utf-8")
    (dest / "evidence" / "VERIFICATION.md").write_text("# Evidence\n", encoding="utf-8")
    (dest / "evidence" / "offsets.json").write_text(
        json.dumps(MINIMAL_OFFSETS, indent=2), encoding="utf-8"
    )
    (dest / ".gitattributes").write_text("* text=auto eol=lf\n", encoding="utf-8")
    (dest / ".github" / "workflows" / "ci.yml").write_text("name: CI\n", encoding="utf-8")
    shutil.copy2(GATE, dest / "tools" / "check_claims.py")


def run_gate(repo: Path) -> subprocess.CompletedProcess[str]:
    # check=False on purpose: a non-zero exit is the expected outcome for most
    # tests here, and it is asserted explicitly rather than raised.
    return subprocess.run(
        [sys.executable, str(repo / "tools" / "check_claims.py")],
        capture_output=True,
        text=True,
        cwd=str(repo),
        check=False,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    dest = tmp_path / "repo"
    build_fixture(dest)
    return dest


def test_valid_fixture_passes(repo: Path) -> None:
    """Control: without a mutation the gate must be green."""
    result = run_gate(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checks passed" in result.stdout


def test_detects_dead_relative_link(repo: Path) -> None:
    readme = repo / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\n[missing](docs/interfaces.md)\n",
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "dead relative link" in result.stdout


def test_detects_missing_required_file(repo: Path) -> None:
    (repo / "evidence" / "VERIFICATION.md").unlink()
    result = run_gate(repo)
    assert result.returncode == 1
    assert "missing required file" in result.stdout


def test_detects_family_counts_not_summing_to_total(repo: Path) -> None:
    path = repo / "evidence" / "offsets.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["image_families"][0]["count"] -= 1  # 2698 -> 2697, sum now 3763
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    result = run_gate(repo)
    assert result.returncode == 1
    assert "sum to" in result.stdout


def test_detects_mapping_total_disagreeing_with_tag_count(repo: Path) -> None:
    path = repo / "evidence" / "offsets.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["dockerhub_tag_count"] = 3000  # inner total still 3764
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    result = run_gate(repo)
    assert result.returncode == 1
    assert "dockerhub_tag_count" in result.stdout


def test_detects_code_ledger_drift(repo: Path) -> None:
    path = repo / "evidence" / "offsets.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["code_cyber_ledger"]["code_rows"] = 2600  # no longer equals parquet_rows.code
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    result = run_gate(repo)
    assert result.returncode == 1
    assert "ledger" in result.stdout


def test_detects_shared_domain_given_per_task_images(repo: Path) -> None:
    """visual has 1 image for 2093 tasks; if it ever gets >= 2093 the prose is wrong."""
    path = repo / "evidence" / "offsets.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data["image_families"]:
        if entry["domain"] == "visual":
            entry["count"] = 2093
    # Rebalance so the sum still matches, isolating the sharing check.
    data["image_mapping_total"] += 2092
    data["dockerhub_tag_count"] += 2092
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    result = run_gate(repo)
    assert result.returncode == 1
    assert "should share images" in result.stdout


def test_detects_unknown_domain(repo: Path) -> None:
    """A new family appearing must not slip through unaccounted."""
    path = repo / "evidence" / "offsets.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["image_families"].append({"family": "mystery-*", "domain": "mystery", "count": 0})
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    result = run_gate(repo)
    assert result.returncode == 1
    assert "domain set changed" in result.stdout


def test_detects_external_cdn_in_blog(repo: Path) -> None:
    (repo / "blog" / "mimo-v2.6-rl.html").write_text(
        '<html><head><link rel="stylesheet" href="https://cdn.example.com/x.css">'
        "<style>a{}</style></head></html>",
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "must be self-contained" in result.stdout


def test_detects_external_script_in_blog(repo: Path) -> None:
    (repo / "blog" / "mimo-v2.6-rl.html").write_text(
        '<html><head><style>a{}</style></head><body><script src="app.js"></script></body></html>',
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1


def test_detects_missing_readme_section(repo: Path) -> None:
    (repo / "README.md").write_text("# Title\n\nno sections here\n", encoding="utf-8")
    result = run_gate(repo)
    assert result.returncode == 1
    assert "missing required section" in result.stdout


def test_detects_unbalanced_code_fences(repo: Path) -> None:
    readme = repo / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\n```bash\nunclosed\n",
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "unbalanced code fences" in result.stdout


def test_detects_readme_not_linking_blog(repo: Path) -> None:
    (repo / "README.md").write_text(
        "# Title\n\n## Findings\n## Reproduce\n## License\n## References\n",
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "does not link the blog post" in result.stdout


def test_detects_missing_index(repo: Path) -> None:
    """The Pages root disappearing must not leave the site dead-ending at a 404."""
    (repo / "index.html").unlink()
    result = run_gate(repo)
    assert result.returncode == 1
    assert "missing required file: index.html" in result.stdout


def test_detects_index_not_linking_post(repo: Path) -> None:
    (repo / "index.html").write_text(
        "<html><head><style>a{}</style></head><body><p>nothing here</p></body></html>",
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "Pages root would hide it" in result.stdout


def test_detects_index_hiding_one_instalment(repo: Path) -> None:
    """Dropping only one of the two instalments must not pass as 'reaches both'."""
    (repo / "index.html").write_text(
        '<html><head><style>a{}</style></head><body>'
        '<a href="blog/mimo-v2.6-rl.html">post</a>'
        "</body></html>",
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "blog/environment-engineering.html" in result.stdout


def test_detects_missing_part1(repo: Path) -> None:
    (repo / "blog" / "environment-engineering.html").unlink()
    result = run_gate(repo)
    assert result.returncode == 1
    assert "missing required file: blog/environment-engineering.html" in result.stdout


def test_detects_part1_not_linking_part2(repo: Path) -> None:
    """A one-directional series still renders fine and still strands the reader."""
    (repo / "blog" / "environment-engineering.html").write_text(
        "<html><head><style>a{}</style></head><body><p>no onward link</p></body></html>",
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "blog/environment-engineering.html: does not link" in result.stdout


def test_detects_part2_not_linking_part1(repo: Path) -> None:
    (repo / "blog" / "mimo-v2.6-rl.html").write_text(
        "<html><head><style>a{}</style></head><body><p>no back link</p></body></html>",
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "blog/mimo-v2.6-rl.html: does not link" in result.stdout


def test_detects_external_asset_in_part1(repo: Path) -> None:
    """Part 1 has no more licence to break offline than Part 2 does."""
    (repo / "blog" / "environment-engineering.html").write_text(
        '<html><head><style>a{}</style>'
        '<link rel="stylesheet" href="https://cdn.example.com/x.css"></head>'
        '<body><a href="mimo-v2.6-rl.html">part 2</a></body></html>',
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "blog/environment-engineering.html: references" in result.stdout


def test_detects_missing_markdown_twin(repo: Path) -> None:
    (repo / "blog" / "mimo-v2.6-rl.md").unlink()
    result = run_gate(repo)
    assert result.returncode == 1
    assert "missing required file: blog/mimo-v2.6-rl.md" in result.stdout


def test_detects_stale_markdown_section(repo: Path) -> None:
    """The HTML gains a section but the Markdown was never regenerated."""
    html = repo / "blog" / "mimo-v2.6-rl.html"
    html.write_text(
        html.read_text(encoding="utf-8").replace(
            "<h2>Section</h2>", "<h2>Section</h2><h2>Added Later</h2>"
        ),
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "regenerate with tools/html_to_markdown.py" in result.stdout


def test_detects_markdown_missing_a_code_block(repo: Path) -> None:
    """A dropped listing must be caught: the Markdown still reads as complete."""
    md = repo / "blog" / "mimo-v2.6-rl.md"
    md.write_text(md.read_text(encoding="utf-8").replace("```bash\necho hi\n```\n", ""),
                  encoding="utf-8")
    result = run_gate(repo)
    assert result.returncode == 1
    assert "fenced blocks" in result.stdout


def test_detects_markdown_missing_a_table_row(repo: Path) -> None:
    md = repo / "blog" / "mimo-v2.6-rl.md"
    md.write_text(md.read_text(encoding="utf-8").replace("| 1 | 2 |\n", ""), encoding="utf-8")
    result = run_gate(repo)
    assert result.returncode == 1
    assert "table lines" in result.stdout


def test_detects_presentation_leaking_into_markdown(repo: Path) -> None:
    """A bad conversion that leaves styling behind is still invalid Markdown."""
    md = repo / "blog" / "mimo-v2.6-rl.md"
    md.write_text(
        md.read_text(encoding="utf-8") + '\n<div class="note">residue</div>\n',
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "presentation leaked" in result.stdout


def test_detects_missing_nojekyll(repo: Path) -> None:
    (repo / ".nojekyll").unlink()
    result = run_gate(repo)
    assert result.returncode == 1
    assert ".nojekyll" in result.stdout


def test_detects_external_asset_in_index(repo: Path) -> None:
    """The landing page has no more licence to break offline than the post does."""
    (repo / "index.html").write_text(
        '<html><head><style>a{}</style>'
        '<script src="https://cdn.example.com/analytics.js"></script></head>'
        '<body><a href="blog/mimo-v2.6-rl.html">post</a></body></html>',
        encoding="utf-8",
    )
    result = run_gate(repo)
    assert result.returncode == 1
    assert "index.html: contains external resource tag" in result.stdout
