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
<body><p>self-contained</p></body></html>
"""


def build_fixture(dest: Path) -> None:
    """Materialise a valid repository at ``dest``."""
    (dest / "blog").mkdir(parents=True, exist_ok=True)
    (dest / "evidence").mkdir(parents=True, exist_ok=True)
    (dest / "tools").mkdir(parents=True, exist_ok=True)
    (dest / ".github" / "workflows").mkdir(parents=True, exist_ok=True)

    (dest / "README.md").write_text(MINIMAL_README, encoding="utf-8")
    (dest / "README.zh-CN.md").write_text(MINIMAL_README, encoding="utf-8")
    (dest / "LICENSE").write_text("Apache License\n", encoding="utf-8")
    (dest / "blog" / "mimo-v2.6-rl.html").write_text(MINIMAL_BLOG, encoding="utf-8")
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
