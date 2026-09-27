# Agentic RL Environment Engineering

English | [简体中文](README.zh-CN.md)

**Reading the MiMo-V2.6 open stack end to end** — the technical report, the released `verl` fork, 7,780 real task rows, and 3,764 Docker images — and mapping every claim about how the RL was actually done onto the file, config value, or environment that implements it.

[![CI](https://github.com/possibleme2026-lang/agentic-rl-environment-engineering/actions/workflows/ci.yml/badge.svg)](https://github.com/possibleme2026-lang/agentic-rl-environment-engineering/actions/workflows/ci.yml)

📄 **[Read the blog post →](blog/mimo-v2.6-rl.html)** — self-contained HTML, no build step, no network.

---

## What this is

Xiaomi released a 44-page report on MiMo-V2.6 describing RL at a scale most teams never reach: 1,568 prompts × 16 rollouts = **25,000 trajectories per step**, 2.7–3.7B tokens per batch, 1M-token contexts, balancing three scaling axes at once — training compute, environment/harness diversity, and *grader* compute.

Reports like this usually stop at the equations. What makes this release unusual is that the recipe is also shipped: a `verl` fork implementing the algorithm, the task environments as runnable Docker images, and the training data with a manifest tying each row to its image.

So the blog is a reading exercise. For each claim in the report, it finds the code or environment that makes it concrete — and where the code contradicts or bounds the report, it says so plainly. This repository holds that blog plus the evidence behind it.

## Findings

Six things the code reveals that the report's equations don't. Full detail with commands and raw output in [`evidence/VERIFICATION.md`](evidence/VERIFICATION.md).

**1. The penalty machinery is fork-specific, despite ByteDance copyright headers.** Three files carrying the report's §4.3.3 behavioural regularization return **404** on upstream `verl` at `v0.9.0` *and* on `main` — yet their headers read *Copyright ByteDance Ltd. and/or its affiliates*. The natural reading: Xiaomi implemented their in-house algorithm against `verl`'s extension points and left the copyright headers pointing at the library owners. Only `rollout_corr_helper.py` is genuinely upstream.

**2. The code anonymises a system the report names.** The fork's comments refer eight times to an undefined, capitalised "**reference RL framework**". The report never uses the phrase once. Report §6.4 names what it means: *"We extend the RL and OPD infrastructure of MiMo-V2-Flash, keeping SGLang and Megatron-LM as the inference and the training engine."* Concrete anchors (`SETUP_FAILED`, the `dirty_repetition` rule, `invalid_reward_value: -999`) confirm it.

**3. Both §4.3.3 equations are implemented term-for-term.** Eq.(4)'s every symbol has a named config field — including `anchor_quantile: 0.3`, deliberately below the median so compression bites harder. Eq.(5)'s α/β are `solve_signed_factors(removed, pos_base, added, neg_base)`, and conservation is enforced by runtime `assert`. Note the code's α/β names are **inverted** relative to the paper's (α = `min_scale`, β = `max_scale`) — read the values, not the letters.

**4. One comment explains why the obvious penalty is harmful.** Setting a bad turn's advantage to −1 "injects net negative mass into the batch, which with **no KL / entropy bonus** flattens the policy and lengthens outputs." The released recipe has `use_kl_loss: false` *and* `entropy_coeff: 0` — nothing counteracts injected negative mass. Mass conservation is a numerical requirement, not an aesthetic one.

**5. Three things the report describes are absent from the open stack** — and the blog says so rather than papering over it:

| Report describes | Open stack has |
|---|---|
| Four decoupled clip bounds, upper limit to **5.0**, entropy-adjusted | `clip_ratio_high` ∈ {0.2, 0.28} only; no schedule code exists |
| GRS + GAR across all code agent tasks | Webdev groupwise grader only; `grep -rn "GAR\|GRS" recipes/` → zero hits |
| Anti-hack: 2-stage detection + full rule set | Regex guard (default **OFF**), stage 1 only; 4 rules listed as `excluded_reference_rules` |

**6. The data-to-environment join is exact.** `image-mapping.jsonl` has **3,764** lines; Docker Hub has **3,764** tags. Composition: 2,698 code + 1,000 cyber + 65 general + 1 webdev. For code and cyber, images and tasks correspond exactly, one to one.

## Reproduce

Everything the blog claims can be re-derived. The two external clones plus one script:

```bash
# 1. Upstream fork (735 .py, plus two submodules)
git clone --depth 1 https://github.com/XiaomiMiMo/verl.git mimo-verl
cd mimo-verl && git submodule update --init --depth 1 \
    third_party/mimoagent-osr third_party/uni_agent && cd ..

# 2. Dataset (5 parquets, 7,780 rows, plus the image manifest)
#    Requires huggingface_hub; any Python with pyarrow can read the files.
python -c "
from huggingface_hub import snapshot_download
snapshot_download('XiaomiMiMo/MiMo-V2.6-RL-oss', local_dir='mimo-oss-data')
"

# 3. Re-run the machine-independent checks in this repo
python tools/check_claims.py
python -m pytest tools/test_check_claims.py -q
```

`tools/check_claims.py` is a standard-library-only gate. It verifies that the README's relative links resolve, that [`evidence/offsets.json`](evidence/offsets.json) is internally consistent (family counts sum to the total, code/cyber images equal task counts, shared-image domains really do share), and that the blog contains no external resource tags so it stays viewable offline.

`tools/test_check_claims.py` is what makes that gate trustworthy. It builds throwaway repositories, injects one deliberate defect into each, and asserts the gate fails — **13 mutations, all detected**. Without it, a gate that silently stopped checking anything would look exactly like a gate that works.

### Docker images

The images are **not** redistributed here — they live on Docker Hub at [`xiaomimimo/mimo-v2.6-rl-oss`](https://hub.docker.com/r/xiaomimimo/mimo-v2.6-rl-oss). Where a container runtime is unavailable, the Registry v2 API works directly:

```bash
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io\
&scope=repository:xiaomimimo/mimo-v2.6-rl-oss:pull" | python -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -sL -H "Authorization: Bearer $TOKEN" \
  "https://registry-1.docker.io/v2/xiaomimimo/mimo-v2.6-rl-oss/manifests/general-agent-env-1" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json"
```

## Repository layout

```
blog/mimo-v2.6-rl.html      the post — single file, inline CSS, zero dependencies
evidence/VERIFICATION.md    raw commands and output for every claim
evidence/offsets.json       machine-readable count ledger, checked in CI
tools/check_claims.py       stdlib-only gate over links, ledger, and blog self-containment
tools/test_check_claims.py  negative tests proving the gate fails on each defect
```

## Scope and limits

This is **analysis, not a reproduction**. Nothing here trains a model; no GPU is involved.

- Every claim traces to a file path, config value, parquet field, image tag, or report section. Line numbers cited in the blog refer to the `pypdf` text extraction of the report, not to PDF page coordinates.
- **One finding is explicitly labelled an open question**: why 925 `general_agent` tasks share a single image while 64 `terminal_bench` tasks each get their own. The split is read directly from the `docker_image` column, but the *mechanism* is inferred from `manifest.json` — no line of code was found that draws the boundary.
- The report's production numbers (MiMo-V2.6-Pro: lr 3×10⁻⁶, 384 experts/layer) describe the internal run, not the released 9B recipe (lr 1×10⁻⁶). They are not interchangeable and the blog keeps them separate.
- Upstream-presence checks record HTTP status codes only; they establish that a path does not exist upstream, not why.

## License

Apache-2.0 — see [LICENSE](LICENSE).

This repository contains **no code from Xiaomi, ByteDance, or `verl-project/verl`**. It is an independent analysis: the blog is original prose, `check_claims.py` is original, and `offsets.json` records facts read from public artifacts. The `verl` fork and the Docker images remain under their own licenses and are the property of their respective owners.

## References

The analysis builds directly on work by others:

- **MiMo-V2.6** — *MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement*, Xiaomi MiMo Team. Technical report, 44 pp. Defines everything analysed here: the three scaling axes (§4), groupwise agentic grading (§4.3), multi-harness training (§4.2.5), reward-hacking mitigation (§4.2.6), and RL infrastructure (§6).
- **`XiaomiMiMo/verl`** — RL training fork at commit `a2ad9f6`, based on [`verl-project/verl`](https://github.com/verl-project/verl) `0.9.0.dev`.
- **`XiaomiMiMo/mimoagent`** — agent harnesses, tools, execution environments and graders. Fork of [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).
- **`XiaomiMiMo/uni-agent`** — model gateway and TransferQueue trajectory capture. Fork of [verl-project/uni-agent](https://github.com/verl-project/uni-agent).
- **ARVO** — the vulnerability-reproduction dataset family behind the cyber tasks.
- **GRPO** — Shao et al., *DeepSeekMath*, 2024. The base algorithm; this stack stays deliberately close to vanilla.
- **Muown** — the Muon + AdamW hybrid optimizer, with row-norm control, used for the mid-training switch.
- **GLM-5.2** — cited by `antihack.py` as the source of the two-stage online guard design.

With thanks to the MiMo team for releasing the environments, verifiers, and images that make this kind of reading possible at all.
