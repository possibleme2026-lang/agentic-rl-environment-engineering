# Verification Evidence

Raw commands and their actual output, backing every claim in
[`blog/mimo-v2.6-rl.html`](../blog/mimo-v2.6-rl.html).

Reproduce from a clean checkout: `bash scripts/verify.sh` re-runs the machine-independent
checks below against a fresh clone of the upstream sources.

---

## 1. Dataset row counts and image-mapping composition

PyArrow against the five released parquets, plus a family histogram over `image-mapping.jsonl`:

```
code            rows=  2698 cols=6
cyber           rows=  1000 cols=5
general-train   rows=   989 cols=6
webdev          rows=  2093 cols=6
music           rows=  1000 cols=5

image-mapping total: 3764

format-code-task (code)             2698
arvo-rl (cyber)                     1000
general-agent-env (general)           65
webdev-rl-opensource (visual)          1
TOTAL                               3764
```

Cross-check against parquet row counts:

```
code rows 2698 -> mapping 2698     # exact
cyber rows 1000 -> mapping 1000    # exact
webdev rows 2093 -> mapping 1
general rows 989 -> mapping 65
```

Docker Hub tag count for `xiaomimimo/mimo-v2.6-rl-oss`: **3764**. Equal to the
`image-mapping.jsonl` line count.

---

## 2. Fork provenance — upstream presence of the penalty files

HTTP status from the GitHub Contents API, both against upstream `verl` at tag `v0.9.0`
and against `main`:

```
verl/trainer/ppo/arvo_penalties.py             404
verl/trainer/ppo/signed_rebalance.py           404
verl/utils/length_penalty.py                   404
verl/trainer/ppo/rollout_corr_helper.py        200
```

`404` means the path does not exist upstream. Repeated on `main` for the two most
important:

```
verl/trainer/ppo/arvo_penalties.py             404
verl/utils/length_penalty.py                   404
```

Copyright headers of the same files, read locally:

```
verl/trainer/ppo/arvo_penalties.py                 Copyright 2025 Bytedance Ltd. and/or its affiliates
verl/trainer/ppo/signed_rebalance.py               Copyright 2026 Bytedance Ltd. and/or its affiliates
verl/trainer/ppo/rollout_corr_helper.py            Copyright 2025 Bytedance Ltd. and/or its affiliates
verl/utils/length_penalty.py                       Copyright 2025 Bytedance Ltd. and/or its affiliates
recipes/arvo/agent_loop.py                         Copyright 2026 Bytedance Ltd. and/or its affiliates
recipes/code/reward.py                             Copyright 2026 Bytedance Ltd. and/or its affiliates
```

---

## 3. The "reference RL framework" anonymisation

Search across the fork's `verl/` tree:

```
$ grep -rn "reference RL framework" mimo-verl/verl/
verl/trainer/config/algorithm.py:94
verl/trainer/config/algorithm.py:114
verl/trainer/config/algorithm.py:120
verl/trainer/config/ppo_trainer.yaml:135
verl/trainer/ppo/signed_rebalance.py:29
verl/trainer/ppo/signed_rebalance.py:31
verl/trainer/ppo/signed_rebalance.py:33
verl/trainer/ppo/v1/trainer_base.py:262
```

Search across the extracted technical report text (44 pages / 132,725 characters):

```
$ grep -c "reference RL framework" _mimo_report.txt
0
```

The report instead names the system in §6.4:

> "We extend the RL and OPD infrastructure of MiMo-V2-Flash, keeping SGLang and
> Megatron-LM as the inference and the training engine, respectively."

Supporting anchors found in the same comments: the error category `SETUP_FAILED`, the rule
name `dirty_repetition`, and `algorithm.invalid_reward_value: -999`.

---

## 4. Dynamic clip bounds: absent from the OSS stack

Every `clip_ratio_high` declaration in the repo:

```
./recipes/code/config/train.yaml:95:    clip_ratio_high: 0.2
./third_party/uni_agent/examples/blackbox_recipes/claude_code/config/claude_code_megatron_v1.yaml:79:    clip_ratio_high: 0.28
./verl/trainer/config/actor/actor.yaml:42:clip_ratio_high: 0.2
./verl/trainer/config/distillation/distillation.yaml:55:  clip_ratio_high: 0.2
./verl/trainer/config/_generated_ppo_{megatron,torchtitan,veomni}_trainer.yaml    0.2
```

Two distinct values: `0.2` and `0.28`. Searches for a schedule:

```
$ grep -rn "dynamic_clip|clip_schedul|entropy_aware|clip_bound|adv_clip" --include=*.py --include=*.yaml .
(no output)
```

The report describes four decoupled bounds with the upper limit reaching 5.0, adjusted by
entropy.

---

## 5. GAR / GRS naming: absent from the released recipes

```
$ grep -rn "GAR|GRS" --include=*.py --include=*.md recipes/
(no output)
```

What *is* present: `recipes/design/grader_service/` (the webdev `group_v1` grader),
`recipes/design/webdev/group_reward.py` (driver-side group rewrite), and
`verl/trainer/ppo/arvo_penalties.py` (the penalty module).

---

## 6. The seven-item configuration contract

`ReferencePenalties.for_training()` in `verl/trainer/ppo/arvo_penalties.py` raises
`ValueError` unless all seven match:

```python
required = {
    "algorithm.adv_estimator": "grpo",
    "algorithm.norm_adv_by_std_in_grpo": False,
    "algorithm.use_kl_in_reward": False,
    "algorithm.filter_groups.enable": True,
    "algorithm.filter_groups.metric": "reward",
    "actor_rollout_ref.actor.use_kl_loss": False,
    "actor_rollout_ref.actor.loss_agg_mode": "prompt-mean",
}
for key, value in required.items():
    if OmegaConf.select(config, key) != value:
        raise ValueError(f"The frozen reference profile requires {key}={value!r}")
```

---

## 7. Anti-hack guard defaults

`third_party/mimoagent-osr/src/mimoagent/agents/antihack.py` (343 lines) — module docstring:

> "Default OFF. An absent or `enabled: false` config is a strict no-op — `inspect` returns
> `None` for every action — so existing runs are byte-for-byte unaffected."

> "Two-stage detection (per the GLM-5.2 blog) is scaffolded but only stage 1 ships:
> **Stage 1 (here):** a regex filter tuned for recall over the known leak vectors. This is
> what runs in v1. **Stage 2 (stub):** an optional LLM judge (`llm_judge_enabled`) that
> confirms the *intent* of a flagged action to keep precision high. Until it is wired to a
> model endpoint, `confirm` conservatively upholds every regex hit."

---

## 8. Router freeze: before / after

Report §5.4, decoder layer 9, 384 experts, MiMo-V2.6-Pro:

| Metric | Trainable router (steps 1 → 20) | Frozen router |
|---|---|---|
| Coefficient of variation | 0.78 → 2.0 | ≈ 0.7 (flat) |
| Peak load (max/mean) | 6× → 16× | ≈ 5.5× (flat) |
| Cold experts (<0.1× mean) | 0.5% → 22% | ≈ 1% (flat) |

---

## 9. Tool-call parser pathology

`_cap_tool_calls` docstring in `recipes/arvo/agent_loop.py`:

> "On Qwen3.5-9B the `qwen3_coder` parser turns a batched multi-call turn into a long run of
> junk: one observed assistant turn parsed as 47 calls -- 3 real `read` calls followed by 44
> named `command` with EMPTY arguments, i.e. `<parameter=command>` being read as
> `<function=command>`. Across 64 trajectories that made 76.6% of all tool calls
> nonexistent-tool calls (9659/12616), each answered with "Unknown tool", which the model
> retried until the 128K budget was gone (89.1% ended on budget exhaustion)."

> "It is NOT a fix -- it also drops legitimate parallel calls -- so the real repair is in the
> parser, for which `_dump_raw_generation` collects the evidence."

---

## 10. General-domain image injection split (inferred, not confirmed in code)

`general-train.parquet` — 989 rows:

```
general_agent    925 rows -> docker_image "general-agent-env-0:oss"      (shared)
terminal_bench    64 rows -> docker_image "general-agent-env-1..65:oss"  (dedicated)
```

**Open question.** This split is read directly from the `docker_image` column. The *mechanism* (baked into image vs injected at runtime via the MCP manifest and
uploads) is inferred from `general-sample/manifest.json`, which specifies a sidecar
container, four MCP servers on ports 39101–39104, six uploads, and
`verifier.command = "python3 /work/run_verify.py"`. No line of code was found that draws
the boundary. Flagged as an open question in the blog rather than asserted.

---

## 11. Image layer shapes

| Image | Size | Layers |
|---|---|---|
| `general-agent-env-1` | 292.5 MB | 8 |
| `arvo-v1-42525170` | 603.1 MB | 39 |
| `webdev-rl-opensource:v2` | 2,629.7 MB | 1 |

`general-agent-env-1` layer blobs were downloaded individually and all eight sha256 digests
verified against the manifest.

---

## 12. Multi-harness mixing

`recipes/code/mimoagent_runner.py`:

```python
digest = hashlib.sha256(f"{seed}\0{int(harness_round)}\0{sample_key}".encode()).digest()
index = int.from_bytes(digest[:8], "big")
label, config_path = specs[index % len(specs)]
```

`scripts/code/env.example`:

```
# N is the GRPO group size. With four harness arms and step-hash mixing, a
# group stays on one arm and the sample rotates across arms between steps, so N
# does not need to be a multiple of four.
# N=16
# MIXED_HARNESS_MODE=step-hash
# MIXED_HARNESS_SEED=20260911
```

Four arms, from `config/agent/code/mix-four-whitebox.yaml`:

| Harness | Agent type | Tools | Protocol |
|---|---|---|---|
| `mini-bash` | `bashonly-agent` | bash-only | chat |
| `mini-mimocode` | `mimocode-agent` | bash, read, write, edit, grep, glob, task | chat |
| `mini-claude-code` | `cc-agent` | Bash, Read, Write, Edit, Grep, Glob | chat |
| `mini-codex` | `codex-agent` | exec_command, apply_patch | responses |

Constant across all four: `cwd: /testbed`, `cpu_limit: 4`, `memory_limit: 8Gi`,
`step_limit: 500`, `environment_class: kubernetes`, `anti_hack_cleanup: true`.

---

## Method notes

- All upstream-presence checks used `curl` against `api.github.com` through the local proxy,
  recording only HTTP status codes.
- Report text was extracted from the PDF with `pypdf`; line numbers cited in the blog refer
  to that extraction, and the quoted passages were re-read from the PDF during review.
- The Docker daemon was unavailable on this machine, so images were fetched directly via the
  Docker Registry v2 API (`auth.docker.io/token` → `registry-1.docker.io/v2/.../blobs/...`)
  and verified by digest.
- Where a claim could not be confirmed in code, it is listed here as an open question rather
  than presented as a finding.
