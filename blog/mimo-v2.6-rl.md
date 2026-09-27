*RL Post-Training · Code Reading*

# How Xiaomi MiMo-V2.6 Actually Does RL

*Reading the MiMo-V2.6 open stack — the technical report, the `XiaomiMiMo/verl` fork, 1,000+ real task environments, and 3,764 Docker images — and mapping every claim to the code that implements it.*

*Evidence-based walkthrough• commit `a2ad9f6`• verl `0.9.0.dev` fork• 5 environments• 3,764 images*

> **Part 2 of 2** · The ground survey — one release, read down to the config values · [← Part 1: the general field guide to environment engineering](environment-engineering.html)

The MiMo-V2.6 technical report describes a large-scale agentic RL run: **1,568 prompts × 16 rollouts = 25,000 trajectories per step**, 2.7–3.7 billion tokens per batch, up to 1M-token contexts, across five task domains and eight agent harnesses. Reports like this usually stop at the equations. What makes this release unusual is that Xiaomi also published a `verl` fork that implements the recipe, plus the actual task environments as Docker images.

So this is a reading exercise. For each claim in the report, I went looking for the code or the environment that makes it concrete — and where the code contradicts the report, I say so.

## 1. What was actually released (and what wasn't)

Five artifacts, all public:

| Artifact | Location | Size / shape |
|---|---|---|
| Technical report | 44-page PDF, §7.2 open-source section | ~133K chars extracted |
| RL training code | `XiaomiMiMo/verl` | 735 `.py`, +326 in submodules |
| Training data | `XiaomiMiMo/MiMo-V2.6-RL-oss` | 7,778 task rows across 5 parquets |
| Task environments | `xiaomimimo/mimo-v2.6-rl-oss` (Docker Hub) | 3,764 image tags |
| Starting model | `MiMo-V2.6-Distill-Qwen-9B` | distilled from Qwen3.5-9B |

The dataset splits, read directly from the parquet files:

| Parquet | Rows | `dataset_type` | Verifier |
|---|---|---|---|
| `code.parquet` | 2,698 | `opensource-code` | Executable tests |
| `cyber.parquet` | 1,000 | `arvo` | Rule checks (ASan/MSan/UBSan) |
| `general-train.parquet` | 989 | `general_agent` (925) + `terminal_bench` (64) | Rubric-based judging |
| `webdev.parquet` | 2,093 | all `category=website` | Visual grading |
| `music.parquet` | 1,000 | zh 504 / en 496; short 504 / long 496 | Rule checks (MIDI) |

That is 7,780 rows — just over the "approximately 7k tasks plus 1k music tasks" the report claims. The five domains also match the report's Table 4 SFT composition and Table 6 evaluation table exactly: Code, Cyber, General, Visual, Music.

> **The single most useful artifact**
>
> `image-mapping.jsonl` — 3,764 lines, each mapping a dataset row's `dataset_image` to a Docker Hub tag:
>
> ```
> {"dataset_image": "arvo-rl:v1-arvo-10055",
>  "dockerhub_image": "docker.io/xiaomimimo/mimo-v2.6-rl-oss:arvo-v1-10055"}
> ```
>
> This file is the **join key between the data and the environments**. Without it, the dataset is a pile of JSON blobs with dangling image references. With it, every task can be reconstituted.

## 2. The provenance puzzle: whose code is this?

Before reading the algorithm, it's worth knowing what you're reading. The fork's `README.md` is refreshingly explicit:

```
This fork adds reproduction code for five RL environments upon
[verl](https://github.com/verl-project/verl) (0.9.0.dev).
```

But that undersells it. Checking every penalty- and consistency-related file for its copyright header:

| File | Copyright | On upstream `verl` v0.9.0? |
|---|---|---|
| `verl/trainer/ppo/arvo_penalties.py` | 2025 ByteDance | **404 — fork-only** |
| `verl/trainer/ppo/signed_rebalance.py` | 2026 ByteDance | **404 — fork-only** |
| `verl/utils/length_penalty.py` | 2025 ByteDance | **404 — fork-only** |
| `verl/trainer/ppo/rollout_corr_helper.py` | 2025 ByteDance | 200 — upstream |

So the files carrying the report's §4.3.3 behavioural regularization are *not* in upstream `verl` — yet they carry ByteDance copyright. The natural reading: Xiaomi needed penalty machinery that matched their in-house RL system, wrote it against `verl`'s extension points, and left the copyright headers pointing at the library owners rather than rewriting them. The `recipes/` directory — the part that is unambiguously Xiaomi's — is where the glue lives.

### The "reference RL framework"

Now the interesting part. Grep the code for how it refers to the system it is porting from:

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

Eight hits. A phrase used consistently, never defined, capitalised like a proper noun. The report itself **never uses this phrase once** (zero hits across all 133K characters). So the code is systematically anonymising a system the paper names outright.

The report names it in §6.4:

> **Report §6.4**
>
> "We extend the RL and OPD infrastructure of MiMo-V2-Flash, keeping **SGLang** and **Megatron-LM** as the inference and the training engine, respectively."

Combined with the concrete anchors in the comments — `SETUP_FAILED` as an error category, a `dirty_repetition` rule name, `invalid_reward_value: -999` — the "reference RL framework" is unambiguously **MiMo's own internal SGLang + Megatron RL stack**, the system that trained MiMo-V2-Flash and then Pro/Flash. The open-source `verl` fork is a *port* of that system's learning-signal design onto a public framework.

> **Why this matters for reading the code**
>
> When a comment says "this replaced the reference RL framework's approach because…", it is describing a design decision made in a system you cannot see. The comment is the *only* surviving rationale. That makes these docstrings unusually high-value documentation — and also means you cannot independently verify them from the open code alone.

## 3. The architecture in one picture

The report frames RL scaling along three axes (§4), with compute proportions measured on the Pro run:

| Axis | Mechanism | Code / config anchor |
|---|---|---|
| **1. Training compute**
43.8% rollout / 43.5% train / 12.7% grader | Large batch, async partial rollout at staleness 4, 2.7–3.7B tokens/step | `train_batch_size`, `N=16`, `MAXLEN=262144` |
| **2. Environment & harness diversity** | 5 domains × 4 mini-harnesses, mixed by step-hash | `config/agent/code/mix-four-whitebox.yaml` |
| **3. Grader compute** | Groupwise agentic grading — GRS (offline) + GAR (online) | `recipes/design/grader_service/` |

The execution model is **agent-centric**, not inference-centric (§6.1). Each sequence runs as an *Agent Loop* that owns the environment lifecycle and calls the inference engine on demand, exposing a request endpoint that the external agent drives. Dialogues are kept both as string prefixes and token sequences, so only new suffixes get tokenised. The trajectory data is a four-level hierarchy:

```
Sample  →  one prompt dispatched by the Sample Mixer (spawns a GRPO group)
  └─ Sequence  →  one Agent Loop execution  ← the group is accept/reject as a whole
       └─ Context   →  one dialogue branch (unit of prefix matching, KV reuse, export)
            └─ Segment  →  one turn: system/user message, model generation, or tool result
                           (only model-generated turns contribute to the loss)
```

This hierarchy is not cosmetic — it's what makes the Penalty Module able to target a decision at the right granularity. Hold that thought for §5.

## 4. GRPO, but engineered

The objective is standard GRPO with token-level importance sampling, Eq.(1):

```
J(θ) = Eq~D, {oi}~μθold [ (1/Σ|oi|) Σi Σt ri,t Mi,t Ai log πθ(oi,t | q, oi,<t) ]
(1)
```

What matters is how it's configured. From the released recipe, `recipes/code/config/train.yaml`:

```
actor:
  use_rollout_log_probs: true        # keep rollout policy's logprobs (needed for IS)
  clip_ratio: 0.2
  clip_ratio_low: 0.2
  clip_ratio_high: 0.2
  clip_ratio_c: 3.0                  # dual-clip PPO lower bound
  loss_agg_mode: prompt-mean         # ← averages at PROMPT level, not over tokens
  entropy_coeff: 0
  use_kl_loss: false
  optim:
    lr: 1e-6
    lr_decay_style: constant

algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: false     # ← no /std normalisation
  use_kl_in_reward: false
  filter_groups:
    enable: true                     # drop all-pass / all-fail groups
```

### Four choices worth pausing on

**`loss_agg_mode: prompt-mean`** — the report is explicit that this "averages the surrogate at the prompt level rather than over all response tokens". The consequence: a 50-turn trajectory does not get 50× the gradient weight of a 1-turn trajectory. Given that §5.5 documents trajectories hitting length limits as a primary failure mode, this is a direct countermeasure to length-driven gradient inflation.

**`norm_adv_by_std_in_grpo: false`** — the advantage is `Ri − mean(R)`, not divided by the group standard deviation. On binary rewards this matters enormously: dividing by `std` of a nearly-degenerate group (e.g. 7 pass, 1 fail) produces a huge advantage for the single outlier. Skipping the division keeps the signal proportional to the actual reward gap. Note this is *load-bearing* — `arvo_penalties.py` raises `ValueError` if it is set otherwise (see §5).

**`clip_ratio_low == clip_ratio_high == 0.2`** — and here the code *contradicts* the report.

> **Discrepancy: the decoupled clip bounds are not in the open stack**
>
> The report describes four decoupled clip boundaries with the upper bound allowed to move to **5.0**, adjusted dynamically by entropy. In the open-source recipe, `clip_ratio_low` and `clip_ratio_high` are both pinned at `0.2` — plain PPO. A repo-wide sweep of every `clip_ratio_high` declaration finds only two distinct values:
>
> ```
> 0.2   — everywhere in the OSS recipe and verl defaults
> 0.28  — third_party/uni_agent/.../claude_code_megatron_v1.yaml
> ```
>
> There is no entropy-driven clip schedule anywhere: searches for `dynamic_clip`, `clip_schedul`, `entropy_aware`, `clip_bound` all return zero hits. **The dynamic clipping is part of the internal stack, not the released one.**

**`filter_groups.enable: true`** — groups that are all-pass or all-fail are dropped. With binary test rewards, such groups have zero advantage variance and therefore produce zero gradient. The report cites this as the "dynamic sampler". Worth knowing the failure mode: dropping groups biases the reported post-filter pass-rate toward 0.5 (a known sampling-bias artefact documented in sibling frameworks).

### Off-policy correction

Because the released recipe uses staleness-4 asynchronous rollouts, training happens on trajectories from older checkpoints. `recipes/code/config/train.yaml` enables the correction block:

```
rollout_correction:
  bypass_mode: false
```

The machinery behind it lives in `verl/trainer/ppo/rollout_corr_helper.py`, which is *upstream verl* (not Xiaomi code). Its docstring states the problem well: it addresses "policy mismatch between rollout and training implementations (e.g., vLLM BFloat16 vs FSDP FP32)", "model update staleness", and general distribution shift. It offers token-level and sequence-level IS weights, plus rejection-sampling filters (`token_k1/2/3`, `seq_sum_k*`, `seq_mean_k*`, `seq_max_k*`), with log-space computation and a fixed `exp(±20)` safety bound to avoid overflow.

The subtlety from §6.4 that *is* implemented in the fork: top-k and top-p sampling renormalise over a restricted candidate set, so training log-probabilities must be renormalised within that same set. `routed_experts` — the MoE expert indices per token — does flow through the fork's `verl/experimental/agent_loop/agent_loop.py`, and `verl/workers/engine/veomni/transformer_impl.py` has an explicit `REPLAY` mode that raises if `routed_experts` is missing rather than silently falling back. The Megatron engine reads it at `transformer_impl.py:1092`.

> **Careful reading**
>
> The fork plumbs `routed_experts` end-to-end and has a strict replay path in the VeOmni engine. But there is also a `compute_moe_lb_metrics` consumer (`metric_utils.py:312`) that treats it as *load-balance telemetry*, gated behind `moe_lb_metrics_interval`. So the field serves two masters: routing replay for consistency, and MoE load-balance monitoring. Both are consistent with §5.4 and §6.4 — just don't assume the presence of the field proves replay is active in any given run.

## 5. The Penalty Module: Eq.(4) and Eq.(5) in code

This is the most rewarding part of the codebase to read, because report §4.3.3 states two equations and the code implements both with matching structure.

### 5.1 The design: detection separated from effect

From §6.1: "*we separate detection from its effect on training. A **Rule** judges a segment, context, or sequence… A **Strategy** binds an action to a level of the hierarchy*."

The strategies are enumerated in `verl/trainer/config/algorithm.py`:

```
MARKED_TOKEN_PENALTY_STRATEGIES = frozenset({
    "monitor",       # record metrics only, zero effect on training
    "mask",          # exclude hit tokens from the loss
    "adv_reduction", # subtract penalty_value from advantages after GRPO norm
    "adv_set",       # replace their advantages with penalty_value
    "adv_signed",    # sign-aware, mass-conserving (the interesting one)
})
```

And the escalation ladder is documented: "a context with no surviving model turns is dropped, a sequence with no surviving context receives zero advantage, and a sample with no surviving sequence is rejected." That's the four-level trajectory hierarchy from §3 earning its keep.

### 5.2 Eq.(4): the group-relative length penalty

```
R̃i = Ri − 1[i ∈ Pq] · X · [ clip( (ℓi/ℓ*q − 1 − δ)/(s − δ), 0, 1 ) ]γ
(4)
```

Where ℓ*<sub>q</sub> is a quantile of the *successful* rollouts' lengths in the group, X is the max deduction, δ the tolerated excess, s the saturation point, γ ≥ 1 the ramp exponent. Now read `verl/utils/length_penalty.py`'s config docstring — every symbol has a named field:

| Paper symbol | Config field | OSS value | Meaning |
|---|---|---|---|
| X | `max_penalty` | `0.2` | Upper bound subtracted from reward |
| δ | `excess_threshold` | `0` | Deadzone — already at anchor ⇒ penalised |
| s | `excess_saturate` | `1` | 200% of anchor hits max penalty |
| γ | `penalty_exponent` | `1.5` | Convex ramp: `t**1.5` |
| B | `anchor_quantile` | `0.3` | p30 of passed rollouts' lengths |
| A | `min_pass_rate` | `0.5` | Skip group unless pass-rate > 0.5 |
| — | `metrics` | `["turns","input_tokens","output_tokens"]` | Which length signals to compare |
| — | `combine` | `"max"` | Fold per-metric excess by worst dimension |
| — | `pass_threshold` | `0.5` | "Passed" = reward ≥ 0.5 |

The docstring even explains *why* the quantile is low: "0.5 = median (default). Lower values (e.g. 0.25 = p25) compress harder." Choosing 0.3 means the anchor sits below the median of successful runs — deliberately aggressive compression of token growth.

And it states the anchoring invariant that makes per-group computation safe: "*One call = one uid group. The accept boundary delivers exactly one full uid group, and both the anchor and the penalty are group-local, so per-group computation is equivalent to whole-batch computation.*"

### 5.3 Eq.(5): sign-aware, mass-conserving advantage rebalancing

Here's the code's own prose rendering of Eq.(5), from the `signed_rebalance.py` module docstring:

> **signed_rebalance.py — module docstring**
>
> "The penalty a marked token receives depends on the sign of its OWNING SEQUENCE's outcome advantage:
>
> - positive sample, marked token → advantage set to **0** (stop reinforcing the span)
> - negative sample, marked token → advantage ***= kappa** (kappa ≥ 1; punish the span harder)
>
> Each sign then conserves its total mass over the whole train batch: the mass removed from positive marked tokens is handed back to the clean tokens of positive samples by one factor `alpha ≥ 1` (clamped at `max_scale`); the mass added on negative marked tokens is taken back from the clean tokens of negative samples by one factor `beta ≤ 1` (clamped at `min_scale`)."

Compare to Eq.(5): α = min(α<sub>max</sub>, 1 + Σ<sub>H+</sub>A / Σ<sub>C+</sub>A) and β = max(β<sub>min</sub>, 1 − (κ−1)Σ<sub>H−</sub>|A| / Σ<sub>C−</sub>|A|). The code's `solve_signed_factors` computes exactly this — with the naming translated (`removed`↔Σ<sub>H+</sub>A, `pos_base`↔Σ<sub>C+</sub>A, `added`↔(κ−1)Σ<sub>H−</sub>|A|, `neg_base`↔Σ<sub>C−</sub>|A|) — and asserts conservation:

```
if not a_cl:
    assert abs(pos_post - pos_pre) <= 1e-6 * max(1.0, abs(pos_pre)), \
        f"positive mass not conserved: {pos_pre} -> {pos_post}"
if not b_cl:
    assert abs(neg_post - neg_pre) <= 1e-6 * max(1.0, abs(neg_pre)), \
        f"negative mass not conserved: {neg_pre} -> {neg_post}"
```

The conservation claim in the report's last sentence ("Each sign's total advantage mass is conserved when neither scale is clipped") is enforced by a runtime assertion. When a scale *is* clamped, the code logs a warning rather than failing — precisely the report's caveat.

> **Why the earlier approach was abandoned**
>
> Same docstring, and this is the kind of detail you only get from the code:
>
> "A whole-turn `adv_set -1` instead injects net negative mass into the batch, which with no KL / entropy bonus flattens the policy and lengthens outputs — the documented reason the reference RL framework replaced it with this."
>
> So the naive penalty (set the bad turn's advantage to −1) is **actively harmful** when `use_kl_loss: false` and `entropy_coeff: 0` — both of which are the OSS recipe's settings. Negative mass with no counterweight drives entropy up and outputs longer. The mass-conserving formulation is the fix.

One extension beyond the internal system is noted honestly: an optional per-row `token_weights` so that "mass" means *gradient* mass under prompt-mean aggregation. "With unit weights the math is identical to the reference RL framework's."

### 5.4 The seven-item configuration contract

`ReferencePenalties.for_training()` validates the entire training config before allowing penalties to be enabled. Any mismatch raises `ValueError`. From `recipes/arvo/REFERENCE_PENALTIES.json` and the validator:

```
{"selected_rules": ["tool_call_error", "length_penalty"],
 "tool_call_error": {
   "strategy": "adv_signed", "level": "segment",
   "negative_multiplier": 2,          # kappa = 2
   "signed_min_scale": 0.5,           # beta_min
   "signed_max_scale": 2.0 },         # alpha_max
 "length_penalty": {
   "enabled": true, "max_penalty": 0.2, "excess_threshold": 0,
   "excess_saturate": 1, "penalty_exponent": 1.5,
   "metrics": ["turns", "input_tokens", "output_tokens"],
   "combine": "max", "pass_threshold": 0.5,
   "anchor_quantile": 0.3, "min_pass_rate": 0.5 },
 "common_algorithm": {
   "norm_adv_by_std_in_grpo": false,
   "loss_agg_mode": "prompt-mean",
   "use_kl_in_reward": false,
   "use_kl_loss": false },
 "excluded_reference_rules": ["tool_name_invalid", "toxic_reasoning",
                              "overlong", "agent_context_filter"]}
```

The `for_training()` check is worth quoting because it reveals what the internal system demanded: `adv_estimator=grpo`, `norm_adv_by_std_in_grpo=False`, `use_kl_in_reward=False`, `filter_groups.enable=True`, `metric="reward"`, `use_kl_loss=False`, `loss_agg_mode="prompt-mean"`. These aren't suggestions — the reference recipe is **hard-coded into the validator**. You can't run the released penalties under a different aggregation scheme, which tells you the authors considered those seven settings part of the algorithm, not tunables.

The `excluded_reference_rules` list is equally informative: `tool_name_invalid`, `toxic_reasoning`, `overlong`, `agent_context_filter` exist in the internal system but are explicitly *not* shipped. Selectively releasing rules is a deliberate choice.

## 6. Groupwise agentic grading: GRS and GAR

This is the third scaling axis, and the report devotes 12.7% of compute to it. Two complementary methods on disjoint task subsets.

### 6.1 GRS — Groupwise Reward Synthesis (offline)

For high-pass-rate tasks, multiple offline rollouts are compared to construct task-specific rubrics, which are then reused to score future rollouts. The final reward is multiplicative, Eq.(2):

```
Ri = Rtesti · Ssoli · Sbehi
(2)
```

Multiplicative rather than additive, deliberately: "This multiplicative form keeps rubric supervision tied to test outcomes. Failed trajectories retain zero reward, while passing trajectories are further distinguished by implementation quality and problem-solving behavior." When every rollout passes, the product of the two rubric scores still separates them — that's the learning signal binary rewards throw away.

The report gives a qualitative justification for why this beats naive patch-size penalties: policies trained with online grading "produce smaller, more precise patches that remained within the requested scope", whereas workarounds such as "broad exports, exception swallowing, relaxed validation" aim to pass tests "but can exceed the scope of the task instructions".

### 6.2 GAR — Groupwise Advantage Redistribution (online)

```
λ = Σj∈P Aj / Σj∈P fjAj ,   A′i = λ fi Ai   if i ∈ P ,   Ai otherwise
(3)
```

An SFT-trained agentic grader sees all trajectories in a group within a shared workspace and ranks passing patches on five dimensions: suitability of approach, precision without omissions or unnecessary fallbacks, minimality, avoidance of unintended effects, and craftsmanship consistent with codebase conventions. Passing trajectories get quality factors f<sub>i</sub> ∈ (0,1]; a common factor redistributes the removed positive mass among them.

The critical design detail, which the report explains and which is easy to miss: "Simply downweighting positive advantages leaves negative advantages unchanged; renormalization restores their balance as a safeguard against excessive entropy growth." So λ is not cosmetic — it preserves the total positive mass so the advantage distribution stays balanced against the negatives.

Also: "When evidence confirms dependence on an external or leaked answer, we reset the trajectory's effective reward to zero and treat it as a failure *before recomputing group statistics*." The ordering matters — the hacked trajectory must not pollute the group mean.

### 6.3 The grader service, and why the visual one is fascinating

`recipes/design/grader_service/src/design_grader/service/server.py` is 1,333 lines serving a single reward: the webdev `group_v1`. The most instructive file is `group_pick.py`, the visual groupwise ranking core:

```
raw    = pick_norm − query_deduct
reward = (raw + 2) / 3          # ∈ [0.067, 1]
```

Ranking is done over `PICK_ROUNDS = 8` rounds of a **Williams row-complete Latin square**, so that every shot appears in every position exactly once and every ordered adjacency exactly once — "measured to flatten the first-position bias from ±0.28 to ±0.03 votes." Position bias in pairwise judging is a real, measurable problem; this is a clean fix.

Noise suppression is layered: if fewer than `PICK_MIN_OK = 5` rounds are valid, the whole group's pick is zeroed; too many near-duplicate comparisons also zero the group; a net vote within a deadzone zeroes a single shot.

> **Why the affine remap exists**
>
> The code documents that `group_v1` used to be the only producer of rewards in [−2, 1]. Consequence: the trainer's pass-rate was *pinned to 0.5*, the reward histogram degenerated, and the value `0.0` became semantically ambiguous (was it "no signal" or "bad"?). Remapping to [0.067, 1] fixes all three.
>
> **Lesson**: if a single reward source can emit values on both sides of the pass threshold, you have accidentally built a reward that cannot express "clearly good".

There's also a **hard runtime gate** (`runtime_gate.py`): a page that fails `node --check` syntax validation, throws an uncaught page error, or hangs gets `runtime = 0.0` — not a soft multiplicative factor. The comment explains why zero and not, say, 0.5: "vis=1.0 × 0.5 still beats a clean vis=0.5 page", so any soft factor lets a broken page outscore an honest one. And the hang case is 0 rather than a drop, because dropping the sample masks it out of the advantage calculation — which would teach the model that freezing the renderer is *safe*. The comment calls this "the `while(1){}` incident".

The driver side (`recipes/design/webdev/group_reward.py`) rewrites group rewards under four rules, and INVALID entries are replaced by the group's mean valid reward — because GRPO only ever looks at `reward − group_mean`, so setting an invalid sample to the mean makes it contribute exactly nothing.

## 7. Multi-harness training: four mini-harnesses

Report §4.2.5 argues that training a single harness couples task-solving strategies to harness-specific implementation details. The natural fix — train on production harnesses like MiMo Code and Codex — is rejected for two reasons: production harnesses wrap the loop in engineering safeguards whose behaviour falls outside the reward signal ("making credit assignment unreliable and leaving reward-unmeasured requirements to simply be ignored"), and their modules are too tightly coupled to vary one mechanism in isolation.

So they built *mini-harnesses* from a common minimal agent loop. In the released code, that is exactly four YAML files:

`config/agent/code/mix-four-whitebox.yaml`

```
harnesses:
  - label: mini-mimocode
    config: ./mini-mimocode.yaml
  - label: mini-bash
    config: ./mini-bash.yaml
  - label: mini-claude-code
    config: ./mini-claude-code.yaml
  - label: mini-codex
    config: ./mini-codex.yaml
```

Reading all four side by side shows the controlled-variation design clearly:

| Harness | Agent type | Tools | Protocol | System prompt |
|---|---|---|---|---|
| `mini-bash` | `bashonly-agent` | bash only | chat | 2 lines, minimal |
| `mini-mimocode` | `mimocode-agent` | bash, read, write, edit, grep, glob, **task** | chat | none |
| `mini-claude-code` | `cc-agent` | Bash, Read, Write, Edit, Grep, Glob | chat | long, with anti-hack clause |
| `mini-codex` | `codex-agent` | `exec_command`, `apply_patch` | **responses** | none |

The variation axes are clean: number of tools (1 → 2 → 6 → 7), whether there's a `task` sub-agent, the wire protocol (chat vs responses), and prompt verbosity. Everything else is held constant — `cwd: /testbed`, `cpu_limit: 4`, `memory_limit: 8Gi`, `step_limit: 500`, `environment_class: kubernetes`, `anti_hack_cleanup: true`, `temperature: 1.0`.

### The mixing rule

From `recipes/code/mimoagent_runner.py`, the step-hash scheduler:

```
digest = hashlib.sha256(
    f"{seed}\0{int(harness_round)}\0{sample_key}".encode()
).digest()
index = int.from_bytes(digest[:8], "big")
label, config_path = specs[index % len(specs)]
```

Crucially, **a GRPO group stays on one arm**, and the sample rotates across arms between steps. The env example states the consequence plainly: "With four harness arms and step-hash mixing, a group stays on one arm and the sample rotates across arms between steps, so N does not need to be a multiple of four."

That's a real constraint, not a nicety. If a group's 16 rollouts were split across 4 harnesses, the group-relative advantage would be comparing scores from *different environments* — the baseline would be meaningless.

On the seed, the env example is emphatic:

> **scripts/code/env.example**
>
> "The seed decides which arm each sample routes to, so changing it changes the *experiment* rather than just the shuffle. This is the reference run's value." — `MIXED_HARNESS_SEED=20260911`

And the docstring of `_select_config_path` explains why there's no single-harness branch: "*There is no single-harness branch: the launcher always enables mixing, and a one-armed run is a spec with one entry. Two code paths for one decision is how a run ends up training on a profile nobody chose.*"

The reported result (Table 7) — mean across 7 harnesses on MiMo Code Bench (mini): Qwen3.5-9B 15.3 → Distill 53.1 → **+ Multi-Harness RL 59.0**, with gains on all 21 dataset–harness pairs including the three held-out harnesses (codex, claude code, mini-swe-agent).

### A measured pathology, honestly documented

One of the best passages in the repo is `_cap_tool_calls` in `recipes/arvo/agent_loop.py`, describing a parser bug found during a real run:

```
On Qwen3.5-9B the `qwen3_coder` parser turns a batched multi-call turn
into a long run of junk: one observed assistant turn parsed as 47 calls
-- 3 real `read` calls followed by 44 named `command` with EMPTY
arguments, i.e. `<parameter=command>` being read as
`<function=command>`. Across 64 trajectories that made 76.6% of all
tool calls nonexistent-tool calls (9659/12616), each answered with
"Unknown tool", which the model retried until the 128K budget was gone
(89.1% ended on budget exhaustion).

Capping at 1 tests that story cheaply: if the junk rate collapses, the
batch parse is the cause. It is NOT a fix -- it also drops legitimate
parallel calls -- so the real repair is in the parser, for which
_dump_raw_generation collects the evidence.
```

Two things to admire here. First, the number 76.6% is shockingly high — nearly four out of five tool calls were invalid. Second, the fix is correctly labelled as *not* a fix: it's a diagnostic lever, shipped off by default, with `_dump_raw_generation` added so the raw emitted XML (which the trajectory dump normally discards, since it keeps only the parser's output) is recoverable.

## 8. Reward hacking: a three-layer defence

§4.2.6 opens with the failure mode precisely: for repository-repair tasks, "a recurring failure mode is **solution leakage**: agents obtain a published fix beyond the intended task context and use it to construct a patch." The report then tabulates five concrete patterns across real repos:

| Pattern | Example task | Observed action |
|---|---|---|
| Install and read a newer release | pytest `conftest.py` paths | `pip install pytest==5.4.3`; inspect source |
| Fetch upstream source | Astropy `TimeSeries` | `curl .../astropy/timeseries/core.py` |
| Clone upstream | Matplotlib `ax.clear()` | `git clone .../matplotlib.git` |
| Look up the solution | Django `MultiValueField` | Read change history of ticket #29205 |
| Probe versions | Sphinx return-type doc | Compare later releases for the fix |

The defence has three layers, and the code implements the third.

**Layer 1 — mid-training alignment data.** Cases were synthesised where "MiMo reflects on the faulty reasoning, revises the relevant turn, and continues with actions grounded in the task specification", with the revised reasoning keeping "the original error recognizable".

**Layer 2 — environment preparation.** Artefact/cache/git cleanup plus network isolation. This is what `anti_hack_cleanup: true` in every mini-harness config activates.

**Layer 3 — adversarial screening and running audit.** Two implementations:

- *Prompt-level*: `mini-claude-code.yaml` includes an explicit instruction that shortcuts "will be detected and scored zero", enumerating the exact vectors — don't fetch upstream, don't pip/npm install a newer release, don't overwrite test files, don't hard-code expected outputs.
- *Runtime interception*: `third_party/mimoagent-osr/src/mimoagent/agents/antihack.py` (343 lines) — an `AntiHackGuard` registered on the agent's action-interceptor chain, inspecting each tool call *before* execution and returning a dummy observation instead of blocking the rollout.

The rationale for interception-over-termination is stated well: "the model just sees an unhelpful result, so the shortcut can't reach the answer and can't inflate the verifiable reward, and the trajectory is never abruptly terminated (which is what destabilizes training when whole rollouts are rejected)."

Implementation details worth noting:

- **Default OFF**, and an absent config is "a strict no-op — `inspect` returns `None` for every action — so existing runs are byte-for-byte unaffected."
- **Only stage 1 ships.** The design follows a two-stage pattern (regex recall filter → LLM judge for precision), but stage 2 is a stub: "Until it is wired to a model endpoint, `confirm` conservatively upholds every regex hit."
- **All three tool schemas are covered.** The `_BASH_FIELDS` table maps exact tool-call names — `Bash`/`bash`/`exec_command`/`apply_patch` scan `command`/`cmd`/`input`/`patch` — across `tools/`, `tools/cc/`, and `tools/mimocode/`. The comment explains the necessity: scanning file tools "closes the non-Bash bypass where the model reads an eval artifact directly via Read/Grep/Glob instead of `cat`", and the lowercase rows exist because "mimocode reuses the lowercase ids of the original catalogue but takes cc's param name".

Net effect: "the logged confirmed-hack share remains below **2%** throughout the whole training process for both Flash and Pro."

## 9. RL infrastructure for 25K trajectories/step

§6 describes four subsystems. The engineering constraints are unusually explicit, which makes this section good reading even if you never run MoE RL.

### Harness Pool — multi-tenant actors

The naive design costs "one file descriptor per actor on Ray's global control store (GCS) node, and a large batch would exhaust the GCS node's file descriptors." The fix: fixed-size pools of persistent host actors, each carrying many concurrent tenants, balanced by in-flight instance count. A host is one process whose tenants share one event loop, one request endpoint, one inference proxy, one tokenizer — and on the environment side, one imported harness codebase.

The failure mode this creates and its mitigation: "A shared event loop would let one blocking call stall every tenant, so blocking work — environment operations and tokenization — runs on background threads."

There's also a hard constraint tying back to §7: "One process can import only one harness codebase, so different codebases run in separate pools." Harness pools receive "configurable shares of a fixed budget of host actors — set once at startup, independent of the training-data mixture that is scheduled every step."

### Payload Porter — disaggregated control/data plane

Each sequence is heavy: token ids, log-probabilities, MoE routing data, top-p sampling indices, and multimodal data. Gathering all of that on one driver node ties batch size to that node's memory. So each sequence is split at rollout finish: the payload is written once to a distributed KV store (Ray object store or TransferQueue), while the driver schedules on lightweight metadata only — scalar rewards, per-context lengths, and keys.

"At group finish, only the fields needed are read from the store: a few columns for the accept-time hook." Then the hook imposes length penalties, computes group-relative advantages, and applies advantage shaping; per-token advantages are written back; groups with all-zero advantages are dropped.

Packing is equally careful: "one packer per training tensor-parallel (TP) group serves every rank in the group. From the unpadded rows in the store, it fetches only those its context-parallel (CP) window touches and cuts out that window alone. The result is shared across the TP group as a single read-only in-memory copy." No full-batch aggregation on the driver, no dense padded intermediates.

Multimodal adds a wrinkle: a single trajectory can accumulate gigabytes of screenshots. So during rollout the Agent Loop ships "only the multi-modal delta between requests", and for training, because the vision encoder is replicated across the TP group while the LLM backbone is sharded, "encoding runs data-parallel first — image items are balanced across ranks independently of where each sequence's tokens land" — then embeddings are redistributed to the ranks holding the corresponding tokens.

### Sample Mixer — the discipline layer

The motivating numbers: across **25 profiled data sources, mean generated tokens vary by 90× and active rollout duration by 66×**. A naive scheduler will simply starve short tasks. Four mechanisms fill the specified training distribution:

1. **Adaptive Rollout Concurrency** — per-source budgets
2. **Adaptive Rollout Scheduling** — selects sources within budgets
3. **Predictive Rollout Dispatch** — places new rollouts across ranks
4. **Sample Replay** — covers startup and recovery

The failure analysis in §5.5 shows how fragile the coupling is: "one harness later produced rollouts less than half as long as those from other harnesses on the same code dataset, skewing estimates in the second and third steps after restart", and biases at startup "exhausted both the GPU and pinned host-memory KV pools."

### Training–inference consistency

Four mechanisms, and the detail that took me a moment to appreciate is the top-p bitmap:

> "For top-p sampling, only the GPU–CPU transfer is dense: we ship a bitmap of fixed shape and full-vocabulary width. The fixed shape avoids a GPU–CPU synchronization; the full width never truncates even a set that spans the whole vocabulary. All later stages are sparse: at a typical top-p of 0.97, a candidate set averages fewer than five tokens."

A fixed-shape full-vocabulary bitmap costs transfer bandwidth but buys two things: no GPU→CPU sync (the shape is known statically) and no truncation edge case. At p=0.97 the set averages <5 tokens, so the sparse downstream work is trivial. It's the right trade for a synchronisation-free async pipeline.

Plus: QDQ (quantize–dequantize) applied to experts after every update so both engines see identical MXFP4 weights; R3 routing replay; context caching with a hierarchical extension that keeps state "in HBM during GPU time and in a pinned host pool during tool time", offload/restore on side CUDA streams "never stalling generation".

### Router freezing — the most striking single result

§5.4 tracks three statistics at decoder layer 9 (384 experts) of MiMo-V2.6-Pro:

| Metric | Trainable router (steps 1→20) | Frozen router |
|---|---|---|
| Coefficient of variation | 0.78 → **2.0** | ≈ 0.7 (flat) |
| Peak load (max/mean) | 6× → **16×** | ≈ 5.5× (flat) |
| Cold experts (<0.1× mean) | 0.5% → **22%** | ≈ 1% (flat) |

All three rise monotonically. The diagnostic is the elegant part: restoring *only* the router parameters to their pre-RL values recovers load balance "while benchmark performance remains unchanged" — proving the collapse is router drift, not degradation of the expert weights. So the router is frozen for RL, and "benchmark performance growing normally" confirms nothing was lost.

> **Note the label**
>
> The report calls this "a severe load-collapse problem" and resolves it by freezing. But in the same paper, GRS/GAR and the penalty module exist partly to *shape* learning during RL. Freezing the router is the one place where the answer is to stop a component from learning at all. Twenty-two percent of experts going cold is not a subtle drift — it's a structural collapse of the model's capacity allocation.

Also worth noting for anyone thinking about stability: §5.5's failure timeline lists GPU memory double-bit errors, a Kubernetes failure crashing Cyber-task pods between steps 15 and 16, and the grader becoming unreachable after step 14. Training also used a gradient clipping threshold of 1.0, and initialized RL by carrying over FP32 master weights and Muown's row state from the SFT checkpoint.

## 10. The task environments, up close

This is where the release is most valuable and least documented. The dataset gives you a Docker image name per task; the Docker Hub repo gives you the image. Let me walk the four environment designs, because they're genuinely different solutions to genuinely different problems.

### Code — 2,698 tasks, one image each

Reading `code.parquet`, each row's `extra_info.instance_json` contains:

```
{"cwd": "/testbed",
 "dataset_type": "opensource-code",
 "docker_image": "format-code-task-000001:latest",
 "instance_id": "...",
 "problem_statement": "...",
 "test_command": "bash /testbed/mimo_test_command.sh",
 "test_patch": "...",              # present in 100% of rows
 "verifier_timeout_sec": 1800}
```

The design is recognisably SWE-bench-lineage: a repo checked out at a historical commit, a bug report as the prompt, a test patch applied at verification time, and the test command as the oracle. Two details stand out. `test_patch` is present on every single row — so the verifier always has an authoritative definition of correctness. And the timeout is 1,800 s (30 minutes) per task.

Note that `data_source=opensource-code` and `ability=swe`, and the harness is `mimo_swe_agent`. The dataset is stored with the task as an *embedded JSON string* in `extra_info.instance_json` rather than as nested struct columns — the dataset loader's comment explains this avoids Arrow struct-union corruption when rows have heterogeneous fields.

### Cyber — 1,000 tasks, 1,000 distinct images

Every row is `dataset_type=arvo` with a unique image (`arvo-rl:v1-arvo-10055`, etc.). The verifier design is the interesting part, from §4.2:

> "We instead extract two attributes from the ground-truth sanitizer (ASan/MSan/UBSan) report: the vulnerability type (e.g., heap-buffer-overflow) and the crash location (the topmost project-level stack frame). A PoC is accepted iff its crash matches both under rule-based string matching, which is deterministic, reproducible, and computationally trivial."

The report is precise about why the two obvious alternatives fail. Fix-binary differential testing (as in CyberGym) "accepts a PoC that crashes the vulnerable binary but not the patched one; an incomplete patch rejects correct PoCs, and unrelated changes between the two commits flip verdicts for reasons unconnected to the bug — **either error corrupts the training gradient**." LLM-based judging "returns different verdicts for the same PoC across runs and cannot serve as a stable reward."

And the key property: the task description is derived from the *same* sanitizer report, "stating the exact type and function in which the crash must occur, so description and verification share a single source of truth." That's why a deterministic string match is sufficient — description and oracle can't diverge.

The environment also gives agents more than CyberGym does: the full compiled fuzzing harness binary, not just source, because "providing it mirrors real vulnerability analysis, where a researcher runs the target under a debugger, inspects memory, and crafts mutations from runtime observations."

The ARO RLO config reflects the task's cost: `max_exec_budget: 1200`, `step_limit: 300`, `pod_timeout: 5h` — and notably `anti_hack_cleanup: false`, because vulnerability reproduction legitimately needs the runtime environment intact.

### General — knowledge work with a multi-container, MCP-driven environment

This is the most architecturally ambitious of the four, and reading one real task makes the design click. The sample I pulled (`general-sample/`) is a commercial-real-estate closing review, and its `manifest.json` specifies:

```
{"cwd": "/work/workspace",
 "uploads": [ ...6 entries, split across main and sidecar containers... ],
 "setup": {"command": "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach",
           "container": "sidecar", "timeout_sec": 300},
 "wait_ports": [39101, 39102, 39103, 39104],
 "mcp_servers": [
   {"name": "dealcloud_disposition_workspace",      "url": "http://127.0.0.1:39101/mcp"},
   {"name": "kpmg_tax_modeling_hub",                "url": "http://127.0.0.1:39102/mcp"},
   {"name": "sharepoint_tax_governance_library",    "url": "http://127.0.0.1:39103/mcp"},
   {"name": "yardi_property_asset_register",        "url": "http://127.0.0.1:39104/mcp"}],
 "verifier": {"command": "python3 /work/run_verify.py",
              "reward_file": "/logs/verifier/reward.json"}}
```

So a "general" task is: a **sidecar container** hosting four **MCP servers** on four ports, a shared workspace, and an agent that must query and reason across all four systems. The domain is simulated commercial real estate — DealCloud deal pipeline, KPMG tax modelling, SharePoint governance, Yardi asset register. That's not a toy; that's a miniature enterprise.

The verifier is the part that shows real craft. `verifier_meta.json` defines six rubric items, four of them `critical`:

| ID | Tier | Weight |
|---|---|---|
| `current_recommended_structure` | critical | 0.20 |
| `documentary_tax_components` | critical | 0.20 |
| `protection_statuses` | critical | 0.25 |
| `controlled_payment_split_and_fallback` | critical | 0.20 |
| `readiness_conclusion` | important | 0.15 |
| `source_data_present` | sanity | — |

Each LLM-judged item ships a `pass_anchor` containing the actual expected numbers, which is what makes it gradeable rather than vibes:

```
"Los Angeles County documentary-transfer-tax component = $9,141.00;
 City of Los Angeles component = $37,395.00;
 approved documentary-transfer-tax summary = $46,536.00."

"proceed with a negotiated installment asset sale using a
 $7,183,750.00 closing principal payment
 and a $21,551,250.00 secured seller note."
```

And here's the genuinely clever bit — the task is built around **designed distractors**. The anchor text explicitly warns which nearby numbers are traps:

```
"The answer must use these installment values, not the $13,200.00 county,
 $54,000.00 city, or $67,200.00 summary belonging to the active
 direct-sale comparison, and not the inactive legacy county amount."

"the partnership-interest transfer is not prioritized. The answer must
 tie the open protections to this controlled recommendation rather than
 treating the DealCloud comparison row's different $8,310,000.00 /
 $6,000,000.00 figures as the approved split."
```

The task contains a *superseded* transaction, an *active but only-a-fallback* comparison, a completed requirement row attached to the wrong (superseded) transaction, and a required diligence item with no active row at all. A model that pattern-matches "find the number" fails; a model that tracks transaction identity, status, and supersession across four systems succeeds. There's a `check_code` using `sqlite3` that asserts specific database states — e.g. that `DC-TXN-25Q3-024` is `installment_sale/recommended` and that `DEC-2025-017` is `approved/INSTALLMENT_ASSET_SALE`.

The reward pipeline is minimal and sane: `run_verify.py` extracts the *last* assistant message from the session jsonl, builds post-state by reading each MCP's `state.db`, calls `verify.py`, writes `reward.json`. `VERIFY_DETERMINISTIC=1` is the default, described as "Use verify()'s own weighted score (single source of truth)".

> **An oddity in the shipped data**
>
> Of the 989 general rows, 925 share a *single* image (`general-agent-env-0:oss`) while the remaining 64 (the `terminal_bench` subset) get 65 dedicated images (`general-agent-env-1` … `-65`).
>
> I inspected `general-agent-env-1` byte-for-byte (all 8 layers, sha256 verified, 292.5 MB). Its Dockerfile is `FROM python:3.12-slim-bookworm`, installs a shared `requirements-shared.txt`, then `COPY . /app`. The `requirements-shared.txt` carries a Chinese comment identifying it as "shared pip dependencies for the 65 terminal_bench candidate images", and the Dockerfile comment says "self-contained terminal_bench candidate image; the build context is this directory." The application layer is small: `workflow_probe.py` is a stdlib-only static fix detector that checks whether specific strings (e.g. `self.temp_output = set()`) are still present.
>
> So the 65 `terminal_bench` images bake the task state into the image, while the 925 `general_agent` tasks share one base image and receive their task via the sidecar/MCP manifest plus uploads. **That's an inference from the data, not something the code states outright** — the split makes sense (generated tasks need per-task state, benchmark tasks need per-task repos) but I could not find the line of code that draws the boundary. Flagging it as an open question rather than asserting it.

### Visual — 2,093 tasks, one 2.63 GB single-layer image

All 2,093 rows are `category=website` and share one image, `webdev-rl-opensource:v2` — whose manifest is a single layer of 2,629.7 MB. Every task is a website brief, and the deliverable location is pinned in the harness config's system prompt: `{{cwd}}/dist/index.html`. Verification is the groupwise visual grader from §6.3.

Compare the three image shapes side by side, because the layer counts tell a story about how each domain's dependency weight is distributed:

| Image | Size | Layers | Interpretation |
|---|---|---|---|
| `general-agent-env-1` | 292.5 MB | 8 | Thin base + shared pip layer + app layer; well-factored |
| `arvo-v1-42525170` | 603.1 MB | 39 | Incremental build layers from compiling the fuzzing harness |
| `webdev-rl-opensource:v2` | 2,629.7 MB | 1 | Flattened/squashed — a pre-baked toolchain snapshot |

The single-layer webdev image is the giveaway: someone took a heavy Node/browser toolchain image and squashed it, so task startup pulls one blob instead of crawling dozens of layers. For a domain where thousands of pods spin up per step, that's the right optimisation.

### Music — pure standard library

`recipes/design/music/scorer/core.py` does MIDI parsing and acoustic/tonal/rhythmic/structural metric computation with nothing but the standard library. No image per task, no external scorer. It's a nice contrast to the webdev path and suggests music was treated as the "cheap and deterministic" end of the environment spectrum. The report notes the internal music benchmark score going 45.7 (SFT) → 52.5 (RL).

## 11. The 3,764-image catalogue

Version tags on Docker Hub: **3,764**. Lines in `image-mapping.jsonl`: **3,764**. Not a coincidence — and the composition explains the dataset shape exactly:

| Image family | Domain | Count |
|---|---|---|
| `format-code-task-*` | Code (SWE) | 2,698 |
| `arvo-rl:v1-arvo-*` | Cyber | 1,000 |
| `general-agent-env-*` | General | 65 |
| `webdev-rl-opensource:v2` | Visual | 1 |
| **Total** | **3,764** |  |

Cross-check against the parquet row counts: code 2,698 ↔ 2,698 images, cyber 1,000 ↔ 1,000 images. Exact. Visual: 2,093 rows ↔ 1 image. General: 989 rows ↔ 65 images. Music contributes no image at all — it uses neither submodule, a point its own README makes explicit.

So the "3,764 images" headline is really "one image per Code task, one per Cyber task, one per terminal_bench task, plus one shared webdev image". Once you see that, the infrastructure cost is legible: a Code RL run needs to pull and run thousands of distinct images concurrently, which is exactly why the Harness Pool uses Kubernetes with per-pod limits (`cpu_limit: 4`, `memory_limit: 8Gi`) and why the env example notes that `MAX_CONCURRENT_SESSIONS` "is bounded by cluster quota rather than by GPU memory."

This mapping is also the practical key to using the release. To reproduce a Code task you need: the parquet row (for `problem_statement`, `test_patch`, `test_command`), the mapped image tag (for the environment), and the harness config (for the agent loop). Three pieces, all public, joined by `image-mapping.jsonl`.

## 12. What the open stack deliberately omits

A release of this kind is defined as much by what's held back. Being precise about the boundary is the difference between "reproducible" and "looks reproducible".

### Absent: the groupwise grading algorithms

```
$ grep -rn "GAR\|GRS" --include=*.py --include=*.md recipes/
(no output)
```

**Zero hits.** The report's §4.3 core contribution — Groupwise Reward Synthesis and Groupwise Advantage Redistribution — does not appear by name anywhere in the released recipe code. What *is* released:

- The **grader service** for one reward (`group_v1` for webdev work, in `recipes/design/grader_service/`), complete with the Latin-square ranking and runtime gate
- The **driver-side group reward rewrite** for webdev (`group_reward.py`, four rewrite rules)
- The **penalty module** implementing §4.3.3 (Eq. 4 and 5)

What is not released: the rubric-generation pipeline that produces GRS solution/behaviour rubrics from offline rollouts, and the online grader that produces GAR's quality factors f<sub>i</sub> over five ranking dimensions. The webdev grader is a domain-specific instance of the pattern, not the general machinery.

### Absent: the dynamic clip schedule

As established in §4: `clip_ratio_high` never exceeds 0.28 anywhere in the repo, and no entropy-driven scheduling code exists. The paper's four decoupled bounds with an upper limit at 5.0 are internal-stack-only.

### Absent: the production run's hyperparameters

The report gives Pro's settings — lr 3×10⁻⁶, staleness 4, Muown with 10 Newton–Schulz iterations. The OSS recipe gives lr 1×10⁻⁶ and `total_training_steps: 200` on a 9B distilled model. These are not the same run and the README doesn't pretend otherwise.

### Present but narrower than the paper suggests

| Report describes | OSS ships |
|---|---|
| Four training + three held-out harnesses | Four mini-harness configs (the held-out ones aren't needed for training) |
| Anti-hack: mid-training data + environment prep + adversarial screening + audit | `anti_hack_cleanup` flag + regex guard (default OFF), stage-1 only, stage-2 an LLM-judge stub |
| Penalty rules incl. `tool_name_invalid`, `toxic_reasoning`, `overlong`, `agent_context_filter` | Two rules shipped; four listed as `excluded_reference_rules` |
| GRS + GAR across all code agent tasks | Webdev groupwise grader + driver rewrite; code recipe uses plain test rewards |

> **What this boundary buys you**
>
> The five-environment GRPO recipe is genuinely end-to-end runnable: launch scripts, env examples, dataset loaders, harness configs, and Docker images all line up with matching row and tag counts. The groupwise grading — arguably the most novel part of the paper — is demonstrated rather than shipped in general form. Read the report for the algorithm; read the repo for the engineering.

## 13. Takeaways

Six things I'd carry forward from this codebase, whether or not you ever train an agentic RL model.

#### 1. Separate detection from effect.

The Rule/Strategy split — a rule judges a segment, a strategy binds `monitor`/`mask`/`adv_reduction`/`adv_set`/`adv_signed` to a hierarchy level — is a genuinely good pattern. It lets you deploy a new detector in `monitor` mode, watch its metrics for a few thousand steps, and only then give it teeth. Most training systems conflate "I noticed this" with "I penalised this", and can never turn a detector on safely.

#### 2. Mass conservation is a numerical requirement, not an aesthetic one.

The code's own account of why naive penalty-setting was abandoned is the most valuable sentence in the repo: setting a bad turn's advantage to −1 "injects net negative mass into the batch, which with no KL / entropy bonus flattens the policy and lengthens outputs." With `use_kl_loss: false` and `entropy_coeff: 0` — the released defaults — there is nothing to counteract injected negative mass. The conservation assertion at the end of `signed_rebalance` encodes this as a testable invariant rather than a comment.

#### 3. Make the invalid path loud.

Infrastructure failures get the sentinel `INVALID_REWARD_VALUE = -999.0`, not 0. The comment: "infra failures are visibly invalid, not silently a model failure." Compare the alternative — a dead pod scoring 0 — which teaches the model that some tasks are simply unlearnable. A -999 sentinel is detectable in metrics and excludable from advantage estimation. Most of the value is in the labelling, not the training.

#### 4. Group-boundary discipline protects the baseline.

One GRPO group stays on one harness. A group is accepted or rejected as a whole. The anchor for the length penalty is computed within the group. Penalties escalate along the hierarchy — context dropped, then sequence zeroed, then sample rejected. In group-relative methods, the *group is the unit of statistical validity*, and every one of these rules is protecting that. Break any of them and the baseline stops meaning anything — which is exactly why the A/B test in Figure 8 shows GAR sustaining pass-rate gains to step 52 while the no-GAR run's gains stall as turns and tokens balloon.

#### 5. Design the discriminator, not just the reward.

The general-domain tasks' `pass_anchor` fields are the clearest artefact of this discipline. They don't just state the right answer; they enumerate the adjacent wrong answers and explain why each is wrong — a superseded transaction, a fallback comparison row, a completion record attached to the wrong entity. That turns an LLM judge from a fuzzy similarity check into a discriminator with a known false-positive surface. Any eval or reward that relies on an LLM grader should carry one.

#### 6. Anonymisation in comments destroys verifiability.

The "reference RL framework" phrase appears eight times and is never defined. The rationale it carries is genuinely useful — but it describes a system that isn't public, so none of it can be checked. This is a small, fixable thing: a one-line note in `NOTICE` mapping the phrase to "MiMo's internal SGLang+Megatron RL stack (report §6.4)" would cost the authors nothing and make a full class of comments verifiable. Worth remembering next time you're tempted to genericise a proper noun in a commit message.

> **The through-line**
>
> Reading this stack end to end, the coherent thesis is that **agentic RL's bottleneck is the learning signal, not the optimizer**. Every notable mechanism in the release attacks signal quality rather than optimisation: GRS/GAR extract quality distinctions binary tests erase; the Penalty Module puts credit on the right tokens instead of spreading outcome reward evenly over thousands of them; the four mini-harnesses prevent strategies from overfitting to one agent loop; multi-harness mixing teaches transfer; the anti-hack layers stop the signal from being gamed. The optimizer — GRPO, prompt-mean, no std normalisation — is deliberately vanilla.
>
> And the one component they chose to *stop* learning — the MoE router — did so because its learning was damaging the model's capacity, not because the optimizer was struggling. Which is the same thesis from the other direction.

---

**Part 1 of 2 · background**

### [Your Agent Is Only as Smart as the World It Trains In](environment-engineering.html)

This instalment reads one release closely. The other one reads the field: what a training-grade environment must be, why difficulty had to stop being an adjective and become a measurement, why a judge can be the weakest link in your reward, and why a benchmark rots the moment you publish it. It closes on an open call — that contributing a world should be as easy as opening a pull request — which is precisely the question the release above is an answer to. Read it first if the MiMo mechanisms feel like solutions in search of a problem.

**Sources.** MiMo-V2.6 technical report (44 pp.). `github.com/XiaomiMiMo/verl` @ `a2ad9f6` (verl `0.9.0.dev` fork, submodules `mimoagent-osr` + `uni_agent` at `--depth 1`). Dataset `XiaomiMiMo/MiMo-V2.6-RL-oss` (5 parquets, 7,780 rows; `image-mapping.jsonl` 3,764 lines; one full `general_agent` environment sample). Docker Hub `xiaomimimo/mimo-v2.6-rl-oss` (3,764 tags; `general-agent-env-1` all 8 blobs sha256-verified; manifest/config for `arvo-v1-42525170` and `webdev-rl-opensource:v2`).

**Method.** Every claim above is traced to a file path, line range, config value, parquet field, image tag, or report section. Three findings contradict or bound the report and are marked as such: the clip-bound discrepancy (§4), the GAR/GRS zero-hit result (§12), and the general-domain image-injection split (§10), which is inferred from the data rather than confirmed in code. Upstream-file checks were performed against `verl-project/verl` at tag `v0.9.0` and on `main`.

**Correction policy.** If you spot an error in the mapping between an equation and its implementation, the code is the authority, and I'd want to know.
