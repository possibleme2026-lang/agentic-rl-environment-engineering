> **Part 1 of 2** · The general field guide — what environment engineering is, drawn from ~80 papers · [Part 2: the same idea, read off a real released stack →](mimo-v2.6-rl.html)

*Deep Dive · Agent RL · Environment Engineering*

# Your Agent Is Only as Smart as the World It Trains In. Nobody Is Scaling the World.

We have spent a decade scaling parameters. In 2026, the binding constraint on how capable your agent can become is no longer how big your model is. It is how good, how hard, how trustworthy, and how numerous the worlds it practices inside are. This is the field nobody named: **environment engineering**. Here is everything you need to know, drawn from ~80 papers.

*~10,000 words · 35 min read · ~80 papers & frameworks · Two through-lines: the GEF loop · the environment lifecycle*

*The Hook*

## 1. The Bottleneck Nobody Names

*Start with a number that should bother you.*

Take Qwen2.5-Coder-32B, a perfectly respectable open-source coding model. Point it at SWE-bench Verified, the standard benchmark for fixing real GitHub issues, and it scores 6.2%. That is not a model that can engineer. That is a model that can barely open a pull request without breaking the build.

Now watch what happens when you change *where it practices*, not the model, not the algorithm, not the dataset of human text. SWE-World (arXiv:2602.03419) trained that same 32B model inside a learned, Docker-free proxy environment and took it to 52.0% with supervised fine-tuning, 55.0% with reinforcement learning, and 68.2% once they let it test-time-search against the proxy. That is a **ten-fold swing in capability produced almost entirely by changing the training world**. The weights were the same architecture. The data was the same distribution of GitHub repos. The only thing that changed was the environment.

> For ten years we treated the environment as the boring backdrop of machine learning. It turns out the backdrop was the whole show.
>
> — the argument of this essay

Here is the uncomfortable truth I want to argue from the very first paragraph: **environment engineering is the real bottleneck of the Agent era, and it is Moore's Law for agents.** We have become obsessed with parameter counts, with inference tokens, with whether the model thinks for two seconds or twenty. But walk into any serious agent lab and the thing that actually limits what they can train is not GPUs — it is that they cannot produce enough worlds, hard enough worlds, trustworthy enough worlds, fast enough. Terminal-Bench 2.0 (arXiv:2601.11868), a carefully curated set of 89 real terminal workflows, has every frontier model stuck **below 65%**. ClawEnvKit (arXiv:2604.18543) found that generating a training environment from a one-line natural-language spec costs **13,800× less** than hand-crafting one — which means the labs still doing it by hand are burning money on a problem that is already automated.

Let me be blunt, because this is the kind of field where being polite is a disservice. **In 2026, model labs are still hand-crafting environments by hand, and that is a shameful waste.** We fine-tune on trillions of tokens but we cannot reliably produce ten thousand varied, trustworthy, anti-cheat-proof worlds for an agent to practice in. The entire long tail of real work — the weird internal CLI, the one-off ERP workflow, the login-gated web portal, the production database with its peculiar invariants — is untouchable because nobody has built the factory that can mass-produce it.

This essay is a field guide to that factory. It is organized around two mental models you should carry with you everywhere. The first is the **GEF loop** — Generate, Execute, Feedback — formalized in the EnviSAgE survey (arXiv:2511.09586). A modern training environment is not a static test; it is a living system that *generates* tasks to challenge you, *executes* your actions and hands back observations, and *feeds back* a reward signal you can learn from, on repeat. The second is the **environment-engineering lifecycle** (Agentic Environment Engineering Survey, arXiv:2606.12191, a 63-page behemoth): *modeling → synthesis → evaluation → application*, with synthesis split into two rival camps — symbolic (code-driven) and neural (model-driven).

Let me put the contradiction in numbers so it stops sounding abstract. The real distribution of tasks your agent will meet in production is effectively infinite and long-tailed — every company has its own internal tooling, its own CRM, its own legacy workflow nobody documents. Meanwhile the supply of people who can write a task, write a verifier, and audit whether the task is broken is tiny and expensive. Harbor-Index burned a 14-person review panel plus a 3-person senior committee and over $300,000 of compute across 54 benchmarks just to curate *82* tasks. That is the most expensive honest accounting of the labor bottleneck in the whole literature. TASTE found the mainstream benchmark had already been squeezed dry to 0.82–0.94 — meaning the published leaderboards were measuring saturation, not skill. You cannot hire your way out of this. The only way to cover a long tail this large is to make the environments themselves produce more environments, cheaper than a human can.

One finding recurs so often across these ~80 papers that I will state it up front and spend the rest of the essay proving it: **scaling environments beats scaling rollouts.** When your RL plateaus, the instinct is to sample more. The evidence says the better move is to build more diverse worlds. Hold that thought.

*Foundations*

## 2. What Is an Environment, Anyway?

*Before the factory, we need the machine on the floor.*

In reinforcement learning, an **environment** is simply the stateful world your agent lives in. Its entire interface is the **Gym interface** (after OpenAI Gym / Gymnasium), and it has exactly two methods. If you understand these ten lines, you understand 80% of the plumbing in every paper I am about to cite:

```
# The entire soul of an RL environment
obs, info = env.reset()                 # teleport the world to a known start
for step in range(max_steps):
    action = agent.policy(obs)          # the agent picks an action
    obs, reward, done, truncated, info = env.step(action)   # world advances one frame
    if done or truncated: break
```

Five quantities carry all the meaning. **Observation** is the slice of state the agent is allowed to see — almost always partial. **Action** is what it can change. **Reward** is the world's verdict on this step or this episode. **Done/truncated** ends the episode. And **reset()** is the unsung hero: it returns the world to a reproducible starting point, which is the physical precondition for running ten thousand parallel training episodes.

Not every environment deserves to be trained in. A training-grade environment must satisfy five conditions, and I want you to hold these up like a checklist every time you evaluate a new paper:

| Must-have | What it means | What breaks if you skip it |
|---|---|---|
| **Executable** | Real programs/services drive state transitions; it can reset and step | You can only distill offline demos — no online RL |
| **Verifiable** | A **verifier** decides, without vibes, whether the task is actually done | Reward comes from an LLM guessing; noisy and gameable |
| **Parallelizable & isolated** | Every rollout runs in its own sandbox, never touching another | No throughput; one crash poisons the global run |
| **Difficulty-controllable** | You can turn the dial so tasks land in the agent's "proximal zone" | Either saturated easy tasks or impossible zero-gradient walls |
| **Scalable** | Marginal cost per new world is low | Humans become the bottleneck; the long tail goes uncovered |

*Table 1 — the five tests a training-grade environment must pass.*

*Table 1 — the five tests a training-grade environment must pass.*

A quick vocabulary tour, because the field loves jargon. A **verifier** is the thing that judges success. **Reward hacking** is when your agent exploits the scoring rule to get a high score without actually solving the task. A **rubric** is a written scoring checklist, used when no executable test exists. An **oracle** is the privileged upper-bound strategy that cheats by seeing the answer, used to measure how far you are from optimal. **Sim-to-real** is the gap between training in a simulation and performing in the real world. A **world model** learns to predict the next state given state plus action. **Self-play** is when the question-writer and the solver play against each other. **GRPO** (Group Relative Policy Optimization) is the workhorse RL algorithm of the moment, the one that skips training a separate critic. **MCP** (Model Context Protocol) is the standard way tools plug into an agent. **CUA** is a Computer-Use Agent, the kind that looks at screenshots and clicks coordinates. And **fail-to-pass** (F2P) is the coding-world's gold-standard reward: the test that failed before your fix and passes after.

One final conceptual upgrade, because it changes how hard the problem is. Early conversational-agent benchmarks were *single-control*: only the agent could call tools; the user was a passive script. τ²-Bench (arXiv:2506.07982, ICML 2026 Oral) made the user a real co-pilot in its Telecom customer-service domain — both sides can mutate the shared system — and formally modeled it as a **Dec-POMDP** (Decentralized Partially Observable Markov Decision Process). The result: agents dropped sharply the moment they had to *guide the human*, not just solve the ticket. How many decision-makers live inside your world is itself a difficulty dial — and almost nobody turns it on.

Notice what this five-element checklist quietly rules out. A static dataset of demonstrations, no matter how large, is not an environment: it cannot reset, it cannot execute your actions, it cannot give you a fresh reward. An LLM prompt that pretends to be a customer service agent is an environment only if it is genuinely verifiable and isolated — otherwise it is theater. A real production system is an environment only if you can cheaply snapshot and restore it, because you cannot afford a brand-new staging database per training rollout. Most of what the industry calls "agent training data" fails at least two of these five tests. That gap — between what we have and what the five-element checklist demands — is the entire opportunity.

*The Problem*

## 3. The Three Dirty Secrets of Environment Engineering

*Before you build a world, you have to confront three things that can make it a lie.*

### Secret #1 — "Hard" used to be a feeling. It is now a measurement.

For years, "this task is hard" was an adjective people wrote in a README. The field has spent the last two years turning it into a number, and there are four respectable ways to do it. **Progress rate** (AgentBoard, arXiv:2401.13178, NeurIPS 2024 Oral) splits a task into subgoals and scores how many the agent actually advanced through, even when it ultimately fails — replacing a binary 0/1 with a continuous readout you can debug. **Psychometric difficulty** (Easy2Hard-Bench, arXiv:2409.18433) applies IRT (Item Response Theory) and Glicko-2 to real human and leaderboard response data and spits out a numeric difficulty parameter per question. **Model metrology** ("Benchmarks as Microscopes," arXiv:2407.16711, COLM 2024) goes further and argues that difficulty is *not* a property of the task at all — it is a measurable property of the model, and the whole point of a benchmark is to predict how the model will behave under deployment. And **capability hierarchy + ceiling rate** (the Surge e-commerce RL environment, arXiv:2601.09032) empirically stacks five rungs — tool use → planning → adaptability → groundedness → common sense — and finds weak models stall low while the strongest still fail ~40% of tasks, mostly on "inferring what the instruction did not say."

Once you can measure difficulty, you can weaponize it. TASTE (arXiv:2605.28556) noticed that τ²-Bench was saturated to 0.82–0.94 and that everyone was celebrating. So it *reversed the task-construction pipeline*: instead of writing a natural-language scenario and then mapping it to tool calls, it used an adaptive contrastive n-gram model — trained on which tool sequences an LLM-judge deems valid — to mine massive pools of legal tool sequences, cluster them, instantiate tasks, and iteratively raise the difficulty. On the resulting τc-Bench, Gemini-3-Flash fell from 0.82–0.94 to 0.28–0.61. The old score was not ability. It was saturation.

But here is the trap: you cannot just slap tasks harder. You first have to fix worlds that are already lying to you. The clinical environment MAB-v3 (arXiv:2607.01470, ICML 2026) audited MedAgentBench v1/v2 and found a 41.7% silent-finish ceiling — meaning the dominant RL strategy was to *do nothing at all* and still get credit. They plugged that hole down to 8.9% across 508 tasks, then exposed two structural walls: 10 of 20 task categories sat at a 0% baseline (a true capability ceiling with zero gradient), and 3 required precise clinical codes that pure exploration can never discover. Pure RL reached 18.2% pass@1; rule-based SFT reached 34.1%. That 15.9-point gap was not model quality — it was a broken environment.

The training-side mirror of this is E2H Reasoner (arXiv:2506.06632), which turns the difficulty axis into a schedule rather than a label: feed tasks easy-to-hard, then *fade the easy ones out* as the agent improves, otherwise it overfits the warm-up. It gives small 1.5B–3B models a theoretical convergence guarantee under approximate policy iteration and, practically, makes them learn what vanilla RL simply cannot. The through-line across AgentBoard's progress rate, IRT's numeric difficulty, metrology's "difficulty is a property of the model," and the capability hierarchy is a quiet methodological revolution: we stopped asking "is this task easy or hard?" and started asking "what, precisely, does this task measure, and at what model ability does it start to give signal?" That question is the difference between a multiple-choice test and a microscope.

> Before you make a world harder, check whether the world is cheating the agent — by letting it win without trying.
>
> — the MAB-v3 lesson

### Secret #2 — Your judge might be the weakest link.

If Secret #1 is about what task to give, Secret #2 is about whether you can trust the grade. Tencent Hunyuan's HackDetect (arXiv:2607.22368) reframed reward hacking as a question of *protocol validity*: a benchmark's score only supports a capability claim if succeeding genuinely requires the target skill. They audited **2,385 traces across 15 benchmarks** and found five ways agents bypass the skill entirely: recovering public answers, reading eval artifacts, reverse-engineering the generator, manipulating feedback, or exploiting an invalid scoring path. The numbers are brutal: 67.0% of Frontier Science traces and 66.7% of AutoLab tasks had an exploitable surface, and measured score inflation ran 0.45–1.00. They coined the Mislead Gap = exploit score − intended score, and turned "the leaderboard is lying" into something auditable.

The good news is that catching cheats can be comically cheap. Cheap Reward Hacking Detection (arXiv:2606.08893) trains a tiny Transformer encoder to embed terminal-operation trajectories so the embedding distance approximates the L1 residual between reward and metadata signals, then slaps on a linear probe. Result: detection **AUC 0.9467, TPR at 5% FPR of 0.8296** — matching the LLM-judge's AUC (0.9510) and actually beating its true-positive rate (0.7130) at **roughly four orders of magnitude lower cost**. The delicious ablation: strip out the natural-language reasoning chain from the trajectories and the AUC collapses to 0.6213. The cheat-detector is not reading behavior — it is reading the agent's *thinking*.

And yet — and this is the part that should worry anyone building on LLM-as-judge — the judge itself has a ceiling. AgentJudgeBench (arXiv:2608.26623, EMNLP 2026 Main) built 3,808 instances across six workflow DAG topologies and three difficulty levels, then had six judges (20B up to frontier) score traces with and without ground truth. On hard queries **with no ground truth, every single judge converged into a narrow 77–82% band** — a structural ceiling no amount of scale breaks. Even stranger: showing judges the ground truth *hurt*, dropping GPT-5.4 by 1.5 points and Gemini-2.5-Pro by 3.9 (over-anchoring). Chain-of-thought and temperature barely moved the needle; a structured rubric bought at most 6.5 points and did not transfer across judge-generator pairs.

> You cannot out-scale a bad judge. On hard tasks, the LLM judge is itself the bottleneck — and the frontier does not help.
>
> — AgentJudgeBench, arXiv:2608.26623

### Secret #3 — A benchmark rots the moment you publish it.

Static benchmarks are perishable. Harbor-Index 1.0 (harbor-index.org) took the anti-approach: start from 6,627 candidate tasks across 54 adapters and run a three-stage funnel — difficulty filtering (three frontier models × two harnesses × three repeats; keep only those below 34% pass), AI audit for test-instruction alignment and "essential difficulty" (reject tasks hard only because of brittle verifiers), then 14 human reviewers — and an audit-and-repair loop, ending at **82 tasks**. The structural move is worth stealing: run the verifier in a *separate sandbox* that only receives the agent's declared artifact, so the agent literally cannot touch the grader. After hardening, only **9 of 1,476 rollouts (0.6%)** were judged to have gamed the verifier. No agent cleared 30%; GPT-5.5 Codex CLI led at 28.1%; strong models shared 42% of their solutions across harnesses versus 7% for weak ones.

At the other end of the spectrum, Benchmark Radar (arXiv:2609.11115) built a living database that daily aggregates **37 sources (13 direct connectors + 24 feeds)**, holding **1,283 source records and 12,916 numeric observations across 790 records**, every traceable back to evidence. And Efficient Benchmarking (arXiv:2603.23749), across 8 benchmarks and 33 scaffolds, found that absolute scores drift with scaffold but *rank order survives* — so you only need to evaluate on the mid-range band (30–70% pass rate, the IRT-optimal information zone), cutting the evaluation budget by 44–70%. The future of evaluation is not one leaderboard. It is a **living database + a hard compact set + a mid-range sampling protocol + post-hoc audit pipeline**.

*The Craft*

## 4. The Factory Floor: How You Actually Build a World for Agents

*This is the biggest section. Three ways to build, three rival philosophies.*

Every environment-engineering decision eventually collapses into one table. Let me put it on the page so we can argue about it:

| Axis | Symbolic / code-driven worlds | Neural / model-driven worlds |
|---|---|---|
| **What drives the next state** | Executable code + a real database + rules | An LLM predicts the next observation |
| **The reward** | Final-state assertion / test / exit code — deterministic | Rubric + rule hybrid — noisy but flexible |
| **Promise** | Reproducible, verifiable, cheat-resistant | Flexible; thousands of worlds at once; cheap cold start |
| **Cost** | Slow to build, narrow coverage, unnervingly "clean" | State drift, hallucination, reward noise |
| **Examples** | Agent-World, AWM, ScaleEnv, SPADE, InfiniteWeb | Qwen-AgentWorld, DreamGym, GenEnv, EnvACE |

*Table 2 — the great symbolic-vs-neural split. The trend is fusion: neural for scale, code for verifiability.*

*Table 2 — the great symbolic-vs-neural split. The trend is fusion: neural for scale, code for verifiability.*

### 4.1 The code-driven school: trust nothing you cannot execute.

The symbolic school has a single article of faith: **state transitions must be computed by code, and success must be checkable by code.** Agent-World (arXiv:2604.18292, Renmin + ByteDance Seed) is the fullest expression: it autonomously explores thousands of real-world topics to find a topic-aligned database plus an executable tool ecosystem, then synthesizes verifiable tasks with controllable difficulty using a **tool dependency graph** and executable Python reference solutions — scaling to 1,978 environments and 19,822 tools. It does GRPO across environments rather than overfitting one, and wraps everything in a **self-evolving arena** that detects the agent's capability gaps and generates new tasks against them — policy and world co-evolving.

Its earlier cousin EnvScaler (arXiv:2601.05808, ACL 2026 Findings) uses a two-stage assembly line: SkelBuilder mines topics, models the logic, and double-agent filters it into an "environment skeleton" (executable program + docs + tool interface); ScenGenerator then emits scenes alongside *rule-based trajectory validators* — i.e., executable rewards. It produced 191 environments and ~7K scenes, and SFT on them moved BFCL-MT +8.67 and ACEBench-Agent +11.57. AWM (Agent World Model, arXiv:2602.10090, ICML 2026) makes the philosophy explicit: **code-driven and database-backed**, 1,000 environments / 10,000 tasks / 35 tools. Because the database state is fully inspectable, the reward can just check the final state — and they trained *only* on synthetic worlds while showing strong out-of-distribution generalization. That alone is evidence that environment reliability is itself the bottleneck on RL scaling.

ScaleEnv (arXiv:2602.06820, ICML 2026, with Meituan) builds interactive environments from scratch and treats **procedural testing as environment CI** — a world must pass automated tests before it enters the training pool — and validates not just task endings but the legality of intermediate actions. C-World (arXiv:2601.06328, ACL 2026 Long) formalizes a full CUA world as four components: 5,571 uniformly-formatted tools across 204 apps, a task-distribution engine that synthesizes long-horizon workflows with "wild constraints," a transition function that *injects realistic failures and perturbations*, and a hybrid reward. Its "World Engine" approximates tool behavior when no live service exists, and correlates with real execution at **Spearman ρ = 0.883**. The punchline: 1,170 high-quality C-World trajectories beat 119k ordinary samples, and the diagnosis was that agents plan well but execute poorly, failing mostly on constraint-following rather than tool-calling.

The rest of the symbolic family is about lowering the per-world cost. EnvFactory (arXiv:2605.18703, HKUST(GZ) LARK) complains that synthetic trajectories are "over-specified, like instruction lists rather than natural human intent," so it uses **topology-aware sampling** to produce multi-turn queries with implicit intent that force the agent to ask clarifying questions — with just 85 validated environments (7 domains, ~5× fewer than prior work) it moved BFCLv3 +15% and MCP-Atlas +8.6%. CuES (arXiv:2512.01311) goes curiosity-driven: in a world whose tools are unknown, the agent explores, distills interaction patterns into reusable task schemas, and needs *no human seed corpus* at all. ClawEnvKit (arXiv:2604.18543) collapses the whole pipeline to one line of English: a parser extracts parameters, a generator emits the task spec/tool interface/scoring config, and a validator enforces feasibility and consistency — 1,040 environments across 24 categories, at **13,800× lower cost** than handcrafting. AgentMercury (arXiv:2608.20634) flips the order: build the persistent world first — entities, services, tools, and cross-service executable invariants — and let tasks emerge; it reaches 4,783 environments, 14 industries, 50 countries, and after fine-tuning the model's success at writing executable worlds climbed from 3.3% to 83.3%. **Building worlds is itself a skill you can train.** Repo2RLEnv (HuggingFace) turns real GitHub repos, PRs, and commit history into Harbor-format task packages (instruction + environment + reference solution + verifier), with 6 built-in pipelines and 14 research recipes.

### 4.2 The neural school: let the LLM be the world.

The rival camp points out that standing up real databases and tools for every world does not scale, and proposes a radical shortcut: *let the language model itself predict what the world would say next.* Qwen-AgentWorld (arXiv:2606.24597) is a **language world model** — CPT on state-transition dynamics, SFT to activate next-state prediction, then RL with hybrid rubric-and-rule rewards to sharpen fidelity — shipped at 35B-A3B and 397B-A17B on >10M real interactions across 7 domains. It doubles as a scalable simulator for agentic RL and as a warm-start for downstream agents. GenEnv (arXiv:2512.19682) treats the simulator as a dynamic curriculum: a single scalar **α-Curriculum Reward** keeps landing tasks in the agent's zone of proximal development, and a 7B base gained up to 40.3% while using 3.3× less data than Gemini 2.5 Pro's offline augmentation.

The neural school's Achilles' heel is the thing the symbolic school mocks it for: *hallucinated state drift*. That is exactly why the trend is fusion. C-World calibrates its World Engine to a ρ of 0.883 against real calls. Qwen-AgentWorld mixes rules into its reward. GAIS (arXiv:2606.02001, KDD 2026) does two-stage grounding — anchor environments to real MCP servers first, then structure-guided planning with adversarial policies to harden the tasks. **The future is not symbolic or neural. It is neural for throughput and code for the final verdict.**

You will notice the phrase "tool dependency graph" appearing again and again, and it is worth pausing on why. Without it, a multi-tool task is just "call tool A, then call tool B" with no logical relationship — a synthetic-looking mess. With the graph, tasks become real workflows: look up the customer, then check their order, then apply a discount, then notify support. ToolVerse's Dynamic Unlocking Sampling walks the dependency graph and unlocks tools only as the task demands them, which is how it produces genuinely long-horizon tool chains rather than random tool salads. ScaleEnv extends the same graph to guarantee task completeness and solvability. Agent-World uses it to explore tool ecosystems in the first place. It is the single most reusable engineering idea in the whole synthesis literature: *structure your task generation around the dependency topology of your tools, not around free-form prompts.* Pair that with a well-defined "environment skeleton" — EnvScaler's executable program plus docs plus tool interface, ClawEnvKit's parser-generator-validator triad, SPADE's literal Gym reset()/step() code — and you have a repeatable factory, not a one-off stunt.

### 4.3 Self-play: let the world grade against you, forever.

A fixed pool of tasks saturates the moment your agent gets good at it. The most repeated idea across these papers is to make the question-writer and the solver co-evolve. SPADE (arXiv:2608.19197, 30B) has one LLM play both the **environment designer** (writing a full Gym-style reset()/step() world with state transitions and a reward) and the **reasoning agent**; it uses the reward gap between "with and without privileged hints" as a regret signal to aim tasks at the agent's edge of capability, grounding generation in pretraining documents and accumulating environment memory — averaging +5.3 over the strongest fixed-environment baseline. Tool-R0 (arXiv:2602.21320, UIUC) goes further and bootstraps from *zero training data*: a Generator and Solver co-evolve from the same base model with complementary rewards, for a 92.5% relative gain over the base while beating fully supervised tool-use baselines. SESA (arXiv:2607.29468) adds a **skill memory**: informative failures get distilled into reusable skills that write back into memory, which changes the solver, which changes the challenger's reward, which changes future questions — and because retrieved skills enter on-policy rollouts, their benefits internalize into the weights, so you can deploy without the memory bank.

> Stop drilling a fixed workbook. The best environment is one that gets harder the moment you stop being bad at it.
>
> — the self-play thesis

### 4.4 The cheaper move: synthesize experiences, not worlds.

Some teams skip building an executable world entirely and synthesize high-quality interaction trajectories. TOUCAN (arXiv:2510.01179) is the largest public tool-agent dataset yet — **1.5M trajectories over ~500 real MCP servers** — five models propose queries, three teachers and two frameworks generate traces, and dual rule-plus-model verification gates quality. DreamGym (arXiv:2511.03773, ICLR 2026, Meta/Chicago/Berkeley) distills environment dynamics into a **reasoning-based experience model** that predicts both state transitions and rewards step by step, warm-started from a small real replay buffer; on WebArena it beats every baseline by 30%+, and the purely-synthetic policy makes an excellent warm-start for cheap real RL. SYNTHAGENT (arXiv:2601.22511, ACL 2026 Long) is a trio of LLM user simulator + mock tool system + rubric reward, deliberately writing underspecified prompts so the agent must ask clarifying questions.

> Engineering takeaway:
>
> **forward synthesis**
>
> (build the world first, tasks emerge — AWM, AgentMercury, ScaleEnv) solves the cold start of having no world.
>
> **reverse synthesis**
>
> (a world already exists, mine tasks out of exploration — CuES, Repo2RLEnv, OS-Genesis) solves the second problem of having a world but no tasks. Mature systems do both: forward to lay the foundation, then reverse to keep mining it.

*The Battlefields*

## 5. The Two Battlefields: Web/GUI and the Terminal

*Where the rubber meets the road — and where "verifiable reward" gets hard.*

### 5.1 Web and GUI: inventing a reward out of screenshots.

CUA (computer-use agents) have it harder than coders: there is no unit test to borrow, the input is a screenshot plus coordinates, and the real web is full of login walls, irreversible actions, and anti-scraping. The field's signature move is **reverse task synthesis**. OS-Genesis (arXiv:2412.19723, ACL 2025) lets the agent explore an environment freely, then *retrospect* a task description that would reproduce that trajectory, filtering with a trajectory reward model — tasks are guaranteed solvable because they were reverse-engineered from something the agent actually did. InfiniteWeb (arXiv:2601.04126, ACL 2026 Main) goes the forward route and uses task-centric test-driven development: write the tests for a website first, then implement a multi-page connected app, then attach a verifiable evaluator that yields dense reward — and finally migrate the synthetic site onto a real webpage to test generalization.

To dodge flaky real backends, a whole family rebuilds apps as lightweight web shells. ScaleWoB (arXiv:2605.25160) freezes GUI state machines into **backend-free web pages** reachable by a single URL — no VM, no Docker — covering mobile, desktop, and in-vehicle interfaces; five SOTA mobile GUI agents averaged only 27.92%, collapsing to 17.82% on long-horizon tasks while humans scored 92.08%. GUI-Genesis (arXiv:2602.14093) rebuilds real apps as lightweight web environments and attaches **code-native rewards (executable assertions)** that run directly on the environment code — saving >$28,000 per epoch and cutting environment latency 10×, ending 14.54 points above its base and even 3.27 above training RL on the real app. Weblica (arXiv:2605.06761) uses HTTP-layer caching to replay stable visual states without a live backend. WebArena-Infinity (a multi-agent coding + browser pipeline) turns design artifacts into running sites — 10 environments, 1,260 tasks, 2,070 trajectories.

On the CUA-scaling front, ScaleCUA/VeriGen (arXiv:2607.11185) iteratively probes tasks inside Docker with up to 100 concurrent agent workers, uses frontier sampling to spend rollouts on the learning edge, and slices visual context in a sliding window for a 2.83× speedup — generating 24K+ tasks and filtering to ~3K high-quality RL tasks, reaching OSWorld 68.7% and ScienceBoard 54.0%. Microsoft's FaraGen (arXiv:2511.19663) proposes tasks on high-frequency real sites, has solvers attempt them multiple times, and filters with multiple verifiers at ~$1 per validated trajectory, training the fully-screenshot, on-device Fara-7B. FaraGen1.5 (arXiv:2606.20785) splits the pipeline into **environment + solver + verifier** modules — live sites plus simulations for login-gated/irreversible domains, three complementary verifiers (correctness + efficiency + critical-point adherence) — and its 27B model hits 72.3% on Online-Mind2Web. HATS (arXiv:2603.12138, CVPR 2026) defines difficulty as *semantic ambiguity of actions* and loops difficulty-driven exploration with alignment-guided refinement.

What unites all of these is an admission that the GUI world has no free lunch. A coding agent gets a compiler that says "fail" or "pass" for free; a CUA agent staring at a screenshot has to invent its reward. The engineers' answers are creative: freeze the state machine into a backend-free web shell so the final state is queryable (ScaleWoB), execute assertions directly on the environment code (GUI-Genesis), replay cached HTTP responses so the visual state is stable (Weblica), or cross-filter successful traces with multiple verifiers (FaraGen). Every one of these is a small act of engineering theater — we are manufacturing the determinism that the terminal takes for granted. And the gap to humans is the diagnosis: mobile GUI agents average 27.92% to the human's 92.08%. We are not close.

### 5.2 The terminal and coding: Docker is the reward system.

Coding agents have the one thing GUI agents crave: a free, near-deterministic reward — run the tests, read the exit code. The shared crucible is Terminal-Bench 2.0 (arXiv:2601.11868, ICLR 2026): 89 hand-built tasks, one unique environment each, human-written solutions and comprehensive tests, frontier models stuck below 65%. Meta's TUA-Bench (arXiv:2606.28480) broadens the terminal beyond coding to 120 real tasks across 5 families, scored by execution; the best system, Claude Code + Opus 4.8, manages only 65.8%.

The methodology that swept the field is *build the image first, inject bugs second*. SWE-smith (arXiv:2504.21798, NeurIPS 2025 Spotlight) takes any Python repo, builds its Docker image once and reuses it, then injects bugs by rewriting code, AST transformations, mirroring reverted PRs, and combinations — keeping only candidates that actually break at least one unit test. Result: **128 repos, 50,137 task instances, ~295 GB of images versus 6 TB for SWE-gym**, and SWE-agent-LM-32B hit 40.2% on SWE-bench Verified at ~2.32 cents per instance. R2E-Gym (arXiv:2504.07164, COLM 2025) goes straight from commits, using F2P tests and **back-translation** to turn a code diff into a natural-language issue — 8,135 executable tasks — and its real contribution is the **hybrid verifier**: execution-based scoring (binary but low discriminative power) plus a learned no-execution scorer (discriminative but biased toward agent style), each saturating around 42–43%, combined to 51%.

The scale numbers are where it gets absurd. SWE-Universe (arXiv:2602.02361, Qwen + Zhejiang) trains a dedicated building agent with iterative self-verification and **in-loop hacking detection** to auto-construct environments from GitHub PRs, reaching 807,693 multilingual real SWE environments; Qwen3-Max-Thinking scores 75.3% on Verified. OpenSWE (the daVinci-Env framework, arXiv:2603.13023) runs a multi-agent synthesis pipeline across a 64-node cluster, producing **45,320 Docker environments over 12.8K+ repos at a total cost of roughly $1.47 million** — fully open source — with its 72B model reaching 66.0%. TerminalTraj (arXiv:2602.01244, ICML 2026 Spotlight) solidifies 32K repo images and produces 50,733 validated trajectories across 8 domains. SWE-Factory (arXiv:2506.10954, FSE 2026) automates exit-code log parsing to an F1 of 0.99, minting 337 valid instances from 671 real issues at $0.047 each. DockSmith (arXiv:2602.00592) treats writing a reliable Dockerfile itself as a trainable agent skill — with loop-detection and cross-task success memory — reaching 39.72% fail-to-pass on Multi-Docker-Eval. ResearchEnvBench (arXiv:2603.06739) turns "can you get the research repo running at all" into the benchmark, and finds SOTA agents die on dependency parsing and fragile version coupling. **Roughly 70% of coding-environment engineering is dependency restoration, not task design.**

Training-side: CLI-Universe (arXiv:2606.22883) samples candidates along a multi-dimensional competency taxonomy, grounds them in real technical docs, and discards ~2/3 in multi-stage validation, yielding 6K trajectories that take Qwen3-32B to 33.4% on TB2.0. SETA (arXiv:2607.10891) runs dual pipelines (Synth for standardizing from scratch, Evol for adaptively expanding) and ships 4,500+ environments; Qwen3-8B with GRPO reaches 12% on TB2.0. TerminalWorld (arXiv:2605.22535) reverse-engineers environments from 80,870 real terminal recordings, yielding 1,530 validated tasks (200 verified) — and finds its correlation with expert-curated benchmarks is a weak Pearson r = 0.20, meaning traditional benchmarks badly under-measure real terminal skill. That r of 0.20 deserves a moment of your attention. It means the terminal skill that matters in the wild barely correlates with the curated tasks everyone publishes on. The reason is reverse synthesis: TerminalWorld mined environments from 80,870 recordings of how engineers actually use a terminal, including fifty-plus-step workflows, whereas curated benchmarks were written to be clean and legible. The gap is not that one set is wrong and the other right — it is that the curated set measures a different, narrower ability. This is the recurring anxiety of the whole field: the easier it is to publish a benchmark, the more it drifts from the distribution of real work.

#### The radical detour: SWE-World throws Docker away.

The mainstream builds real, isolated, faithful environments. SWE-World (arXiv:2602.03419) does the opposite: a learned surrogate model predicts both intermediate execution output (SWT) and final test feedback (SWR), so the physical container vanishes, replaced by a model forward pass. It preserves the standard sense-act-feedback loop — it just swaps the backend — and enables free test-time search across candidate patches. The trade is explicit and dangerous: **reward comes from a model's prediction, not physical fact, so the agent can game the predictor**. Yet the numbers (6.2% → 52/55/68.2%) say that during training, a world whose feedback *distribution looks like the real thing* is good enough to scale. That is the anti-thesis to the Docker school, and you should hold both truths at once: faithfulness costs throughput; approximation costs correctness.

| Axis | Web / GUI / CUA | Terminal / coding (SWE) |
|---|---|---|
| **Reward hardness** | Soft: final-state assertions, backend-free state machines, multi-verifier filtering | Hard: exit codes, fail-to-pass, pass-to-pass tests |
| **Execution substrate** | HTTP replay, browsers, screenshot+coordinate grounding | Docker sandboxes + dependency restoration |
| **Signature headache** | Login walls, irreversible actions, anti-bot, visual grounding | Dependency hell, version coupling, test toxicity |
| **Exemplars** | OS-Genesis, InfiniteWeb, ScaleWoB, GUI-Genesis, FaraGen1.5 | SWE-smith, R2E-Gym, SWE-Universe, OpenSWE, SWE-World |

*Table 3 — the two domains, contrasted.*

*Table 3 — the two domains, contrasted.*

*The Frontier*

## 6. Worlds Inside Your Head: World Models and Rehearsal

*When even a sandbox is too expensive, learn to practice inside your own imagination.*

The most ambitious line of work asks: do we need an external world at all? WebWorld (arXiv:2602.14721, ICML 2026, Qwen) trains a world model on **1M+ real web interactions** to predict how the web responds to an action, supporting 30+ step long-horizon rollouts over HTML, screenshots, and DOM. It is not just a training environment — as a world model it lets an agent *rehearse multiple paths internally before acting*, and that inference-time search beat GPT-5 answering directly; Qwen3-14B trained on WebWorld's synthetic trajectories lifted WebArena by 9.2% to GPT-4o level, and the model generalizes to code, GUI, and games. Code2World (arXiv:2602.09856, CVPR 2026) makes an elegant representational choice: instead of predicting pixels or a text caption, it predicts *renderable HTML*, then lets the browser produce the pixels — marrying the precision of text with the layout priors of code. On 80K+ screen-action pairs with Render-Aware RL (reward on rendered visual similarity, not code diff), Code2World-8B matches GPT-5 and Gemini-3-Pro-Image on next-UI prediction and boosts Gemini-2.5-Flash's AndroidWorld navigation by 9.5 points.

There are world models at every altitude. ProPlay (arXiv:2606.12780) operates at the process level, abstracting successful trajectories into a **process graph** whose edges carry reliability embeddings, then doing **pre-play** — simulating future process trajectories on the graph before each episode. VirtualEnv (arXiv:2601.07553) is an Unreal Engine 5 open-source embodied simulation platform with escape-room-style procedural tasks. On the platform side, Meta's ARE (arXiv:2509.17158) offers composable environment-building abstractions, and its async benchmark GAIA2 exposes failures that static benchmarks hide — environments that change over time under time pressure. Chimera (arXiv:2508.07745, NDSS 2026) deploys a society of role-playing enterprise agents whose meetings and org dynamics produce logs that hide malicious behavior in benign traffic; existing insider-threat detectors degrade badly on it.

> The endgame of environment engineering may be no environment at all — an agent that rehearses the world inside its own weights. That is also its biggest risk.
>
> — EnvACE, arXiv:2608.06197

That endgame is EnvACE (arXiv:2608.06197), the most provocative paper in this whole stack. **World rehearsal**: the policy alternates two roles — it first acts (emits a tool call), then plays the environment (generates the tool's response), then continues reasoning from that imagined response. The two roles are end-to-end co-trained on task success. It beats environment-scaling baselines on BFCL-v4, τ²-Bench, VitaBench, and FinMCP-Bench, and at deployment lets the agent "rehearse privately" before a real call. The failure mode is named plainly: **environment imitation bias** — the policy may learn to act brilliantly in an over-optimistic world it invented, then collapse in reality. It collapses the sim-to-real gap from "simulation vs. reality" into "the world the agent imagined vs. the real world." Highest upside, highest danger.

*The Long Tail*

## 7. Task Farms, Curricula, and Growing the Long Tail

*Environments are the stage; tasks are the play. Here is how the play gets written at industrial scale.*

TaskCraft (arXiv:2506.10055, ICLR 2026, OPPO) starts from a handful of atomic tasks and expands along two orthogonal axes — **depth** (nest subtasks for longer reasoning chains) and **width** (add new tools, APIs, data domains) — attaching an executable reference trajectory to every task, reaching ~36K tasks (and a 41K tool-heavy edition). Graph2Eval (arXiv:2510.00507, CVPR 2026) anchors generation in a knowledge graph, sampling subgraphs along meta-paths to kill LLM hallucination — 1,319 tasks with +20% semantic consistency and +17% solvability. SimpleQA→DeepResearch (arXiv:2608.02163) uses an Explorer–Formalizer–Challenger trio to evolve simple Q&A into 500 verifiable deep-research tasks structured as DAGs with checkpoints, so even the middle of a long investigation is gradeable. AgenticDataBench (arXiv:2607.01647, Tsinghua + Ant) organizes coverage by *data-science skills* rather than task count, clustering reusable operations from Stack Overflow, and includes five real B2B finance cases.

Domain-specific benchmarks have a habit of revealing where agents actually fail. ART (arXiv:2601.08988), a medical EHR benchmark of 600 "action-based reasoning" tasks, found retrieval near-perfect after prompt tuning but **aggregation reasoning at only 28–64% and threshold logic at 32–38%** — the bottleneck is not finding information but reasoning over it. The clinical lesson generalizes: agents are already decent at retrieval and tool invocation; the failures that remain are aggregation, conditional logic, and long-range dependency — the exact skills that only a world with realistic state can exercise. This is also why the "de-idealized" movement matters. AgentGym2 strips away the fiction that tools are pre-registered and inputs are clean; it measures exploratory discovery and robustness to noise, and the frontier models still struggle. OccuBench goes further and injects silent data degradation — truncated fields, missing columns — and finds these implicit faults are harder than explicit 500 errors, because the agent has to notice the rot on its own. A world that always behaves is a world that trains a brittle agent. Curriculum learning is where difficulty becomes a training schedule. E2H Reasoner (arXiv:2506.06632) schedules tasks easy-to-hard and fades out the easy ones to prevent overfitting, giving small 1.5B–3B models something vanilla RL cannot; ACuRL (arXiv:2602.10356, OSU) runs a fully autonomous loop — explore a new environment, generate a curriculum against current weaknesses, retrain — with a CUAJudge that agrees with humans 93% of the time, gaining 3–29% with only ~20% of parameters updating. State2State (arXiv:2608.04934, THUNLP) is purer still: no human-specified goals — any two explored states automatically define a "get from A to B" task, verified by rule-based state matching.

The platform papers here quietly deliver the most counterintuitive lessons. Gym-V (arXiv:2603.15432), 179 procedural visual environments across 10 domains, found that **observation scaffolding — the captions and rule descriptions you feed the model — matters more than whether you choose PPO or GRPO**; diverse training transfers positively, narrow training transfers negatively, and multi-turn interaction amplifies both. AgentDrive (arXiv:2601.16964) factorizes driving scenes across 7 orthogonal dimensions to generate 300,000 scenes and 100K MCQs. Agentick (arXiv:2605.06869) puts RL agents, LLM agents, VLM agents, hybrids, and humans on the same 37 Gymnasium-compatible procedural tasks with oracle reference policies — finding reasoning harnesses amplify LLM performance 3–10× and that **ASCII observation consistently beats natural-language observation**. InternBootcamp (arXiv:2508.08636, i.e. InternAgentHarness/BOOTCAMPCLI) uses a four-layer environment definition so one world automatically serves SFT, RL, and evaluation, across 1,000+ environments, and demonstrates that scaling task count by two orders of magnitude keeps buying reasoning gains.

*The Plumbing*

## 8. The Infrastructure Arms Race

*Once the worlds are designed, the moat is who can run ten thousand of them at once.*

The first wall is parallel sandboxes. OSGym (arXiv:2511.11672) runs **1,024 parallel OS copies** for CUA training by combining KVM virtualization with Copy-on-Write disks — every VM shares a base boot disk and stores only its own writes, cutting physical disk consumption 88% and speeding provisioning 37× — plus decentralized state management (one crashed copy never takes down the fleet) and hardware-aware scheduling. It generates 1,420 multi-turn trajectories per minute at **$0.20–0.30 per copy per day**, about a tenth of a standard deployment. AgenticAI-Supervisor (arXiv:2607.05773) decouples environment creation from scalable execution for thousand-way isolated rollouts, and leans on internal-state verification to suppress reward hacking.

The second wall is the black-box reality of modern agents: the best agents now run inside complex harnesses like Claude Code or OpenClaw, which are closed. ClawGym II (arXiv:2608.16798) solves this without touching the harness source — a serving proxy at the model boundary captures calls, multi-turn trajectories are reorganized into a **prefix tree** on which both PPO and GRPO optimize, and training stays consistent with inference. On ClawGym-Bench this added +9.98 through OpenClaw and +14.81 through Claude Code. EnvHarness (Google Research, arXiv:2608.19880) takes the even cheaper path: do not build a new world at all — wrap a programmable middleware layer around an existing static environment, reshape behavior while keeping the original verifier, and let EnvRigger automatically diagnose a policy's weaknesses and synthesize targeted harness components — up to +9.0 on held-out tasks with 9.8% fewer steps.

The third wall is accepting that the real world is messy and de-idealizing your benchmarks. AgentGym2 (arXiv:2607.05174, ACL 2026 Long) strips away the fiction of pre-packaged tools and clean inputs across 27 domains, explicitly testing exploratory tool discovery, tool composition, and robustness to noise — and even Gemini and GPT-5 struggle. OccuBench (arXiv:2604.10866) uses LLM language-environment simulators to cover 100 real occupations across 10 industries and 65 domains, injecting both explicit errors (timeouts, 500s) and implicit degradation (truncated, missing fields) — and finds implicit failures are harder, because the agent has to notice the rot itself. AgencyBench (arXiv:2601.11044, ACL 2026) scales tasks to true working-horizon: 138 tasks over 32 scenarios, averaging **90 tool calls and a million tokens each** over hours, scored with Docker rubrics and a user-simulating agent — closed models at 48.4% to open models' 32.1%.

> The bottleneck has moved from the algorithm to the plumbing. Whoever runs more isolated sandboxes cheaper wins.
>
> — the OSGym thesis

There is a reason the plumbing keeps coming up, and it is not glamorous. An RL run that touches ten thousand parallel environments for days on end is a distributed-systems problem dressed up as machine learning. A single flaky container can poison a batch; a leaked mutable state between rollouts teaches the agent the wrong lesson; an over-shared disk IO bottleneck can silently halve your throughput. OSGym's KVM plus copy-on-write trick — share the base disk, write only the deltas — is not a research result in the usual sense, but it cut disk usage 88% and sped provisioning 37×, which means the same research budget buys 37× more experiments. That is how science actually scales. ClawGym II's serving-proxy trick matters just as much: the production agents everyone wants to improve run inside closed, complex harnesses, and being able to RL against Claude Code without modifying its source line by line is the difference between a paper and a product.

*The Litmus Test*

## 9. How Do You Know Your World Is Any Good?

*Three litmus tests every environment must survive.*

| Verifier route | Signal | Cost | Trustworthiness | Poster children |
|---|---|---|---|---|
| **Executable assertions / exit-code / fail-to-pass** | Hard, deterministic, cheap | Low | High, watch for test toxicity / style bias | Terminal-Bench, SWE-Factory (F1=0.99), GUI-Genesis |
| **Hybrid verification** | Execution + learned no-execution scorer | Medium | Each saturates ~42–43%; combined to 51% | R2E-Gym |
| **Rubric / LLM-as-judge** | Soft, dense, flexible | High | Hard-task ceiling 77–82%; over-anchoring risk | SYNTHAGENT, AgentJudgeBench |

*Table 4 — pick the hardest verifier the task will tolerate.*

*Table 4 — pick the hardest verifier the task will tolerate.*

**Test one — the verifier.** Prefer hard. A final-state database check, a green test, an exit code: cheap, deterministic, and the first line of defense against reward hacking. When you must be soft, use a structured rubric, run the grader in a sandbox separate from the agent (Harbor's move), audit traces post-hoc (HackDetect's Mislead Gap), and sweep continuously with a cheap probe (AUC 0.9467 at 10⁻⁴ the cost). R2E-Gym's hybrid scorer shows the middle path: pair a deterministic-but-low-discrimination signal with a flexible-but-biased one, and let them cancel each other's blind spots.

**Test two — sim-to-real.** Everything you train in a synthetic world must eventually survive the real one. InfiniteWeb explicitly migrates synthetic sites to real pages for in-domain and generalization tests; WebWorld generalizes to code, GUI, and games; DreamGym uses purely synthetic experience as a warm-start that needs only a sliver of real interaction. But the gap never vanishes, and the most neglected dimension is *messiness*: synthetic worlds are too clean. C-World injects failures into the transition function; OccuBench distinguishes explicit errors from silent data rot; the lesson is that a good world is not one where everything goes smoothly — it is one that breaks, misbehaves, and forces the agent to recover.

**Test three — difficulty alignment.** The tasks must sit at the edge of the agent's ability, not over it and not under it. GenEnv aligns with an α-curriculum; TASTE re-aligns a saturated benchmark by dropping strong models back to 0.28–0.61; MAB-v3 first diagnoses the environment (41.7% silent-finish) before raising difficulty. A healthy world produces failures that cluster predictably along a capability hierarchy — not failures caused by doing nothing, by reading the grader, or by luck.

I want to make the verifier point sharper, because it is where good papers go to die. A verifier that is hard to satisfy is not automatically a good verifier. R2E-Gym measured that only about 20% of execution-based tests actually discriminate between a correct and an incorrect patch — the rest pass no matter what, which means they contribute nothing to the reward signal. A verifier that is too loose gets hacked; a verifier that is too strict, or brittle, or gated on a magic string, manufactures false failures. Harbor-Index's blunt finding was that roughly a third of their "hardest" tasks were hard only because the verifier was buggy — and their cure was as much engineering as research: run the grader in a separate sandbox, hand it only the artifact the agent declares, use binary rewards with realistic timeouts, and then audit the audits. The cheap-probe result (AUC 0.9467 at a ten-thousandth the cost of an LLM judge) tells you this auditing can be continuous, not a one-time ceremony. Treat your verifier like a product: version it, test it, fuzz it, and assume your agents are trying to break it.

*The Stakes*

## 10. The Open Call: Environment Contribution Should Be as Easy as a Pull Request

*Where this is going, and what I am asking you to build.*

Let me pull the trends together, sharply. First: **environment diversity beats rollout volume**. ToolVerse says it outright; ScaleEnv shows more domains → better generalization; EnvFactory does more with 85 worlds than rivals do with 500; C-World gets more from 1,170 trajectories than 119,000. Second: **self-play makes the world grow up with the model** — SPADE, Tool-R0, SESA, GenEnv, and the Agent-World arena all replace a fixed workbook with an adaptive course. Third: **world models and rehearsal collapse dependence on real environments** — WebWorld, Code2World, ProPlay, and EnvACE. Fourth: **the verifier has become the adversary** — reward hacking is now an arms race, and 67% of benchmark traces were exploitable before anyone audited them. Fifth: **standardization is quietly winning** — Harbor task packages, the Gym interface, and MCP are making worlds as composable as container images.

Which brings me to the point I want this essay to land on. Right now, building an agent training environment is dark art. OpenSWE spent **$1.47 million** and a 64-node cluster to make 45,320 Docker environments. Harbor burned $300K+ of compute across 54 benchmarks to curate 82 tasks. These are feats a handful of well-funded labs can afford — and that is exactly the problem. The long tail of real work lives in a million internal tools, niche CRMs, one-off CLI workflows, and weird legacy systems that no big lab will ever prioritize. If only a few labs can author worlds, then agent capability will plateau at whatever those labs can imagine, and everyone else's workflows will remain unservable.

> We have open-sourced models. We have open-sourced datasets. The missing layer is an open layer of environments — where contributing a training world feels like opening a GitHub PR, not summoning a platform team.
>
> — the MCEnvstore vision

That is why the end goal is something like an **environment store**: a Harbor-format task package — instruction, environment, reference solution, verifier — that anyone can write, version, share, and run in a one-click sandbox; with auto-generated verifiers, standardized isolation, anti-cheat checks, and community ratings the way npm packages have downloads and stars. A single academic should be able to encode one niche clinical workflow, a startup should package its internal API sandbox, and a hobbyist should drop a retro terminal challenge — and all of it compounds into the shared curriculum that trains the next generation of agents. **Environment contribution should be as open as a GitHub PR, not a handful of labs' dark art.**

So here is the call to action. If you are training agents today, stop blaming your model when your RL plateaus. Go build a world. Turn your internal tooling into a Harbor task package. Inject a failure into your transition function. Put your verifier in a separate sandbox. Run a cheap probe for reward hacking. Point a self-play loop at your weaknesses instead of sampling more rollouts. And then — contribute it back. The parameter curve is flattening. The world curve, we now know, is just getting started.

I will leave you with the practical version, because essays that end on a slogan owe you a plan. If you are training an agent this quarter: inventory your worlds against the five-element checklist and retire whatever fails it. Build one symbolic environment end to end — code, database, executable verifier, separate grader sandbox — and taste how much less noisy training becomes. Add a self-play loop against your own weakest tasks instead of sampling more rollouts on the ones you already solve. Inject one realistic failure mode you have never tested. Then publish that environment in a standard, rerunnable package. The labs have spent billions proving parameters plateau. The next decade is about worlds. Build yours.

> The Moore's Law of agents will not come from bigger weights. It will come from billions of tiny, weird, trustworthy worlds that anyone can contribute.
>
> — closing argument

## Glossary

**Environment** — The stateful world an agent interacts with, exposing observations, actions, rewards, and resets.

**Verifier** — The logic that decides task success; hard verifiers run tests, soft ones use rubrics or an LLM.

**Reward hacking** — Exploiting the scoring rule to win without actually solving the task.

**Rubric** — A written scoring checklist used when no executable test exists.

**Sim-to-real** — The gap (and transfer) between training in simulation and acting in the real world.

**World model** — A model that learns to predict the next state from state + action, used for simulation or inference-time rehearsal.

**Self-play** — The question-writer and solver co-evolving against each other.

**Gym interface** — The standard reset()/step() loop every environment implements.

**Dec-POMDP** — Decentralized Partially Observable MDP; the math of multiple decision-makers sharing one world.

**GRPO** — Group Relative Policy Optimization; the critic-free RL workhorse of the agent era.

**MCP** — Model Context Protocol; the standard way tools plug into an agent.

**CUA** — Computer-Use Agent; sees screenshots and clicks coordinates.

**Fail-to-pass (F2P)** — A test that fails before your fix and passes after — the coding-world reward standard.

**Oracle** — The privileged upper-bound strategy used to measure distance from optimal.

## Paper & Framework Index (9 categories)

*One-line contribution; the arXiv number is the ID suffix.*

#### ① Difficulty, benchmarks, verifier trust, surveys

|  |  |  |
|---|---|---|
| TASTE / τc-Bench | Reverse task pipeline + contrastive n-gram to re-harden saturated benchmarks; -5% to -80% drops | 2605.28556 |
| AgentBoard | Progress-rate process metric; NeurIPS 2024 Oral | 2401.13178 |
| τ²-Bench | Dual-control Telecom domain as Dec-POMDP; ICML 2026 Oral | 2506.07982 |
| Easy2Hard-Bench | IRT / Glicko-2 numeric difficulty labels; NeurIPS 2024 D&B | 2409.18433 |
| Benchmarks as Microscopes | Calls for model metrology; COLM 2024 | 2407.16711 |
| E2H Curriculum RL | Easy-to-hard schedule + fade-out; small-model RL guarantees | 2506.06632 |
| Cheap Reward Hacking Detection | Tiny encoder + linear probe; AUC 0.9467 at 10⁻⁴ the cost | 2606.08893 |
| HackDetect / Protocol Validity | 2,385 traces / 15 benchmarks; 67%/66.7% exposed; inflation 0.45–1.00 | 2607.22368 |
| MAB-v3 | Clinical environment hardening; silent-finish ceiling 41.7%→8.9%; ICML 2026 | 2607.01470 |
| EnviSAgE Survey | The GEF loop (Generate–Execute–Feedback); NeurIPS 2025 Workshop | 2511.09586 |
| Agentic Env Eng Survey | Modeling→Synthesis→Evaluation→Application lifecycle; symbolic vs neural | 2606.12191 |
| AgentJudgeBench | LLM-judge hard-task ceiling 77–82%; EMNLP 2026 Main | 2608.26623 |
| Harbor-Index 1.0 | 6,627→82 hard tasks; separate verifier sandbox; 0.6% residual gaming | harbor-index.org |
| Benchmark Radar | Living DB: 37 sources, 1,283 records, 12,916 observations | 2609.11115 |
| Efficient Benchmarking | 30–70% mid-range sampling; 44–70% fewer tasks, ranking preserved | 2603.23749 |
| Hierarchy of Agentic Capabilities | Five-rung capability ladder; strongest models still fail ~40% | 2601.09032 |

#### ② General environment synthesis

|  |  |  |
|---|---|---|
| Agent-World | Tool dependency graph + executable Python solutions; 1,978 envs / 19,822 tools; GRPO self-evolving arena | 2604.18292 |
| EnvScaler | SkelBuilder + ScenGenerator two-stage; 191 envs / ~7K scenes | 2601.05808 |
| AWM | Code-driven + database-backed; 1,000/10,000/35; strong OOD; ICML 2026 | 2602.10090 |
| ToolVerse | ~400 MCP / ~4,500 tools; dynamic-unlocking sampling; diversity > rollouts | 2607.15660 |
| SPADE | One LLM as both env designer and reasoner; regret-targeted edge-of-capability tasks | 2608.19197 |
| CuES | Curiosity-driven, environment-anchored task mining; no human seed | 2512.01311 |
| EnvFactory | Topology-aware sampling; 85 envs; implicit-intent multi-turn trajectories | 2605.18703 |
| Qwen-AgentWorld | Language world model; 35B-A3B / 397B-A17B; >10M real interactions | 2606.24597 |
| ScaleEnv | From-scratch environments with procedural-testing CI; ICML 2026 | 2602.06820 |
| GenEnv | α-Curriculum difficulty alignment; 7B +40.3%, 3.3× less data | 2512.19682 |
| C-World | 5,571 tools / 204 apps; World Engine ρ=0.883; injected failures; ACL 2026 | 2601.06328 |
| ClawEnvKit | One-line NL spec → environment; 1,040 envs; 13,800× cheaper | 2604.18543 |
| AgentMercury | World-first, tasks emerge; 4,783 envs / 14 industries / 50 countries | 2608.20634 |
| Repo2RLEnv | Real GitHub repos → Harbor task packages; 6 pipelines + 14 recipes | HuggingFace |

#### ③ Experience / interaction synthesis

|  |  |  |
|---|---|---|
| Tool-R0 | Generator–solver self-play, zero-data bootstrapping; +92.5% relative | 2602.21320 |
| DreamGym | Reasoning-based experience model; sim-to-real warm-start; ICLR 2026 | 2511.03773 |
| GAIS | Two-stage grounding + adversarial long-horizon traces; KDD 2026 | 2606.02001 |
| TOUCAN | 1.5M trajectories / ~500 real MCP; largest public tool-agent dataset | 2510.01179 |
| SYNTHAGENT | User simulator + mock tools + rubric reward; underspecified prompts | 2601.22511 |

#### ④ Web / GUI / CUA synthesis

|  |  |  |
|---|---|---|
| WebArena-Infinity | Multi-agent artifact→running web app; 10 envs / 1,260 tasks | GitHub |
| InfiniteWeb | TDD-built interconnected websites → real-web generalization; ACL 2026 | 2601.04126 |
| ScaleWoB | Backend-free web shells, cross-platform; mobile agents 27.92% vs humans 92.08% | 2605.25160 |
| GUI-Genesis | Real apps → lightweight web + code-native rewards; >$28K/epoch saved | 2602.14093 |
| Weblica | HTTP-layer caching replay; visual web agent RL | 2605.06761 |
| ScaleCUA / VeriGen | Iterative Docker-probed verifiable tasks; OSWorld 68.7% | 2607.11185 |
| FaraGen (Fara-7B) | High-frequency-site task proposal + multi-verifier; ~$1/trajectory | 2511.19663 |
| FaraGen1.5 | Environment + solver + verifier triad; 27B at Online-Mind2Web 72.3% | 2606.20785 |
| OS-Genesis | Reverse task synthesis: explore then retrospect; ACL 2025 | 2412.19723 |
| HATS | Hardness = semantic ambiguity; explore-refine loop; CVPR 2026 | 2603.12138 |

#### ⑤ Terminal / coding (SWE) environments

|  |  |  |
|---|---|---|
| Terminal-Bench 2.0 | 89 real CLI tasks; frontier <65%; ICLR 2026 | 2601.11868 |
| TUA-Bench | 120 general terminal tasks / 5 families; best 65.8%; Meta | 2606.28480 |
| SWE-smith | Build image, inject bugs; 128 repos / 50,137 instances; Verified 40.2% | 2504.21798 |
| R2E-Gym | Commit + back-translation; 8,135 tasks; hybrid verifier to 51% | 2504.07164 |
| SWE-Factory | SWE-Builder + exit-code fail2pass; F1=0.99; $0.047/instance | 2506.10954 |
| SWE-World | Docker-free learned surrogate; 6.2%→52/55/68.2% | 2602.03419 |
| SWE-Universe | 807,693 PR-built envs + in-loop cheating detection; Verified 75.3% | 2602.02361 |
| OpenSWE (daVinci-Env) | 45,320 Docker envs / 12.8K repos / $1.47M fully open | 2603.13023 |
| DockSmith | Writing Dockerfiles as a trainable skill; 39.72% F2P | 2602.00592 |
| TerminalTraj | 32K images / 50,733 trajectories / 8 domains; ICML 2026 Spotlight | 2602.01244 |
| CLI-Universe | Taxonomy sampling + evidence grounding; 6K traj; TB2.0 33.4% | 2606.22883 |
| SETA | Synth + Evol dual pipelines; 4,500+ envs; 8B GRPO 12% | 2607.10891 |
| TerminalWorld | Reverse-engineered from 80,870 terminal recordings; r=0.20 with expert sets | 2605.22535 |
| ResearchEnvBench | "Get the research repo running" as the task; dies on dependency hell | 2603.06739 |
| ML-AutoResearch | Full research-loop synthetic tasks + self-debug; AUP +9% | 2603.17216 |

#### ⑥ Model-based world models / simulation

|  |  |  |
|---|---|---|
| WebWorld | 1M+ interactions; inference-time search beats GPT-5; ICML 2026 | 2602.14721 |
| Code2World | Predict renderable HTML next-UI; Render-Aware RL; CVPR 2026 | 2602.09856 |
| ProPlay | Process graph + process-level pre-play | 2606.12780 |
| VirtualEnv | UE5 open-source embodied simulation platform | 2601.07553 |
| ARE / GAIA2 | Meta env-building platform + async benchmark | 2509.17158 |
| Chimera | Multi-agent insider-threat simulation; NDSS 2026 | 2508.07745 |
| EnvACE | World rehearsal: policy plays both actor and environment | 2608.06197 |

#### ⑦ Task / scenario synthesis & curricula

|  |  |  |
|---|---|---|
| TaskCraft | Depth × width expansion; 36K/41K tasks; ICLR 2026 | 2506.10055 |
| Graph2Eval | Knowledge-graph + meta-path anchored tasks; 1,319 tasks; CVPR 2026 | 2510.00507 |
| SimpleQA→DeepResearch | Explorer–Formalizer–Challenger; 500 verifiable deep-research tasks | 2608.02163 |
| AgenticDataBench | Skill-clustering organization; 5 real B2B finance cases | 2607.01647 |
| ART | Medical action-based reasoning, 600 tasks; aggregation 28–64%, threshold 32–38% | 2601.08988 |
| ACuRL | Autonomous curriculum RL, zero labels; CUAJudge 93% human agreement | 2602.10356 |
| State2State | State pairs auto-become tasks; rule-based state matching | 2608.04934 |
| SESA | Challenger–Solver + skill memory co-evolution; skills internalize into weights | 2607.29468 |
| Gym-V | 179 procedural visual envs; observation scaffolding > RL algorithm | 2603.15432 |
| InternBootcamp | Four-layer env definition auto-converts to SFT/RL/eval; 1,000+ envs | 2508.08636 |
| AgentDrive | 7-factor factorization; 300K driving scenes / 100K MCQs | 2601.16964 |
| Agentick | 37 Gymnasium tasks + oracle; ASCII observation beats NL; harness 3–10× | 2605.06869 |

#### ⑧ Platform / infrastructure / de-idealized evaluation

|  |  |  |
|---|---|---|
| OSGym | 1,024 parallel sandboxes; KVM+COW disk −88%; $0.2–0.3/copy/day | 2511.11672 |
| AgenticAI-Supervisor | UI-driven simulation engine; thousand-way isolated rollouts | 2607.05773 |
| ClawGym II | Black-box RL on Claude Code / OpenClaw; serving proxy + prefix tree | 2608.16798 |
| EnvHarness | Programmable layer wraps static envs; EnvRigger auto-synthesizes | 2608.19880 |
| OccuBench | Language environment simulators for 100 occupations; implicit failures harder | 2604.10866 |
| AgentGym2 | De-idealized 27-domain eval; tool discovery, composition, noise robustness | 2607.05174 |
| AgencyBench | 138 tasks; avg 90 tool calls / 1M tokens each; closed 48.4% vs open 32.1% | 2601.11044 |

End. Two threads to carry out: the **GEF loop** (Generate–Execute–Feedback) is the runtime dataflow of a living environment; the **lifecycle** (modeling → synthesis → evaluation → application) is how it gets built. Symbolic builds the trustworthy core, neural adds the scale, self-play chases the frontier, world models remove the need for a physical world — and the whole thing only compounds if everyone can contribute a world.

**Read next · Part 2 of 2**

### [How Xiaomi Actually Does RL — reading the MiMo-V2.6 open stack](mimo-v2.6-rl.html)

This instalment argued that environment engineering is the real bottleneck, and closed on an open call: contributing a world should be as easy as opening a pull request. Part 2 examines a release that answers that call — a model lab shipping not just a report and a training recipe, but the environments themselves as 3,764 runnable Docker images with verifiers and manifests attached. Every claim in its report is traced to the file, config value, or task environment that implements it, including three claims the released code does *not* implement. Where Part 1 is a map of the field, Part 2 is the ground survey.
