# Agentic RL Environment Engineering

[English](README.md) | 简体中文

**把 MiMo-V2.6 的开源栈完整读一遍** —— 技术报告、开源的 `verl` fork、7,780 条真实任务、3,764 个 Docker 镜像 —— 并把每一条「RL 到底是怎么做的」论断，落到实现它的那个文件、配置值或环境上。

[![CI](https://github.com/possibleme2026-lang/agentic-rl-environment-engineering/actions/workflows/ci.yml/badge.svg)](https://github.com/possibleme2026-lang/agentic-rl-environment-engineering/actions/workflows/ci.yml)

📄 **[阅读正文 →](blog/mimo-v2.6-rl.html)** —— 单文件 HTML，无需构建，无需联网。

---

## 这是什么

小米发布了一份 44 页的 MiMo-V2.6 报告，描述了一个多数团队达不到的 RL 规模：1,568 个 prompt × 16 条 rollout = **每步 25,000 条轨迹**，每批 27–37 亿 token，上下文最长 100 万，同时平衡三个扩展维度 —— 训练算力、环境与 harness 多样性、以及**判分算力**。

这类报告通常止于公式。这次的不同在于配方也一起开源了：一个实现了该算法的 `verl` fork、可作为容器直接运行的 task 环境、以及一份把每行数据对应到镜像的训练数据清单。

所以正文是一次「读代码」练习：报告里每一条论断，都去找让它落地的代码或环境 —— 代码与报告矛盾、或给报告划出边界的地方，直接讲明。本仓库就是这篇正文加它背后的证据。

## 主要发现

代码揭示了六件公式不会告诉你的事。完整命令与原始输出见 [`evidence/VERIFICATION.md`](evidence/VERIFICATION.md)。

**1. 惩罚机制是 fork 独有的，尽管版权头写着 ByteDance。** 承载报告 §4.3.3 行为正则化的三个文件，在上游 `verl` 的 `v0.9.0` **和** `main` 上都是 **404** —— 但它们的版权头写着 *Copyright ByteDance Ltd. and/or its affiliates*。合理的解读是：小米对着 `verl` 的扩展点实现了自己内部的算法，版权头没改。只有 `rollout_corr_helper.py` 真正来自上游。

**2. 代码在系统地匿名化一个论文点名了的系统。** fork 的注释里出现八次未定义、且被当作专有名词大写的「**reference RL framework**」，而论文里全文零次。论文 §6.4 点明了它指什么：*"We extend the RL and OPD infrastructure of MiMo-V2-Flash, keeping SGLang and Megatron-LM as the inference and the training engine."* 加上 `SETUP_FAILED`、`dirty_repetition`、`invalid_reward_value: -999` 这几个具体锚点可以确证。

**3. §4.3.3 的两个公式都逐项落地了。** Eq.(4) 的每个符号都有对应配置字段 —— 包括 `anchor_quantile: 0.3`，刻意低于中位数以便压得更狠。Eq.(5) 的 α/β 就是 `solve_signed_factors(removed, pos_base, added, neg_base)`，并用运行时 `assert` 强制质量守恒。注意代码里 α/β 的名称与论文**相反**（α = `min_scale`，β = `max_scale`）—— 按值判断，别按字母。

**4. 一句注释解释了为什么「显而易见的惩罚」是有害的。** 把一整轮坏轨迹的 advantage 设成 −1 会「向 batch 注入净负质量，在**无 KL、无 entropy bonus** 时把策略拉平并让输出变长」。而开源的 recipe 恰好是 `use_kl_loss: false` **且** `entropy_coeff: 0` —— 没有任何东西对冲注入的负质量。**质量守恒是数值要求，不是审美要求。**

**5. 报告描述的三样东西，开源栈里没有** —— 正文如实写出，而不是含糊带过：

| 报告描述 | 开源栈实际有 |
|---|---|
| 四个解耦 clip 边界，上界到 **5.0**，按 entropy 调 | `clip_ratio_high` 只有 {0.2, 0.28}；调度代码零命中 |
| 全部 code agent 任务上的 GRS + GAR | 只有 webdev 的 groupwise grader；`grep -rn "GAR\|GRS" recipes/` 零命中 |
| 两阶段 anti-hack + 完整规则集 | 正则守卫（**默认关**），只 ship stage 1；4 条规则列在 `excluded_reference_rules` |

**6. 数据到环境的对应是精确的。** `image-mapping.jsonl` **3,764** 行，Docker Hub **3,764** 个 tag。构成：2,698 code + 1,000 cyber + 65 general + 1 webdev。其中 code 与 cyber 的镜像数与任务数**精确一一对应**。

## 复现

正文里的每条论断都可以自行重导。两个外部 clone 加一个脚本：

```bash
# 1. 上游 fork（735 个 .py，另含两个 submodule）
git clone --depth 1 https://github.com/XiaomiMiMo/verl.git mimo-verl
cd mimo-verl && git submodule update --init --depth 1 \
    third_party/mimoagent-osr third_party/uni_agent && cd ..

# 2. 数据集（5 个 parquet，7,780 行，另含镜像清单）
#    需要 huggingface_hub；任何带 pyarrow 的 Python 都能读这些文件
python -c "
from huggingface_hub import snapshot_download
snapshot_download('XiaomiMiMo/MiMo-V2.6-RL-oss', local_dir='mimo-oss-data')
"

# 3. 重跑本仓库的机器无关检查
python tools/check_claims.py
```

`tools/check_claims.py` 是纯标准库的守卫。它校验 README 的相对链接是否落地、[`evidence/offsets.json`](evidence/offsets.json) 内部是否自洽（各族计数之和等于总数、code/cyber 镜像数等于任务数、共享镜像的域确实在共享），以及正文是否不含外部资源标签从而能离线打开。

### Docker 镜像

镜像**不在本仓库分发**，它们位于 Docker Hub 的 [`xiaomimimo/mimo-v2.6-rl-oss`](https://hub.docker.com/r/xiaomimimo/mimo-v2.6-rl-oss)。在没有容器运行时的环境下，可以直接走 Registry v2 API：

```bash
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io\
&scope=repository:xiaomimimo/mimo-v2.6-rl-oss:pull" | python -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -sL -H "Authorization: Bearer $TOKEN" \
  "https://registry-1.docker.io/v2/xiaomimimo/mimo-v2.6-rl-oss/manifests/general-agent-env-1" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json"
```

## 仓库结构

```
blog/mimo-v2.6-rl.html      正文 —— 单文件、内联 CSS、零依赖
evidence/VERIFICATION.md    每条论断的原始命令与输出
evidence/offsets.json       机器可读的计数台账，CI 中校验
tools/check_claims.py       纯标准库守卫：链接、台账、正文自包含性
```

## 范围与限制

这是**分析，不是复现**。这里不训练任何模型，不涉及 GPU。

- 每条论断都可追溯到文件路径、配置值、parquet 字段、镜像 tag 或报告章节。正文引用的行号对应报告经 `pypdf` 提取后的文本，不是 PDF 页内坐标。
- **有一处发现被明确标为未解问题**：为什么 925 个 `general_agent` 任务共用一个镜像，而 64 个 `terminal_bench` 任务各自独占一个。这个分界是直接从 `docker_image` 列读出来的，但**机制**是从 `manifest.json` 推断的 —— 没有在代码里找到划出这条线的位置。
- 报告里的生产数字（MiMo-V2.6-Pro：lr 3×10⁻⁶、每层 384 专家）描述的是内部 run，不是开源的 9B recipe（lr 1×10⁻⁶）。两者不可互换，正文严格分开。
- 上游存在性检查只记录 HTTP 状态码；它证明某个路径在上游不存在，不解释原因。

## 许可证

Apache-2.0 —— 见 [LICENSE](LICENSE)。

本仓库**不含任何来自小米、字节跳动或 `verl-project/verl` 的代码**。这是独立分析：正文是原创文字，`check_claims.py` 是原创代码，`offsets.json` 记录的是从公开产物中读出的事实。`verl` fork 与 Docker 镜像仍适用其各自的许可证，归各自所有者所有。

## 参考

本分析直接建立在他人工作之上：

- **MiMo-V2.6** —— *MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement*，小米 MiMo 团队。技术报告，44 页。本文分析的一切都由它定义：三个扩展维度（§4）、groupwise agentic grading（§4.3）、multi-harness training（§4.2.5）、reward hacking 缓解（§4.2.6）、RL 基础设施（§6）。
- **`XiaomiMiMo/verl`** —— RL 训练 fork，commit `a2ad9f6`，基于 [`verl-project/verl`](https://github.com/verl-project/verl) `0.9.0.dev`。
- **`XiaomiMiMo/mimoagent`** —— agent harness、工具、执行环境与 grader。fork 自 [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)。
- **`XiaomiMiMo/uni-agent`** —— 模型网关与 TransferQueue 轨迹采集。fork 自 [verl-project/uni-agent](https://github.com/verl-project/uni-agent)。
- **ARVO** —— cyber 任务背后的漏洞复现数据集家族。
- **GRPO** —— Shao et al., *DeepSeekMath*, 2024。基础算法；本栈刻意贴近原版。
- **Muown** —— Muon + AdamW 混合优化器，带 row-norm 控制，用于 mid-training 的切换。
- **GLM-5.2** —— `antihack.py` 引用它作为两阶段在线守卫设计的来源。

感谢 MiMo 团队开源这些环境、验证器与镜像 —— 正是它们让这种程度的代码阅读成为可能。
