# Agent 安全 Benchmark 调研与交接文档

> 目标：为自研 Agent 做安全评测，要求"按 Agent 流程不同节点分别攻击、分项打分"。
> 本文档所有条目均已通过 arXiv / GitHub 官方仓库交叉核验，最后更新日期以文末为准。

---

## 一、Agent 流程节点与攻击面对照

先约定一套通用 Agent 流水线节点，用于统一各 Benchmark 的覆盖范围描述：

| 节点编号 | 节点名 | 说明 |
|---|---|---|
| N1 | 用户输入 (User Prompt) | 直接注入 / 越狱 |
| N2 | 系统提示 / 工具描述 (System & Tool Spec) | 工具描述投毒、schema 篡改 |
| N3 | 检索 / 外部数据 (Retrieval & Web) | 间接注入（网页、邮件、文档） |
| N4 | 工具返回结果 (Tool Output) | 间接注入、返回值污染 |
| N5 | 记忆 / 上下文 (Memory) | 记忆投毒、跨会话污染 |
| N6 | 规划 / 决策 (Planning) | 目标劫持、路径改写 |
| N7 | 工具调用与执行 (Tool Execution) | 越权调用、参数篡改、MCP 攻击 |
| N8 | 多 Agent 协作 (Multi-Agent) | 委托越权、跨 Agent 传播 |

---

## 二、核心推荐 Benchmark（已逐项核验）

### 1. AgentDojo —— 事实标准，首选

| 项 | 内容 |
|---|---|
| 名称 | AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents |
| 论文 | NeurIPS 2024 D&B Track，arXiv:2406.13352 |
| 仓库 | https://github.com/ethz-spylab/agentdojo （772★，MIT） |
| 主页 / 排行榜 | https://agentdojo.spylab.ai/results/ |
| 出品方 | ETH Zurich SPY Lab + Invariant Labs |
| 覆盖节点 | **N1（直接注入）、N3/N4（间接注入，主战场）、N6（目标劫持）、N7（工具调用）** |
| 规模 | 97 个用户任务、629 个安全测试用例，覆盖 4 个套件：workspace / banking / travel / slack |
| 核验方式 | arXiv v3 (2024-11)、GitHub 首页 README、PyPI `pip install agentdojo` |
| License | MIT |
| 评分指标 | **① Utility under attack**（有攻击时任务完成率）<br>**② Security Score**（抗注入成功率，= 1 − ASR；越高越好）<br>按 4 个套件、按攻击类型分别出分 |
| 攻击方式 | `important_instructions` / `tool_knowledge` / `ignore_previous` / `direct` / `system_message` / `injecagent`，可自定义并注册新攻击 |
| 防御方式 | `tool_filter` / `spotlighting_with_delimiting` / `repeat_user_prompt` / `transformers_pi_detector`，可插入自研防御 |

**接入使用**：

```bash
pip install agentdojo
# 或需要 prompt injection detector：
pip install "agentdojo[transformers]"

# 全套件跑
python -m agentdojo.scripts.benchmark \
    --model gpt-4o-2024-05-13 \
    --defense tool_filter --attack tool_knowledge

# 单套件、单任务跑
python -m agentdojo.scripts.benchmark -s workspace \
    -ut user_task_0 -ut user_task_1 \
    --model gpt-4o-2024-05-13 \
    --defense tool_filter --attack tool_knowledge
```

**接入你自研 Agent**：实现 `AgentPipeline` 抽象类（在 `src/agentdojo/agent_pipeline/` 下），把你 Agent 的 LLM 调用、工具调用替换进去即可。文档见 https://agentdojo.spylab.ai/。

**建议**：作为主评测；结果自动分套件、分攻击面出分，直接对应节点 N1/N3/N4/N7。

---

### 2. AgentDyn —— 第二基准，2026 事实新标准

| 项 | 内容 |
|---|---|
| 名称 | AgentDyn: Are Your Agent Security Defenses Deployable in Real-World Dynamic Environments? |
| 论文 | arXiv:2602.03117（2026 年发布） |
| 仓库 | https://github.com/SaFo-Lab/AgentDyn （75★，MIT） |
| 出品方 | Washington University in St. Louis / SaFo-Lab（Chaowei Xiao 组） |
| 覆盖节点 | **N3/N4（动态间接注入，含"有用的第三方指令"）、N6（开放式任务规划）、N7** |
| 规模 | 60 个开放式任务、560 个注入用例，覆盖 Shopping / GitHub / DailyLife 三套件 |
| 核验方式 | GitHub 仓库 README（AgentDojo 团队致谢标注确认），arXiv v2 (2026-05) |
| License | MIT |
| 评分指标 | 沿用 AgentDojo 的 Utility / ASR 分项体系；额外区分"过度防御 (over-defense)" |
| 与 AgentDojo 关系 | **基于 AgentDojo 框架扩展**，兼容 AgentDojo 原有 4 个套件；使用同一 `benchmark.py` 脚本 |

**接入使用**：

```bash
git clone https://github.com/SaFo-Lab/AgentDyn
cd AgentDyn
pip install -e .

# 跑 shopping 套件
python -m agentdojo.scripts.benchmark -s shopping \
    --model GPT_4O_2024_08_06 \
    --defense tool_filter --attack important_instructions
```

**建议**：作为 AgentDojo 的进阶补充，重点评估防御在"真实动态环境+有用的第三方指令"下的部署可行性。已被 StepGuard、AgentFlow、DRIFT、AuthGraph 等主流防御论文并列引用。

---

### 3. Agent-SafetyBench —— 风险广度覆盖

| 项 | 内容 |
|---|---|
| 名称 | Agent-SafetyBench: Evaluating the Safety of LLM Agents |
| 论文 | arXiv:2412.14470（2024-12） |
| 仓库 | https://github.com/thu-coai/Agent-SafetyBench （157★，MIT） |
| 出品方 | 清华 CoAI 组 |
| 覆盖节点 | **N1/N3/N4/N7**（跨节点，按风险类别组织；N5/N6 覆盖待论文确认） |
| 规模 | 349 个交互环境、2000 个测试用例；**8 类安全风险 × 10 类失败模式** |
| 核验方式 | GitHub API 直接查询、arXiv 摘要 |
| License | MIT |
| 评分指标 | Safety Score（0–100），按 8 类风险、10 类失败模式分别出分 |
| 特点 | 覆盖 16 个主流 LLM Agent，公开结果：**无一 Agent 安全分超过 60%** |

**接入使用**：

```bash
git clone https://github.com/thu-coai/Agent-SafetyBench
cd Agent-SafetyBench
pip install -r requirements.txt

# 1) 生成结果：改 evaluation/eval.sh 里的 model_name（API 模型走 OpenRouter）
cd evaluation
bash eval.sh

# 2) 打分：用官方 scorer 模型 ShieldAgent 计算 Safety Score
cd ../score
bash eval_with_shield.sh
```

**建议**：用来做**风险类别覆盖度**测试，与 AgentDojo（注入专项）互补。

---

### 4. InjecAgent —— 工具输出节点专项

| 项 | 内容 |
|---|---|
| 名称 | InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents |
| 论文 | ACL 2024 Findings，arXiv:2403.02691 |
| 仓库 | https://github.com/uiuc-kang-lab/InjecAgent （162★，MIT） |
| 出品方 | UIUC Kang Lab |
| 覆盖节点 | **N4（工具返回，间接注入专项）** |
| 规模 | 1054 用例，17 个用户工具 + 62 个攻击者工具 |
| 核验方式 | arXiv v3 (2024-08)、GitHub API |
| License | MIT |
| 评分指标 | **ASR-valid** / **ASR-all**，按两类攻击目标（① 直接危害用户 ② 私密数据窃取）分别打分 |
| 特点 | GPT-4 (ReAct) 基线 ASR 约 24%，加"hacking prompt"后近乎翻倍 |

**接入使用**：

```bash
git clone https://github.com/uiuc-kang-lab/InjecAgent
cd InjecAgent
export PYTHONPATH=.
pip install -r requirements.txt
python3 src/evaluate_prompted_agent.py \
    --model_type GPT --model_name gpt-3.5-turbo-0613 \
    --setting base --prompt_type InjecAgent \
    --use_cache
```

**建议**：仅测"间接注入"这一点时更快、更聚焦，是很多防御论文的必测项。

---

### 5. BIPIA —— 间接注入"首个基准"

| 项 | 内容 |
|---|---|
| 名称 | BIPIA: Benchmarking and Defending Against Indirect Prompt Injection Attacks on LLMs |
| 论文 | KDD 2025，arXiv:2312.14197 |
| 仓库 | https://github.com/microsoft/BIPIA （155★，License: Other/自定义） |
| 出品方 | Microsoft Research Asia + USTC |
| 覆盖节点 | **N3/N4（外部内容注入）** |
| 规模 | 首个 IPI 基准，覆盖 5 类任务模板：email / table / QA / abstract / code；可大规模合成衍生样本（相关研究已生成 143 万+ 样本） |
| 核验方式 | GitHub API、arXiv 摘要（KDD 2025） |
| License | **自定义（GitHub 标注 Other，非标准开源协议，商用前需确认）** |
| 评分指标 | **ASR**（攻击成功率），按任务类型分别统计；论文中白盒防御可将 ASR 降至近零 |
| 特点 | 早期 IPI 事实标准之一，被 InstructDetector、CommandSans 等大量防御论文作为基准引用 |

**接入使用**：

```bash
git clone https://github.com/microsoft/BIPIA
cd BIPIA
# 数据集与生成脚本在仓库内，README 含评估与防御的复现命令
```

**建议**：与 InjecAgent 二选一即可（都测 N3/N4）；BIPIA 更偏"LLM 内容注入安全"，InjecAgent 更偏"工具集成 Agent"。

---

### 6. AgentHarm —— 有害能力基线

| 项 | 内容 |
|---|---|
| 名称 | AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents |
| 论文 | ICLR 2025，arXiv:2410.09024 |
| 数据集 | https://huggingface.co/datasets/ai-safety-institute/AgentHarm |
| 代码 | https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/agentharm |
| 出品方 | UK AI Security Institute + Gray Swan AI + EPFL + CMU |
| 覆盖节点 | **N1（越狱）+ N7（多步工具调用完成度）** |
| 规模 | 110 个明显恶意任务（含增强共 440 个），**11 类危害**：欺诈、网络犯罪、骚扰、毒品、性、版权、仇恨、虚假信息等 |
| 核验方式 | HuggingFace 数据集页面、arXiv v3 (2025-04) |
| License | MIT + 附加条款（仅限用于提升 AI 安全性研究） |
| 评分指标 | **① Refusal Rate**（拒绝率）<br>**② Harm Score**（真被越狱后能否完成多步任务）<br>按 11 类危害分别出分 |
| 特点 | 测的是"Agent 能被诱导做多坏"，不测按节点攻击，与 AgentDojo **互补而非替代** |

**接入使用**：

```bash
pip install inspect_ai==0.3.25
# 从 UKGovernmentBEIS/inspect_evals 仓库的 agentharm 目录运行：
git clone https://github.com/UKGovernmentBEIS/inspect_evals
cd inspect_evals/src/inspect_evals/agentharm
python inspect_eval.py --split test_public \
    --model_name openai/gpt-4o-2024-08-06 --prompt_template_name empty
# 良性对照：
python inspect_eval.py --split test_public_benign \
    --model_name openai/gpt-4o-2024-08-06 --prompt_template_name empty
```

**建议**：作为"有害能力上限"基线。**注意：它不满足"按节点分别打分"的要求**，只作为整体能力评估。

---

### 7. WASP —— 网页 / UI Agent 专用

| 项 | 内容 |
|---|---|
| 名称 | WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks |
| 论文 | arXiv:2504.18575（2025-04） |
| 仓库 | https://github.com/facebookresearch/wasp （98★，**已 2026-07 归档为只读**） |
| 出品方 | Meta / Facebook AI Research |
| 覆盖节点 | **N3（网页 DOM 注入）、N7（浏览器操作）** |
| 规模 | 基于 VisualWebArena / WebArena 的 Reddit + GitLab 真实网站环境，端到端 |
| 核验方式 | GitHub 仓库 README、arXiv v2 (2025-05) |
| License | **CC-BY-NC 4.0**（仅非商业使用，需注意） |
| 评分指标 | ① 攻击部分成功率（up to 86%）<br>② 攻击目标完全达成率（明显更低，量化"安全的偶然性"） |
| 支持的 Agent | GPT-4o / GPT-4o-mini / Claude-3.5 Sonnet Computer Use / Claude-3.7 |
| 特点 | 需要 Docker 部署完整 WebArena 环境；单次跑 4-6 小时 |

**接入使用**：

```bash
git clone https://github.com/facebookresearch/wasp
cd wasp/webarena_prompt_injections
bash setup.sh
# 需先按 visualwebarena/environment_docker/README.md 部署 Docker 环境
export DATASET=webarena_prompt_injections
export REDDIT="<your_reddit_domain>:9999"
export GITLAB="<your_gitlab_domain>:8023"
export OPENAI_API_KEY=sk-...

python run.py --config configs/experiment_config.raw.json \
    --model gpt-4o \
    --system-prompt configs/system_prompts/wa_p_som_cot_id_actree_3s.json \
    --output-dir ./results \
    --output-format webarena
```

**建议**：**仅当你的 Agent 是网页浏览 / Computer-Use 类型才用**。License 是 CC-BY-NC，不能商用。仓库已归档但代码可用。

---

### 8. Uber ADR / ADR-Bench —— MCP & 生产级检测评测

| 项 | 内容 |
|---|---|
| 名称 | Agentic AI Detection and Response (ADR) + ADR-Bench |
| 论文 | MLSys 2026（Industry Track），arXiv:2605.17380 |
| 仓库 | https://github.com/uber/ADR （1501★，Apache-2.0） |
| 出品方 | Uber |
| 覆盖节点 | **N7 为主（MCP 工具调用）**，含 N3/N4/N5 部分 |
| 规模 | ADR-Bench：**302 任务 × 17 种攻击技术 × 133 个 MCP 服务器** |
| 核验方式 | GitHub API 直接查询，Uber 生产 10 个月部署（>7200 主机、>10000 会话/日） |
| License | Apache-2.0 |
| 评分指标 | **F1（precision / recall）**，与 ALRPHFS / GuardAgent / LlamaFirewall 三个基线对比；Uber 凭证检测模块上报 97.2% precision（针对凭证泄露分类任务） |
| 特点 | 既是攻击基准，也是配套检测器；对 MCP 生态覆盖最全 |

**接入使用**：

```bash
git clone https://github.com/uber/ADR
cd ADR
# 详见 README，包含 sensor / explorer / detector 三个模块
# ADR-Bench 用例放在 benchmarks 目录，可独立跑
```

**建议**：如果你的 Agent 走 MCP 协议，这是必测项；同时能作为检测器基线。

---

## 三、辅助工具（非 Benchmark 但对"按节点攻击"有用）

### PyRIT (Microsoft)

| 项 | 内容 |
|---|---|
| 名称 | Python Risk Identification Tool for generative AI |
| 仓库 | https://github.com/microsoft/PyRIT （**4361★**，MIT） |
| 类型 | 攻击生成 / 红队框架（非 benchmark） |
| 覆盖节点 | 全节点：可对 N1/N3/N5/N7 分别生成攻击 |
| 用途 | 当上述 benchmark 不够用、要针对自研 Agent 的**特定节点**自定义攻击时使用；可跑批评分 |
| 接入 | `pip install pyrit`，配 `Target + Attack + Scorer` 三段式 |

---

## 四、推荐组合（按 Agent 类型）

| 你的 Agent 类型 | 主评测 | 补充评测 | 有害能力基线 | 红队工具 |
|---|---|---|---|---|
| 通用工具型（RAG + Tool Use） | **AgentDojo** | AgentDyn + InjecAgent/BIPIA | AgentHarm | PyRIT |
| MCP / 企业 Agent | **AgentDojo + Uber ADR** | Agent-SafetyBench | AgentHarm | PyRIT |
| 网页 / Computer-Use | **WASP** | AgentDojo | AgentHarm | PyRIT |
| 多领域综合 | **AgentDojo + AgentDyn** | Agent-SafetyBench + InjecAgent/BIPIA | AgentHarm | PyRIT |

---

## 五、评分指标横向对照

| Benchmark | 主指标 | 分项维度 | 是否含"效用" |
|---|---|---|---|
| AgentDojo | Utility under attack / Security Score (1−ASR) | 4 套件 × N 种攻击 | ✅ |
| AgentDyn | Utility / ASR / Over-defense | 3 套件 × N 种攻击 | ✅ |
| Agent-SafetyBench | Safety Score (0–100) | 8 类风险 × 10 类失败模式 | ❌（仅安全） |
| InjecAgent | ASR-valid / ASR-all | 2 类攻击目标 | ❌ |
| BIPIA | ASR | 5 类任务模板 | ❌ |
| AgentHarm | Refusal Rate / Harm Score | 11 类危害 | ❌ |
| WASP | Partial ASR / Full ASR | 端到端多目标 | ✅ |
| ADR-Bench | F1 / Precision / Recall | 17 种攻击技术 | ❌ |

---

## 六、已核验但**撤回**的项（记录用，避免下次重复调研）

| 名称 | 撤回原因 |
|---|---|
| AgentAttacker | arXiv 全字符串搜索仅 1 条不相关命中，GitHub 无匹配仓库，无法核实存在 |
| BAG (Benchmark for Agentic Guardrails) | arXiv 无结果，GitHub 无对应仓库 |

---

## 七、快速上手建议（一周落地方案）

1. **Day 1–2**：装 AgentDojo，把你的 Agent 包装成 `AgentPipeline` 接入，跑通 workspace 套件。
2. **Day 3**：跑 AgentDojo 全部 4 套件 × 全部 attack，拿到基础分。按套件（对应节点 N3/N4/N7）和攻击类型分别看分。
3. **Day 4**：接 InjecAgent（或 BIPIA），专项复测间接注入。
4. **Day 5**：接 AgentDyn（同一脚本），复测动态环境下的表现。
5. **Day 6**：接 AgentHarm，出有害能力基线。
6. **Day 7**：如涉及 MCP，接 Uber ADR-Bench；如涉及网页，接 WASP。汇总报告。

---

## 八、参考文献

- AgentDojo: Debenedetti et al., NeurIPS 2024, arXiv:2406.13352
- AgentDyn: Li et al., arXiv:2602.03117 (2026)
- Agent-SafetyBench: Zhang et al., arXiv:2412.14470
- InjecAgent: Zhan et al., ACL 2024 Findings, arXiv:2403.02691
- BIPIA: Yi et al., KDD 2025, arXiv:2312.14197
- AgentHarm: Andriushchenko et al., ICLR 2025, arXiv:2410.09024
- WASP: Evtimov et al., arXiv:2504.18575
- Uber ADR: Li et al., MLSys 2026, arXiv:2605.17380
- OWASP Top 10 for Agentic Applications 2026 (Dec 2025)

---

*文档核验方式：所有仓库通过 GitHub API 或直接访问 README 校验，所有论文通过 arXiv 摘要页校验；星数、License、代码路径均为核验时的实测值。*

---

## 九、修订记录

| 日期 | 修订内容 |
|---|---|
| 2026-08-27 | 初版成文 |
| 2026-08-27 | 二轮复核：① 更正 InjecAgent 接入命令（`evaluate_prompted_agent.py` + `export PYTHONPATH=.`）；② 更正 Agent-SafetyBench 接入命令（`evaluation/eval.sh` + `score/eval_with_shield.sh`，评分用官方 ShieldAgent）；③ 更正 AgentHarm 接入命令（按 HF 官方 README 的 `inspect_eval.py` 用法）；④ 新增 BIPIA（KDD 2025，`microsoft/BIPIA`）；⑤ 章节重新编号 |
| 2026-08-27 | 三轮复核：① 更正 AgentDojo 评分指标名称（Security Score = 1−ASR，而非 ASR；完整攻击名确认为 `important_instructions`/`tool_knowledge`/`ignore_previous`/`direct`/`system_message`/`injecagent`）；② 去除 Agent-SafetyBench N5 覆盖的未核实声明（保守标注）；③ 补全 AgentHarm 接入步骤（需先 `git clone inspect_evals`）；④ 限定 Uber ADR 97.2% precision 的适用范围（凭证检测任务，非整体指标） |
