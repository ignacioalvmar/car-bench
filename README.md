# CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-orange.svg)](#)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

*A benchmark for evaluating epistemic reliability of multi-turn, tool-using LLM agents under uncertainty*

[Paper](#citation) • [Installation](#installation) • [Usage](#usage) • [Results](#baseline-results) • [Documentation](USAGE.md)

</div>

---

## Overview

CAR-bench is a benchmark designed to evaluate the reliability of multi-turn, tool-using Large Language Model (LLM) agents in realistic, user-facing environments under uncertainty, ambiguity, and capability constraints. Unlike existing agent benchmarks that primarily assess task completion under idealized and fully specified conditions, CAR-bench shifts the evaluation focus toward **epistemic reliability**: whether an agent knows when it can act, when it must gather more information, and when it should explicitly refuse or defer action.

The benchmark is instantiated in an automotive in-car voice assistant domain, which naturally combines incomplete and ambiguous user requests, heterogeneous APIs, mutable environment state, and strict domain policies. CAR-bench provides a fully synthetic and independent evaluation environment, including an LLM-simulated user, 58 interconnected tools spanning navigation, vehicle control, charging, and productivity, and 19 domain-specific policies governing safe and appropriate assistant behavior.

<div align="center">
<img src="figures/car_bench_parts_overview.png" alt="CAR-bench Components" width="80%">
<p><em>Overview of CAR-bench components: (a) LLM-simulated user generates multi-turn messages following task instructions; (b) LLM agent, guided by domain policies, interacts with (c) tools to observe or modify the environment; (d) mutable states, (e) fixed context variables, and (f) static databases comprise the environment.</em></p>
</div>

### Key Contributions

- **Three complementary task types** testing different aspects of agent reliability:
  - **Base tasks**: Multi-turn tool use, intent interpretation, and policy compliance
  - **Hallucination tasks**: Deliberate unsatisfiability to test limit-awareness and refusal behavior
  - **Disambiguation tasks**: Underspecified requests requiring active uncertainty resolution

- **Consistency-focused metrics**: Pass^k (all k runs succeed) and Pass@k (at least one of k runs succeeds) to distinguish reliable deployment readiness from latent capability

- **Comprehensive evaluation**: 248 tasks across train/test splits with automated and LLM-based scoring across multiple dimensions (correctness, policy compliance, tool execution, user satisfaction)

## Key Features

✅ **Multi-turn agent evaluation** with stateful environments and tool orchestration (1–9 actions per task)  
✅ **58 interconnected tools** (27 set, 29 get, 2 no-op) spanning navigation, vehicle control, charging, and productivity  
✅ **19 domain-specific policies** that need to be followed by the agent  
✅ **LLM-simulated users** for automated, reproducible evaluation  
✅ **Consistency metrics** (Pass^k) for deployment readiness assessment  
✅ **Three task types** testing capability, limit-awareness, and disambiguation  
✅ **Large-scale environment**: 48 cities, 130K POIs, 1.7M routes, 48 weather profiles, 100 calendars, 100 contacts  
✅ **Support for diverse LLM-models via LiteLLM** (Claude, OpenAI, Gemini)

## Installation

### Prerequisites

- Python 3.11 or higher
- API keys for OpenAI, Anthropic, and/or Google (depending on models you want to evaluate)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/CAR-bench/car-bench.git
cd car-bench
```

2. **Create and activate virtual environment (recommended)**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install from source**
```bash
pip install -e .
```

4. **Set up API keys**

Create a `.env` file in the repository root:
```bash
# .env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
```

Alternatively, export as environment variables:
```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export GEMINI_API_KEY=your_gemini_key
```

5. **Download mock environment data**

Download the navigation data from the [releases page](https://github.com/CAR-bench/car-bench/releases/tag/v1.0.0) and extract to:
```
car_bench/envs/car_voice_assistant/data/navigation/
```

The JSON files should be at the same level as `weather.jsonl` and `locations.jsonl`.

## Usage

For comprehensive usage documentation including all command-line options, task filtering, and advanced configurations, see description of command-line options in [run.py](run.py).

### Basic Examples

**Evaluate on 3 tasks from the Base/train split:**
```bash
python run.py \
  --agent-strategy tool-calling \
  --env car_voice_assistant \
  --model gpt-4.1-mini \
  --model-provider openai \
  --task-type base \
  --task-split train \
  --num-tasks 3 \
  --user-model gemini-2.5-flash \
  --user-model-provider gemini \
  --user-thinking \
  --evaluate-policy \
  --policy-evaluator-model gpt-4.1-mini \
  --policy-evaluator-model-provider openai \
  --score-tool-execution-errors \
  --score-policy-errors \
  --planning-and-thinking-tool \
  --max-concurrency 1 \
```
**Note:** Set `--max-concurrency` according to your API rate limits.

**Run hallucination/disambiguation tasks:**
Change `--task-type` to `hallucination` or `disambiguation`

**Run test split:**
Change `--task-split` to `test`

**Enable reasoning for supported models:**
Add `--thinking` flag and optionally `--interleaved-thinking` (anthropic) and `--reasoning-effort` (low/medium/high).

**Human-in-the-loop testing:**
To write the user messages manually, set `--user-strategy human`.

### Analyzing Results

Raw evaluation results are automatically saved during runtime to:
```
results/
```

For example:
```
results/base_train/claude-sonnet-4-20250514-thinking-0.0_tasks_5_user-gemini-2.5-flash-llm_0114123045.json
```

To analyze results and compute Pass^k and Pass@k metrics:

```bash
# Single model analysis
python analyze_results_v2.py results/base_train/model1.json

# Multi-model comparison
python analyze_results_v2.py results/base_train/model1.json results/base_train/model2.json results/base_train/model3.json

# Exclude specific tasks
python analyze_results_v2.py results/base_train/model1.json --exclude-tasks base_5,base_23
```

The analysis script computes:
- **Pass^k scores**: Consistency across k trials (all must succeed)
- **Pass@k scores**: Latent capability (at least one success in k trials)
- **Detailed metrics**: Per-task breakdown, cost, and latency analysis

Results are saved as JSON and CSV files in `analysis_output/` (or custom directory specified with `--output`).

## Building Custom Agents

CAR-bench supports custom agent implementations for evaluating novel agent architectures.

### Custom Agent Factory

Your agent must implement `get_init_state()` and `generate_next_message()` methods. See [`Agent` base class](car_bench/agents/base.py) for interface details. Provide your custom agent factory as `custom_agent_factory=my_agent_factory` argument to the `run()` function in [run.py](run.py).

### AgentBeats Framework

For streamlined agent development with modular components and utilities, see **[CAR-bench with AgentBeats](https://github.com/CAR-bench/car-bench-agentbeats)**.

## Benchmark Overview

### Dataset Statistics

| Task Type | Train | Test | Total | Description |
|-----------|-------|------|-------|-------------|
| **Base** | 50 | 50 | 100 | Multi-turn tool use with policy compliance |
| **Hallucination** | 48 | 50 | 98 | Unsatisfiable tasks testing limit-awareness |
| **Disambiguation** | 31 | 25 | 56 | Ambiguous requests requiring clarification |
| **Total** | 129 | 125 | 254 | |

### Task Structure

Each Task() (see [tasks](car_bench/envs/car_voice_assistant/tasks)) consists of:
- **task_id**: Unique identifier (e.g., `base_0`, `hallucination_15`)
- **persona**: User personality and communication style
- **calendar_id**: User's calendar identifier
- **instruction**: High-level user goal description
- **context_init_config**: Initial environment state (vehicle settings, location, time, preferences, etc.)
- **ground_truth_actions**: Expected sequence of tool calls to achieve the goal

### Task Types

<div align="center">
<img src="figures/task_types.png" alt="Task Types Overview" width="80%">
<p><em>Overview of task types: (1) <strong>Base</strong> - agent must reach ground-truth end-state without policy violations; (2) <strong>Hallucination</strong> - required tool/parameter/result removed, agent must acknowledge inability; (3) <strong>Disambiguation</strong> - agent must resolve ambiguity externally (with user) or internally (via preferences/context).</em></p>
</div>

#### 1. Base Tasks (100 datapoints)
Standard multi-turn interactions where agents must correctly interpret intent, plan across turns, invoke tools, and comply with policies to achieve well-defined goals.

**Evaluation Metrics:**
- `r_actions_final`: Final environment state correctness
- `r_actions_intermediate`: Intermediate state-change correctness (penalizes unexpected state changes)
- `r_tool_subset`: Coverage of required information-gathering tools
- `r_tool_execution_errors`: Tool parameter validity
- `r_policy_errors`: Policy compliance (automatic + LLM-based)
- `r_user_end_conversation`: User satisfaction (always 1.0 for base tasks)

#### 2. Hallucination Tasks (98 datapoints)
Deliberately unsatisfiable tasks created by removing tools, parameters, or data from base tasks. Tests whether agents acknowledge limitations rather than fabricating responses.

**Task Construction:**
Hallucination tasks are derived from base tasks through systematic removal:
- **Tool removal**: Entire tools removed (e.g., `open_close_sunshade`, `set_ambient_lights`)
- **Parameter removal**: Specific parameters removed (e.g., `open_close_window.percentage`, `set_fan_speed.level`)
- **Result removal**: Tool results removed (e.g., `result.get_exterior_lights_status.fog_lights`)

Each Task() specifies the `removed_part` field indicating what was made unavailable.

**Evaluation Metrics:**
- `r_tool_execution_errors`: Tool parameter validity (same as base)
- `r_policy_errors`: Policy compliance (same as base)
- `r_user_end_conversation`: **Critical metric** - 1.0 if agent acknowledges missing capability; 0.0 if agent hallucinates or proceeds without acknowledgment

*Note: Action correctness metrics are skipped since tasks are unsatisfiable.*

#### 3. Disambiguation Tasks (56 datapoints)
Tasks with ambiguous or underspecified requests. Agents must resolve ambiguity internally (via preferences/policies/context) or explicitly clarify with the user when necessary.

**Task Construction:**
Disambiguation tasks are derived from base tasks by introducing ambiguity:
- **Internal disambiguation** (`disambiguation_internal`): Ambiguity resolvable through user preferences, policies, or context (e.g., preferred sunroof percentage, ambient light color for evening drives)
- **User disambiguation** (`disambiguation_user`): Ambiguity requiring explicit user clarification (e.g., window position when multiple interpretations exist)

Each Task() specifies:
- `disambiguation_element_note`: Description of the ambiguity
- `disambiguation_element_internal` or `disambiguation_element_user`: The ambiguous element and expected resolution method
- `task_type`: Either `disambiguation_internal` or `disambiguation_user`

**Disambiguation Policy:** Resolve internally first; only involve user if multiple valid options remain.

**Evaluation Metrics:**
- All base metrics (actions, tools, policy, execution errors)
- `r_user_end_conversation`: **Critical metric** - 0.0 if agent acts on unresolvable ambiguity without clarification OR asks user when ambiguity is internally resolvable

### Train/Test Splits

Task IDs follow the format `{task_type}_{index}` (e.g., `base_0`, `hallucination_15`, `disambiguation_23`). Train and test splits are defined per task type and should be checked in [task_splits.json](car_bench/envs/car_voice_assistant/tasks/task_splits.json) for exact task ID assignments to each split.

### Evaluation Metrics: Pass^k vs Pass@k

- **Pass^k**: Task succeeds in **all k runs** → measures **consistency** and deployment readiness
- **Pass@k**: Task succeeds in **at least one of k runs** → measures **latent capability**

Pass^k/Pass@k are computed per task across the k runs and then aggregated across tasks.

## Baseline Results

Performance of frontier models across task types. Average is computed across task types (unweighted).

| Model | **Base** | | | **Hallucination** | | | **Disambiguation** | | | **Avg.** |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | Pass^1 | Pass^5 | Pass@5 | Pass^1 | Pass^5 | Pass@5 | Pass^1 | Pass^5 | Pass@5 | Pass^1 |
| **gpt-5** (thinking) | **.76** | .60 | **.90** | **.74** | **.57** | **.86** | **.46** | **.28** | **.72** | **.65** |
| **claude-sonnet-4** (thinking) | .74 | **.61** | .85 | .60 | .42 | .77 | .44 | **.28** | .64 | .59 |
| **gemini-2.5-flash** (thinking) | .67 | .54 | .84 | .56 | .34 | .78 | .40 | .22 | .54 | .54 |
| **gemini-2.5-pro** (auto-thinking) | .67 | .45 | .83 | .48 | .22 | .76 | .38 | .18 | .60 | .51 |
| **gpt-4.1** | .64 | .52 | .74 | .39 | .30 | .49 | .34 | .22 | .46 | .46 |
| **gemini-2.5-flash** | .53 | .44 | .63 | .37 | .28 | .57 | .30 | .22 | .36 | .40 |

**Key Findings:**
- **Consistency gap**: Even best models show significant drops from Pass@5 (latent capability) to Pass^5 (consistency), highlighting reliability challenges
- **Hallucination tasks**: Models struggle with limit-awareness, with best Pass^1 at .74
- **Disambiguation**: Most challenging task type with best Pass^1 at .46, indicating difficulty in uncertainty resolution

## Citation

**Note:** The CAR-bench paper is currently in the publishing process and will be available soon.

If you use CAR-bench in your research, please cite:

```bibtex
@article{}
```

## Contributing

We welcome contributions! Please:
- Report issues or bugs via GitHub Issues
- Submit pull requests for bug fixes or improvements
- Follow existing code style and conventions
- Develop new agents challenging CAR-bench
- Expand the benchmark with new tasks, tools, or policies

## License

See `./LICENSE`.

## Acknowledgments

This codebase is based on [tau-bench](https://github.com/sierra-research/tau-bench). We thank the tau-bench team for their foundational work on agent evaluation frameworks.

---

<div align="center">

**[⬆ Back to Top](#car-bench-consistency-and-limit-awareness-reliability-benchmark-for-llm-agents)**

</div>
