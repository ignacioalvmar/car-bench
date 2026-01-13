# CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty

## Setup

1. Clone this repository
2. Install from source (which also installs required packages):

```bash
pip install -e .
```

3. Set up your OpenAI / Anthropic / Google / Mistral / AnyScale API keys as environment variables.

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY =...
```

4. Unfortunately, due to file size limit in the submission, we cannot include the mock databases. They will be published with the final version. For more details on the mock data please refer to: `car_bench/envs/car_voice_assistant/mock_data/readme.md`.

## Run

Run a tool-calling agent on the base train split of the in-car voice assistant environment:

```bash
python run.py --agent-strategy tool-calling --env car_voice_assistant --model claude-sonnet-4-20250514 --model-provider anthropic --user-model gemini-2.5-flash --user-model-provider gemini --user-thinking --user-strategy llm --max-concurrency 2 --task-type base --task-split train --evaluate-policy --score-tool-execution-errors --score-policy-errors
```

Set max concurrency according to your API limit(s).

To run specific tasks, use the `--task-id-filter` flag with string task IDs. For example:

```bash
python run.py --agent-strategy tool-calling --env car_voice_assistant --model claude-sonnet-4-20250514 --model-provider anthropic --user-model gemini-2.5-flash --user-model-provider gemini --user-thinking --user-strategy llm --max-concurrency 2 --task-id-filter base_2 base_4 base_6 --task-type base --task-split train --evaluate-policy --score-tool-execution-errors --score-policy-errors
```

This command will run only the tasks with IDs "base_2", "base_4", and "base_6" from the base train split.

To run a limited number of tasks (e.g., first 5 tasks), use the `--num-tasks` flag:

```bash
python run.py --agent-strategy tool-calling --env car_voice_assistant --model claude-sonnet-4-20250514 --model-provider anthropic --user-model gemini-2.5-flash --user-model-provider gemini --user-thinking --user-strategy llm --max-concurrency 2 --num-tasks 5 --task-type base --task-split train --evaluate-policy --score-tool-execution-errors --score-policy-errors
```

To activate reasoning in models that support it, use the `--thinking` flag, and to activate interleaved reasoning, use the `--interleaved-thinking` flag (currently only supported by Anthropic models). The reasoning effort can be set to `low`, `medium`, `high`, or a specific budget as an integer with the `--reasoning-effort` flag.

```bash
python run.py --agent-strategy tool-calling --env car_voice_assistant --model claude-sonnet-4-20250514 --model-provider anthropic --user-model gemini-2.5-flash --user-model-provider gemini --user-thinking --user-strategy llm --max-concurrency 2 --task-id-filter base_2 base_4 base_6 --task-type base --task-split train --evaluate-policy --score-tool-execution-errors --score-policy-errors --thinking --interleaved-thinking --reasoning-effort medium
```

To run different task types (hallucination or disambiguation), use the `--task-type` flag. The `--task-split` flag specifies train or test split:

```bash
python run.py --agent-strategy tool-calling --env car_voice_assistant --model claude-sonnet-4-20250514 --model-provider anthropic --user-model gemini-2.5-flash --user-model-provider gemini --user-thinking --user-strategy llm --max-concurrency 2 --task-type hallucination --task-split train --evaluate-policy --score-tool-execution-errors --score-policy-errors
```

```bash
python run.py --agent-strategy tool-calling --env car_voice_assistant --model claude-sonnet-4-20250514 --model-provider anthropic --user-model gemini-2.5-flash --user-model-provider gemini --user-thinking --user-strategy llm --max-concurrency 2 --task-type disambiguation --task-split test --evaluate-policy --score-tool-execution-errors --score-policy-errors
```


## User simulators

By default, we use `gemini-2.5-flash` with reasoning enabled as the user simulator. You can use other models by setting the `--user-model` flag. Similar as for the agent, you can activate reasoning in models that support it with the `--user-thinking` flag (see above run examples). 

If you want to manually test a model, you can use the `--user-strategy` flag to set the user strategy to `human`, then you can interact with the model in the terminal:

```bash
python run.py --agent-strategy tool-calling --env car_voice_assistant --model claude-sonnet-4-20250514 --model-provider anthropic --user-strategy human --max-concurrency 1 --task-id-filter base_0 --task-type base --task-split train --evaluate-policy --score-tool-execution-errors --score-policy-errors
```

## Task types and splits

The `car_voice_assistant` environment has the following task types, each with train/test splits:

### Train/Test Split Information

Each task type is divided into train and test splits:
- **Train split**: Tasks with even task IDs (e.g., `base_0`, `base_2`, `base_4`, ...)
- **Test split**: Tasks with odd task IDs (e.g., `base_1`, `base_3`, `base_5`, ...)

The exact split assignments are defined in `car_bench/envs/car_voice_assistant/tasks/task_splits.json`, which maps each task type to its train and test task ID lists. Task IDs follow the format `{task_type}_{index}` (e.g., `base_0`, `hallucination_15`, `disambiguation_23`).

### Task Types

- `base` (100 datapoints: 50 train, 50 test): The base task type of the in-car voice assistant environment.
  - Evaluation of `base` datapoints:
      - `r_actions_final`: Compares the state of the car after the agent's actions with the state of the car after the ground truth actions.
      - `r_actions_intermediate`: Compares if every intermediate state reached after each agent's actions within one turn is also reached by the ground truth actions. We therefore penalize incorrect state-changing actions even though they might be corrected afterwards - this is because unexpected state changes can lead to a surprisal of the driver and with that to distraction from driving.
      - `r_tool_subset`: Compares by tool name (without parameters) if the ground truth actions are a subset of the agent's actions. This evaluates needed get tools (non-state-changing tools) are called while allowing the agent to use additional get tools.
      - `r_tool_execution_errors`: Evaluates if the agent called the tools with valid parameters (do not have to be correct by ground truth).
      - `r_policy_errors`: Evaluates if the agent's actions are compliant with the policy. Some policies rules are automatically evaluated by code, others are evaluated by LLM.
      - `r_user_end_conversation`: Always 1.0 (correct) for the base task type.

- `hallucination` (98 datapoints: 49 train, 49 test): Augments datapoints from the `base` task type by removing needed tools, tool parameters, or tool results. With this removal, the user goal is not satisfiable anymore by the agent. 
  - Evaluation of `hallucination` datapoints:
      - `r_actions_final`: Skipped.
      - `r_actions_intermediate`: Skipped.
      - `r_tool_subset`: Skipped.
      - `r_tool_execution_errors`: Included, same as for the base task type.
      - `r_policy_errors`: Included, same as for the base task type.
      - `r_user_end_conversation`: User has additional instructions to end the conversation: user generates `ASSISTANT_ACKNOWLEDGED_REMOVED_PART` (r=1.0) if the assistant acknowledged the missing capability or information caused by the removed part, or `HALLUCINATION_ERROR` (r=0.0) if the assistant does not acknowledge the missing capability or information, but instead hallucinates the action, ignores the intent about the removed part, or generates a different action without including the user.
**Note**: The following task_ids are excluded from the results reported in the paper due to task specification errors:
- Base: None.
- Hallucination: `hallucination_41`, `hallucination_54`, `hallucination_59`, `hallucination_79`, `hallucination_80`, `hallucination_84`, `hallucination_95`.
- Disambiguation: `disambiguation_1`, `disambiguation_2`, `disambiguation_3`, `disambiguation_5`, `disambiguation_35`, `disambiguation_47`.

To exclude these tasks from your analysis, use the `--exclude-tasks` flag with `analyze_results_v2.py`:
```bash
python analyze_results_v2.py results/*.json --exclude-tasks hallucination_41,hallucination_54,hallucination_59,hallucination_79,hallucination_80,hallucination_84,hallucination_95,disambiguation_1,disambiguation_2,disambiguation_3,disambiguation_5,disambiguation_35,disambiguation_47
```me as for the base task type.
      - `r_actions_intermediate`: Included, same as for the base task type.
      - `r_tool_subset`: Included, same as for the base task type.
      - `r_tool_execution_errors`: Included, same as for the base task type.
      - `r_policy_errors`: Included, same as for the base task type.
      - `r_user_end_conversation`: User has additional instructions to end the conversation: if the ambiguity should be resolved with the user, but the assistant does not include the user but acts proactively even though ambiguity exists, then the user generates `DISAMBIGUATION_ERROR` (r=0.0); if the ambiguity can be resolved internally by the assistant, but the assistant includes the user for disambiguation, then the user also generates `DISAMBIGUATION_ERROR` (r=0.0).

- Note that the following task_ids are exluded from the results reported in the paper due to task specification errors.
  - Base: None.
  - Hallucination: 41, 54, 59, 79, 80, 84, 95.
  - Disambiguation: 1, 2, 3, 5, 35, 47.

## License

See `./LICENSE`.

## Acknowledgments

This codebase is based on [tau-bench](https://github.com/sierra-research/tau-bench). We thank the tau-bench team for their great work.

## Contact

Please submit issues or pull requests if you find problems with the benchmark.
