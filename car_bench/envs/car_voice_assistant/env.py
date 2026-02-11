import json
from typing import List, Optional, Union

from car_bench.envs.base import Env
from car_bench.envs.car_voice_assistant.mock_data import load_data
from car_bench.envs.car_voice_assistant.tools import (
    ALL_TOOLS,
)
from car_bench.envs.car_voice_assistant.wiki import WIKI
from car_bench.envs.policy_evaluator import PolicyEvaluatorStrategy
from car_bench.envs.user.user import UserStrategy
from car_bench.types import Action, Task


# Default HuggingFace dataset repo ID
HF_DATASET_REPO_ID = "johanneskirmayr/car-bench-dataset"


def _load_tasks(
    task_type: str,
    task_split: str,
    repo_id: str = HF_DATASET_REPO_ID,
) -> List[Task]:
    """Load tasks from HuggingFace dataset repo.

    The HF dataset has configs like 'tasks_base', 'tasks_disambiguation',
    'tasks_hallucination' with train/test splits already separated.
    Fields 'context_init_config', 'actions', and 'removed_part' are stored
    as JSON strings and need to be parsed back.
    """
    from datasets import load_dataset

    config_name = f"tasks_{task_type}"
    print(f"Loading tasks from HuggingFace: {repo_id} / {config_name} / {task_split}")
    ds = load_dataset(repo_id, config_name, split=task_split)

    tasks = []
    for row in ds:
        d = dict(row)
        # Parse JSON string fields back to Python objects
        d["context_init_config"] = json.loads(d["context_init_config"])
        d["actions"] = [
            Action(**a) for a in json.loads(d["actions"])
        ]
        if d.get("removed_part") is not None:
            d["removed_part"] = json.loads(d["removed_part"])
        tasks.append(Task(**d))

    print(f"Loaded {len(tasks)} tasks from HuggingFace ({config_name}/{task_split})")
    return tasks


class MockCarVoiceAssistantDomainEnv(Env):
    def __init__(
        self,
        user_strategy: Union[str, UserStrategy] = UserStrategy.LLM,
        policy_evaluator_strategy: Union[
            str, UserStrategy
        ] = PolicyEvaluatorStrategy.LLM,
        user_model: str = "gpt-4.1-mini",
        policy_evaluator_model: str = "gpt-4.1-mini",
        user_provider: Optional[str] = None,
        policy_evaluator_provider: Optional[str] = None,
        user_thinking: bool = False,
        task_type: str = "base",
        task_split: str = "train",
        task_index: Optional[int] = None,
        evaluate_policy: Optional[bool] = False,
        score_tool_execution_errors: Optional[bool] = False,
        score_policy_errors: Optional[bool] = False,
        use_user_as_a_tool_tools: bool = False,
    ):
        # Load tasks from HuggingFace
        tasks = _load_tasks(task_type, task_split)

        all_tools_plus_user_as_a_tool = None
        wiki_with_user_as_a_tool = None
        if use_user_as_a_tool_tools:
            all_tools_plus_user_as_a_tool = ALL_TOOLS_PLUS_USER_AS_A_TOOL
            wiki_with_user_as_a_tool = WIKI

        super().__init__(
            data_load_func=load_data,
            tools=all_tools_plus_user_as_a_tool or ALL_TOOLS,
            tasks=tasks,
            wiki=wiki_with_user_as_a_tool or WIKI,
            user_strategy=user_strategy,
            user_model=user_model,
            user_thinking=user_thinking,
            policy_evaluator_strategy=policy_evaluator_strategy,
            policy_evaluator_model=policy_evaluator_model,
            user_provider=user_provider,
            policy_evaluator_provider=policy_evaluator_provider,
            task_index=task_index,
            evaluate_policy=evaluate_policy,
            score_tool_execution_errors=score_tool_execution_errors,
            score_policy_errors=score_policy_errors,
        )
        self.terminate_tools = ["call_phone_by_number"]
