import json
from pathlib import Path
from typing import Optional, Union

from car_bench.envs.base import Env
from car_bench.envs.car_voice_assistant.mock_data import load_data
from car_bench.envs.car_voice_assistant.tools import (
    ALL_TOOLS,
)
from car_bench.envs.car_voice_assistant.wiki import WIKI
from car_bench.envs.policy_evaluator import PolicyEvaluatorStrategy
from car_bench.envs.user.user import UserStrategy


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
        # Load tasks based on task_type
        match task_type:
            case "base":
                from car_bench.envs.car_voice_assistant.tasks.tasks_base import (
                    TASKS as all_tasks,
                )
            case "hallucination":
                from car_bench.envs.car_voice_assistant.tasks.tasks_hallucination import (
                    TASKS as all_tasks,
                )
            case "disambiguation":
                from car_bench.envs.car_voice_assistant.tasks.tasks_disambiguation import (
                    TASKS as all_tasks,
                )
            case _:
                raise ValueError(f"Unknown task type: {task_type}")

        # Filter tasks based on task_split
        tasks_dir = Path(__file__).parent / "tasks"
        splits_file = tasks_dir / "task_splits.json"
        
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        split_key = f"{task_type}_{task_split}"
        if split_key not in splits:
            raise ValueError(f"Unknown split combination: {split_key}")
        
        # Get the task IDs for this split
        split_task_ids = set(splits[split_key])
        
        # Filter tasks to only include those in the split
        tasks = [task for task in all_tasks if task.task_id in split_task_ids]

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
