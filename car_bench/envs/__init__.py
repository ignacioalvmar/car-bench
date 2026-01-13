from typing import Optional, Union

from car_bench.envs.base import Env
from car_bench.envs.user.user import UserStrategy


def get_env(
    env_name: str,
    user_strategy: Union[str, UserStrategy],
    user_model: str,
    task_type: str,
    task_split: str = "train",
    user_provider: Optional[str] = None,
    task_index: Optional[int] = None,
    policy_evaluator_provider: Optional[str] = None,
    policy_evaluator_strategy: Optional[Union[str, UserStrategy]] = None,
    policy_evaluator_model: Optional[str] = None,
    evaluate_policy: Optional[bool] = False,
    score_tool_execution_errors: Optional[bool] = False,
    score_policy_errors: Optional[bool] = False,
    use_user_as_a_tool_tools: Optional[bool] = False,
    user_thinking: bool = False,
) -> Env:
    if env_name == "car_voice_assistant":
        from car_bench.envs.car_voice_assistant import MockCarVoiceAssistantDomainEnv

        return MockCarVoiceAssistantDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            policy_evaluator_strategy=policy_evaluator_strategy,
            policy_evaluator_model=policy_evaluator_model,
            task_type=task_type,
            task_split=task_split,
            user_provider=user_provider,
            policy_evaluator_provider=policy_evaluator_provider,
            user_thinking=user_thinking,
            task_index=task_index,
            evaluate_policy=evaluate_policy,
            score_tool_execution_errors=score_tool_execution_errors,
            score_policy_errors=score_policy_errors,
            use_user_as_a_tool_tools=use_user_as_a_tool_tools,
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
