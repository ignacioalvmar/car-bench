import asyncio
import json
import random
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from car_bench.envs.policy_evaluator import (
    PolicyEvaluatorStrategy,
    load_policy_evaluator,
    policy_errors_during_runtime,
)
from car_bench.envs.tool import Tool
from car_bench.envs.tool_execution_error_evaluator import (
    tool_execution_errors_during_runtime,
)
from car_bench.envs.user.user import UserStrategy, load_user
from car_bench.envs.user.user_end_conversation import end_conversation_failure
from car_bench.types import (
    RESPOND_ACTION_NAME,
    USER_AS_A_TOOL_ACTION_NAMES,
    Action,
    EnvInfo,
    EnvResetResponse,
    EnvResponse,
    RewardInfo,
    RewardResult,
    Task,
    TaskType,
)

ToHashable = Union[
    str, int, float, Dict[str, "ToHashable"], List["ToHashable"], Set["ToHashable"]
]
Hashable = Union[str, int, float, Tuple["Hashable"], Tuple[Tuple[str, "Hashable"]]]


def to_hashable(item: ToHashable) -> Hashable:
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item


def consistent_hash(
    value: Hashable,
) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()


class Env(object):
    def __init__(
        self,
        data_load_func: Callable[[], Dict[str, Any]],
        tools: List[Type[Tool]],
        tasks: List[Task],
        wiki: str,
        user_strategy: Union[str, UserStrategy],
        user_model: str,
        user_provider: Optional[str] = None,
        user_thinking: bool = False,
        task_index: Optional[int] = None,
        policy_evaluator_strategy: Optional[Union[str, PolicyEvaluatorStrategy]] = None,
        policy_evaluator_model: Optional[str] = None,
        policy_evaluator_provider: Optional[str] = None,
        evaluate_policy: Optional[bool] = False,
        score_tool_execution_errors: Optional[bool] = False,
        score_policy_errors: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.data_load_func = data_load_func
        self.data = data_load_func()
        self.tools_map: Dict[str, Type[Tool]] = {
            tool.get_info()["function"]["name"]: tool for tool in tools
        }
        self.tools_info = [tool.get_info() for tool in tools]
        self.terminate_tools = []
        self.tasks = tasks
        if task_index is not None:
            self.task_index = task_index
        else:
            self.task_index = random.randint(0, len(tasks) - 1)
        self.task = tasks[self.task_index]
        self.wiki = wiki
        self.user = load_user(
            user_strategy=user_strategy,
            model=user_model,
            provider=user_provider,
            user_thinking=user_thinking,
        )
        self.policy_evaluator = load_policy_evaluator(
            policy_evaluator_strategy=policy_evaluator_strategy,
            model=policy_evaluator_model,
            provider=policy_evaluator_provider,
        )
        self.actions: List[Action] = []
        self.state_hashes: List[str] = []  # Track intermediate state hashes
        self.evaluate_policy = evaluate_policy
        self.score_tool_execution_errors = score_tool_execution_errors
        self.score_policy_errors = score_policy_errors

    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        if task_index is None:
            task_index = random.randint(0, len(self.tasks))
        self.task_index = task_index
        self.data = self.data_load_func()
        self.task = self.tasks[task_index]
        self.actions = []

        # Initialize state hash tracking with initial state
        from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
            context_state,
        )

        initial_state_hash = self.get_context_state_hash(context_state.get())
        self.state_hashes = [initial_state_hash]

        initial_observation = self.user.reset(
            persona=self.task.persona,
            instruction=self.task.instruction,
            task_type=self.task.task_type,
            removed_part=self.task.removed_part,
            disambiguation_element_internal=self.task.disambiguation_element_internal,
        )
        return EnvResetResponse(
            observation=initial_observation, info=EnvInfo(task=self.task, source="user")
        )

    def _record_state_hash_if_needed(self):
        """Record state hash after tool actions that might change context state."""
        from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
            context_state,
        )

        current_state_hash = self.get_context_state_hash(context_state.get())
        self.state_hashes.append(current_state_hash)

    def step(self, action: Action, messages: List[Dict]) -> EnvResponse:
        self.actions.append(action)

        info = EnvInfo(task=self.task)
        reward = 0
        done = False
        if action.name in USER_AS_A_TOOL_ACTION_NAMES:
            pass
        elif action.name in self.tools_map:
            try:
                observation = self.tools_map[action.name].invoke(
                    data=self.data, **action.kwargs
                )
            except Exception as e:
                observation = f"Error: {e}"
            info.source = action.name
        else:
            observation = f"Unknown action {action.name}"
            info.source = action.name

        return EnvResponse(observation=observation, reward=reward, done=done, info=info)

    async def _run_action(self, action: Action, tools_map, data):
        """Helper function to run a single action asynchronously."""
        if action.name in tools_map:
            try:
                observation = tools_map[action.name].invoke(data=data, **action.kwargs)
                source = action.name
                is_terminate = action.name in self.terminate_tools
            except Exception as e:
                observation = f"Error: {e}"
                source = action.name
                is_terminate = False
                if self.task.removed_part:
                    removed_parameters = [
                        removed_part
                        for removed_part in self.task.removed_part
                        if "." in removed_part and removed_part.split(".")[0] == action.name
                    ]
                    if removed_parameters:
                        observation = f"Error: {e}. Note that this tool argument is currently removed so that this tool can not be used. Do not try to call this tool with this argument as it will result in an error."
                # tool_names_removed_part = [removed_part.split('.')[0] for removed_part in self.task.removed_part] if self.task.removed_part else []
                # if action.name in tool_names_removed_part:
                #     observation = f"Error: {e}. Note that this tool argument is currently removed so that this tool can not be used. Do not try to call this tool with this paramter as it will result in an error."
        else:
            observation = f"Unknown action {action.name}"
            source = action.name
            is_terminate = False

        return {
            "observation": observation,
            "source": source,
            "is_terminate": is_terminate,
        }

    async def steps(self, actions: List[Action], messages: List[Dict]) -> EnvResponse:
        """Run multiple actions asynchronously and return a combined response."""
        self.actions.extend(actions)

        info = EnvInfo(task=self.task)
        reward = 0
        done = False
        observations = []

        # Handle USER_AS_A_TOOL_ACTION_NAMES separately if it's the first action
        new_user_message = None
        if actions and actions[0].name in USER_AS_A_TOOL_ACTION_NAMES:
            # Record state hash after tools of one turn are performed
            self._record_state_hash_if_needed()
            new_user_message = self.user.step(actions[0].kwargs["content"])
            observations.append(new_user_message)
            info.source = "user"
            done = "###STOP###" in new_user_message
            # Do not process any other action
            remaining_actions = None
        else:
            remaining_actions = actions

        # Process remaining actions asynchronously
        if remaining_actions:
            # DEBUG
            for action in remaining_actions:
                if not action.name == "planning_tool":
                    print(f"🔧 Running tool: {action}")
            # Create tasks for all remaining actions
            tasks = [
                self._run_action(action, self.tools_map, self.data)
                for action in remaining_actions
            ]

            # Gather results
            results = await asyncio.gather(*tasks)

            for i, result in enumerate(results):
                # DEBUG
                if result["source"] == "planning_tool":
                    try:
                        plan_result = json.loads(result["observation"])
                        from car_bench.envs.car_voice_assistant.tools.cross_domain.planning import (
                            pretty_print_plan,
                        )

                        formatted_plan = pretty_print_plan(
                            plan_result["result"]["plan_details"]
                        )
                        print(f"🔧📝 Plan: {formatted_plan}")
                    except Exception as e:
                        print(f"🔧 Error parsing plan: {e}")
                else:
                    print(f"🔧 Tool outputs: {result['observation']}")
                observations.append(result["observation"])
                if result["is_terminate"]:
                    done = True

            # If there were multiple sources, combine them
            if new_user_message is not None:
                info.source = "user+" + "+".join(result["source"] for result in results)
            else:
                info.source = "+".join(result["source"] for result in results)

        if done:
            if not actions[0].name in USER_AS_A_TOOL_ACTION_NAMES:
                # Extract tool_call_ids from the last assistant message to match with tool results
                # This is required by OpenAI/litellm API - tool responses must have tool_call_id
                tool_call_ids = []
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        tool_call_ids = [tc["id"] for tc in msg["tool_calls"]]
                        break
                
                for i, action in enumerate(remaining_actions):
                    tool_msg = {
                        "role": "tool",
                        "name": action.name,
                        "content": observations[i],
                    }
                    # Add tool_call_id if available (required by OpenAI API spec)
                    if i < len(tool_call_ids):
                        tool_msg["tool_call_id"] = tool_call_ids[i]
                    messages.append(tool_msg)
            reward_res = self.calculate_reward(messages)
            reward = reward_res.reward
            info.reward_info = reward_res
            info.user_cost = self.user.get_total_cost()

        return EnvResponse(
            observation=observations, reward=reward, done=done, info=info
        )

    def run_steps(self, actions: List[Action], messages: List[Dict]) -> EnvResponse:
        """Synchronous wrapper for the asynchronous steps method that works in threads."""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        # Set it as the current event loop for this thread
        asyncio.set_event_loop(loop)
        try:
            # Run the async function in this loop
            return loop.run_until_complete(self.steps(actions, messages))
        finally:
            # Clean up
            loop.close()

    def get_data_hash(self) -> str:
        return consistent_hash(to_hashable(self.data))

    def get_context_state_hash(self, context_state) -> str:
        return consistent_hash(to_hashable(context_state))

    def calculate_reward(self, messages: List[Dict]) -> RewardResult:
        reward = 1.0
        performed_actions = self.actions.copy()

        # ==== State-based Reward (skipped if task type is hallucination) ====
        if (
            self.task.task_type == TaskType.HALLUCINATION_MISSING_TOOL
            or self.task.task_type == TaskType.HALLUCINATION_MISSING_TOOL_PARAMETER
            or self.task.task_type == TaskType.HALLUCINATION_MISSING_TOOL_RESPONSE
        ):
            r_actions_final = None
            r_actions_intermediate = None
            r_actions = None  # Legacy for backward compatibility
            gt_vehicle_state_hash = "Skipped"
        else:
            from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
                ContextState,
                context_state,
            )

            context_state_hash = self.get_context_state_hash(context_state.get())
            climate_temperature_driver = context_state.get().climate_temperature_driver
            climate_temperature_passenger = (
                context_state.get().climate_temperature_passenger
            )
            difference_is_more_than_3_degrees = (
                abs(climate_temperature_driver - climate_temperature_passenger) > 3
            )

            # Check if the vehicle state changes are correct by performing the ground truth actions and compare the edited vehicle state. If they are not correct, then we set the reward to 0.
            # Generate ground truth intermediate state hashes by temporarily setting context_state to a fresh vehicle state to perform ground truth actions, later restore the vehicle state to the state before the token was created.
            gt_context_state = ContextState()
            token = context_state.set(gt_context_state)
            context_state.get().update_state(**self.task.context_init_config)

            # Record initial state hash for ground truth
            gt_state_hashes = [self.get_context_state_hash(context_state.get())]

            for action in self.task.actions:

                self.step(action, messages)
                # Record state hash after each ground truth action
                if (
                    action.name in self.tools_map
                    and action.name not in USER_AS_A_TOOL_ACTION_NAMES
                ):
                    current_gt_hash = self.get_context_state_hash(context_state.get())
                    gt_state_hashes.append(current_gt_hash)

            gt_vehicle_state_hash = self.get_context_state_hash(context_state.get())

            # Restore the vehicle state to the state before the token was created.
            context_state.reset(token)

            # Check final state correctness (compare final state hashes)
            final_actual_state_hash = (
                self.state_hashes[-1] if self.state_hashes else context_state_hash
            )
            r_actions_final = context_state_hash == gt_vehicle_state_hash
            if not r_actions_final:
                reward = 0.0

            # Check intermediate states correctness (all intermediate states must be valid)
            gt_state_hashes_set = set(gt_state_hashes)
            actual_state_hashes_set = set(self.state_hashes)
            r_actions_intermediate = actual_state_hashes_set.issubset(
                gt_state_hashes_set
            )
            if not r_actions_intermediate:
                reward = 0.0

            # Legacy field for backward compatibility
            r_actions = r_actions_final and r_actions_intermediate
        # ========

        # ==== Tool Subset Evaluation Reward (skipped if task type is hallucination) ====
        gt_action_names = {
            action.name
            for action in self.task.actions
            if action.name != RESPOND_ACTION_NAME
        }
        performed_action_names = {action.name for action in performed_actions}
        if (
            self.task.task_type == TaskType.HALLUCINATION_MISSING_TOOL
            or self.task.task_type == TaskType.HALLUCINATION_MISSING_TOOL_PARAMETER
            or self.task.task_type == TaskType.HALLUCINATION_MISSING_TOOL_RESPONSE
        ):
            r_tool_subset = None
            tool_subset_missing_tools = None
        else:
            # Calculate missing tools from ground truth
            tool_subset_missing_tools = list(gt_action_names - performed_action_names)
            r_tool_subset = gt_action_names.issubset(performed_action_names)
            if not r_tool_subset:
                r_tool_subset = 0.0
                reward = 0.0
            else:
                r_tool_subset = 1.0
        # ========

        # ==== Tool-Execution-Error-Calculations ====
        tool_execution_errors = tool_execution_errors_during_runtime.get()
        if self.score_tool_execution_errors and len(tool_execution_errors) > 0:
            r_tool_execution = 0.0
            reward = 0.0
        else:
            r_tool_execution = 1.0
        # ========

        # ==== End-Conversation-Failure-Calculations ====
        end_conversation_fail = end_conversation_failure.get()
        if len(end_conversation_fail) > 0:
            reward = 0.0
            r_user_end_conversation = 0.0
            end_conversation_keyword = end_conversation_fail[0][
                "conversation_control_keyword"
            ]
        else:
            r_user_end_conversation = 1.0
            end_conversation_keyword = None
        # ========

        # ==== Output-string-based Reward ====
        r_outputs = None
        outputs = {}
        # ========

        # ==== Policy-Error-Calculations (skipped if task type is hallucination) ====
        if (
            self.task.task_type == TaskType.HALLUCINATION_MISSING_TOOL
            or self.task.task_type == TaskType.HALLUCINATION_MISSING_TOOL_PARAMETER
            or self.task.task_type == TaskType.HALLUCINATION_MISSING_TOOL_RESPONSE
        ):
            policy_llm_errors = None
            policy_errors_aut = None
            r_policy = None
        else:
            # LLM-based Policy Error Calculation
            if self.evaluate_policy:
                r_policy = 1.0
                from car_bench.envs.car_voice_assistant.wiki import WIKI_LLM_POL_LINES

                traj_messages = (
                    messages[1:] if messages[0]["role"] == "system" else messages
                )
                traj_messages_core = []
                # Filter traj_messages to only include specified keys
                for msg in traj_messages:
                    filtered_msg = {}
                    for key in ["role", "content", "name", "tool_calls"]:
                        if key in msg:
                            if (
                                key == "content"
                                and "tool_calls" in msg
                                and msg["tool_calls"]
                            ):
                                filtered_msg[key] = ""
                            else:
                                filtered_msg[key] = msg[key]
                    traj_messages_core.append(filtered_msg)
                pol_llm_evaluation_results = []
                pol_llm_evaluation_scores = []
                for line in WIKI_LLM_POL_LINES:
                    # make llm request with litellm, create system prompt, input
                    if "REQUIRES_CONFIRMATION" in line:
                        if (
                            len(
                                performed_action_names.intersection(
                                    {
                                        "send_email",
                                        "open_close_trunk_door",
                                        "set_head_lights_high_beams",
                                    }
                                )
                            )
                            > 0
                        ):
                            line = (
                                line
                                + " The tools that require confirmation are: 'send_email', 'open_close_trunk_door', and 'set_head_lights_high_beams'."
                            )
                        else:
                            continue
                    elif (
                        "windows are requested by the user to open more than 25% (absolute position) and AC is ON in that moment"
                        in line
                    ):
                        if (
                            len(
                                performed_action_names.intersection(
                                    {"open_close_window"}
                                )
                            )
                            > 0
                        ):
                            line = (
                                line
                                + " Consider that you can only know the AC status if it was turned on **before** the user request to open the windows. Mark not applicable, if th AC was not requested to turn on before, also  mark not applicable if the windows are closed or requested to close. Also this policy is one-way: applies only if user requests to open the windows, turn on AC request are handled in another policy."
                            )
                        else:
                            continue
                    elif (
                        "In certain weather conditions, the vehicle control actions"
                        in line
                    ):
                        if (
                            len(
                                performed_action_names.intersection(
                                    {"open_close_sunroof", "set_fog_lights"}
                                )
                            )
                            == 0
                        ):
                            continue
                    elif "user sets the temperature to a single seat zone" in line:
                        if (
                            len(
                                performed_action_names.intersection(
                                    {"set_climate_temperature"}
                                )
                            )
                            == 0
                            or not difference_is_more_than_3_degrees
                        ):
                            continue
                    elif (
                        "route is presented in detail (fastest route, shortest route, or upon user detail request)"
                        in line
                    ):
                        if (
                            len(
                                performed_action_names.intersection(
                                    {"get_routes_from_start_to_destination"}
                                )
                            )
                            > 0
                        ):
                            line = (
                                line
                                + " For this policy, first reason which routes were presented in detail, then evaluate if these routes include a toll road - if true this would be included in the list of road types else it's not present, then evaluate the policy for the routes identified to be presented in detail."
                            )
                        else:
                            continue
                    elif (
                        "user asks for a multi-stop route and does not specify the route selection"
                        in line
                    ):
                        if (
                            len(
                                performed_action_names.intersection(
                                    {"get_routes_from_start_to_destination"}
                                )
                            )
                            == 0
                        ):
                            continue
                    evaluation_result = self.policy_evaluator.evaluate_llm(
                        policy=line, trajectory=str(traj_messages_core)
                    )
                    pol_llm_evaluation_results.append(json.loads(evaluation_result))
                    pol_llm_evaluation_scores.append(
                        json.loads(evaluation_result)["policy_followed"]
                    )
                policy_llm_errors = [
                    eval_result["reasoning"]
                    for eval_result in pol_llm_evaluation_results
                    if not eval_result["policy_followed"]
                ]
            else:
                policy_llm_errors = []
                r_policy = None

            # AUT-based Policy Error Calculation
            if self.evaluate_policy:
                self.policy_evaluator.evaluate_aut(trajectory=traj_messages)
                policy_errors_aut = policy_errors_during_runtime.get()
            else:
                policy_errors_aut = []
            if self.score_policy_errors and (
                len(policy_errors_aut) > 0 or len(policy_llm_errors) > 0
            ):
                r_policy = 0.0
                reward = 0.0

        info = RewardInfo(
            r_actions=r_actions,
            r_actions_final=r_actions_final,
            r_actions_intermediate=r_actions_intermediate,
            r_tool_subset=r_tool_subset,
            tool_subset_missing_tools=tool_subset_missing_tools,
            r_tool_execution=r_tool_execution,
            tool_execution_errors=tool_execution_errors,
            r_policy=r_policy,
            policy_llm_errors=policy_llm_errors,
            policy_aut_errors=policy_errors_aut,
            r_user_end_conversation=r_user_end_conversation,
            end_conversation_keyword=end_conversation_keyword,
            r_outputs=r_outputs,
            outputs=outputs,
        )
        return RewardResult(reward=reward, info=info, actions=performed_actions)
