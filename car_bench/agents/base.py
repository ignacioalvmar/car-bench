import abc
from typing import Any, Dict, List, Optional, Tuple

from car_bench.envs.base import Env
from car_bench.types import AgentState, SolveResult


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_init_state(
        self, system_prompt: str, initial_observation: str
    ) -> AgentState:
        """
        Get the initial state of the agent.

        Args:
            system_prompt: System prompt (domain policy/wiki)
            initial_observation: Initial user message/observation

        Returns:
            Initial AgentState with messages and metrics initialized
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_next_message(
        self, state: AgentState, tools_info: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], AgentState]:
        """
        Generate the next message from the agent.

        Args:
            state: Current agent state with messages and metrics
            tools_info: List of tool definitions (already prepared by orchestrator)

        Returns:
            Tuple of (next_message, updated_state)
        """
        raise NotImplementedError

