"""
Live-sim backend for CAR-bench: back the vehicle state with a running prompt-drive
simulator over a WebSocket, keeping Python authoritative over the hashed truth.
See car-bench-compat-plan.md (in the prompt-drive repo) §4.

Typical use (see interactive_app.py --live-sim / CAR_BENCH_LIVE_SIM):

    from car_bench.envs.car_voice_assistant.live_sim import PromptDriveClient, init_live_context_state, finish
    client = PromptDriveClient(port=8765, token="...").start()
    client.wait_for_sim()
    tokens = init_live_context_state(env, idx, client)
    # ... run the task ...
    divergences = finish(tokens)
"""

from .client import PromptDriveClient, SimTimeout, SimUnavailable
from .lifecycle import finish, init_live_context_state
from .proxies import LiveContextState, LiveFixedContext, LiveSimDivergence

__all__ = [
    "PromptDriveClient",
    "SimUnavailable",
    "SimTimeout",
    "LiveContextState",
    "LiveFixedContext",
    "LiveSimDivergence",
    "init_live_context_state",
    "finish",
]
