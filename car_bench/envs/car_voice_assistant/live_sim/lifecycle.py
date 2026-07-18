"""
Per-task wiring for the live-sim backend (car-bench-compat-plan.md §4.8).

`init_live_context_state` is a drop-in variant of interactive_app._init_context_state
that installs LiveContextState / LiveFixedContext, resets the sim to the task's
`context_init_config` BEFORE binding write-through (so the local config application
doesn't double-send), then verifies the sim's returned snapshot matches the local
state at t0. No changes to the env, tools, orchestrator, or reward code are needed.
"""

from car_bench.envs.car_voice_assistant.context.dynamic_context_state import context_state
from car_bench.envs.car_voice_assistant.context.fixed_context import fixed_context
from car_bench.envs.car_voice_assistant.tasks.task_config import TaskConfig, task_config
from car_bench.envs.policy_evaluator import policy_errors_during_runtime
from car_bench.envs.tool_execution_error_evaluator import tool_execution_errors_during_runtime
from car_bench.envs.user.user_end_conversation import end_conversation_failure

from .proxies import LiveContextState, LiveFixedContext, _value_eq


def init_live_context_state(env, idx, client, verify="per_write", benchmark=True,
                            ambience=True, ambience_condition=None, strict=False):
    """Initialize context vars for a task, backed by the live sim. Mirrors
    interactive_app._init_context_state exactly except for the Live proxies and
    the sim reset. Returns a tokens dict (compatible with the upstream
    _reset_context_state) plus the bound `ctx`/`fx` proxies and any t0
    divergences."""
    task = env.tasks[idx]

    default_task_config = TaskConfig()
    token_task_config = task_config.set(default_task_config)
    task_config.get().update_state(calendar_id=task.calendar_id)

    ctx = LiveContextState()
    fx = LiveFixedContext()
    token_context = context_state.set(ctx)
    token_fixed = fixed_context.set(fx)

    # 1) Reset the sim to the task config FIRST (before binding write-through),
    #    so the local update_state calls below don't double-send.
    reset_snapshot = None
    if client is not None:
        opts = {"benchmark": benchmark, "ambience": ambience}
        if ambience_condition is not None:
            opts["ambienceCondition"] = ambience_condition
        ok, res = client.try_call("vehicle.reset", task.context_init_config, opts)
        if ok and isinstance(res, dict) and res.get("ok"):
            reset_snapshot = (res.get("value") or {}).get("dynamic")

    # 2) Apply the flat config to both models locally (still unbound -> no send),
    #    exactly as upstream does.
    fx.update_state(**task.context_init_config)
    ctx.update_state(**task.context_init_config)

    # 3) Bind write-through for subsequent tool calls.
    ctx.bind(client, verify=verify, strict=strict)
    fx.bind(client, verify="off", strict=strict)

    # 4) t0 parity check: the sim's post-reset snapshot must match local state.
    t0_divergences = []
    if reset_snapshot is not None:
        local = ctx.model_dump(mode="json")
        for k in type(ctx).model_fields:
            if not _value_eq(local.get(k), reset_snapshot.get(k)):
                entry = {"kind": "t0_mismatch", "key": k, "local": local.get(k), "remote": reset_snapshot.get(k)}
                t0_divergences.append(entry)
                ctx._record_divergence(entry)

    # Wiki placeholders (identical to upstream).
    if "{{placeholder_location_based_on_task_context_init_config}}" in env.wiki:
        env.wiki = env.wiki.replace(
            "{{placeholder_location_based_on_task_context_init_config}}",
            fx.current_location.model_dump_json(),
        )
    if "{{placeholder_datetime_based_on_task_context_init_config}}" in env.wiki:
        env.wiki = env.wiki.replace(
            "{{placeholder_datetime_based_on_task_context_init_config}}",
            fx.current_datetime.model_dump_json(),
        )

    token_policy = policy_errors_during_runtime.set([])
    token_tool_exec = tool_execution_errors_during_runtime.set([])
    token_end_conv = end_conversation_failure.set([])

    return {
        "token_context": token_context,
        "token_fixed": token_fixed,
        "token_task_config": token_task_config,
        "token_policy": token_policy,
        "token_tool_exec": token_tool_exec,
        "token_end_conv": token_end_conv,
        "ctx": ctx,
        "fx": fx,
        "t0_divergences": t0_divergences,
    }


def finish(tokens):
    """End-of-task parity check. Returns the full divergence list for run info."""
    ctx = tokens.get("ctx")
    if ctx is None:
        return []
    try:
        ctx.verify_all()
    except Exception:
        pass
    return ctx.divergences
