"""
Interactive Task UI for CAR-bench.

A web-based interface where a human can play through benchmark tasks,
taking the role of either the user (driver) or the agent (in-car assistant).

Usage:
    pip install flask
    python interactive_app.py [--task-type base] [--task-split train] [--port 5000]
"""

import argparse
import contextvars
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request

from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
    ContextState,
    context_state,
)
from car_bench.envs.car_voice_assistant.context.fixed_context import (
    FixedContext,
    fixed_context,
)
from car_bench.envs.car_voice_assistant.mock_data import car_va_data_manager, load_data
from car_bench.envs.car_voice_assistant.tasks.task_config import TaskConfig, task_config
from car_bench.envs.car_voice_assistant.tools import ALL_TOOLS
from car_bench.envs.car_voice_assistant.wiki import WIKI, WIKI_RAW
from car_bench.envs.policy_evaluator import policy_errors_during_runtime
from car_bench.envs.tool_execution_error_evaluator import (
    tool_execution_errors_during_runtime,
)
from car_bench.envs.tool_manipulation import remove_tool_elements
from car_bench.envs.user.user_end_conversation import end_conversation_failure
from car_bench.types import (
    RESPOND_ACTION_NAME,
    USER_AS_A_TOOL_ACTION_NAMES,
    Action,
    TaskType,
)

app = Flask(__name__, template_folder="templates")

# ---------------------------------------------------------------------------
# Task loading with HuggingFace fallback to local result files
# ---------------------------------------------------------------------------

# Map from task_type URL param to result directories to search
_RESULT_DIR_MAP = {
    "base": ["base_v2", "base"],
    "hallucination": ["hallucination_train_v2", "hallucination"],
    "disambiguation": ["disambiguation_v2", "disambiguation"],
}

from car_bench.types import Task


def _load_tasks_from_hf(task_type: str, task_split: str) -> List[Task]:
    """Try loading tasks from HuggingFace dataset."""
    from car_bench.envs.car_voice_assistant.env import _load_tasks

    return load_tasks(task_type, task_split)


def _load_tasks_from_results(task_type: str) -> List[Task]:
    """Fallback: extract unique tasks from local result files."""
    results_base = os.path.join(os.path.dirname(__file__), "results")
    tasks_by_id: Dict[str, Task] = {}

    dirs = _RESULT_DIR_MAP.get(task_type, [task_type])
    for d in dirs:
        dirpath = os.path.join(results_base, d)
        if not os.path.isdir(dirpath):
            continue
        for fname in os.listdir(dirpath):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                for entry in data:
                    info = entry.get("info", {})
                    task_dict = info.get("task")
                    if not task_dict or not isinstance(task_dict, dict):
                        continue
                    # Ensure task_id exists
                    if "task_id" not in task_dict:
                        top_id = entry.get("task_id")
                        if top_id is not None:
                            task_dict["task_id"] = f"{task_type}_{top_id}"
                        else:
                            continue
                    tid = task_dict["task_id"]
                    if tid in tasks_by_id:
                        continue
                    # Parse actions if they are dicts
                    if "actions" in task_dict and task_dict["actions"]:
                        task_dict["actions"] = [
                            Action(**a) if isinstance(a, dict) else a
                            for a in task_dict["actions"]
                        ]
                    tasks_by_id[tid] = Task(**task_dict)
            except Exception:
                continue
        if tasks_by_id:
            break  # Use first directory that has data

    return sorted(tasks_by_id.values(), key=lambda t: t.task_id)


def load_tasks(task_type: str, task_split: str) -> List[Task]:
    """Load tasks, falling back to local result files if HuggingFace is unavailable."""
    try:
        tasks = _load_tasks_from_hf(task_type, task_split)
        if tasks:
            return tasks
    except Exception as e:
        print(f"HuggingFace loading failed ({e}), falling back to local results...")

    tasks = _load_tasks_from_results(task_type)
    if tasks:
        print(f"Loaded {len(tasks)} {task_type} tasks from local result files")
    return tasks


# ---------------------------------------------------------------------------
# Global state for the interactive session
# ---------------------------------------------------------------------------

_session: Dict[str, Any] = {}


def _get_tools_info() -> List[Dict[str, Any]]:
    """Get tool info list from ALL_TOOLS."""
    return [tool.get_info() for tool in ALL_TOOLS]


def _get_tools_map() -> Dict[str, Any]:
    """Get tool name -> tool class map."""
    return {tool.get_info()["function"]["name"]: tool for tool in ALL_TOOLS}


def _build_wiki_for_task(task) -> str:
    """Build wiki text with placeholders filled from task context."""
    wiki = WIKI
    ctx = task.context_init_config

    # Build a temporary fixed context to get location/datetime
    fc = FixedContext()
    fc.update_state(**ctx)

    wiki = wiki.replace(
        "{{placeholder_location_based_on_task_context_init_config}}",
        fc.current_location.model_dump_json(),
    )
    wiki = wiki.replace(
        "{{placeholder_datetime_based_on_task_context_init_config}}",
        fc.current_datetime.model_dump_json(),
    )
    return wiki


def _init_session_context(task):
    """Initialize context vars for the interactive session."""
    default_task_config = TaskConfig()
    token_task_config = task_config.set(default_task_config)
    task_cfg = task_config.get()
    task_cfg.update_state(calendar_id=task.calendar_id)

    default_context_state = ContextState()
    default_fixed_context = FixedContext()
    token_context = context_state.set(default_context_state)
    token_fixed = fixed_context.set(default_fixed_context)

    vehicle_ctx = context_state.get()
    fixed_ctx = fixed_context.get()
    fixed_ctx.update_state(**task.context_init_config)
    vehicle_ctx.update_state(**task.context_init_config)

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
    }


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("interactive.html")


@app.route("/api/task-types")
def api_task_types():
    """Return available task types and splits."""
    return jsonify(
        {
            "task_types": ["base", "hallucination", "disambiguation"],
            "task_splits": ["train", "test"],
        }
    )


@app.route("/api/tasks")
def api_tasks():
    """Return list of tasks for a given type/split."""
    task_type = request.args.get("type", "base")
    task_split = request.args.get("split", "train")
    try:
        tasks = load_tasks(task_type, task_split)
        task_list = []
        for i, t in enumerate(tasks):
            task_list.append(
                {
                    "index": i,
                    "task_id": t.task_id,
                    "task_type": t.task_type.value,
                    "instruction": t.instruction,
                    "persona": t.persona,
                    "removed_part": t.removed_part,
                    "disambiguation_element_internal": t.disambiguation_element_internal,
                    "disambiguation_element_user": t.disambiguation_element_user,
                }
            )
        return jsonify({"tasks": task_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/start", methods=["POST"])
def api_start():
    """Start an interactive session for a given task."""
    data = request.json
    task_type = data.get("task_type", "base")
    task_split = data.get("task_split", "train")
    task_index = data.get("task_index", 0)

    try:
        # Initialize mock data (best-effort: vehicle tools work without it)
        try:
            car_va_data_manager.initialize()
        except Exception as e:
            print(f"Mock data init failed ({e}). "
                  "Navigation/communication tools may not work, but vehicle tools will.")

        tasks = load_tasks(task_type, task_split)
        task = tasks[task_index]

        # Init context vars
        tokens = _init_session_context(task)

        # Build wiki with placeholders filled
        wiki = _build_wiki_for_task(task)

        # Build tools info, removing tools as needed for hallucination tasks
        tools_info = _get_tools_info()
        if task.removed_part:
            tools_info = remove_tool_elements(
                tools_info, _get_tools_info(), task.removed_part
            )

        # Get initial context state
        vehicle_state = context_state.get().model_dump()
        fixed_state = fixed_context.get().model_dump()

        # Store session
        _session.clear()
        _session["task"] = task
        _session["tasks"] = tasks
        _session["task_index"] = task_index
        _session["wiki"] = wiki
        _session["tools_info"] = tools_info
        _session["tokens"] = tokens
        _session["messages"] = []  # conversation history
        _session["step"] = 0
        _session["done"] = False
        _session["tools_map"] = _get_tools_map()

        # Build ground-truth actions for reference
        gt_actions = [
            {"name": a.name, "kwargs": a.kwargs} for a in task.actions
        ]

        return jsonify(
            {
                "status": "ok",
                "task": {
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "instruction": task.instruction,
                    "persona": task.persona,
                    "removed_part": task.removed_part,
                    "disambiguation_element_internal": task.disambiguation_element_internal,
                    "disambiguation_element_user": task.disambiguation_element_user,
                    "context_init_config": task.context_init_config,
                    "ground_truth_actions": gt_actions,
                },
                "wiki": wiki,
                "tools": tools_info,
                "vehicle_state": vehicle_state,
                "fixed_context": fixed_state,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/execute-tool", methods=["POST"])
def api_execute_tool():
    """Execute a tool call (human playing as agent)."""
    if not _session or _session.get("done"):
        return jsonify({"error": "No active session or session is done"}), 400

    data = request.json
    tool_name = data.get("tool_name")
    tool_args = data.get("tool_args", {})

    tools_map = _session["tools_map"]
    mock_data = load_data()

    if tool_name not in tools_map:
        return jsonify({"error": f"Unknown tool: {tool_name}"}), 400

    try:
        result = tools_map[tool_name].invoke(data=mock_data, **tool_args)
        # Update session with the action
        action = Action(name=tool_name, kwargs=tool_args)

        # Get updated state
        vehicle_state = context_state.get().model_dump()

        # Record in messages as a tool call + tool response
        tool_call_id = f"call_{_session['step']}"
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                }
            ],
        }
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result if isinstance(result, str) else json.dumps(result),
        }
        _session["messages"].append(assistant_msg)
        _session["messages"].append(tool_msg)
        _session["step"] += 1

        return jsonify(
            {
                "status": "ok",
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": result if isinstance(result, str) else json.dumps(result),
                "vehicle_state": vehicle_state,
                "step": _session["step"],
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "tool_name": tool_name}), 500


@app.route("/api/send-message", methods=["POST"])
def api_send_message():
    """Send a message from the agent to the user (human playing as agent),
    or from the user to the agent (human playing as user)."""
    if not _session or _session.get("done"):
        return jsonify({"error": "No active session or session is done"}), 400

    data = request.json
    role = data.get("role", "assistant")  # "assistant" or "user"
    content = data.get("content", "")

    if not content.strip():
        return jsonify({"error": "Empty message"}), 400

    msg = {"role": role, "content": content}
    _session["messages"].append(msg)
    _session["step"] += 1

    # Check for stop signals
    done = False
    if "###STOP###" in content:
        done = True
        _session["done"] = True

    return jsonify(
        {
            "status": "ok",
            "message": msg,
            "step": _session["step"],
            "done": done,
        }
    )


@app.route("/api/end-session", methods=["POST"])
def api_end_session():
    """End the current session."""
    if not _session:
        return jsonify({"error": "No active session"}), 400

    _session["done"] = True

    # Collect any runtime errors
    policy_errors = []
    tool_exec_errors = []
    try:
        policy_errors = policy_errors_during_runtime.get()
        tool_exec_errors = tool_execution_errors_during_runtime.get()
    except Exception:
        pass

    return jsonify(
        {
            "status": "ok",
            "messages": _session["messages"],
            "step": _session["step"],
            "policy_errors": policy_errors,
            "tool_execution_errors": tool_exec_errors,
        }
    )


@app.route("/api/state")
def api_state():
    """Get current session state."""
    if not _session:
        return jsonify({"error": "No active session"}), 400

    try:
        vehicle_state = context_state.get().model_dump()
        fixed_state = fixed_context.get().model_dump()
    except Exception:
        vehicle_state = {}
        fixed_state = {}

    policy_errors = []
    tool_exec_errors = []
    try:
        policy_errors = policy_errors_during_runtime.get()
        tool_exec_errors = tool_execution_errors_during_runtime.get()
    except Exception:
        pass

    return jsonify(
        {
            "vehicle_state": vehicle_state,
            "fixed_context": fixed_state,
            "messages": _session.get("messages", []),
            "step": _session.get("step", 0),
            "done": _session.get("done", False),
            "policy_errors": policy_errors,
            "tool_execution_errors": tool_exec_errors,
        }
    )


@app.route("/api/wiki")
def api_wiki():
    """Get the wiki/policy text for the current session."""
    if not _session:
        return jsonify({"error": "No active session"}), 400
    return jsonify({"wiki": _session.get("wiki", ""), "wiki_raw": WIKI_RAW})


@app.route("/api/tools")
def api_tools():
    """Get available tools info for the current session."""
    if not _session:
        return jsonify({"error": "No active session"}), 400
    return jsonify({"tools": _session.get("tools_info", [])})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Interactive CAR-bench Task UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Pre-load mock data (best-effort)
    print("Pre-loading mock data...")
    try:
        car_va_data_manager.initialize()
        print("Mock data ready.")
    except Exception as e:
        print(f"Mock data loading failed ({e}). "
              "Vehicle tools will work, but navigation/communication tools may not.")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
