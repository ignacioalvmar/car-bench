"""
WP4/WP5 tests for the live-sim adapter (car_bench/envs/car_voice_assistant/live_sim).

Part A (unit, no sim): hash/str/model_dump parity of LiveContextState vs
ContextState; write-through mirroring to a mock client; read-back verification;
graceful degradation when the sim is unreachable; thread-safety smoke.

Part B (integration): the real PromptDriveClient (websockets server) + the
headless prompt-drive sim harness (scripts/test-socket-sim.js) over real WS —
update_state mirrors to the sim's VehicleState with no divergence, and
vehicle_reset round-trips.

Run: python tests/test_live_sim.py   (exit 0 = pass). Requires `websockets`;
Part B additionally requires `node` and the built prompt-drive bundles.
"""

import os
import random
import subprocess
import sys
import threading
import types

# Make the repo root importable when run as a script (car_bench isn't pip-installed here).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub litellm so importing the context models doesn't require the full agent stack.
if "litellm" not in sys.modules:
    _m = types.ModuleType("litellm")
    _m.completion = lambda *a, **k: None
    sys.modules["litellm"] = _m

from car_bench.envs.base import consistent_hash, to_hashable  # noqa: E402
from car_bench.envs.car_voice_assistant.context.dynamic_context_state import ContextState  # noqa: E402
from car_bench.envs.car_voice_assistant.context.fixed_context import FixedContext, fixed_context  # noqa: E402
from car_bench.envs.policy_evaluator import policy_errors_during_runtime  # noqa: E402
from car_bench.envs.car_voice_assistant.live_sim.proxies import (  # noqa: E402
    LiveContextState, LiveFixedContext, _value_eq,
)

# update_state(waypoints_id=...) reads these ContextVars (start-of-route policy hook).
fixed_context.set(FixedContext())
policy_errors_during_runtime.set([])

failures = []


def check(name, cond):
    print(("  ok  : " if cond else "  FAIL: ") + name)
    if not cond:
        failures.append(name)


# --- sample field values that satisfy the ContextState constraints -----------
FIELD_CHOICES = {
    "sunroof_position": [0, 50, 100], "sunshade_position": [0, 25, 100],
    "trunk_door_position": ["closed", "OPEN", "CLOSE"],
    "window_driver_position": [0, 30, 100], "window_passenger_position": [0, 100],
    "reading_light_driver": [True, False], "fog_lights": [True, False],
    "head_lights_low_beams": [True, False], "head_lights_high_beams": [True, False],
    "ambient_light": ["OFF", "YELLOW", "CYAN"],
    "climate_temperature_driver": [16, 20, 21.5, 28],
    "climate_temperature_passenger": [18, 22.5],
    "steering_wheel_heating": [0, 1, 3], "seat_heating_driver": [0, 2],
    "fan_speed": [0, 3, 5], "window_front_defrost": [True, False],
    "fan_airflow_direction": ["FEET", "WINDSHIELD_HEAD_FEET"],
    "air_conditioning": [True, False], "air_circulation": ["AUTO", "RECIRCULATION"],
    "navigation_active": [True, False],
    "waypoints_id": [[], ["loc_a", "loc_b"]],
    "email_addresses_sent_mail_to": [[], ["a@b.c"]],
}


class MockClient:
    """Records calls; simulates a faithful mirror unless `corrupt` is set."""
    def __init__(self, reachable=True, corrupt=None):
        self.calls = []
        self.reachable = reachable
        self.corrupt = corrupt or {}   # key -> wrong value to return on get
        self._store = {}

    def try_call(self, op, *args):
        self.calls.append((op, args))
        if not self.reachable:
            return False, {"error": "SimUnavailable"}
        if op == "vehicle.set":
            self._store.update(args[0])
            return True, {"ok": True, "value": {"changed": list(args[0].keys())}}
        if op == "vehicle.get":
            keys = args[0] if args and args[0] else list(self._store.keys())
            val = {}
            for k in keys:
                val[k] = self.corrupt[k] if k in self.corrupt else self._store.get(k)
            return True, {"ok": True, "value": val}
        return True, {"ok": True, "value": None}


def part_a():
    print("Part A — unit (no sim)")

    # 1) hash / str / model_dump parity over random field sets
    rng = random.Random(42)
    parity_ok = True
    for _ in range(200):
        vals = {k: rng.choice(v) for k, v in FIELD_CHOICES.items() if rng.random() < 0.5}
        a = ContextState(); a.update_state(**vals)
        b = LiveContextState(); b.update_state(**vals)   # unbound: pure passthrough
        if str(a) != str(b) or a.model_dump() != b.model_dump():
            parity_ok = False; break
        if consistent_hash(to_hashable(a)) != consistent_hash(to_hashable(b)):
            parity_ok = False; break
    check("LiveContextState str/dump/hash == ContextState (200 random states)", parity_ok)
    check("private attrs hidden from model_fields/dump",
          "_pd_client" not in LiveContextState.model_fields and "_pd_client" not in LiveContextState().model_dump())

    # 2) client=None -> pure passthrough, no divergences
    c = LiveContextState()
    c.update_state(fan_speed=4, ambient_light="BLUE")
    check("unbound update_state applies locally", c.fan_speed == 4 and c.ambient_light.value == "BLUE")
    check("unbound update_state records no divergence", c.divergences == [])

    # 3) mock client: mirrors changed keys as JSON, read-back verifies clean
    mock = MockClient()
    c = LiveContextState().bind(mock, verify="per_write")
    c.update_state(fan_speed=3, climate_temperature_driver=21.5, ambient_light="YELLOW")
    set_calls = [a for a in mock.calls if a[0] == "vehicle.set"]
    check("update_state mirrored one vehicle.set", len(set_calls) == 1)
    sent = set_calls[0][1][0]
    check("mirror sent changed keys as JSON-safe dump",
          sent == {"fan_speed": 3, "climate_temperature_driver": 21.5, "ambient_light": "YELLOW"})
    check("clean read-back -> no divergence", c.divergences == [])

    # 4) read-back mismatch -> divergence, local state intact
    mock2 = MockClient(corrupt={"fan_speed": 999})
    c = LiveContextState().bind(mock2, verify="per_write")
    c.update_state(fan_speed=3)
    check("mismatch read-back records a divergence", any(d["kind"] == "mismatch" for d in c.divergences))
    check("local state intact despite mismatch", c.fan_speed == 3)

    # 5) unreachable sim -> degrade with warning, state intact
    mock3 = MockClient(reachable=False)
    c = LiveContextState().bind(mock3, verify="per_write")
    c.update_state(seat_heating_driver=2)
    check("unreachable sim -> sim_unreachable divergence", any(d["kind"] == "sim_unreachable" for d in c.divergences))
    check("local state intact despite unreachable sim", c.seat_heating_driver == 2)

    # 6) strict mode raises
    from car_bench.envs.car_voice_assistant.live_sim.proxies import LiveSimDivergence
    c = LiveContextState().bind(MockClient(reachable=False), verify="per_write", strict=True)
    raised = False
    try:
        c.update_state(fan_speed=1)
    except LiveSimDivergence:
        raised = True
    check("strict mode raises LiveSimDivergence on divergence", raised)

    # 7) thread-safety smoke: concurrent update_state (client=None) stays consistent
    c = LiveContextState()
    def worker(v):
        for _ in range(50):
            c.update_state(fan_speed=v)
    ts = [threading.Thread(target=worker, args=(i % 6,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    check("concurrent update_state leaves a valid value", 0 <= c.fan_speed <= 5)

    # 8) LiveFixedContext mirrors to vehicle.fixed.set
    mock4 = MockClient()
    fx = LiveFixedContext().bind(mock4, verify="off")
    fx.update_state(state_of_charge=55, car_color="GREEN")
    check("fixed proxy mirrors to vehicle.fixed.set",
          any(a[0] == "vehicle.fixed.set" for a in mock4.calls))

    # 9) _value_eq tolerates int/float
    check("_value_eq(20, 20.0)", _value_eq(20, 20.0) and not _value_eq(20, 21))


def part_b():
    print("Part B — integration (real PromptDriveClient + node sim harness)")
    try:
        from car_bench.envs.car_voice_assistant.live_sim.client import PromptDriveClient
    except Exception as e:
        check("PromptDriveClient import (websockets installed)", False)
        print("   ", e)
        return

    pd_root = os.environ.get("PROMPT_DRIVE_ROOT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prompt-drive")
    sim = os.path.join(pd_root, "scripts", "test-socket-sim.js")
    if not os.path.exists(sim):
        print("   SKIP: sim harness not found at", sim)
        return

    token = "cb-live-token"
    client = PromptDriveClient(port=8770, token=token, call_timeout=3.0).start()
    node = subprocess.Popen(["node", sim, "ws://127.0.0.1:8770", token],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        try:
            client.wait_for_sim(timeout=15)
            check("sim connected to PromptDriveClient", True)
        except Exception:
            check("sim connected to PromptDriveClient", False)
            return

        # bind a proxy and drive a real tool-style write
        c = LiveContextState().bind(client, verify="per_write")
        c.update_state(fan_speed=3, ambient_light="BLUE", climate_temperature_driver=22.5)
        check("live update_state -> no divergence (sim mirrored + read-back ok)", c.divergences == [])

        ok, res = client.try_call("vehicle.get", ["fan_speed", "ambient_light", "climate_temperature_driver"])
        val = (res or {}).get("value", {}) if ok else {}
        check("sim VehicleState reflects the write",
              val.get("fan_speed") == 3 and val.get("ambient_light") == "BLUE" and val.get("climate_temperature_driver") == 22.5)

        # reset round-trip
        ok, res = client.try_call("vehicle.reset",
                                  {"fan_speed": 5, "head_lights_low_beams": True, "state_of_charge": 42},
                                  {"benchmark": True, "ambience": False})
        snap = (res or {}).get("value", {}) if ok else {}
        check("vehicle.reset returns snapshot", bool(snap) and snap.get("dynamic", {}).get("fan_speed") == 5)
    finally:
        node.terminate()
        try:
            node.communicate(timeout=5)
        except Exception:
            node.kill()
        client.stop()


if __name__ == "__main__":
    part_a()
    part_b()
    print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILURE(S): {failures}")
    sys.exit(0 if not failures else 1)
