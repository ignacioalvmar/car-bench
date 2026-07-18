"""
Live-sim proxies for CAR-bench state (car-bench-compat-plan.md §4.1 / §4.8).

`LiveContextState` / `LiveFixedContext` subclass the real Pydantic models and add
ONLY private attributes (PrivateAttr) — no new fields — so `str()`, `model_dump()`
and therefore the reward hash stay byte-identical to the stock models. Every
`update_state` runs the real validation first (super()), then write-through-mirrors
the changed keys to the running prompt-drive sim and (optionally) reads them back
to verify. Divergences are recorded as warnings and never mutate local state, so
the sim can be flaky, slow, or absent without ever corrupting evaluation.
"""

from pydantic import PrivateAttr

from car_bench.envs.car_voice_assistant.context.dynamic_context_state import ContextState
from car_bench.envs.car_voice_assistant.context.fixed_context import FixedContext


class LiveSimDivergence(Exception):
    """Raised (only in strict mode) when the sim mirror diverges from the truth."""


def _value_eq(a, b):
    """Value equality tolerant of JSON int/float (20 == 20.0) and list order."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_value_eq(x, y) for x, y in zip(a, b))
    return a == b


class _LiveMixin:
    """Shared write-through + verification. Host classes declare the PrivateAttrs."""

    def bind(self, client, verify="per_write", strict=False):
        """Attach a PromptDriveClient. `verify`: per_write | per_task | off."""
        self._pd_client = client
        self._pd_verify = verify
        self._pd_strict = strict
        return self

    @property
    def divergences(self):
        return list(self._pd_divergences)

    def _record_divergence(self, entry):
        self._pd_divergences.append(entry)
        if self._pd_strict:
            raise LiveSimDivergence(entry)

    def _mirror(self, op, changed):
        """Send changed keys to the sim; record (never raise, unless strict) on failure."""
        client = self._pd_client
        if client is None or not changed:
            return
        diff = self.model_dump(mode="json", include=set(changed))
        ok, res = client.try_call(op, diff)
        if not ok:
            self._record_divergence({"kind": "sim_unreachable", "op": op, "keys": list(changed), "detail": res})
            return
        if isinstance(res, dict) and res.get("ok") is False:
            self._record_divergence({"kind": "set_rejected", "op": op, "keys": list(changed), "result": res})
            return
        if self._pd_verify == "per_write":
            self._verify(changed)

    def _verify(self, keys):
        client = self._pd_client
        if client is None:
            return
        ok, res = client.try_call("vehicle.get", list(keys))
        if not ok or not isinstance(res, dict) or not res.get("ok"):
            self._record_divergence({"kind": "readback_failed", "keys": list(keys), "detail": res})
            return
        remote = res.get("value", {}) or {}
        local = self.model_dump(mode="json", include=set(keys))
        for k in keys:
            if not _value_eq(local.get(k), remote.get(k)):
                self._record_divergence({"kind": "mismatch", "key": k, "local": local.get(k), "remote": remote.get(k)})

    def verify_all(self):
        """Full-state parity check (used at per_task cadence and at finish)."""
        client = self._pd_client
        if client is None:
            return []
        keys = list(type(self).model_fields.keys())
        self._verify(keys)
        return self.divergences


class LiveContextState(_LiveMixin, ContextState):
    # Private only — never appears in model_fields / __str__ / model_dump.
    _pd_client: object = PrivateAttr(default=None)
    _pd_verify: str = PrivateAttr(default="per_write")
    _pd_strict: bool = PrivateAttr(default=False)
    _pd_divergences: list = PrivateAttr(default_factory=list)

    def update_state(self, **kwargs):
        super().update_state(**kwargs)  # real validation + waypoints policy hook
        changed = [k for k in kwargs if k in type(self).model_fields]
        self._mirror("vehicle.set", changed)


class LiveFixedContext(_LiveMixin, FixedContext):
    _pd_client: object = PrivateAttr(default=None)
    _pd_verify: str = PrivateAttr(default="off")  # fixed context isn't hashed
    _pd_strict: bool = PrivateAttr(default=False)
    _pd_divergences: list = PrivateAttr(default_factory=list)

    def update_state(self, **kwargs):
        super().update_state(**kwargs)
        changed = [k for k in kwargs if k in type(self).model_fields]
        self._mirror("vehicle.fixed.set", changed)
