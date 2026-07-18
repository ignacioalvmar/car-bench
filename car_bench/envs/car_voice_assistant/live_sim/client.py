"""
PromptDriveClient — the Python side of the prompt-drive WebSocket transport
(car-bench-compat-plan.md §4.4 / Appendix A).

The browser (sim) connects OUT to this server; the client exposes a synchronous,
thread-safe API so CAR-bench tools — which run in worker threads under
`Env.run_steps` — can call facade ops with `<100 ms` round trips. A single
asyncio loop runs in a daemon thread; `call()` marshals into it via
`run_coroutine_threadsafe`. Requires the `websockets` package.
"""

import asyncio
import json
import threading

try:
    import websockets
except ImportError as _e:  # pragma: no cover - dependency guard
    websockets = None
    _IMPORT_ERROR = _e


class SimUnavailable(Exception):
    """No sim is connected."""


class SimTimeout(Exception):
    """The sim did not answer within the call timeout."""


class PromptDriveClient:
    def __init__(self, host="127.0.0.1", port=8765, token=None, call_timeout=2.0):
        if websockets is None:  # pragma: no cover
            raise RuntimeError("PromptDriveClient requires the 'websockets' package") from _IMPORT_ERROR
        self.host = host
        self.port = port
        self.token = token
        self.call_timeout = call_timeout

        self._loop = asyncio.new_event_loop()
        self._thread = None
        self._server = None
        self._ws = None                      # current sim connection
        self._next_id = 1
        self._pending = {}                   # id -> asyncio.Future (loop thread)
        self._subscribers = {}               # event -> [fn]
        self._on_connect = []                # fn(client) run in a worker thread
        self._connected = threading.Event()
        self._started = threading.Event()

    # --- lifecycle ----------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run_loop, name="promptdrive-ws", daemon=True)
        self._thread.start()
        self._started.wait(timeout=10)
        return self

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._started.set()
        self._loop.run_forever()

    async def _start_server(self):
        self._server = await websockets.serve(self._handler, self.host, self.port)

    def stop(self):
        def _close():
            if self._server is not None:
                self._server.close()
        try:
            self._loop.call_soon_threadsafe(_close)
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass

    def wait_for_sim(self, timeout=60.0):
        if not self._connected.wait(timeout=timeout):
            raise SimUnavailable("no sim connected within %.1fs" % timeout)
        return True

    def is_connected(self):
        return self._ws is not None

    # --- server handler (loop thread) --------------------------------------
    async def _handler(self, ws):
        try:
            hello_raw = await ws.recv()
            hello = json.loads(hello_raw)
            if hello.get("kind") != "hello":
                await ws.close()
                return
            if self.token and hello.get("token") != self.token:
                await ws.close(code=4001, reason="bad token")
                return
            self._ws = ws
            self._connected.set()
            # Run connect callbacks off the loop thread (they may call self.call).
            for fn in list(self._on_connect):
                threading.Thread(target=self._safe, args=(fn, self), daemon=True).start()
            async for raw in ws:
                try:
                    d = json.loads(raw)
                except Exception:
                    continue
                if d.get("ns") != "promptdrive":
                    continue
                kind = d.get("kind")
                if kind == "res":
                    fut = self._pending.pop(d.get("id"), None)
                    if fut is not None and not fut.done():
                        fut.set_result(d.get("result"))
                elif kind == "event":
                    self._dispatch_event(d.get("event"), d.get("payload"))
        except Exception:
            pass
        finally:
            if self._ws is ws:
                self._ws = None
                self._connected.clear()

    def _dispatch_event(self, event, payload):
        for fn in list(self._subscribers.get(event, [])) + list(self._subscribers.get("*", [])):
            self._safe(fn, payload, event)

    @staticmethod
    def _safe(fn, *args):
        try:
            fn(*args)
        except Exception:
            pass

    # --- request/response (thread-safe) ------------------------------------
    async def _request(self, op, args, timeout):
        if self._ws is None:
            raise SimUnavailable("no sim connected")
        rid = self._next_id
        self._next_id += 1
        fut = self._loop.create_future()
        self._pending[rid] = fut
        try:
            await self._ws.send(json.dumps({"ns": "promptdrive", "kind": "req", "id": rid, "op": op, "args": list(args)}))
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise SimTimeout("op %r timed out" % op)
        finally:
            self._pending.pop(rid, None)

    def call(self, op, *args, timeout=None):
        """Synchronous facade call. Raises SimUnavailable / SimTimeout."""
        t = self.call_timeout if timeout is None else timeout
        fut = asyncio.run_coroutine_threadsafe(self._request(op, args, t), self._loop)
        try:
            return fut.result(timeout=t + 1.0)
        except (SimUnavailable, SimTimeout):
            raise
        except (asyncio.TimeoutError, TimeoutError):
            raise SimTimeout("op %r timed out (outer)" % op)

    def try_call(self, op, *args):
        """Never raises. Returns (ok: bool, result_or_error)."""
        try:
            return True, self.call(op, *args)
        except Exception as e:
            return False, {"error": type(e).__name__, "message": str(e)}

    # --- facade sugar -------------------------------------------------------
    def get(self, path=None):
        return self.call("get", path) if path is not None else self.call("get")

    def vehicle_get(self, keys=None):
        return self.call("vehicle.get", keys) if keys is not None else self.call("vehicle.get")

    def vehicle_set(self, diff):
        return self.call("vehicle.set", diff)

    def vehicle_reset(self, init_config, opts=None):
        return self.call("vehicle.reset", init_config, opts or {})

    def vehicle_snapshot(self):
        return self.call("vehicle.snapshot")

    def benchmark(self, on, opts=None):
        return self.call("vehicle.benchmark", bool(on), opts or {})

    def nav_display(self, meta):
        return self.call("vehicle.navDisplay", meta)

    # --- events -------------------------------------------------------------
    def subscribe(self, event, fn):
        self._subscribers.setdefault(event, []).append(fn)
        return lambda: self._subscribers.get(event, []).remove(fn)

    def on_sim_connected(self, fn):
        self._on_connect.append(fn)
        return lambda: self._on_connect.remove(fn)
