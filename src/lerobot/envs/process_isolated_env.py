"""Process-isolated VectorEnv wrapper for LIBERO.

Runs each SyncVectorEnv in a separate ``spawn``ed process so that the
environment's EGL rendering context never coexists with CUDA in the
policy process.

The parent process **never** imports ``libero`` / ``robosuite`` / MuJoCo.
Only plain config dicts cross the process boundary — the child imports
everything from scratch.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections import defaultdict
from typing import Any, Callable

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------------------
#  Public entry point — called from factory.py instead of create_libero_envs
# ---------------------------------------------------------------------------


def create_process_isolated_libero_envs(
    task: str,
    n_envs: int,
    camera_name: str,
    init_states: bool,
    gym_kwargs: dict[str, Any] | None,
    control_mode: str,
    episode_length: int | None,
) -> dict[str, dict[int, "ProcessIsolatedVectorEnv"]]:
    """Create LIBERO envs, each in its own child process.

    Mirrors the return type of ``create_libero_envs`` but every VectorEnv
    lives in a separate ``spawn``-ed process.  The parent never imports
    ``libero``.
    """
    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)

    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    # Discover task IDs in a short-lived child (avoids importing libero here).
    suite_task_ids = _discover_tasks(suite_names, task_ids_filter)

    print(
        f"Creating process-isolated LIBERO envs | suites={suite_names} "
        f"| n_envs(per task)={n_envs} | init_states={init_states}"
    )

    out: dict[str, dict[int, ProcessIsolatedVectorEnv]] = defaultdict(dict)
    for suite_name in suite_names:
        for tid in suite_task_ids[suite_name]:
            config = {
                "suite_name": suite_name,
                "task_id": tid,
                "n_envs": n_envs,
                "camera_name": camera_name,
                "init_states": init_states,
                "gym_kwargs": gym_kwargs,
                "control_mode": control_mode,
                "episode_length": episode_length,
            }
            out[suite_name][tid] = ProcessIsolatedVectorEnv(config)
            print(f"  Built isolated env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    return {suite: dict(task_map) for suite, task_map in out.items()}


# ---------------------------------------------------------------------------
#  Task-ID discovery  (short-lived spawn child — no MuJoCo init)
# ---------------------------------------------------------------------------


def _discover_tasks(
    suite_names: list[str],
    task_ids_filter: list[int] | None,
) -> dict[str, list[int]]:
    """Spawn a tiny child to discover how many tasks each suite has."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(
        target=_discover_worker,
        args=(child_conn, suite_names, task_ids_filter),
        daemon=True,
    )
    p.start()
    child_conn.close()
    try:
        status, payload = parent_conn.recv()
    except EOFError:
        p.join(timeout=10)
        raise RuntimeError(
            f"Task-discovery child died (exit code {p.exitcode}). "
            "Check stderr for details."
        )
    p.join(timeout=10)
    parent_conn.close()
    if status == "error":
        raise RuntimeError(f"Task discovery failed:\n{payload}")
    return payload


def _discover_worker(conn, suite_names, task_ids_filter):
    """Child: import libero, return {suite: [task_ids]}."""
    try:
        from lerobot.envs.libero import _get_suite, _select_task_ids

        result = {}
        for name in suite_names:
            suite = _get_suite(name)
            total = len(suite.tasks)
            result[name] = _select_task_ids(total, task_ids_filter)
        conn.send(("ok", result))
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
#  Per-env child process
# ---------------------------------------------------------------------------


def _env_worker(conn: mp.connection.Connection, config: dict[str, Any]) -> None:
    """Entry point for each env child process.

    Imports libero from scratch (clean ``spawn``), creates a single
    SyncVectorEnv, then serves commands over the pipe.
    """
    import faulthandler
    import sys

    faulthandler.enable(file=sys.stderr, all_threads=True)

    env = None
    try:
        import gymnasium as gym
        from lerobot.envs.libero import _get_suite, _make_env_fns, _parse_camera_names

        suite = _get_suite(config["suite_name"])
        camera_names = _parse_camera_names(config["camera_name"])
        fns = _make_env_fns(
            suite=suite,
            suite_name=config["suite_name"],
            task_id=config["task_id"],
            n_envs=config["n_envs"],
            camera_names=camera_names,
            episode_length=config["episode_length"],
            init_states=config["init_states"],
            gym_kwargs=config["gym_kwargs"],
            control_mode=config["control_mode"],
        )
        env = gym.vector.SyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)

        meta = _collect_metadata(env)
        conn.send(("ready", meta))

        # Command loop.
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break

            cmd = msg[0]
            try:
                if cmd == "reset":
                    _, kwargs = msg
                    result = env.reset(**kwargs)
                    conn.send(("ok", result))
                elif cmd == "step":
                    _, action = msg
                    result = env.step(action)
                    conn.send(("ok", result))
                elif cmd == "call":
                    _, name = msg
                    result = env.call(name)
                    conn.send(("ok", result))
                elif cmd == "render":
                    frames = [e.render() for e in env.envs]
                    conn.send(("ok", frames))
                elif cmd == "close":
                    break
                else:
                    conn.send(("error", f"unknown command: {cmd}"))
            except Exception:
                tb = traceback.format_exc()
                print(f"[ProcessIsolatedEnv child] error in {cmd!r}:\n{tb}", file=sys.stderr)
                conn.send(("error", tb))
    except Exception:
        tb = traceback.format_exc()
        print(f"[ProcessIsolatedEnv child] fatal error:\n{tb}", file=sys.stderr)
        try:
            conn.send(("error", tb))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        conn.close()


def _collect_metadata(env) -> dict[str, Any]:
    """Gather picklable metadata from the live env for the parent proxy."""
    sub_attrs: list[dict[str, Any]] = []
    for e in env.envs:
        attrs: dict[str, Any] = {}
        for name in ("task_description", "task", "_max_episode_steps"):
            if hasattr(e, name):
                val = getattr(e, name)
                if not callable(val):
                    attrs[name] = val
        attrs["__class_name__"] = type(e).__name__
        sub_attrs.append(attrs)

    # Grab metadata from the first sub-env (used for render_fps etc.)
    metadata = getattr(env.envs[0], "metadata", {}) if env.envs else {}

    return {
        "num_envs": env.num_envs,
        "observation_space": env.observation_space,
        "single_observation_space": env.single_observation_space,
        "action_space": env.action_space,
        "single_action_space": env.single_action_space,
        "sub_env_attrs": sub_attrs,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
#  Proxy objects for env.envs[i]
# ---------------------------------------------------------------------------


class _SubEnvProxy:
    """Lightweight stand-in for ``env.envs[i]`` that carries cached attributes."""

    def __init__(self, attrs: dict[str, Any], render_fn: Callable[[], np.ndarray]) -> None:
        self._attrs = attrs
        self._render_fn = render_fn

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and name not in ("_max_episode_steps", "_attrs", "_render_fn"):
            raise AttributeError(name)
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(f"Sub-env proxy has no attribute {name!r}") from None

    def render(self) -> np.ndarray:
        return self._render_fn()


# ---------------------------------------------------------------------------
#  Parent-side wrapper
# ---------------------------------------------------------------------------


class ProcessIsolatedVectorEnv:
    """A VectorEnv whose underlying environments live in a separate process.

    Presents the same interface that ``rollout()`` and helper functions in
    ``lerobot.envs.utils`` expect.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._closed = False

        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._process = ctx.Process(
            target=_env_worker,
            args=(child_conn, config),
            daemon=True,
        )
        self._process.start()
        child_conn.close()

        # Wait for the child to be ready.
        try:
            status, payload = self._parent_conn.recv()
        except EOFError:
            self._process.join(timeout=30)
            raise RuntimeError(
                f"Child process died before sending data (exit code: {self._process.exitcode}). "
                "Check stderr for the child traceback."
            )
        if status == "error":
            raise RuntimeError(f"Child process failed during init:\n{payload}")
        assert status == "ready"
        meta: dict[str, Any] = payload

        self._num_envs: int = meta["num_envs"]
        self.observation_space = meta["observation_space"]
        self.single_observation_space = meta["single_observation_space"]
        self.action_space = meta["action_space"]
        self.single_action_space = meta["single_action_space"]
        self.metadata = meta.get("metadata", {})

        # Build proxy objects for env.envs[i].
        self.envs: list[_SubEnvProxy] = []
        for i, attrs in enumerate(meta["sub_env_attrs"]):
            proxy = _SubEnvProxy(attrs, render_fn=lambda idx=i: self._render_one(idx))
            self.envs.append(proxy)

    # -- Core VectorEnv interface ---------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def unwrapped(self):
        """Return self so that ``env.unwrapped.metadata`` works."""
        return self

    def reset(self, *, seed: list[int] | int | None = None, options: dict | None = None) -> tuple:
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options
        return self._send("reset", kwargs)

    def step(self, action: np.ndarray) -> tuple:
        return self._send("step", action)

    def call(self, name: str) -> Any:
        return self._send("call", name)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._parent_conn.send(("close",))
        except (BrokenPipeError, OSError):
            pass
        self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.kill()
        self._parent_conn.close()

    # -- Rendering helpers ----------------------------------------------------

    def render(self) -> list[np.ndarray]:
        """Render all sub-envs and return list of frames."""
        return self._send("render")

    def _render_one(self, idx: int) -> np.ndarray:
        frames = self.render()
        return frames[idx]

    # -- Internal -------------------------------------------------------------

    def _send(self, cmd: str, payload: Any = None) -> Any:
        if self._closed:
            raise RuntimeError("Env is closed")
        self._parent_conn.send((cmd, payload) if payload is not None else (cmd,))
        try:
            status, result = self._parent_conn.recv()
        except EOFError:
            self._process.join(timeout=10)
            raise RuntimeError(
                f"Child process died during {cmd!r} (exit code: {self._process.exitcode}). "
                "Check stderr for the child traceback."
            )
        if status == "error":
            raise RuntimeError(f"Error in child process:\n{result}")
        return result

    def __del__(self) -> None:
        self.close()
