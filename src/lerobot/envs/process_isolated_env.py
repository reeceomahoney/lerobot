"""Process-isolated VectorEnv wrapper.

Runs a SyncVectorEnv in a separate process so that the environment's
rendering context (e.g. EGL for MuJoCo/LIBERO) does not interfere with
CUDA in the main (policy) process.  Communication happens over a single
``multiprocessing.Connection`` pair — one command at a time, fully
synchronous.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import traceback
from typing import Any, Callable, Sequence

import cloudpickle
import gymnasium as gym
import numpy as np


# ---- Child process ----------------------------------------------------------


def _child_main(
    conn: mp.connection.Connection,
    env_fns_bytes: bytes,
    autoreset_mode: Any,
) -> None:
    """Entry point for the worker process that owns the VectorEnv."""
    import sys

    env: gym.vector.SyncVectorEnv | None = None
    try:
        env_fns = pickle.loads(env_fns_bytes)
        env = gym.vector.SyncVectorEnv(env_fns, autoreset_mode=autoreset_mode)

        # Send back metadata the parent needs for proxying.
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
                print(f"[ProcessIsolatedEnv child] error handling {cmd!r}:\n{tb}", file=sys.stderr)
                conn.send(("error", tb))
    except Exception:
        tb = traceback.format_exc()
        print(f"[ProcessIsolatedEnv child] fatal error during init:\n{tb}", file=sys.stderr)
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


def _collect_metadata(env: gym.vector.SyncVectorEnv) -> dict[str, Any]:
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

    return {
        "num_envs": env.num_envs,
        "observation_space": env.observation_space,
        "single_observation_space": env.single_observation_space,
        "action_space": env.action_space,
        "single_action_space": env.single_action_space,
        "sub_env_attrs": sub_attrs,
    }


# ---- Proxy objects for env.envs[i] -----------------------------------------


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


# ---- Parent-side wrapper ----------------------------------------------------


class ProcessIsolatedVectorEnv:
    """A VectorEnv whose underlying environments live in a separate process.

    Presents the same interface that ``rollout()`` and helper functions in
    ``lerobot.envs.utils`` expect:

    * ``reset(seed=...)`` / ``step(action)``
    * ``call(name)``
    * ``num_envs``  (int property)
    * ``envs``      (list of proxy objects with cached attributes)
    * ``close()``
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Any]],
        autoreset_mode: Any = None,
    ) -> None:
        self._closed = False

        if autoreset_mode is None:
            autoreset_mode = gym.vector.AutoresetMode.NEXT_STEP

        # Serialize env_fns with cloudpickle (handles closures, lambdas, etc.)
        # so they survive the spawn boundary.
        env_fns_bytes = cloudpickle.dumps(list(env_fns))

        ctx = mp.get_context("spawn")  # spawn to get a clean CUDA/EGL state
        self._parent_conn, child_conn = ctx.Pipe()
        self._process = ctx.Process(
            target=_child_main,
            args=(child_conn, env_fns_bytes, autoreset_mode),
            daemon=True,
        )
        self._process.start()
        child_conn.close()  # parent doesn't use the child end

        # Wait for the child to be ready.
        try:
            status, payload = self._parent_conn.recv()
        except EOFError:
            self._process.join(timeout=10)
            raise RuntimeError(
                f"Child process died before sending data (exit code: {self._process.exitcode}). "
                "Check stderr for the child traceback. This often means the environment "
                "crashed during creation (e.g. EGL/MuJoCo init failure)."
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

        # Build proxy objects for env.envs[i].
        self.envs: list[_SubEnvProxy] = []
        for i, attrs in enumerate(meta["sub_env_attrs"]):
            proxy = _SubEnvProxy(attrs, render_fn=lambda idx=i: self._render_one(idx))
            self.envs.append(proxy)

    # -- Core VectorEnv interface ---------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._num_envs

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
        """Render a single sub-env (used by _SubEnvProxy.render)."""
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
