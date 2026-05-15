# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Episodic rollout strategy: per-episode recording with manual env reset."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock

import numpy as np

from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.stdin_keys import ESC, StdinKeyListener
from lerobot.utils.utils import log_say

from ..configs import EpisodicStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)


class EpisodicStrategy(RolloutStrategy):
    """Per-episode autonomous rollout with manual environment reset.

    Each iteration of the outer loop is one episode:

    1. Reset the inference engine (clear hidden state / RTC queue).
    2. Run the policy until either ``max_episode_steps`` is reached or
       the user presses ``end_key``.
    3. Smoothly interpolate the robot back to its captured initial
       position via :meth:`RolloutStrategy._return_to_initial_position`.
    4. Block until the user labels the episode as success (``success_key``)
       or failure (``failure_key``); the label is written to every frame's
       ``success`` column.
    5. Save the episode and (optionally) push to the Hub.

    Requires ``streaming_encoding=True`` (enforced in config validation)
    so ``dataset.add_frame`` does not block the control loop.
    """

    config: EpisodicStrategyConfig

    def __init__(self, config: EpisodicStrategyConfig):
        super().__init__(config)
        self._end_episode = Event()
        self._discard_episode = Event()
        self._mark_success = Event()
        self._mark_failure = Event()
        self._listener = None
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._episode_lock = Lock()

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)
        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="episodic-push")
        self._setup_keyboard(ctx.runtime.shutdown_event)
        logger.info(
            "Episodic strategy ready (max_episode_steps=%d, end='%s', discard='%s', success='%s', failure='%s')",
            self.config.max_episode_steps,
            self.config.end_key,
            self.config.discard_key,
            self.config.success_key,
            self.config.failure_key,
        )

    def run(self, ctx: RolloutContext) -> None:
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        interpolator = self._interpolator
        engine = self._engine
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)
        play_sounds = cfg.play_sounds
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        max_steps = self.config.max_episode_steps
        start_time = time.perf_counter()
        episodes_since_push = 0

        with VideoEncodingManager(dataset):
            while not ctx.runtime.shutdown_event.is_set():
                if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                    logger.info("Duration limit reached (%.0fs)", cfg.duration)
                    break

                # Fresh state at the start of each episode — the human resets
                # the environment too, so policy memory shouldn't carry over.
                engine.reset()
                interpolator.reset()
                self._warmup_flushed = False
                self._cached_obs_processed = None
                self._end_episode.clear()
                self._discard_episode.clear()
                self._mark_success.clear()
                self._mark_failure.clear()
                engine.resume()

                logger.info(
                    "Episode %d started (end='%s', discard='%s')",
                    dataset.num_episodes + 1,
                    self.config.end_key,
                    self.config.discard_key,
                )
                log_say(f"Episode {dataset.num_episodes + 1} started", play_sounds)

                steps = self.run_episode(ctx, control_interval, max_steps, features, task_str)

                discarded = self._discard_episode.is_set()
                if discarded:
                    with self._episode_lock:
                        dataset.clear_episode_buffer()
                    logger.info("Episode discarded (steps=%d)", steps)
                    log_say("Episode discarded", play_sounds)

                # Re-arm so `discard_key` works again during the label prompt
                # below (it's the operator's escape hatch for misclicks).
                self._discard_episode.clear()

                # Pause inference and drift the arm home, then wait for the
                # operator to reset the scene.  The same post-episode path
                # runs whether the episode was saved or discarded so the
                # robot doesn't barrel into the next attempt mid-motion.
                engine.pause()
                logger.info("Returning robot to initial position...")
                if ctx.hardware.initial_position:
                    self._return_to_initial_position(ctx.hardware)

                if discarded:
                    logger.info(
                        "Reset the environment, then press '%s' or '%s' to continue (ESC to stop).",
                        self.config.success_key,
                        self.config.failure_key,
                    )
                    log_say("Reset the environment", play_sounds)
                else:
                    logger.info(
                        "Reset the environment, then press '%s' for success, '%s' for failure, or '%s' to discard (ESC to stop).",
                        self.config.success_key,
                        self.config.failure_key,
                        self.config.discard_key,
                    )
                    log_say("Mark success or failure", play_sounds)
                while (
                    not self._mark_success.is_set()
                    and not self._mark_failure.is_set()
                    and not self._discard_episode.is_set()
                    and not ctx.runtime.shutdown_event.is_set()
                ):
                    precise_sleep(0.05)

                if ctx.runtime.shutdown_event.is_set():
                    if not discarded:
                        # Drop the unlabeled episode rather than guess a label.
                        with self._episode_lock:
                            dataset.clear_episode_buffer()
                        logger.info("Shutdown before labeling — episode discarded")
                    break

                if not discarded and self._discard_episode.is_set():
                    # Operator chose to throw away the just-finished episode
                    # at the label prompt instead of marking it.
                    with self._episode_lock:
                        dataset.clear_episode_buffer()
                    logger.info("Episode discarded at label prompt (steps=%d)", steps)
                    log_say("Episode discarded", play_sounds)
                    discarded = True

                if discarded:
                    continue

                success_label = self._mark_success.is_set()
                buf = dataset.writer.episode_buffer
                buf["success"] = [
                    np.array([success_label], dtype=bool) for _ in range(buf["size"])
                ]

                end_reason = (
                    "user" if self._end_episode.is_set()
                    else ("max_steps" if max_steps > 0 and steps >= max_steps else "shutdown")
                )
                with self._episode_lock:
                    dataset.save_episode()
                logger.info(
                    "Episode %d saved (steps=%d, reason=%s, success=%s)",
                    dataset.num_episodes,
                    steps,
                    end_reason,
                    success_label,
                )
                log_say(
                    f"Episode {dataset.num_episodes} saved as {'success' if success_label else 'failure'}",
                    play_sounds,
                )

                episodes_since_push += 1
                if (
                    self.config.upload_every_n_episodes > 0
                    and episodes_since_push >= self.config.upload_every_n_episodes
                ):
                    self.background_push(dataset, cfg)
                    episodes_since_push = 0

    def run_episode(
        self,
        ctx: RolloutContext,
        control_interval: float,
        max_steps: int,
        features: dict,
        task_str: str,
    ) -> int:
        """Inner per-episode control loop.  Returns steps recorded."""
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        interpolator = self._interpolator
        steps = 0

        while not ctx.runtime.shutdown_event.is_set():
            if self._end_episode.is_set() or self._discard_episode.is_set():
                break
            if max_steps > 0 and steps >= max_steps:
                logger.info("Max episode steps reached (%d)", max_steps)
                break

            loop_start = time.perf_counter()
            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)

            if action_dict is not None:
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                # `success` is a placeholder here; the real episode-level
                # label is patched into the buffer after the user marks it.
                frame = {
                    **obs_frame,
                    **action_frame,
                    "task": task_str,
                    "success": np.array([False], dtype=bool),
                }
                dataset.add_frame(frame)
                steps += 1

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                )

        return steps

    def teardown(self, ctx: RolloutContext) -> None:
        play_sounds = ctx.runtime.cfg.play_sounds
        logger.info("Stopping episodic recording")
        log_say("Stopping episodic recording", play_sounds)

        if self._listener is not None:
            logger.info("Stopping keyboard listener")
            self._listener.stop()

        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()
            if ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
                logger.info("Pushing final dataset to hub...")
                if safe_push_to_hub(
                    ctx.data.dataset,
                    tags=ctx.runtime.cfg.dataset.tags,
                    private=ctx.runtime.cfg.dataset.private,
                ):
                    logger.info("Dataset uploaded to hub")
                    log_say("Dataset uploaded to hub", play_sounds)

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Episodic strategy teardown complete")

    def _setup_keyboard(self, shutdown_event: Event) -> None:
        end_key = self.config.end_key
        discard_key = self.config.discard_key
        success_key = self.config.success_key
        failure_key = self.config.failure_key

        def on_press(ch: str) -> None:
            if ch == end_key:
                self._end_episode.set()
            elif ch == discard_key:
                self._discard_episode.set()
                self._end_episode.set()
            elif ch == success_key:
                self._mark_success.set()
            elif ch == failure_key:
                self._mark_failure.set()
            elif ch == ESC:
                # Treat ESC as both end-of-episode and shutdown so the
                # inner loop unblocks promptly.
                self._end_episode.set()
                shutdown_event.set()

        self._listener = StdinKeyListener(on_press)
        if not self._listener.start():
            self._listener = None
            logger.warning("Keyboard listener disabled (stdin not a TTY)")
            return
        logger.info(
            "Keyboard listener started (end='%s', discard='%s', success='%s', failure='%s', ESC=stop)",
            end_key,
            discard_key,
            success_key,
            failure_key,
        )

    def background_push(self, dataset, cfg) -> None:
        if self._push_executor is None:
            return

        if self._pending_push is not None and not self._pending_push.done():
            logger.info("Previous push still in progress; queueing next")

        def push():
            try:
                with self._episode_lock:
                    if safe_push_to_hub(
                        dataset,
                        tags=cfg.dataset.tags if cfg.dataset else None,
                        private=cfg.dataset.private if cfg.dataset else False,
                    ):
                        logger.info("Background push to hub complete")
            except Exception as e:
                logger.error("Background push failed: %s", e)

        self._pending_push = self._push_executor.submit(push)
        logger.info("Background push task submitted")
