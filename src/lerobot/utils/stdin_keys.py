"""Stdin-based single-key listener.

Unlike pynput, this only sees keys typed into the controlling terminal
(the focused tmux pane / shell window), not OS-wide input — so background
panes and other windows won't trigger callbacks.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import termios
import tty
from threading import Event, Thread
from typing import Callable

logger = logging.getLogger(__name__)

ESC = "\x1b"


class StdinKeyListener:
    """Daemon-thread keystroke listener that reads single chars from stdin.

    The callback receives the typed character. Escape (and any trailing
    CSI sequence such as arrow keys) is delivered as the single string
    ``"\\x1b"`` — exported as :data:`ESC` for convenience.
    """

    def __init__(self, on_press: Callable[[str], None]):
        self.on_press = on_press
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.saved_attrs: list | None = None
        self.fd: int | None = None

    def start(self) -> bool:
        """Start the listener.  Returns False if stdin is not a TTY."""
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY — key listener disabled")
            return False
        self.fd = sys.stdin.fileno()
        try:
            self.saved_attrs = termios.tcgetattr(self.fd)
        except termios.error as e:
            logger.warning("Could not read terminal attrs: %s", e)
            return False
        tty.setcbreak(self.fd)
        self.thread = Thread(target=self.run, name="stdin-keys", daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        self.stop_event.set()
        if self.fd is not None and self.saved_attrs is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved_attrs)
            except termios.error:
                pass
        if self.thread is not None:
            self.thread.join(timeout=0.5)

    def run(self) -> None:
        assert self.fd is not None
        while not self.stop_event.is_set():
            r, _, _ = select.select([self.fd], [], [], 0.1)
            if not r:
                continue
            try:
                ch = os.read(self.fd, 1).decode(errors="ignore")
            except OSError:
                break
            if not ch:
                continue
            if ch == ESC:
                # Drain any trailing CSI bytes (arrow keys etc.) so the next
                # read isn't desynchronised; we still report a bare ESC.
                r2, _, _ = select.select([self.fd], [], [], 0.01)
                if r2:
                    try:
                        os.read(self.fd, 16)
                    except OSError:
                        pass
                try:
                    self.on_press(ESC)
                except Exception:
                    logger.exception("stdin key callback failed")
            else:
                try:
                    self.on_press(ch)
                except Exception:
                    logger.exception("stdin key callback failed")
