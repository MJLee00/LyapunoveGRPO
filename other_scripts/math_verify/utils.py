# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os
import sys
import time
from functools import wraps
from typing import Literal

from math_verify.errors import TimeoutException

TIMEOUT_WARNING_SHOWN = False
logger = logging.getLogger(__name__)


TimeoutMode = Literal["normal", "advanced"]


def _timeout_posix_normal(timeout_seconds: int):
    # Unix-like approach: signal.alarm
    import signal

    def decorator(func):
        def handler(signum, frame):
            raise TimeoutException("Operation timed out!")

        def wrapper(*args, **kwargs):
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_seconds)
            try:
                return func(*args, **kwargs)
            finally:
                # Cancel the alarm and restore previous handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


def _timeout_posix_advanced(timeout_seconds: int):
    import signal

    # Rearm interval after the first timeout signal fires.
    #
    # Why: In some environments (e.g. when garbage collection triggers weakref callbacks),
    # exceptions raised from a signal handler can be *ignored* by CPython (printed as
    # "Exception ignored in: ..."). Since `alarm()`/`setitimer()` is one-shot by default,
    # this would effectively "lose" the timeout signal and allow the wrapped function to
    # run forever. By re-arming, we guarantee another chance for the TimeoutException to
    # escape from a non-ignoring frame.
    #
    # Keep it small but not too aggressive to avoid busy signaling.
    rearm_seconds = min(0.1, max(0.01, timeout_seconds / 10))

    def decorator(func):
        def handler(signum, frame):
            # If this exception ends up being ignored (e.g. weakref callbacks),
            # make sure we get another SIGALRM soon.
            try:
                signal.setitimer(signal.ITIMER_REAL, rearm_seconds)
            except Exception:
                # Best-effort; we still raise below.
                pass
            raise TimeoutException("Operation timed out!")

        @wraps(func)
        def wrapper(*args, **kwargs):
            old_handler = signal.getsignal(signal.SIGALRM)
            old_timer = signal.getitimer(signal.ITIMER_REAL)
            start_time = time.monotonic()
            old_unraisablehook = getattr(sys, "unraisablehook", None)
            signal_handler_installed = False

            def _unraisablehook(args):
                # Suppress noisy "Exception ignored in: ..." messages when
                # SIGALRM happens to fire inside weakref callbacks / __del__.
                #
                # This does *not* affect the actual timeout behavior; it is
                # purely for log hygiene in long-running processes.
                if isinstance(getattr(args, "exc_value", None), TimeoutException):
                    return
                if old_unraisablehook is not None:
                    old_unraisablehook(args)

            try:
                signal.signal(signal.SIGALRM, handler)
                signal_handler_installed = True
                if old_unraisablehook is not None:
                    sys.unraisablehook = _unraisablehook

                # Use setitimer() for finer granularity than alarm() and to
                # simplify restoring a previous timer.
                signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
                return func(*args, **kwargs)
            finally:
                if signal_handler_installed:
                    # Cancel the alarm and restore previous handler
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)

                    # Restore a previously active timer if it existed.
                    #
                    # Note: we approximate by subtracting the time spent in `func`.
                    # If it should have already fired, we disable it instead of
                    # trying to "catch up" by firing it immediately.
                    if old_timer != (0.0, 0.0):
                        elapsed = time.monotonic() - start_time
                        remaining = max(0.0, old_timer[0] - elapsed)
                        if remaining > 0.0:
                            signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])
                if old_unraisablehook is not None:
                    sys.unraisablehook = old_unraisablehook

        return wrapper

    return decorator


def _timeout_windows(timeout_seconds: int):
    # Windows approach: use multiprocessing
    from multiprocessing import Process, Queue

    def decorator(func):
        def wrapper(*args, **kwargs):
            q = Queue()

            def run_func(q, args, kwargs):
                try:
                    result = func(*args, **kwargs)
                    q.put((True, result))
                except Exception as e:
                    q.put((False, e))

            p = Process(target=run_func, args=(q, args, kwargs))
            p.start()
            p.join(timeout_seconds)

            if p.is_alive():
                # Timeout: Terminate the process
                p.terminate()
                p.join()
                raise TimeoutException("Operation timed out!")

            # If we got here, the process completed in time.
            success, value = q.get()
            if success:
                return value
            else:
                # The child raised an exception; re-raise it here
                raise value

        return wrapper

    return decorator


def timeout(timeout_seconds: int | None = 10, mode: TimeoutMode = "normal"):  # noqa: C901
    """A decorator that applies a timeout to the decorated function.

    Args:
        timeout_seconds (int): Number of seconds before timing out the decorated function.
            Defaults to 10 seconds.
        mode (Literal["normal", "advanced"]): Timeout mode.
            - "normal": Original implementation. Uses `signal.alarm` on POSIX.
            - "advanced": Enhanced POSIX implementation with `signal.setitimer`,
              automatic rearm, and `sys.unraisablehook` suppression.

    Notes:
        On Unix systems, uses a signal-based alarm approach which is more efficient as it doesn't require spawning a new process.
        On Windows systems, uses a multiprocessing-based approach since signal.alarm is not available. This will incur a huge performance penalty.
    """
    if timeout_seconds is None or timeout_seconds <= 0:

        def no_timeout_decorator(func):
            return func

        return no_timeout_decorator

    if mode not in ("normal", "advanced"):
        raise ValueError(f"Unknown timeout mode: {mode}. Expected one of ('normal', 'advanced').")

    if os.name == "posix":
        if mode == "advanced":
            return _timeout_posix_advanced(timeout_seconds)
        return _timeout_posix_normal(timeout_seconds)

    return _timeout_windows(timeout_seconds)
