
from __future__ import annotations

import time
from collections.abc import Callable

Outcome = str

def await_stage(
    running: Callable[[], bool],
    recorded: Callable[[], bool],
    *,
    start_timeout_s: float,
    finish_timeout_s: float,
    poll_s: float = 300.0,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    report: Callable[[str], None] | None = None,
) -> Outcome:

    def say(message: str) -> None:
        if report is not None:
            report(message)

    up = running()
    if not up:
        if recorded():
            say("the awaited stage already recorded a verdict; not waiting to start")
            return "already-finished"
        say("waiting for the awaited stage to START")
        started = now()
        while not up:
            if recorded():
                return "already-finished"
            if now() - started >= start_timeout_s:
                return "never-started"
            sleep(poll_s)
            up = running()

    say("the awaited stage is up; waiting for it to finish")
    started = now()
    while running():
        if now() - started >= finish_timeout_s:
            return "never-finished"
        sleep(poll_s)
    return "finished"
