
from __future__ import annotations

from collections.abc import Iterable

ENTRYPOINT = "scripts/emergent/train.py"

_NOT_THE_RUN = ("chain_next_stage.py", "record_completion.py", "python -c", "-c import")

def run_is_training(run_id: str, lines: Iterable[str]) -> bool:

    for line in lines:
        if ENTRYPOINT not in line:
            continue
        if any(marker in line for marker in _NOT_THE_RUN):
            continue

        if f"{ENTRYPOINT} " not in line and not line.rstrip().endswith(ENTRYPOINT):
            continue
        fields = line.split()
        executed = {
            fields[index + 1]
            for index, field in enumerate(fields[:-1])
            if field == "--run-id"
        }
        if run_id in executed:
            return True
    return False

def training_processes(run_id: str) -> bool:

    import subprocess

    result = subprocess.run(
        ["ps", "-eo", "args"], capture_output=True, text=True, check=False
    )
    return run_is_training(run_id, result.stdout.splitlines())
