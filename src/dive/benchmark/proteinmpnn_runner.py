
from __future__ import annotations

from dive.codirect_paths import joined

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dive.signed_value.roots import DENIED_PREFIXES, RUNTIME_PYTHON

class ProteinMPNNRunnerError(RuntimeError):
    pass

PROTEINMPNN_DIR = Path(
    joined('PROTEINA_COMPLEXA_ROOT', 'community_models', 'ProteinMPNN')
)
VANILLA_WEIGHTS = PROTEINMPNN_DIR / "vanilla_model_weights"
WEIGHT_FILE = VANILLA_WEIGHTS / "v_48_020.pt"

@dataclass(frozen=True, slots=True)
class ProteinMPNNRun:
    pdb_path: str
    fasta_path: str
    sequences: tuple[str, ...]
    returncode: int
    stderr: str
    command: tuple[str, ...]

def run_proteinmpnn(
    pdb_path: Path,
    output_dir: Path,
    *,
    num_seq: int = 1,
    seed: int = 42,
    timeout_s: int = 300,
) -> ProteinMPNNRun:
    pdb = Path(pdb_path)
    dest = Path(output_dir)
    if any(str(dest).startswith(str(prefix)) for prefix in DENIED_PREFIXES):
        raise ProteinMPNNRunnerError(f"denied output {dest}")
    if not pdb.is_file():
        raise ProteinMPNNRunnerError(f"missing pdb {pdb}")
    if not WEIGHT_FILE.is_file():
        raise ProteinMPNNRunnerError(f"missing official weights {WEIGHT_FILE}")
    dest.mkdir(parents=True, exist_ok=True)
    command = (
        str(RUNTIME_PYTHON),
        str(PROTEINMPNN_DIR / "protein_mpnn_run.py"),
        "--pdb_path",
        str(pdb),
        "--out_folder",
        str(dest),
        "--num_seq_per_target",
        str(num_seq),
        "--sampling_temp",
        "0.1",
        "--seed",
        str(seed),
        "--path_to_model_weights",
        str(VANILLA_WEIGHTS) + "/",
        "--model_name",
        "v_48_020",
        "--batch_size",
        "1",
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(PROTEINMPNN_DIR),
    )
    fasta_hits = sorted(dest.rglob("*.fa")) + sorted(dest.rglob("*.fasta"))
    sequences: list[str] = []
    fasta_path = fasta_hits[0] if fasta_hits else dest / "missing.fa"
    if fasta_hits:
        text = fasta_hits[0].read_text(encoding="utf-8")
        chunks: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith(">"):
                if current:
                    chunks.append("".join(current))
                    current = []
            else:
                current.append(line.strip())
        if current:
            chunks.append("".join(current))
        sequences = chunks
    (dest / "subprocess.json").write_text(
        json.dumps(
            {
                "command": list(command),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ProteinMPNNRun(
        pdb_path=str(pdb),
        fasta_path=str(fasta_path),
        sequences=tuple(sequences),
        returncode=completed.returncode,
        stderr=completed.stderr,
        command=command,
    )
