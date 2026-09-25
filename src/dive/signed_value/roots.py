
from __future__ import annotations

import sys
from pathlib import Path

from dive.codirect_paths import ROOT, cache_dir, joined

EXECUTION_HOST = "local"

REPO_ROOT = ROOT

EVIDENCE_ROOT = cache_dir()

DCC_MIRROR_ROOT = cache_dir("dcc-mirror")

UPSTREAM_ROOT = joined(
    "PROTEINA_COMPLEXA_ROOT", default=ROOT / "vendor" / "proteina-complexa")
AF2_ROOT = joined("CODIRECT_AF2_ROOT", default=ROOT / "external" / "af2")
RUNTIME_PYTHON = Path(sys.executable)

DENIED_PREFIXES: tuple[Path, ...] = ()

EMERGENT_UPSTREAM_ROOT = UPSTREAM_ROOT
EMERGENT_UPSTREAM_COMMIT = "32b71ae1d9a8767414eae3921cf35969430db82f"

EMERGENT_EVIDENCE_ROOT = EVIDENCE_ROOT / "emergent"

EMERGENT_BULK_ROOT = joined("CODIRECT_STRUCTURE_ROOT", default=ROOT / "structures")
