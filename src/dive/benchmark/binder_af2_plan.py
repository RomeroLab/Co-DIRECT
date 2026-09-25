
from __future__ import annotations

from dive.codirect_paths import cache_dir

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dive.benchmark.binder_af2_cells import (
    binder_chain,
    hotspot_target_positions,
    chain_residues,
    chain_sequence,
    parse_mpnn_fasta,
)

BENCH_ROOT = Path(
    cache_dir('emergent', 'benchmarks', 'codirect-public-dev-v2')
)

METHODS = ("rfd", "proteina", "proteina_self")

@dataclass
class Cell:
    example_id: str
    pdb_id: str
    method: str
    replicate: int
    seed: int
    sequence_source: str
    sequence: str
    sequence_span_length: int
    sequence_resolved_length: int
    unresolved_positions: list[int]
    binder_chain: str
    target_chains: list[str]
    target_template: str
    generated_complex: str
    generated_binder_chain: str
    hotspots: list[str]
    hotspot_target_positions: list[int]
    hotspots_unmapped: list[str]
    hotspot_source: str | None
    fasta: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.method}_{self.pdb_id}"

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _proteina_dir(example_id: str) -> Path:
    return BENCH_ROOT / "proteina/binder" / example_id

def _rfd_dir(example_id: str) -> Path | None:
    _, pdb_id, chain_spec = example_id.split(":")
    for candidate in (
        BENCH_ROOT / "rfdiffusion" / f"binder_{pdb_id}_{chain_spec}",
        BENCH_ROOT / "rfdiffusion" / f"binder_{pdb_id}",
    ):
        if (candidate / "design_0.pdb").is_file():
            return candidate
    return None

def _hotspots(path: Path) -> tuple[list[str], str | None]:
    if not path.is_file():
        return [], None
    source = None
    tokens: list[str] = []
    for line in _read(path).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("source="):
            source = line.split("=", 1)[1]
            continue
        tokens.extend(t for t in line.split(",") if t)
    return tokens, source

def build_cells(example_ids: list[str], out_dir: Path) -> list[Cell]:

    from dive.benchmark.binder_af2_cells import renumber_target_template

    out_dir.mkdir(parents=True, exist_ok=True)
    cells: list[Cell] = []
    for example_id in example_ids:
        _, pdb_id, _ = example_id.split(":")
        pdir = _proteina_dir(example_id)
        complex_pdbs = [
            p for p in sorted(pdir.glob("*.pdb")) if not p.name.endswith(".native_target.pdb")
        ]
        if len(complex_pdbs) != 1:
            raise ValueError(f"{example_id}: expected one complex pdb, got {complex_pdbs!r}")
        complex_pdb = complex_pdbs[0]
        native_pdb = pdir / complex_pdb.name.replace(".pdb", ".native_target.pdb")

        complex_text = _read(complex_pdb)
        native_text = _read(native_pdb)
        target_chains = list(chain_residues(native_text))
        b_chain = binder_chain(chain_residues(complex_text), target_chains)

        template = out_dir / f"target_{pdb_id}_template.pdb"
        template.write_text(renumber_target_template(native_text, target_chains), encoding="utf-8")

        rfd_dir = _rfd_dir(example_id)
        hotspots, hotspot_source = _hotspots(rfd_dir / "hotspots.txt") if rfd_dir else ([], None)

        positions, unmapped = hotspot_target_positions(native_text, target_chains, hotspots)
        common = dict(
            hotspots=hotspots,
            hotspot_target_positions=positions,
            hotspots_unmapped=unmapped,
            hotspot_source=hotspot_source,
        )

        proteina_fasta = pdir / "if1_designed/seqs" / complex_pdb.name.replace(".pdb", ".fa")
        if proteina_fasta.is_file():
            parsed = parse_mpnn_fasta(_read(proteina_fasta))
            cells.append(
                Cell(
                    example_id=example_id, pdb_id=pdb_id, method="proteina",
                    replicate=1, seed=42001,
                    sequence_source="proteinmpnn_if1", sequence=parsed.designed_sequence,
                    sequence_span_length=parsed.span_length,
                    sequence_resolved_length=parsed.resolved_length,
                    unresolved_positions=parsed.unresolved_positions,
                    binder_chain=b_chain, target_chains=target_chains,
                    target_template=str(template), generated_complex=str(complex_pdb),
                    generated_binder_chain=b_chain,
                    fasta=str(proteina_fasta), **common,
                )
            )

        self_seq = chain_sequence(complex_text, b_chain)
        cells.append(
            Cell(
                example_id=example_id, pdb_id=pdb_id, method="proteina_self",
                replicate=1, seed=42001,
                sequence_source="proteina_generated_sequence", sequence=self_seq.replace("X", ""),
                sequence_span_length=len(self_seq),
                sequence_resolved_length=len(self_seq.replace("X", "")),
                unresolved_positions=[i for i, c in enumerate(self_seq) if c == "X"],
                binder_chain=b_chain, target_chains=target_chains,
                target_template=str(template), generated_complex=str(complex_pdb),
                generated_binder_chain=b_chain,
                fasta=None, **common,
            )
        )

        if rfd_dir is not None:
            rfd_fasta = rfd_dir / "if1_chainA/seqs/design_0.fa"
            rfd_design = rfd_dir / "design_0.pdb"
            rfd_chains = chain_residues(_read(rfd_design))
            rfd_target_chains = list(chain_residues(_read(rfd_dir / "target.pdb")))
            rfd_binder = binder_chain(rfd_chains, rfd_target_chains)
            if rfd_fasta.is_file():
                parsed = parse_mpnn_fasta(_read(rfd_fasta))
                notes = []
                if parsed.designed_chains != [rfd_binder]:
                    notes.append(
                        f"IF designed {parsed.designed_chains!r} but the diffused "
                        f"chain is {rfd_binder!r}"
                    )
                cells.append(
                    Cell(
                        example_id=example_id, pdb_id=pdb_id, method="rfd",
                        replicate=1, seed=42001,
                        sequence_source="proteinmpnn_if1", sequence=parsed.designed_sequence,
                        sequence_span_length=parsed.span_length,
                        sequence_resolved_length=parsed.resolved_length,
                        unresolved_positions=parsed.unresolved_positions,
                        binder_chain=b_chain, target_chains=target_chains,
                        target_template=str(template), generated_complex=str(rfd_design),
                        generated_binder_chain=rfd_binder,
                        fasta=str(rfd_fasta), notes=notes, **common,
                    )
                )
    return cells

def write_plan(cells: list[Cell], path: Path) -> None:
    path.write_text(
        json.dumps({"cells": [asdict(c) | {"name": c.name} for c in cells]}, indent=2),
        encoding="utf-8",
    )
