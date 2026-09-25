
from __future__ import annotations

import json
from pathlib import Path

def load_templates(jobs_jsonl: Path) -> dict:

    out = {}
    with jobs_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            out.setdefault(j["task_id"], j)
    return out

def job_from_meta(meta: dict, template: dict, *, method_id: str = "ANTIC") -> dict:

    seq = meta["sequence"]
    dlen = int(meta.get("design_length") or len(seq))
    design_seq = seq[:dlen]
    task = meta["task_id"]
    idx = int(meta["candidate_index"])
    cid = f"{method_id}__{task}__candidate_{idx}"
    chains = []
    replaced = False
    for ch in template["chains"]:
        c = dict(ch)
        if c.get("role") == "design" and c.get("kind") == "protein":
            c["sequence"] = design_seq
            replaced = True
        chains.append(c)
    if not replaced:
        raise ValueError(f"template for {task} has no design protein chain")
    job = dict(template)
    job["candidate_id"] = cid
    job["generation_id"] = cid
    job["method_id"] = method_id
    job["task_id"] = task
    job["family"] = meta.get("family") or template.get("family")
    job["chains"] = chains
    prov = dict(template.get("provenance") or {})
    pdb = meta.get("pdb")
    if pdb:
        prov["backbone_path"] = pdb
    prov["candidate_index"] = idx
    if meta.get("motif_positions"):
        prov["motif_positions"] = meta["motif_positions"]
    job["provenance"] = prov
    return job
