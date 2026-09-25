
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
_TIMEOUT_SECONDS = 60

class MiningError(RuntimeError):
    pass

def _urlopen(url: str) -> bytes:

    with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
        return response.read()

def _run(query: dict, *, opener: Callable[[str], bytes]) -> dict | None:

    url = f"{SEARCH_URL}?json={urllib.parse.quote(json.dumps(query))}"
    try:
        body = opener(url)
    except OSError as error:
        raise MiningError(f"search failed: {error}") from error
    if not body or not body.strip():
        return None
    try:
        return json.loads(body)
    except ValueError as error:
        raise MiningError(f"search response is not JSON: {error}") from error

def search_count(query: dict, *, opener: Callable[[str], bytes] = _urlopen) -> int:

    payload = _run(query, opener=opener)
    return 0 if payload is None else int(payload.get("total_count", 0))

def search_ids(
    query: dict,
    *,
    opener: Callable[[str], bytes] = _urlopen,
    rows: int = 100,
    start: int = 0,
) -> tuple[str, ...]:

    paged = dict(query)
    options = dict(paged.get("request_options") or {})
    options["paginate"] = {"start": int(start), "rows": int(rows)}
    options.setdefault("results_content_type", ["experimental"])
    paged["request_options"] = options

    payload = _run(paged, opener=opener)
    if payload is None:
        return ()
    return tuple(str(r["identifier"]) for r in payload.get("result_set") or [])

def _text(attribute: str, operator: str, value: object) -> dict:

    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }

def candidate_query(*, min_deposit: str, max_resolution: float) -> dict:

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                _text("rcsb_entry_info.polymer_entity_count_protein", "equals", 2),
                _text(
                    "rcsb_entry_info.selected_polymer_entity_types",
                    "exact_match",
                    "Protein (only)",
                ),
                _text(
                    "rcsb_entry_info.resolution_combined",
                    "less_or_equal",
                    float(max_resolution),
                ),
                _text("rcsb_accession_info.deposit_date", "greater_or_equal", min_deposit),
            ],
        },
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"]},
    }

def pre_cutoff_cluster_query(group_id: str, *, before: str) -> dict:

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                _text(
                    "rcsb_polymer_entity_group_membership.group_id",
                    "exact_match",
                    group_id,
                ),
                _text("rcsb_accession_info.deposit_date", "less", before),
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {"results_content_type": ["experimental"]},
    }
