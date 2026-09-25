
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from dive.data.contracts import Partition
from dive.data.similarity import NeighborIndex, SimilarityEdge
from dive.data.split import SCORED_PARTITIONS, SplitBundle

class SplitAuditError(RuntimeError):
    pass

def assert_no_cross_partition_edges(
    bundle: SplitBundle, edges: Iterable[SimilarityEdge]
) -> None:

    violations = []
    for edge in edges:
        left = bundle.assignments.get(edge.left)
        right = bundle.assignments.get(edge.right)
        if left is None or right is None:
            continue
        if left in SCORED_PARTITIONS and right in SCORED_PARTITIONS and left is not right:
            violations.append(f"{edge.left} ({left}) ~ {edge.right} ({right}) via {edge.kind}")
    if violations:
        raise SplitAuditError(
            f"{len(violations)} cross-partition edge(s); first: {violations[0]}"
        )

def assert_official_test_is_held_out(
    bundle: SplitBundle, official_test: Iterable[str]
) -> None:

    leaked = [
        parent_id
        for parent_id in official_test
        if bundle.assignments.get(parent_id) in {Partition.TRAIN, Partition.VALIDATION}
    ]
    if leaked:
        raise SplitAuditError(
            f"{len(leaked)} official test group(s) in train/validation: {sorted(leaked)[:5]}"
        )

def assert_no_legacy_in_strict_blind(
    bundle: SplitBundle, legacy: Iterable[str], edges: Sequence[SimilarityEdge]
) -> None:

    legacy = set(legacy)
    index = NeighborIndex(edges)
    offenders = []
    for parent_id, views in bundle.views.items():
        if not views.get("strict"):
            continue
        if parent_id in legacy:
            offenders.append(f"{parent_id} (is legacy)")
        elif index.neighbors(parent_id) & legacy:
            offenders.append(f"{parent_id} (neighbors legacy)")
    if offenders:
        raise SplitAuditError(
            f"{len(offenders)} legacy-tainted group(s) in the strict blind view: "
            f"{offenders[:5]}"
        )

def assert_transformed_examples_have_parents(
    bundle: SplitBundle, example_ids: Iterable[str], parent_of: Mapping[str, str]
) -> None:

    orphans = [
        example_id
        for example_id in example_ids
        if parent_of.get(example_id) not in bundle.assignments
    ]
    if orphans:
        raise SplitAuditError(
            f"{len(orphans)} example(s) without an assigned parent: {sorted(orphans)[:5]}"
        )

def build_audit_report(
    bundle: SplitBundle,
    *,
    edges: Sequence[SimilarityEdge],
    legacy: Iterable[str],
    extra: Mapping[str, object] | None = None,
) -> dict:

    legacy = set(legacy)

    edge_counts: dict[str, int] = {}
    for edge in edges:
        edge_counts[edge.kind] = edge_counts.get(edge.kind, 0) + 1

    exclusion_reasons: dict[str, int] = {}
    for parent_id, partition in bundle.assignments.items():
        if partition.value.startswith("excluded"):
            reason = bundle.reasons.get(parent_id, "unrecorded")
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    effective: dict[str, dict[str, int]] = {}
    for parent_id, partition in bundle.assignments.items():
        family = str(bundle.families[parent_id])
        stats = effective.setdefault(
            family, {"train": 0, "validation": 0, "standard": 0, "strict": 0, "temporal_clean": 0}
        )
        if partition is Partition.TRAIN:
            stats["train"] += 1
        if partition is Partition.VALIDATION:
            stats["validation"] += 1
        views = bundle.views.get(parent_id, {})
        for view in ("standard", "strict", "temporal_clean"):
            if views.get(view):
                stats[view] += 1
    for stats in effective.values():
        stats["strict_blind"] = stats["strict"]

    cross_family = sum(
        1
        for edge in edges
        if edge.left in bundle.families
        and edge.right in bundle.families
        and bundle.families[edge.left] is not bundle.families[edge.right]
    )

    report = {
        "seed": bundle.seed,
        "counts": bundle.counts(),
        "edge_counts": edge_counts,
        "edge_total": len(edges),
        "cross_family_edges": cross_family,
        "exclusion_reasons": exclusion_reasons,
        "effective_sizes": effective,
        "legacy_groups": len(legacy),
        "view_definitions": {
            "standard": "the complete frozen benchmark, legacy status disclosed",
            "strict": "internally deduplicated unseen subset with no direct project-train edge",
            "temporal_clean": "the strict subset deposited after the verified cutoffs",
        },
    }
    if extra:
        report.update(extra)
    return report

def render_audit_markdown(report: Mapping[str, object]) -> str:

    lines = ["# Emergent-leadership split audit", ""]
    lines.append(f"Seed `{report['seed']}`.")
    lines.append("")

    lines.append("## Partition counts")
    lines.append("")
    counts = report["counts"]
    partitions = sorted({p for family in counts.values() for p in family})
    lines.append("| family | " + " | ".join(partitions) + " |")
    lines.append("|---|" + "---|" * len(partitions))
    for family in sorted(counts):
        row = [str(counts[family].get(p, 0)) for p in partitions]
        lines.append(f"| {family} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Effective sizes per view")
    lines.append("")
    lines.append("| family | train | validation | standard | strict | temporal_clean |")
    lines.append("|---|---|---|---|---|---|")
    for family, stats in sorted(report["effective_sizes"].items()):
        lines.append(
            f"| {family} | {stats['train']} | {stats['validation']} | "
            f"{stats['standard']} | {stats['strict']} | {stats['temporal_clean']} |"
        )
    lines.append("")
    for view, meaning in report["view_definitions"].items():
        lines.append(f"- **{view}** — {meaning}")
    lines.append("")

    lines.append("## Edges")
    lines.append("")
    lines.append(f"{report['edge_total']} total, {report['cross_family_edges']} cross-family.")
    lines.append("")
    for kind, count in sorted(report["edge_counts"].items()):
        lines.append(f"- `{kind}`: {count}")
    lines.append("")

    lines.append("## Exclusion reasons")
    lines.append("")
    if report["exclusion_reasons"]:
        for reason, count in sorted(report["exclusion_reasons"].items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("No group was excluded.")
    lines.append("")
    return "\n".join(lines)
