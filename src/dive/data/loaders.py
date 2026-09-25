
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dive.data.roles import ROLE_COLUMNS

LOADER_COLUMNS: tuple[str, ...] = ("example_id", "path", "generated", "context", "target")

FORBIDDEN_LOADER_COLUMNS: frozenset[str] = frozenset(
    {
        "family",
        "partition",
        "outer_partition",
        "parent_id",
        "view_standard",
        "view_strict",
        "view_temporal_clean",
        "selection_reason",
        "cdr_h3",
        "cdr_h3_sequence",
        "catalytic",
        "interface",
    }
)

class LoaderError(RuntimeError):
    pass

def loader_frame(
    manifest,
    path_of: Callable[[object], object],
    roles_of: Callable[[object], object] | None = None,
):

    import pandas as pd

    records = []
    for row in manifest.itertuples():
        path = path_of(row)
        if path is None:
            continue
        roles = roles_of(row) if roles_of is not None else None
        if roles_of is not None and roles is None:
            continue
        record = {

            "example_id": str(row.example_id),
            "path": str(path),
        }
        record.update(
            roles.as_columns() if roles is not None else dict.fromkeys(ROLE_COLUMNS, "")
        )
        records.append(record)

    frame = pd.DataFrame(records, columns=list(LOADER_COLUMNS))
    assert_loader_manifest_is_clean(frame)
    return frame

def assert_loader_manifest_is_clean(frame) -> None:

    columns = list(getattr(frame, "columns", []))
    missing = [c for c in LOADER_COLUMNS if c not in columns]
    if missing:
        raise LoaderError(f"loader manifest is missing required column(s) {missing}")

    forbidden = sorted(set(columns) & FORBIDDEN_LOADER_COLUMNS)
    if forbidden:
        raise LoaderError(
            f"loader manifest carries semantic column(s) {forbidden}; "
            f"StructureDataset copies every column except path and id into the "
            f"sample, so these would reach the model batch"
        )

    extra = sorted(set(columns) - set(LOADER_COLUMNS))
    if extra:
        raise LoaderError(
            f"loader manifest carries unexpected column(s) {extra}; only "
            f"{list(LOADER_COLUMNS)} may cross this boundary"
        )

def assert_split_is_explicit(datamodule) -> None:

    if getattr(datamodule, "val_metadata_file", None):
        return
    raise LoaderError(
        "val_metadata_file is not set, so StructureDataModule would split by row "
        "position -- iloc[:n] for train and the remainder for validation, at a "
        "train_split fraction. The frozen manifests already assigned every parent "
        "group under a leakage-safe rule, and that fallback would overrule them by "
        "row order"
    )

def write_loader_manifests(
    manifest_root: Path,
    destination: Path,
    structure_path_of: Callable[[str, object], object],
    roles_of: Callable[[str, object], object] | None = None,
    partitions: tuple[str, ...] = ("train", "validation"),
    only: set[str] | None = None,
) -> dict[str, dict[str, int]]:

    import pandas as pd

    destination.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}

    for path in sorted(Path(manifest_root).glob("*.parquet")):
        family = path.stem
        frame = pd.read_parquet(path)
        family_summary: dict[str, int] = {}
        for partition in partitions:
            rows = frame[frame.partition == partition]
            if only is not None:
                rows = rows[rows.example_id.astype(str).isin(only)]
            reduced = loader_frame(
                rows,
                lambda row: structure_path_of(family, row),
                (lambda row: roles_of(family, row)) if roles_of else None,
            )
            out = destination / f"{family}_{partition}.parquet"
            reduced.to_parquet(out, index=False)
            family_summary[partition] = len(reduced)
            family_summary[f"{partition}_unstaged"] = len(rows) - len(reduced)
        summary[family] = family_summary
    return summary
