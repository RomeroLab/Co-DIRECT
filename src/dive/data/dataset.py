
from __future__ import annotations

from collections.abc import Callable, Sequence

from dive.data.roles import ROLE_COLUMNS, Selector

class DatasetError(RuntimeError):
    pass

class DatasetTotalDropError(DatasetError):
    pass

DEFAULT_TOTAL_DROP_CHECK_AFTER = 32

_MAX_IDS_PER_REASON = 5

class RoleAwareStructureDataset:

    def __init__(
        self,
        metadata,
        transforms: Sequence = (),
        *,
        drop_waters: bool = True,
        select_role_chains: bool = True,
        use_parse: bool = False,
        parser_args: dict | None = None,
        atomize_non_polymers: bool = False,
        ligand_pipeline: bool = False,
        ligand_pipeline_args: dict | None = None,
        structure_loader: Callable[[str], object] | None = None,
        sample_encoder: Callable[..., object] | None = None,
        total_drop_check_after: int = DEFAULT_TOTAL_DROP_CHECK_AFTER,
    ) -> None:
        missing = [c for c in ROLE_COLUMNS if c not in metadata.columns]
        if missing:
            raise DatasetError(
                f"metadata is missing role column(s) {missing}; without them there "
                f"is nothing to mask and the task is undefined"
            )
        if "path" not in metadata.columns:
            raise DatasetError("metadata has no `path` column")
        self._metadata = metadata.reset_index(drop=True)
        self._transforms = list(transforms)
        self._drop_waters = drop_waters
        self._select_role_chains = select_role_chains
        self._use_parse = use_parse
        self._parser_args = dict(parser_args or {})
        self._atomize_non_polymers = atomize_non_polymers

        self._ligand_pipeline = ligand_pipeline
        self._ligand_pipeline_args = dict(ligand_pipeline_args or {})
        if ligand_pipeline:
            if not use_parse:
                raise DatasetError(
                    "ligand_pipeline needs use_parse=True: it consumes the "
                    "atomworks parse output, and without parse there is none"
                )
            if atomize_non_polymers:
                raise DatasetError(
                    "ligand_pipeline and atomize_non_polymers are opposite "
                    "policies -- one keeps the ligand as a typed entity, the "
                    "other merges it into the protein-style array"
                )
        if structure_loader is not None and not callable(structure_loader):
            raise DatasetError("structure_loader must be callable or None")
        self._structure_loader = structure_loader

        if sample_encoder is not None and not callable(sample_encoder):
            raise DatasetError("sample_encoder must be callable or None")
        self._sample_encoder = sample_encoder
        if type(total_drop_check_after) is not int or total_drop_check_after < 1:
            raise DatasetError("total_drop_check_after must be a positive integer")
        self._total_drop_check_after = total_drop_check_after
        self._attempted = 0
        self._kept = 0
        self._drops: dict[str, dict] = {}

    def _record_drop(self, reason: str, example_id: str, error: BaseException | None):
        entry = self._drops.setdefault(
            reason,
            {"count": 0, "example_ids": [], "first_error": None, "error_types": {}},
        )
        entry["count"] += 1
        if len(entry["example_ids"]) < _MAX_IDS_PER_REASON:
            entry["example_ids"].append(example_id)
        if error is not None:
            name = type(error).__name__
            entry["error_types"][name] = entry["error_types"].get(name, 0) + 1
            if entry["first_error"] is None:
                entry["first_error"] = f"{name}: {error}"
        self._maybe_raise_total_drop()
        return None

    def _maybe_raise_total_drop(self) -> None:
        if self._kept or self._attempted < self._total_drop_check_after:
            return
        summary = "; ".join(
            f"{reason} x{entry['count']}"
            + (f" ({entry['first_error']})" if entry["first_error"] else "")
            for reason, entry in sorted(self._drops.items())
        )
        raise DatasetTotalDropError(
            f"kept 0 of {self._attempted} attempted examples; every one was dropped. "
            f"Reasons: {summary}. This is a wiring failure, not an empty split -- "
            f"an empty stream here would otherwise be reported as success."
        )

    def drop_report(self) -> dict:

        return {
            "attempted": self._attempted,
            "kept": self._kept,
            "dropped": self._attempted - self._kept,
            "by_reason": {
                reason: {
                    "count": entry["count"],
                    "example_ids": list(entry["example_ids"]),
                    "first_error": entry["first_error"],
                    "error_types": dict(entry["error_types"]),
                }
                for reason, entry in sorted(self._drops.items())
            },
        }

    def _load_with_ligand_pipeline(self, path: str):

        from atomworks.io.parser import parse
        from proteinfoundation.datasets.pipelines.complexa_ligand import (
            build_protein_ligand_transform_pipeline,
        )

        args = {
            k: v for k, v in self._parser_args.items()
            if k not in ("cache_dir", "save_to_cache", "load_from_cache")
        }
        cache_dir = self._parser_args.get("cache_dir")
        result = parse(
            path,
            cache_dir=cache_dir,
            save_to_cache=cache_dir is not None,
            load_from_cache=cache_dir is not None,
            **args,
        )

        if result.get("assemblies") and result["assemblies"].get("1"):
            result["atom_array"] = result["assemblies"]["1"][0]
        elif result.get("asym_unit") is not None:
            unit = result["asym_unit"]
            result["atom_array"] = unit[0] if isinstance(unit, list) else unit
        else:
            raise DatasetError(f"no structure in parse result for {path}")

        pipeline = build_protein_ligand_transform_pipeline(
            is_inference=True, **self._ligand_pipeline_args
        )
        data = pipeline(result)
        atom_array = data.get("atom_array") if hasattr(data, "get") else None
        if atom_array is None:
            raise DatasetError(f"ligand pipeline produced no atom_array for {path}")
        return atom_array, data

    @property
    def ligand_pipeline(self) -> bool:

        return self._ligand_pipeline

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, index: int):
        row = self._metadata.iloc[index]
        self._attempted += 1
        example_id = str(row["example_id"])

        atomworks_data = None
        try:
            if self._ligand_pipeline:
                atom_array, atomworks_data = self._load_with_ligand_pipeline(
                    str(row["path"])
                )
            elif self._structure_loader is None:
                from proteinfoundation.datasets.structure_data import load_structure

                atom_array = load_structure(
                    str(row["path"]),
                    use_parse=self._use_parse,
                    **self._parser_args,
                )
            else:
                atom_array = self._structure_loader(str(row["path"]))
        except Exception as error:
            return self._record_drop("structure_load", example_id, error)

        if self._drop_waters:
            from atomworks.io.transforms.atom_array import remove_waters

            atom_array = remove_waters(atom_array)

        if self._atomize_non_polymers:
            try:
                atom_array = _atomize_non_polymers(atom_array)
            except Exception as error:
                return self._record_drop("atomize_non_polymers", example_id, error)

        if self._select_role_chains and not self._ligand_pipeline:
            atom_array = _keep_role_chains(atom_array, row)
            if atom_array is None or len(atom_array) == 0:
                return self._record_drop("role_chains_empty", example_id, None)

        try:
            encoder = self._sample_encoder
            if encoder is None:
                from proteinfoundation.datasets.structure_data import (
                    atomarray_to_atom37 as encoder,
                )
            sample = encoder(
                atom_array,
                sample_id=str(row["example_id"]),
                atomworks_data=atomworks_data,
            )
        except Exception as error:
            return self._record_drop("atom37_encoding", example_id, error)

        for column in ROLE_COLUMNS:
            setattr(sample, column, str(row[column]))

        for transform in self._transforms:
            name = type(transform).__name__
            try:
                sample = transform(sample)
            except Exception as error:
                return self._record_drop(f"transform:{name}", example_id, error)
            if sample is None:
                return self._record_drop(
                    f"transform_returned_none:{name}", example_id, None
                )
        self._kept += 1
        return sample

def _keep_role_chains(atom_array, row):

    import numpy as np

    wanted = set()
    for column in ("context", "target"):
        for entry in str(row[column]).split(","):
            if entry:
                wanted.add(Selector.parse(entry).chain)
    if not wanted:
        return atom_array

    keep = np.isin(atom_array.chain_id, list(wanted))
    if not keep.any():
        return None
    return atom_array[keep]

def _atomize_non_polymers(atom_array):

    from atomworks.constants import STANDARD_AA
    from atomworks.ml.transforms.atom_array import AddGlobalAtomIdAnnotation
    from atomworks.ml.transforms.atomize import (
        AtomizeByCCDName,
        FlagNonPolymersForAtomization,
    )
    from atomworks.ml.transforms.filters import RemoveHydrogens

    data = {"atom_array": atom_array}
    for transform in (
        RemoveHydrogens(),
        FlagNonPolymersForAtomization(),
        AddGlobalAtomIdAnnotation(allow_overwrite=True),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
    ):
        data = transform(data)
    return data["atom_array"]
