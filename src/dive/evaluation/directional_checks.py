
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

CHECK_SCHEMA_VERSION = 2
EXECUTABLE_ROUTES = (
    "adaptive",
    "none",
    "z_to_x",
    "x_to_z",
    "bidirectional",
)

_FAMILIES = ("binder", "ame", "antibody")
_IDENTITY_KEYS = frozenset({"family", "example_id", "parent_id", "cell_hash"})
_VALIDITY_KEYS = frozenset(
    {
        "corruption_count",
        "route_order",
        "route_call_counts",
        "corruption_digest_mismatch_count",
        "input_digest_mismatch_count",
        "time_digest_mismatch_count",
        "target_digest_mismatch_count",
        "mask_digest_mismatch_count",
        "observed_route_mismatch_count",
        "shape_mismatch_count",
        "dtype_mismatch_count",
        "denominator_mismatch_count",
        "valid_generated_residues",
        "non_finite_count",
        "adaptive_tap_count",
        "adaptive_simplex_violation_count",
        "hard_one_hot_violation_count",
        "none_bridge_residual_nonzero_count",
    }
)
_PRESERVATION_KEYS = frozenset(
    {
        "candidate_none_loss_sum",
        "frozen_parent_loss_sum",
        "valid_generated_residues",
        "conditioning_mutation_count",
        "conditioning_elements",
        "candidate_none_non_finite_count",
        "frozen_parent_non_finite_count",
    }
)
_ROOT_KEYS = frozenset({"schema_version", "identity", "validity", "preservation"})
_MISMATCH_COUNT_KEYS = (
    "corruption_digest_mismatch_count",
    "input_digest_mismatch_count",
    "time_digest_mismatch_count",
    "target_digest_mismatch_count",
    "mask_digest_mismatch_count",
    "observed_route_mismatch_count",
    "shape_mismatch_count",
    "dtype_mismatch_count",
    "denominator_mismatch_count",
    "non_finite_count",
    "adaptive_simplex_violation_count",
    "hard_one_hot_violation_count",
    "none_bridge_residual_nonzero_count",
)

class DirectionalCheckError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class JointPreservationResult:

    passed: bool
    equal_family_ratio: float
    family_ratios: tuple[tuple[str, float], ...]

@dataclass(frozen=True, slots=True)
class _ParsedCheck:
    family: str
    example_id: str
    parent_id: str
    cell_hash: str
    corruption_count: int
    route_order: tuple[str, ...]
    route_call_counts: tuple[tuple[str, int], ...]
    mismatch_counts: tuple[int, ...]
    valid_generated_residues: int
    adaptive_tap_count: int
    candidate_none_loss_sum: float
    frozen_parent_loss_sum: float
    preservation_valid_generated_residues: int
    conditioning_mutation_count: int
    conditioning_elements: int
    candidate_none_non_finite_count: int
    frozen_parent_non_finite_count: int

def _exact_dict(name: str, value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise DirectionalCheckError(f"{name} must be a built-in dict")
    if any(type(key) is not str for key in value):
        raise DirectionalCheckError(f"{name} keys must be built-in strings")
    return value

def _exact_list(name: str, value: object) -> list[object]:
    if type(value) is not list:
        raise DirectionalCheckError(f"{name} must be a built-in list")
    return value

def _exact_keys(name: str, value: dict[str, object], expected: frozenset[str]) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise DirectionalCheckError(f"{name} schema: {'; '.join(details)}")

def _exact_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise DirectionalCheckError(f"{name} must be an integer, not bool")
    if value < 0:
        raise DirectionalCheckError(f"{name} must be nonnegative")
    return value

def _exact_float(name: str, value: object) -> float:
    if type(value) is not float:
        raise DirectionalCheckError(f"{name} must be a float")
    if not math.isfinite(value):
        raise DirectionalCheckError(f"{name} must be finite")
    if value < 0.0 or (value == 0.0 and math.copysign(1.0, value) < 0.0):
        raise DirectionalCheckError(f"{name} must be nonnegative and canonical")
    return value

def _string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise DirectionalCheckError(f"{name} must be a non-empty string")
    return value

def _cell_hash(name: str, value: object) -> str:
    result = _string(name, value)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise DirectionalCheckError(f"{name} must be a lowercase 64-hex identity")
    return result

def _identity_tuple(name: str, value: object) -> tuple[str, str, str, str]:
    if type(value) is not tuple or len(value) != 4:
        raise DirectionalCheckError(f"{name} must be a four-item built-in tuple")
    family = _string(f"{name}[0]", value[0])
    if family not in _FAMILIES:
        raise DirectionalCheckError(f"{name}[0] is an unknown family")
    return (
        family,
        _string(f"{name}[1]", value[1]),
        _string(f"{name}[2]", value[2]),
        _cell_hash(f"{name}[3]", value[3]),
    )

def _assert_unique_identities(
    name: str, identities: tuple[tuple[str, str, str, str], ...]
) -> None:
    if len(set(identities)) != len(identities):
        raise DirectionalCheckError(f"{name} contains duplicate identities")
    cell_hashes = tuple(identity[3] for identity in identities)
    if len(set(cell_hashes)) != len(cell_hashes):
        raise DirectionalCheckError(f"{name} contains duplicate cell_hash values")

def _parse_check(value: object, *, name: str) -> _ParsedCheck:
    record = _exact_dict(name, value)
    _exact_keys(name, record, _ROOT_KEYS)
    if (
        _exact_int(f"{name}.schema_version", record["schema_version"])
        != CHECK_SCHEMA_VERSION
    ):
        raise DirectionalCheckError(f"{name}.schema_version must be exactly 2")

    identity = _exact_dict(f"{name}.identity", record["identity"])
    _exact_keys(f"{name}.identity", identity, _IDENTITY_KEYS)
    family = _string(f"{name}.identity.family", identity["family"])
    if family not in _FAMILIES:
        raise DirectionalCheckError(f"{name}.identity.family is unknown")

    validity = _exact_dict(f"{name}.validity", record["validity"])
    _exact_keys(f"{name}.validity", validity, _VALIDITY_KEYS)
    route_order = _exact_list(f"{name}.validity.route_order", validity["route_order"])
    if any(type(route) is not str for route in route_order):
        raise DirectionalCheckError(f"{name}.validity.route_order must contain strings")
    route_counts = _exact_dict(
        f"{name}.validity.route_call_counts", validity["route_call_counts"]
    )
    _exact_keys(
        f"{name}.validity.route_call_counts", route_counts, frozenset(EXECUTABLE_ROUTES)
    )
    parsed_route_counts = tuple(
        (
            route,
            _exact_int(
                f"{name}.validity.route_call_counts.{route}", route_counts[route]
            ),
        )
        for route in EXECUTABLE_ROUTES
    )
    mismatch_counts = tuple(
        _exact_int(f"{name}.validity.{field}", validity[field])
        for field in _MISMATCH_COUNT_KEYS
    )

    preservation = _exact_dict(f"{name}.preservation", record["preservation"])
    _exact_keys(f"{name}.preservation", preservation, _PRESERVATION_KEYS)
    frozen_parent_loss_sum = _exact_float(
        f"{name}.preservation.frozen_parent_loss_sum",
        preservation["frozen_parent_loss_sum"],
    )
    if frozen_parent_loss_sum <= 0.0:
        raise DirectionalCheckError(
            f"{name}.preservation.frozen_parent_loss_sum must be positive"
        )
    return _ParsedCheck(
        family=family,
        example_id=_string(f"{name}.identity.example_id", identity["example_id"]),
        parent_id=_string(f"{name}.identity.parent_id", identity["parent_id"]),
        cell_hash=_cell_hash(f"{name}.identity.cell_hash", identity["cell_hash"]),
        corruption_count=_exact_int(
            f"{name}.validity.corruption_count", validity["corruption_count"]
        ),
        route_order=tuple(route_order),
        route_call_counts=parsed_route_counts,
        mismatch_counts=mismatch_counts,
        valid_generated_residues=_exact_int(
            f"{name}.validity.valid_generated_residues",
            validity["valid_generated_residues"],
        ),
        adaptive_tap_count=_exact_int(
            f"{name}.validity.adaptive_tap_count", validity["adaptive_tap_count"]
        ),
        candidate_none_loss_sum=_exact_float(
            f"{name}.preservation.candidate_none_loss_sum",
            preservation["candidate_none_loss_sum"],
        ),
        frozen_parent_loss_sum=frozen_parent_loss_sum,
        preservation_valid_generated_residues=_exact_int(
            f"{name}.preservation.valid_generated_residues",
            preservation["valid_generated_residues"],
        ),
        conditioning_mutation_count=_exact_int(
            f"{name}.preservation.conditioning_mutation_count",
            preservation["conditioning_mutation_count"],
        ),
        conditioning_elements=_exact_int(
            f"{name}.preservation.conditioning_elements",
            preservation["conditioning_elements"],
        ),
        candidate_none_non_finite_count=_exact_int(
            f"{name}.preservation.candidate_none_non_finite_count",
            preservation["candidate_none_non_finite_count"],
        ),
        frozen_parent_non_finite_count=_exact_int(
            f"{name}.preservation.frozen_parent_non_finite_count",
            preservation["frozen_parent_non_finite_count"],
        ),
    )

def _counterfactual_failure_count(record: _ParsedCheck) -> int:
    return int(
        record.corruption_count != 1
        or record.route_order != EXECUTABLE_ROUTES
        or any(count != 1 for _, count in record.route_call_counts)
        or any(record.mismatch_counts)
        or record.valid_generated_residues <= 0
        or record.adaptive_tap_count != 3
        or record.valid_generated_residues
        != record.preservation_valid_generated_residues
    )

def counterfactual_validity(record: Mapping[str, object]) -> dict[str, int]:

    parsed = _parse_check(record, name="record")
    return {"failure_count": _counterfactual_failure_count(parsed), "denominator": 1}

def validate_check_inventory(
    records: Sequence[Mapping[str, object]],
    *,
    expected_identities: Sequence[tuple[str, str, str, str]] | None = None,
) -> tuple[_ParsedCheck, ...]:

    if type(records) is not list:
        raise DirectionalCheckError("records must be a built-in list")
    if not records:
        raise DirectionalCheckError("records must not be empty")
    parsed = tuple(
        _parse_check(record, name=f"records[{index}]")
        for index, record in enumerate(records)
    )
    identities = tuple(
        (record.family, record.example_id, record.parent_id, record.cell_hash)
        for record in parsed
    )
    _assert_unique_identities("records", identities)
    if expected_identities is not None:
        if type(expected_identities) not in (list, tuple):
            raise DirectionalCheckError(
                "expected_identities must be a built-in list or tuple"
            )
        expected = tuple(
            _identity_tuple(f"expected_identities[{index}]", identity)
            for index, identity in enumerate(expected_identities)
        )
        _assert_unique_identities("expected_identities", expected)
        if len(identities) != len(expected) or set(identities) != set(expected):
            raise DirectionalCheckError(
                "records is incomplete or inconsistent with expected_identities"
            )
    return parsed

def joint_preservation(
    records: Sequence[Mapping[str, object]],
) -> JointPreservationResult:

    parsed = validate_check_inventory(records)
    present_families = {record.family for record in parsed}
    missing_families = tuple(
        family for family in _FAMILIES if family not in present_families
    )
    if missing_families:
        raise DirectionalCheckError(
            f"records missing families {list(missing_families)}"
        )

    ratios: list[tuple[str, float]] = []
    for family in _FAMILIES:
        try:
            candidate_sum = math.fsum(
                record.candidate_none_loss_sum
                for record in parsed
                if record.family == family
            )
            parent_sum = math.fsum(
                record.frozen_parent_loss_sum
                for record in parsed
                if record.family == family
            )
        except OverflowError as error:
            raise DirectionalCheckError(
                f"records.{family} has a non-finite aggregate"
            ) from error
        if (
            not math.isfinite(candidate_sum)
            or not math.isfinite(parent_sum)
            or parent_sum <= 0.0
        ):
            raise DirectionalCheckError(
                f"records.{family} has a non-finite or nonpositive parent denominator"
            )
        ratio = candidate_sum / parent_sum
        if not math.isfinite(ratio):
            raise DirectionalCheckError(
                f"records.{family} has a non-finite preservation ratio"
            )
        ratios.append((family, ratio))

    equal_family_ratio = math.fsum(ratio for _, ratio in ratios) / len(_FAMILIES)
    if not math.isfinite(equal_family_ratio):
        raise DirectionalCheckError("records has a non-finite equal-family ratio")
    integrity = all(
        record.preservation_valid_generated_residues > 0
        and record.valid_generated_residues
        == record.preservation_valid_generated_residues
        and record.conditioning_mutation_count == 0
        and record.conditioning_elements > 0
        and record.candidate_none_non_finite_count == 0
        and record.frozen_parent_non_finite_count == 0
        for record in parsed
    )
    return JointPreservationResult(
        passed=integrity and equal_family_ratio <= 1.02,
        equal_family_ratio=equal_family_ratio,
        family_ratios=tuple(ratios),
    )

MUTABLE_SOURCE_REALM_EXCLUSIONS: Mapping[str, type] = MappingProxyType({})
