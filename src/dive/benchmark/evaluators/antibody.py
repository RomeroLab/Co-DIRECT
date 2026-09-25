
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import atan2, degrees

import numpy as np

from dive.benchmark.evaluators.contracts import MetricUnavailable, MetricValue

_PRIMARY_NAME = "cdr_h3_ca_rmsd"
_MANIFEST_HASH_LENGTH = 64
_DEFAULT_CHI_ATOMS = {
    "chi1": (("N", "CA", "CB", "CG"),),
    "chi2": (("CA", "CB", "CG", "CD"),),
}

_RESIDUE_CHI_ATOMS = {
    "VAL": {"chi1": (("N", "CA", "CB", "CG1"), ("N", "CA", "CB", "CG2"))},
    "LEU": {"chi2": (("CA", "CB", "CG", "CD1"), ("CA", "CB", "CG", "CD2"))},
    "ASP": {"chi2": (("CA", "CB", "CG", "OD1"), ("CA", "CB", "CG", "OD2"))},
    "PHE": {"chi2": (("CA", "CB", "CG", "CD1"), ("CA", "CB", "CG", "CD2"))},
    "TYR": {"chi2": (("CA", "CB", "CG", "CD1"), ("CA", "CB", "CG", "CD2"))},
}
_VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
    "SE": 1.90,
}
BLANK_INSERTION_CODE = ""
_BLANK_INSERTION_ALIASES = frozenset({BLANK_INSERTION_CODE, ".", "?", " "})

def normalize_insertion_code(value: object) -> str:

    if value is None:
        return BLANK_INSERTION_CODE
    if type(value) is not str:
        raise ValueError("insertion code must be an explicit string")
    if value in _BLANK_INSERTION_ALIASES:
        return BLANK_INSERTION_CODE
    if not value or any(character.isspace() for character in value):
        raise ValueError("insertion code must not contain whitespace")
    return value

@dataclass(frozen=True, slots=True, order=True)
class ResidueKey:

    chain_id: str
    auth_seq_id: int
    insertion_code: str = BLANK_INSERTION_CODE

    def __post_init__(self) -> None:
        if type(self.chain_id) is not str or not self.chain_id:
            raise ValueError("residue key requires a non-empty chain")
        if type(self.auth_seq_id) is not int:
            raise ValueError("residue key requires an integer auth_seq_id")
        object.__setattr__(
            self, "insertion_code", normalize_insertion_code(self.insertion_code)
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "auth_seq_id": self.auth_seq_id,
            "chain_id": self.chain_id,
            "insertion_code": self.insertion_code,
        }

@dataclass(frozen=True, slots=True, order=True)
class AtomKey:

    residue_key: ResidueKey
    atom_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.residue_key, ResidueKey):
            raise ValueError("atom key requires a ResidueKey")
        if type(self.atom_name) is not str or not self.atom_name:
            raise ValueError("atom key requires a non-empty atom name")
        if any(character.isspace() for character in self.atom_name):
            raise ValueError("atom name must not contain whitespace")

    def to_mapping(self) -> dict[str, object]:
        payload = self.residue_key.to_mapping()
        payload["atom_name"] = self.atom_name
        return payload

@dataclass(frozen=True, slots=True)
class AtomRecord:

    chain: str
    residue: str
    atom: str
    coordinates: tuple[float, float, float]
    element: str | None = None
    residue_name: str | None = None
    insertion_code: str = BLANK_INSERTION_CODE

    @property
    def residue_key(self) -> ResidueKey:
        return ResidueKey(self.chain, int(self.residue), self.insertion_code)

    @property
    def identity(self) -> AtomKey:
        return AtomKey(self.residue_key, self.atom)

@dataclass(frozen=True, slots=True)
class FrozenAntibodyRoles:

    heavy_chain: str
    cdr_h3: tuple[ResidueKey, ...]
    antigen_chains: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class AntibodyComplex:
    native: tuple[AtomRecord, ...]
    predicted: tuple[AtomRecord, ...]
    roles: FrozenAntibodyRoles | Mapping[str, object]
    evaluator_manifest_hash: str

@dataclass(frozen=True, slots=True)
class SidechainAccuracy:
    chi1: float | MetricUnavailable
    chi2: float | MetricUnavailable

@dataclass(frozen=True, slots=True)
class AntibodyEvaluation:
    primary: MetricValue | MetricUnavailable
    secondary: Mapping[str, MetricValue | MetricUnavailable]

def cdr_h3_ca_rmsd(
    predicted: Sequence[AtomRecord],
    native: Sequence[AtomRecord],
    roles: FrozenAntibodyRoles | Mapping[str, object],
    *,
    evaluator_manifest_hash: str,
) -> float | MetricUnavailable:

    manifest_hash = _manifest_hash(evaluator_manifest_hash)
    frozen = _frozen_h3_roles(roles, manifest_hash)
    if isinstance(frozen, MetricUnavailable):
        return frozen
    predicted_atoms = _atom_index(predicted, manifest_hash)
    native_atoms = _atom_index(native, manifest_hash)
    if isinstance(predicted_atoms, MetricUnavailable):
        return predicted_atoms
    if isinstance(native_atoms, MetricUnavailable):
        return native_atoms
    identities = tuple(AtomKey(key, "CA") for key in frozen.cdr_h3)
    common = tuple(
        identity
        for identity in identities
        if identity in predicted_atoms and identity in native_atoms
    )
    if len(common) < 3:
        return _unavailable(
            manifest_hash,
            "insufficient_common_cdr_h3_ca_atoms",
            "fewer than three common frozen CDR-H3 C-alpha atoms",
        )
    predicted_points = _coordinates(predicted_atoms[identity] for identity in common)
    native_points = _coordinates(native_atoms[identity] for identity in common)
    if predicted_points is None or native_points is None:
        return _unavailable(
            manifest_hash, "nonfinite_coordinates", "CDR-H3 coordinates must be finite"
        )
    return _kabsch_rmsd(predicted_points, native_points)

def all_atom_rmsd(
    predicted: Sequence[AtomRecord],
    native: Sequence[AtomRecord],
    *,
    evaluator_manifest_hash: str,
) -> float | MetricUnavailable:

    manifest_hash = _manifest_hash(evaluator_manifest_hash)
    predicted_atoms = _atom_index(predicted, manifest_hash)
    native_atoms = _atom_index(native, manifest_hash)
    if isinstance(predicted_atoms, MetricUnavailable):
        return predicted_atoms
    if isinstance(native_atoms, MetricUnavailable):
        return native_atoms
    common = tuple(sorted(set(predicted_atoms) & set(native_atoms)))
    if not common:
        return _unavailable(
            manifest_hash, "no_common_named_atoms", "no common named atoms"
        )
    predicted_points = _coordinates(predicted_atoms[identity] for identity in common)
    native_points = _coordinates(native_atoms[identity] for identity in common)
    if predicted_points is None or native_points is None:
        return _unavailable(
            manifest_hash,
            "nonfinite_coordinates",
            "all-atom coordinates must be finite",
        )
    return float(
        np.sqrt(np.mean(np.sum((predicted_points - native_points) ** 2, axis=1)))
    )

def sidechain_accuracy(
    predicted: Sequence[AtomRecord],
    native: Sequence[AtomRecord],
    *,
    evaluator_manifest_hash: str,
) -> SidechainAccuracy:

    manifest_hash = _manifest_hash(evaluator_manifest_hash)
    predicted_atoms = _atom_index(predicted, manifest_hash)
    native_atoms = _atom_index(native, manifest_hash)
    if isinstance(predicted_atoms, MetricUnavailable):
        return SidechainAccuracy(predicted_atoms, predicted_atoms)
    if isinstance(native_atoms, MetricUnavailable):
        return SidechainAccuracy(native_atoms, native_atoms)
    return SidechainAccuracy(
        _chi_accuracy("chi1", predicted_atoms, native_atoms, manifest_hash),
        _chi_accuracy("chi2", predicted_atoms, native_atoms, manifest_hash),
    )

def antigen_contact_recovery(
    predicted: Sequence[AtomRecord],
    native: Sequence[AtomRecord],
    roles: FrozenAntibodyRoles | Mapping[str, object],
    *,
    cutoff: float = 5.0,
    evaluator_manifest_hash: str,
) -> float | MetricUnavailable:

    manifest_hash = _manifest_hash(evaluator_manifest_hash)
    frozen = _frozen_interface_roles(roles, manifest_hash)
    if isinstance(frozen, MetricUnavailable):
        return frozen
    if cutoff <= 0 or not np.isfinite(cutoff):
        return _unavailable(
            manifest_hash,
            "invalid_contact_cutoff",
            "contact cutoff must be finite and positive",
        )
    predicted_atoms = _atom_index(predicted, manifest_hash)
    native_atoms = _atom_index(native, manifest_hash)
    if isinstance(predicted_atoms, MetricUnavailable):
        return predicted_atoms
    if isinstance(native_atoms, MetricUnavailable):
        return native_atoms
    native_contacts = _contact_pairs(native_atoms, frozen, cutoff, manifest_hash)
    if isinstance(native_contacts, MetricUnavailable):
        return native_contacts
    if not native_contacts:
        return _unavailable(
            manifest_hash,
            "no_native_antigen_contacts",
            "no frozen native H3-antigen contacts",
        )
    predicted_contacts = _contact_pairs(predicted_atoms, frozen, cutoff, manifest_hash)
    if isinstance(predicted_contacts, MetricUnavailable):
        return predicted_contacts
    return len(native_contacts & predicted_contacts) / len(native_contacts)

def clash_rate(
    predicted: Sequence[AtomRecord],
    roles: FrozenAntibodyRoles | Mapping[str, object],
    *,
    evaluator_manifest_hash: str,
) -> float | MetricUnavailable:

    manifest_hash = _manifest_hash(evaluator_manifest_hash)
    frozen = _frozen_interface_roles(roles, manifest_hash)
    if isinstance(frozen, MetricUnavailable):
        return frozen
    atoms = _atom_index(predicted, manifest_hash)
    if isinstance(atoms, MetricUnavailable):
        return atoms
    h3, antigen = _role_atoms(atoms, frozen)
    if not h3 or not antigen:
        return _unavailable(
            manifest_hash,
            "missing_frozen_role_atoms",
            "missing frozen H3 or antigen atoms",
        )
    pairs = [(left, right) for left in h3 for right in antigen]
    if not all(_finite_atom(atom) for pair in pairs for atom in pair):
        return _unavailable(
            manifest_hash, "nonfinite_coordinates", "clash coordinates must be finite"
        )
    overlaps = sum(
        _distance(left, right) < _vdw_radius(left) + _vdw_radius(right) - 0.4
        for left, right in pairs
    )
    return overlaps / len(pairs)

def evaluate_antibody(complex_record: AntibodyComplex) -> AntibodyEvaluation:

    manifest_hash = _manifest_hash(complex_record.evaluator_manifest_hash)
    primary_raw = cdr_h3_ca_rmsd(
        complex_record.predicted,
        complex_record.native,
        complex_record.roles,
        evaluator_manifest_hash=manifest_hash,
    )
    primary = _metric(_PRIMARY_NAME, "lower", primary_raw, manifest_hash)
    sidechains = sidechain_accuracy(
        complex_record.predicted,
        complex_record.native,
        evaluator_manifest_hash=manifest_hash,
    )
    secondary_raw: dict[str, float | MetricUnavailable] = {
        "all_atom_rmsd": all_atom_rmsd(
            complex_record.predicted,
            complex_record.native,
            evaluator_manifest_hash=manifest_hash,
        ),
        "chi1_accuracy": sidechains.chi1,
        "chi2_accuracy": sidechains.chi2,
        "antigen_contact_recovery": antigen_contact_recovery(
            complex_record.predicted,
            complex_record.native,
            complex_record.roles,
            evaluator_manifest_hash=manifest_hash,
        ),
        "clash_rate": clash_rate(
            complex_record.predicted,
            complex_record.roles,
            evaluator_manifest_hash=manifest_hash,
        ),
    }
    directions = {
        "all_atom_rmsd": "lower",
        "chi1_accuracy": "higher",
        "chi2_accuracy": "higher",
        "antigen_contact_recovery": "higher",
        "clash_rate": "lower",
    }
    secondary = {
        name: _metric(name, directions[name], value, manifest_hash)
        for name, value in secondary_raw.items()
    }
    return AntibodyEvaluation(primary=primary, secondary=secondary)

def _frozen_h3_roles(
    roles: FrozenAntibodyRoles | Mapping[str, object],
    manifest_hash: str,
) -> FrozenAntibodyRoles | MetricUnavailable:
    if isinstance(roles, FrozenAntibodyRoles):
        candidate = roles
    elif isinstance(roles, Mapping):
        cdr_h3 = roles.get("cdr_h3")
        antigen_chains = roles.get("antigen_chains", roles.get("antigen", ()))
        heavy_chain = roles.get("heavy_chain")
        if not isinstance(heavy_chain, str) or not heavy_chain:
            return _unavailable(
                manifest_hash,
                "missing_frozen_heavy_chain_role",
                "frozen heavy-chain role is absent",
            )
        if isinstance(antigen_chains, str):
            antigen_chains = tuple(
                chain for chain in antigen_chains.split(",") if chain
            )
        if not isinstance(cdr_h3, Sequence) or isinstance(cdr_h3, str):
            return _unavailable(
                manifest_hash,
                "missing_frozen_cdr_h3_role",
                "frozen CDR-H3 role is absent",
            )
        if not isinstance(antigen_chains, Sequence) or isinstance(antigen_chains, str):
            return _unavailable(
                manifest_hash,
                "missing_frozen_antigen_role",
                "frozen antigen role is absent",
            )
        try:
            candidate = FrozenAntibodyRoles(
                heavy_chain=heavy_chain,
                cdr_h3=tuple(_as_residue_key(item) for item in cdr_h3),
                antigen_chains=tuple(str(chain) for chain in antigen_chains),
            )
        except (IndexError, KeyError, TypeError, ValueError):
            return _unavailable(
                manifest_hash,
                "invalid_frozen_cdr_h3_role",
                "frozen CDR-H3 role has invalid identities",
            )
    else:
        return _unavailable(
            manifest_hash, "missing_frozen_cdr_h3_role", "frozen CDR-H3 role is absent"
        )
    try:
        cdr_h3 = tuple(_as_residue_key(item) for item in candidate.cdr_h3)
    except (IndexError, KeyError, TypeError, ValueError):
        return _unavailable(
            manifest_hash,
            "invalid_frozen_cdr_h3_role",
            "frozen CDR-H3 role has invalid identities",
        )
    candidate = FrozenAntibodyRoles(
        heavy_chain=candidate.heavy_chain,
        cdr_h3=cdr_h3,
        antigen_chains=candidate.antigen_chains,
    )
    if not candidate.cdr_h3:
        return _unavailable(
            manifest_hash, "missing_frozen_cdr_h3_role", "frozen CDR-H3 role is absent"
        )
    if any(key.chain_id != candidate.heavy_chain for key in candidate.cdr_h3):
        return _unavailable(
            manifest_hash,
            "invalid_frozen_cdr_h3_role",
            "CDR-H3 identities disagree with the frozen heavy chain",
        )
    if len(set(candidate.cdr_h3)) != len(candidate.cdr_h3):
        return _unavailable(
            manifest_hash,
            "duplicate_frozen_cdr_h3_residue",
            "frozen CDR-H3 role repeats a residue",
        )
    return candidate

def _frozen_interface_roles(
    roles: FrozenAntibodyRoles | Mapping[str, object], manifest_hash: str
) -> FrozenAntibodyRoles | MetricUnavailable:
    candidate = _frozen_h3_roles(roles, manifest_hash)
    if isinstance(candidate, MetricUnavailable):
        return candidate
    if not candidate.antigen_chains:
        return _unavailable(
            manifest_hash,
            "missing_frozen_antigen_role",
            "frozen antigen role is absent",
        )
    return candidate

def _atom_index(
    atoms: Sequence[AtomRecord],
    manifest_hash: str,
) -> dict[AtomKey, AtomRecord] | MetricUnavailable:
    indexed: dict[AtomKey, AtomRecord] = {}
    for atom in atoms:
        if not isinstance(atom, AtomRecord) or not atom.chain or not atom.residue:
            return _unavailable(
                manifest_hash,
                "invalid_atom_record",
                "atom records require non-empty chain, residue, and atom identities",
            )
        if not atom.atom or type(atom.insertion_code) is not str:
            return _unavailable(
                manifest_hash,
                "invalid_atom_record",
                "atom records require non-empty chain, residue, and atom identities",
            )
        try:
            identity = atom.identity
        except (TypeError, ValueError):
            return _unavailable(
                manifest_hash,
                "invalid_atom_record",
                "atom records require non-empty chain, residue, and atom identities",
            )
        if identity in indexed:
            return _unavailable(
                manifest_hash,
                "duplicate_atom_identity",
                "atom identities must be unique",
            )
        indexed[identity] = atom
    return indexed

def _coordinates(atoms: Sequence[AtomRecord] | object) -> np.ndarray | None:
    array = np.asarray([atom.coordinates for atom in atoms], dtype=float)
    return (
        array
        if array.ndim == 2 and array.shape[1:] == (3,) and np.isfinite(array).all()
        else None
    )

def _kabsch_rmsd(predicted: np.ndarray, native: np.ndarray) -> float:
    predicted_centered = predicted - predicted.mean(axis=0)
    native_centered = native - native.mean(axis=0)
    left, _, right_transpose = np.linalg.svd(predicted_centered.T @ native_centered)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    residual = predicted_centered @ rotation - native_centered
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

def _chi_accuracy(
    chi: str,
    predicted: Mapping[AtomKey, AtomRecord],
    native: Mapping[AtomKey, AtomRecord],
    manifest_hash: str,
) -> float | MetricUnavailable:
    residue_keys = sorted(
        {key.residue_key for key in predicted} & {key.residue_key for key in native}
    )
    comparisons: list[bool] = []
    for residue_key in residue_keys:
        residue_name = _residue_name(residue_key, native)
        definitions = _RESIDUE_CHI_ATOMS.get(residue_name, _DEFAULT_CHI_ATOMS).get(
            chi, ()
        )
        if not definitions:
            continue
        predicted_angles = _chi_angles(predicted, residue_key, definitions)
        native_angles = _chi_angles(native, residue_key, definitions)
        if not predicted_angles or not native_angles:
            continue
        if any(not np.isfinite(angle) for angle in (*predicted_angles, *native_angles)):
            return _unavailable(
                manifest_hash,
                "nonfinite_coordinates",
                f"{chi} coordinates must be finite",
            )
        difference = min(
            abs(_wrapped_angle(predicted_angle - native_angle))
            for predicted_angle in predicted_angles
            for native_angle in native_angles
        )
        comparisons.append(difference <= 20.0)
    if not comparisons:
        return _unavailable(
            manifest_hash, f"no_common_{chi}_atoms", f"no common named atoms for {chi}"
        )
    return sum(comparisons) / len(comparisons)

def _residue_name(residue_key: ResidueKey, atoms: Mapping[AtomKey, AtomRecord]) -> str:
    return next(
        (
            (atom.residue_name or "").upper()
            for key, atom in atoms.items()
            if key.residue_key == residue_key
        ),
        "",
    )

def _chi_angles(
    atoms: Mapping[AtomKey, AtomRecord],
    residue_key: ResidueKey,
    definitions: Sequence[tuple[str, str, str, str]],
) -> tuple[float, ...]:
    angles = []
    for definition in definitions:
        quadruplet = [atoms.get(AtomKey(residue_key, name)) for name in definition]
        if any(atom is None for atom in quadruplet):
            continue
        points = _coordinates(quadruplet)
        if points is None:
            return (float("nan"),)
        angles.append(_dihedral(points))
    return tuple(angles)

def _dihedral(points: np.ndarray) -> float:
    b0, b1, b2 = points[1] - points[0], points[2] - points[1], points[3] - points[2]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return degrees(atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))

def _wrapped_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0

def _contact_pairs(
    atoms: Mapping[AtomKey, AtomRecord],
    roles: FrozenAntibodyRoles,
    cutoff: float,
    manifest_hash: str,
) -> set[tuple[AtomKey, AtomKey]] | MetricUnavailable:
    h3, antigen = _role_atoms(atoms, roles)
    if not h3 or not antigen:
        return _unavailable(
            manifest_hash,
            "missing_frozen_role_atoms",
            "missing frozen H3 or antigen atoms",
        )
    if not all(_finite_atom(atom) for atom in (*h3, *antigen)):
        return _unavailable(
            manifest_hash, "nonfinite_coordinates", "contact coordinates must be finite"
        )
    return {
        (left.identity, right.identity)
        for left in h3
        for right in antigen
        if _distance(left, right) <= cutoff
    }

def _role_atoms(
    atoms: Mapping[AtomKey, AtomRecord], roles: FrozenAntibodyRoles
) -> tuple[tuple[AtomRecord, ...], tuple[AtomRecord, ...]]:
    h3_residues = frozenset(roles.cdr_h3)
    antigen_chains = frozenset(roles.antigen_chains)
    h3 = tuple(atom for atom in atoms.values() if atom.residue_key in h3_residues)
    antigen = tuple(atom for atom in atoms.values() if atom.chain in antigen_chains)
    return h3, antigen

def _distance(left: AtomRecord, right: AtomRecord) -> float:
    return float(
        np.linalg.norm(np.asarray(left.coordinates) - np.asarray(right.coordinates))
    )

def _finite_atom(atom: AtomRecord) -> bool:
    return _coordinates((atom,)) is not None

def _vdw_radius(atom: AtomRecord) -> float:
    element = (atom.element or atom.atom[:2]).strip().upper()
    return _VDW_RADII.get(element, _VDW_RADII.get(element[:1], 1.70))

def _metric(
    name: str,
    direction: str,
    value: float | MetricUnavailable,
    manifest_hash: str,
) -> MetricValue | MetricUnavailable:
    if isinstance(value, MetricUnavailable):
        return MetricUnavailable(name, value.reason_code, value.detail, manifest_hash)
    return MetricValue(name, value, direction, manifest_hash)

def _as_residue_key(item: object) -> ResidueKey:
    if isinstance(item, ResidueKey):
        return item
    if isinstance(item, Mapping):
        return ResidueKey(
            str(item["chain_id"]),
            int(item["auth_seq_id"]),
            str(item.get("insertion_code", BLANK_INSERTION_CODE)),
        )
    if isinstance(item, Sequence) and not isinstance(item, str):
        if len(item) == 2:
            return ResidueKey(str(item[0]), int(item[1]), BLANK_INSERTION_CODE)
        if len(item) == 3:
            return ResidueKey(str(item[0]), int(item[1]), str(item[2]))
    raise TypeError("invalid residue key")

def _manifest_hash(value: str) -> str:
    if len(value) != _MANIFEST_HASH_LENGTH or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(
            "evaluator_manifest_hash must be a lowercase SHA-256 semantic hash"
        )
    return value

def _unavailable(
    manifest_hash: str, reason_code: str, detail: str
) -> MetricUnavailable:
    return MetricUnavailable(_PRIMARY_NAME, reason_code, detail, manifest_hash)
