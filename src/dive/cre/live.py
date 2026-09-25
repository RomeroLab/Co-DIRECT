
from __future__ import annotations

import numpy as np
import torch

from dive.ccr.candidate import ATOM37_NAMES, SLOT_OF_NAME
from dive.ccr.decode import mask_absent_atoms
from dive.cre.exchange import N_FEATURES, build_patch_candidates, patch_message
from dive.cre.wiring import apply_variant, message_tensor

BASIN_ANGLES_DEG = (-60.0, 60.0, 180.0)

ENVIRONMENT_RADIUS = 8.0

RESTYPES = "ARNDCQEGHILKMFPSTWYV"

def dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))

def measured_chi(residue_xyz, code) -> tuple[float, ...]:

    from dive.ccr.candidate import ONE_TO_THREE, n_chi, _constants

    rc = _constants()
    placed = {
        name: residue_xyz[SLOT_OF_NAME[name]] for name in SLOT_OF_NAME
        if np.isfinite(residue_xyz[SLOT_OF_NAME[name]]).all()
    }
    angles = []
    for quartet in rc.chi_angles_atoms[ONE_TO_THREE[code]]:
        if not all(name in placed for name in quartet):
            break
        angles.append(dihedral(*[placed[name] for name in quartet]))
    count = n_chi(code)
    angles += [np.deg2rad(180.0)] * (count - len(angles))
    return tuple(angles[:count])

ELEMENT_OF_SLOT = [name[0] for name in ATOM37_NAMES]

class AmeExchange:

    def __init__(self, *, variant: str = "full", environment_radius: float = ENVIRONMENT_RADIUS):
        self.variant = variant
        self.environment_radius = float(environment_radius)
        self.stats = {
            "calls": 0, "examples": 0, "residues_scored": 0,
            "residues_with_candidates": 0, "candidates_built": 0,
            "examples_without_condition": 0,
        }

    def __call__(self, batch: dict, clean: dict, model=None, autoencoder=None) -> torch.Tensor | None:
        self.stats["calls"] += 1
        mask = batch["mask"]
        device = mask.device
        batch_size, n_residues = mask.shape

        decoder = autoencoder if autoencoder is not None else getattr(model, "autoencoder", None)
        if decoder is None:
            return None
        with torch.no_grad():
            decoded = decoder.decode(clean["local_latents"], clean["bb_ca"], mask)
            coors = mask_absent_atoms(
                decoded["coors_nm"], decoded["atom_mask"].bool(), mask.bool()
            )
        coordinates = coors.detach().cpu().numpy() * 10.0
        residue_types = decoded["residue_type"].detach().cpu().numpy()
        residue_mask = mask.detach().cpu().numpy().astype(bool)

        x_motif = batch.get("x_motif")
        motif_mask = batch.get("motif_mask")
        if x_motif is None or motif_mask is None:
            return None

        supplied = x_motif.detach().cpu().numpy() * 10.0
        supplied_mask = motif_mask.detach().cpu().numpy().astype(bool)

        given = batch.get("motif_residue_mask")
        if given is not None:
            motif_residue_mask = given.detach().cpu().numpy().astype(bool)
            assignment = [
                np.nonzero(motif_residue_mask[i])[0] for i in range(batch_size)
            ]
        else:
            assignment = [
                self._assign(coordinates[i], residue_mask[i], supplied[i], supplied_mask[i])
                for i in range(batch_size)
            ]

        out = torch.zeros(batch_size, n_residues, N_FEATURES, dtype=torch.float32)
        for example in range(batch_size):
            self.stats["examples"] += 1
            indices = assignment[example]
            if indices is None or len(indices) == 0:
                self.stats["examples_without_condition"] += 1
                continue
            per_residue: dict[int, np.ndarray] = {}
            for row, index in enumerate(indices):
                if row >= supplied.shape[1] or index < 0:
                    continue
                required = [
                    (slot, supplied[example, row, slot])
                    for slot in range(supplied_mask.shape[2])
                    if supplied_mask[example, row, slot]
                ]
                if not required:
                    continue
                self.stats["residues_scored"] += 1
                code = self._code(residue_types[example, index])
                if code is None:
                    continue
                environment, elements = self._environment(
                    coordinates[example], residue_mask[example], index
                )
                centres = self._basins(code, coordinates[example, index])
                candidates = build_patch_candidates(
                    residue_xyz=coordinates[example, index],
                    code=code,
                    required=required,
                    environment=environment,
                    environment_elements=elements,
                    chi_centres=centres,
                )
                if not candidates:
                    continue
                self.stats["residues_with_candidates"] += 1
                self.stats["candidates_built"] += len(candidates)
                per_residue[int(index)] = patch_message(candidates)
            if per_residue:
                out[example] = message_tensor(
                    per_residue, n_residues=n_residues
                )[0]
        return apply_variant(out.to(device), self.variant)

    @staticmethod
    def _assign(coordinates, residue_mask, supplied, supplied_mask):

        from scipy.optimize import linear_sum_assignment

        live = np.nonzero(residue_mask)[0]
        rows = [k for k in range(supplied.shape[0]) if supplied_mask[k].any()]
        if not rows or len(live) < len(rows):
            return None
        cap = 10.0
        cost = np.zeros((len(rows), len(live)))
        for i, k in enumerate(rows):
            slots = np.nonzero(supplied_mask[k])[0]
            target = supplied[k][slots]
            block = coordinates[live][:, slots, :]
            distance = np.linalg.norm(block - target[None], axis=-1)
            distance = np.where(np.isfinite(distance), distance, cap)
            cost[i] = np.minimum(distance, cap).sum(axis=1)
        _, columns = linear_sum_assignment(cost)
        chosen = np.full(supplied.shape[0], -1, dtype=int)
        for i, column in zip(rows, columns):
            chosen[i] = int(live[column])
        return chosen

    @staticmethod
    def _code(residue_type) -> str | None:
        value = int(residue_type)
        return RESTYPES[value] if 0 <= value < 20 else None

    @staticmethod
    def _basins(code: str, residue_xyz) -> list[tuple[float, ...]]:

        from dive.cre.exchange import anchored_basins

        return anchored_basins(measured_chi(residue_xyz, code))

    def _environment(self, coordinates, residue_mask, index):

        ca = coordinates[index, SLOT_OF_NAME["CA"]]
        if not np.isfinite(ca).all():
            return np.zeros((0, 3)), []
        keep = np.array(residue_mask, dtype=bool).copy()
        low, high = max(0, int(index) - 1), min(len(keep), int(index) + 2)
        keep[low:high] = False
        finite = np.isfinite(coordinates).all(axis=-1) & keep[:, None]
        distances = np.linalg.norm(coordinates - ca, axis=-1)
        selected = finite & (distances <= self.environment_radius)
        rows, slots = np.nonzero(selected)
        if len(rows) == 0:
            return np.zeros((0, 3)), []
        return coordinates[rows, slots], [ELEMENT_OF_SLOT[s] for s in slots]
