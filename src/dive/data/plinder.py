
from __future__ import annotations

import io
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dive.data.contracts import ContractError, EligibilityFailure

BUCKET_URL = "https://storage.googleapis.com/plinder/2024-06/v2/systems"

READ_BUFFER_BYTES = 1 << 14

@dataclass(frozen=True, slots=True)
class ResolvedSystem:

    system_id: str
    scaffold_chains: tuple[tuple[str, str], ...]
    ligand_smiles: str

    @property
    def scaffold_sequence(self) -> str:

        if len(self.scaffold_chains) != 1:
            raise ContractError(
                f"{self.system_id} has {len(self.scaffold_chains)} receptor chains"
            )
        return self.scaffold_chains[0][1]

    def as_row(self) -> dict[str, object]:
        return {
            "scaffold_chains": list(self.scaffold_chains),
            "ligand_smiles": self.ligand_smiles,
        }

def _parts(system_id: str) -> list[str]:
    parts = system_id.split("__")
    if len(parts) != 4:
        raise ValueError(
            f"malformed system id {system_id!r}; expected pdb__assembly__receptor__ligand"
        )
    return parts

def shard_for(system_id: str) -> str:

    return system_id[1:3]

def receptor_chain_of(system_id: str) -> str:

    return _parts(system_id)[2]

def receptor_chains_of(system_id: str) -> tuple[str, ...]:

    return tuple(part for part in receptor_chain_of(system_id).split("_") if part)

def ligand_chain_of(system_id: str) -> str:
    return _parts(system_id)[3]

def ligand_chains_of(system_id: str) -> tuple[str, ...]:
    return tuple(part for part in ligand_chain_of(system_id).split("_") if part)

def parse_sequences_fasta(text: str) -> dict[str, str]:

    sequences: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].strip()
            sequences.setdefault(current, [])
        elif current is not None:
            sequences[current].append(line)
    return {chain: "".join(parts) for chain, parts in sequences.items()}

def smiles_from_molblock(molblock: str) -> str | None:

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromMolBlock(molblock)
    return Chem.MolToSmiles(mol) if mol is not None else None

def resolve_system(
    system_id: str, fasta_text: str, molblock: str
) -> ResolvedSystem | EligibilityFailure:

    chains = receptor_chains_of(system_id)
    sequences = parse_sequences_fasta(fasta_text)

    resolved: list[tuple[str, str]] = []
    for chain in chains:
        sequence = sequences.get(chain, "").strip()
        if not sequence:
            return EligibilityFailure(
                "missing_scaffold_sequence",
                f"receptor chain {chain!r} absent or empty in {sorted(sequences)}",
                system_id,
            )
        resolved.append((chain, sequence))

    if not resolved:
        return EligibilityFailure(
            "missing_scaffold_sequence", "the system id names no receptor chain", system_id
        )

    smiles = smiles_from_molblock(molblock)
    if smiles is None:
        return EligibilityFailure(
            "unparseable_ligand", "RDKit could not read the ligand mol block", system_id
        )

    return ResolvedSystem(
        system_id=system_id, scaffold_chains=tuple(resolved), ligand_smiles=smiles
    )

def read_system_members(archive: zipfile.ZipFile, system_id: str) -> tuple[str, str]:

    prefix = f"{system_id}/"
    names = [name for name in archive.namelist() if name.startswith(prefix)]
    if not names:
        raise KeyError(f"{system_id} is not present in this shard")

    fasta_name = f"{prefix}sequences.fasta"
    if fasta_name not in names:
        raise KeyError(f"{fasta_name} missing")

    ligand_names = sorted(n for n in names if f"{prefix}ligand_files/" in n)
    if not ligand_names:
        raise KeyError(f"{system_id} has no ligand file")

    candidates = [f"{prefix}ligand_files/{chain}.sdf" for chain in ligand_chains_of(system_id)]
    ligand_name = next((c for c in candidates if c in ligand_names), ligand_names[0])

    return (
        archive.read(fasta_name).decode("utf-8", "replace"),
        archive.read(ligand_name).decode("utf-8", "replace"),
    )

class RangeFile(io.RawIOBase):

    def __init__(self, size: int, fetch: Callable[[int, int], bytes]) -> None:
        self._size = size
        self._fetch = fetch
        self._pos = 0
        self.bytes_read = 0
        self.requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self._size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, buffer) -> int:
        wanted = len(buffer)
        if wanted <= 0 or self._pos >= self._size:
            return 0
        end = min(self._pos + wanted, self._size) - 1
        data = self._fetch(self._pos, end)
        self._pos += len(data)
        self.bytes_read += len(data)
        self.requests += 1
        buffer[: len(data)] = data
        return len(data)

def open_remote_shard(shard: str, *, bucket_url: str = BUCKET_URL, timeout: int = 120):

    url = f"{bucket_url}/{shard}.zip"
    try:
        head = urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=timeout
        )
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise FileNotFoundError(f"no shard {shard!r} in the bucket") from error
        raise
    size = int(head.headers["Content-Length"])

    def fetch(start: int, end: int) -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        return urllib.request.urlopen(request, timeout=timeout).read()

    handle = RangeFile(size=size, fetch=fetch)
    return zipfile.ZipFile(io.BufferedReader(handle, READ_BUFFER_BYTES)), handle

def open_local_shard(path: Path) -> tuple[zipfile.ZipFile, None]:

    return zipfile.ZipFile(path), None
