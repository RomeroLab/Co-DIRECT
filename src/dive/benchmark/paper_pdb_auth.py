
from __future__ import annotations

import ctypes
import errno
import hashlib
import math
import os
import secrets
import stat
from dataclasses import dataclass, field
from pathlib import Path

from dive.benchmark.contracts import ArtifactIdentity

class PdbAuthenticationError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class PdbAtom:

    chain: str
    residue: int
    resname: str
    atom: str
    coordinate: tuple[float, float, float]

    def __post_init__(self) -> None:
        if type(self.chain) is not str or len(self.chain) != 1:
            raise PdbAuthenticationError("PDB atom chain must be one character")
        if type(self.residue) is not int:
            raise PdbAuthenticationError("PDB atom residue must be an integer")
        if type(self.resname) is not str or not self.resname or len(self.resname) > 3:
            raise PdbAuthenticationError("PDB atom residue name is invalid")
        if type(self.atom) is not str or not self.atom or len(self.atom) > 4:
            raise PdbAuthenticationError("PDB atom name is invalid")
        if (
            type(self.coordinate) is not tuple
            or len(self.coordinate) != 3
            or any(type(value) is not float for value in self.coordinate)
            or not all(math.isfinite(value) for value in self.coordinate)
        ):
            raise PdbAuthenticationError("PDB atom coordinate must be finite")

@dataclass(frozen=True, slots=True)
class AuthenticatedPdb:

    identity: ArtifactIdentity
    content: bytes
    atoms: tuple[PdbAtom, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ArtifactIdentity):
            raise PdbAuthenticationError("authenticated PDB identity is invalid")
        if type(self.content) is not bytes:
            raise PdbAuthenticationError("authenticated PDB content must be bytes")
        if (
            hashlib.sha256(self.content).hexdigest() != self.identity.sha256
            or len(self.content) != self.identity.size_bytes
        ):
            raise PdbAuthenticationError(
                "authenticated PDB content differs from its identity"
            )
        if (
            type(self.atoms) is not tuple
            or not self.atoms
            or any(not isinstance(atom, PdbAtom) for atom in self.atoms)
        ):
            raise PdbAuthenticationError("authenticated PDB atoms are invalid")
        atom_identities = tuple(
            (atom.chain, atom.residue, atom.atom) for atom in self.atoms
        )
        if len(set(atom_identities)) != len(atom_identities):
            raise PdbAuthenticationError(
                "authenticated PDB atom identity is duplicated"
            )
        residue_names: dict[tuple[str, int], str] = {}
        for atom in self.atoms:
            residue_identity = (atom.chain, atom.residue)
            claimed = residue_names.setdefault(residue_identity, atom.resname)
            if claimed != atom.resname:
                raise PdbAuthenticationError(
                    "authenticated PDB residue has multiple residue names"
                )

@dataclass(frozen=True, slots=True)
class PrivateBinderInputs:

    complex_pdb: ArtifactIdentity
    native_target: ArtifactIdentity
    _directory_descriptor: int = field(repr=False, compare=False)
    _complex_descriptor: int = field(repr=False, compare=False)
    _native_descriptor: int = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.complex_pdb, ArtifactIdentity) or not isinstance(
            self.native_target, ArtifactIdentity
        ):
            raise PdbAuthenticationError("private binder input identity is invalid")
        descriptors = (
            self._directory_descriptor,
            self._complex_descriptor,
            self._native_descriptor,
        )
        if len(set(descriptors)) != 3 or any(
            descriptor < 0 for descriptor in descriptors
        ):
            raise PdbAuthenticationError("private binder input descriptors are invalid")
        pid = os.getpid()
        if Path(self.complex_pdb.path) != Path(
            f"/proc/{pid}/fd/{self._complex_descriptor}"
        ) or Path(self.native_target.path) != Path(
            f"/proc/{pid}/fd/{self._native_descriptor}"
        ):
            raise PdbAuthenticationError("private binder input paths are invalid")
        try:
            directory = os.fstat(self._directory_descriptor)
            complex_file = os.fstat(self._complex_descriptor)
            native_file = os.fstat(self._native_descriptor)
        except OSError as error:
            raise PdbAuthenticationError(
                "private binder input descriptor is invalid"
            ) from error
        if (
            not stat.S_ISDIR(directory.st_mode)
            or directory.st_uid != os.geteuid()
            or stat.S_IMODE(directory.st_mode) != 0o700
        ):
            raise PdbAuthenticationError(
                "private binder input directory descriptor is invalid"
            )
        for retained in (complex_file, native_file):
            if (
                not stat.S_ISREG(retained.st_mode)
                or retained.st_nlink != 1
                or retained.st_uid != os.geteuid()
                or stat.S_IMODE(retained.st_mode) != 0o400
            ):
                raise PdbAuthenticationError(
                    "private binder input file descriptor is invalid"
                )

    @property
    def directory_descriptor(self) -> int:

        return self._directory_descriptor

    @property
    def complex_descriptor(self) -> int:

        return self._complex_descriptor

    @property
    def native_descriptor(self) -> int:

        return self._native_descriptor

    def close(self) -> None:

        errors: list[OSError] = []
        for name in (
            "_complex_descriptor",
            "_native_descriptor",
            "_directory_descriptor",
        ):
            descriptor = getattr(self, name)
            if descriptor < 0:
                continue
            object.__setattr__(self, name, -1)
            try:
                os.close(descriptor)
            except OSError as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def __enter__(self) -> PrivateBinderInputs:
        if (
            self._directory_descriptor < 0
            or self._complex_descriptor < 0
            or self._native_descriptor < 0
        ):
            raise PdbAuthenticationError("private binder inputs are closed")
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

def _parse_atom_line(
    line: str, *, label: str, line_number: int, expected_serial: int
) -> PdbAtom:
    if len(line) != 80 or line[:6] != "ATOM  ":
        raise PdbAuthenticationError(
            f"{label} line {line_number} does not use the frozen ATOM layout"
        )
    if line[6:11] != f"{expected_serial:>5d}":
        raise PdbAuthenticationError(
            f"{label} line {line_number} has a noncanonical atom serial"
        )
    try:
        residue = int(line[22:26])
    except ValueError as error:
        raise PdbAuthenticationError(
            f"{label} line {line_number} has a malformed residue number"
        ) from error
    atom = line[12:16].strip()
    alternate_location = line[16]
    resname = line[17:20].strip()
    chain = line[21].strip()
    if alternate_location != " ":
        raise PdbAuthenticationError(
            f"{label} line {line_number} uses an alternate atom location"
        )
    if not atom or not resname or not chain:
        raise PdbAuthenticationError(
            f"{label} line {line_number} has an empty atom identity field"
        )
    atom_name = atom if len(atom) == 4 else f" {atom}"
    if (
        line[11] != " "
        or line[12:16] != f"{atom_name:<4}"
        or line[20] != " "
        or line[17:20] != f"{resname:>3}"
        or line[21] != f"{chain:>1}"
        or line[22:26] != f"{residue:>4d}"
        or line[26] != " "
        or line[27:30] != "   "
        or line[54:60] != "  1.00"
        or line[60:66] != "  0.00"
        or line[66:76] != "          "
        or line[76:78] != f"{atom[0]:>2}"
        or line[78:80] != "  "
    ):
        raise PdbAuthenticationError(
            f"{label} line {line_number} differs from the frozen ATOM layout"
        )
    try:
        fields = tuple(line[start : start + 8] for start in (30, 38, 46))
        coordinate = tuple(float(field) for field in fields)
    except ValueError as error:
        raise PdbAuthenticationError(
            f"{label} line {line_number} has malformed coordinates"
        ) from error
    if len(coordinate) != 3 or not all(math.isfinite(value) for value in coordinate):
        raise PdbAuthenticationError(
            f"{label} line {line_number} has nonfinite coordinates"
        )
    if any(
        field != f"{value:>8.3f}"
        for field, value in zip(fields, coordinate, strict=True)
    ):
        raise PdbAuthenticationError(
            f"{label} line {line_number} has noncanonical coordinates"
        )
    return PdbAtom(chain, residue, resname, atom, coordinate)

def _validate_ter_line(
    line: str,
    *,
    label: str,
    line_number: int,
    expected_serial: int,
    preceding_atom: PdbAtom,
) -> None:
    expected = (
        f"TER   {expected_serial:5d}      "
        f"{preceding_atom.resname:>3} {preceding_atom.chain:>1}"
    )
    if line != expected:
        raise PdbAuthenticationError(
            f"{label} line {line_number} does not match the preceding residue TER"
        )

def _parse_authenticated_content(content: bytes, *, label: str) -> tuple[PdbAtom, ...]:
    if not content.endswith(b"\n") or b"\r" in content:
        raise PdbAuthenticationError(f"{label} is not canonical LF-framed content")
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PdbAuthenticationError(f"{label} is not strict UTF-8") from error
    lines = text[:-1].split("\n")
    if not lines or lines[0] != "MODEL        1":
        raise PdbAuthenticationError(f"{label} does not begin with frozen MODEL 1")
    if len(lines) < 4 or lines[-2:] != ["ENDMDL", "END"]:
        raise PdbAuthenticationError(f"{label} does not end with frozen ENDMDL/END")
    atoms: list[PdbAtom] = []
    active_chain: str | None = None
    closed_chains: set[str] = set()
    preceding_atom: PdbAtom | None = None
    expected_serial = 1
    for line_number, line in enumerate(lines[1:-2], start=2):
        if line.startswith("ATOM  "):
            atom = _parse_atom_line(
                line,
                label=label,
                line_number=line_number,
                expected_serial=expected_serial,
            )
            if active_chain is None:
                if atom.chain in closed_chains:
                    raise PdbAuthenticationError(
                        f"{label} line {line_number} reopens a terminated chain"
                    )
                active_chain = atom.chain
            elif atom.chain != active_chain:
                raise PdbAuthenticationError(
                    f"{label} line {line_number} changes chain without TER"
                )
            atoms.append(atom)
            preceding_atom = atom
        elif line.startswith("TER   "):
            if active_chain is None or preceding_atom is None:
                raise PdbAuthenticationError(
                    f"{label} line {line_number} has a premature or repeated TER"
                )
            _validate_ter_line(
                line,
                label=label,
                line_number=line_number,
                expected_serial=expected_serial,
                preceding_atom=preceding_atom,
            )
            closed_chains.add(active_chain)
            active_chain = None
            preceding_atom = None
        else:
            raise PdbAuthenticationError(
                f"{label} line {line_number} is an unsupported PDB record"
            )
        expected_serial += 1
    if not atoms or active_chain is not None or preceding_atom is not None:
        raise PdbAuthenticationError(f"{label} is missing its final TER")
    return tuple(atoms)

def read_authenticated_pdb(
    identity: ArtifactIdentity, *, label: str
) -> AuthenticatedPdb:

    if not isinstance(identity, ArtifactIdentity):
        raise PdbAuthenticationError(f"{label} identity is invalid")
    if type(label) is not str or not label:
        raise PdbAuthenticationError("PDB authentication label must be nonempty")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = -1
    try:
        descriptor = os.open(identity.path, flags)
        retained_stat = os.fstat(descriptor)
        if not stat.S_ISREG(retained_stat.st_mode):
            raise PdbAuthenticationError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
        content = b"".join(chunks)
        if digest.hexdigest() != identity.sha256 or size != identity.size_bytes:
            raise PdbAuthenticationError(f"{label} identity drifted during read")
        final_descriptor_stat = os.fstat(descriptor)
        try:
            final_path_stat = os.lstat(identity.path)
        except OSError as error:
            raise PdbAuthenticationError(
                f"{label} path cannot be authenticated after read"
            ) from error
        if (
            not stat.S_ISREG(final_path_stat.st_mode)
            or (retained_stat.st_dev, retained_stat.st_ino)
            != (final_descriptor_stat.st_dev, final_descriptor_stat.st_ino)
            or (retained_stat.st_dev, retained_stat.st_ino)
            != (final_path_stat.st_dev, final_path_stat.st_ino)
            or final_descriptor_stat.st_size != size
        ):
            raise PdbAuthenticationError(f"{label} path changed during read")
    except PdbAuthenticationError:
        raise
    except OSError as error:
        raise PdbAuthenticationError(f"cannot read authenticated {label}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    atoms = _parse_authenticated_content(content, label=label)
    return AuthenticatedPdb(identity=identity, content=content, atoms=atoms)

_FINAL_DIRECTORY_NAME = "binder-inputs"
_TRANSACTION_PREFIX = ".binder-inputs-txn-"
_RENAME_NOREPLACE = 1

def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise PdbAuthenticationError("private snapshot write made no progress")
        view = view[written:]

def _create_snapshot(directory_descriptor: int, name: str, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o400, dir_fd=directory_descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PdbAuthenticationError("private snapshot is not create-new regular")
        _write_all(descriptor, content)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _directory_leaf(value: str, *, label: str) -> bytes:
    if not value or value in {".", ".."} or "/" in value or "\x00" in value:
        raise PdbAuthenticationError(f"invalid private {label} directory name")
    return os.fsencode(value)

def _rename_directory_noreplace(
    parent_descriptor: int, source_leaf: str, destination_leaf: str
) -> None:

    source = _directory_leaf(source_leaf, label="transaction")
    destination = _directory_leaf(destination_leaf, label="final")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as error:
        raise PdbAuthenticationError(
            "atomic no-replace PDB publication is unavailable"
        ) from error
    renameat2 = getattr(libc, "renameat2", None)
    ctypes.set_errno(0)
    if renameat2 is not None:
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_descriptor,
            source,
            parent_descriptor,
            destination,
            _RENAME_NOREPLACE,
        )
    else:
        syscall = getattr(libc, "syscall", None)
        syscall_number = {"x86_64": 316, "aarch64": 276}.get(os.uname().machine)
        if syscall is None or syscall_number is None:
            raise PdbAuthenticationError(
                "atomic no-replace PDB publication is unavailable"
            )
        result = syscall(
            ctypes.c_long(syscall_number),
            ctypes.c_int(parent_descriptor),
            source,
            ctypes.c_int(parent_descriptor),
            destination,
            ctypes.c_uint(_RENAME_NOREPLACE),
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number, "private binder input directory already exists"
        )
    if error_number in {0, errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
        raise PdbAuthenticationError("atomic no-replace PDB publication is unavailable")
    raise OSError(error_number, os.strerror(error_number))

def _assert_published_directory(
    parent_descriptor: int, directory_descriptor: int
) -> None:
    retained = os.fstat(directory_descriptor)
    published = os.stat(
        _FINAL_DIRECTORY_NAME,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(retained.st_mode)
        or not stat.S_ISDIR(published.st_mode)
        or (retained.st_dev, retained.st_ino) != (published.st_dev, published.st_ino)
        or stat.S_IMODE(retained.st_mode) != 0o700
        or stat.S_IMODE(published.st_mode) != 0o700
    ):
        raise PdbAuthenticationError("published binder input directory drifted")

def _assert_retained_work_path(work_path: Path, parent_descriptor: int) -> None:
    retained = os.fstat(parent_descriptor)
    try:
        named = os.lstat(work_path)
    except OSError as error:
        raise PdbAuthenticationError("binder work directory path drifted") from error
    if (
        not stat.S_ISDIR(retained.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or (retained.st_dev, retained.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise PdbAuthenticationError("binder work directory path drifted")

def _reauthenticate_snapshot(
    directory_descriptor: int,
    *,
    name: str,
    expected: bytes,
) -> tuple[ArtifactIdentity, int]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        retained = os.fstat(descriptor)
        if (
            not stat.S_ISREG(retained.st_mode)
            or retained.st_nlink != 1
            or stat.S_IMODE(retained.st_mode) != 0o400
        ):
            raise PdbAuthenticationError("published private snapshot is not immutable")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
        observed = b"".join(chunks)
        published = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            (retained.st_dev, retained.st_ino) != (published.st_dev, published.st_ino)
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o400
            or observed != expected
            or size != len(expected)
            or digest.hexdigest() != hashlib.sha256(expected).hexdigest()
        ):
            raise PdbAuthenticationError("published private snapshot drifted")
        os.lseek(descriptor, 0, os.SEEK_SET)
        identity = ArtifactIdentity(
            f"/proc/{os.getpid()}/fd/{descriptor}",
            digest.hexdigest(),
            size,
        )
        kept = descriptor
        descriptor = -1
        return identity, kept
    finally:
        if descriptor >= 0:
            os.close(descriptor)

def publish_private_binder_inputs(
    *,
    complex_pdb: AuthenticatedPdb,
    native_target: AuthenticatedPdb,
    work_dir: Path,
) -> PrivateBinderInputs:

    if not isinstance(complex_pdb, AuthenticatedPdb) or not isinstance(
        native_target, AuthenticatedPdb
    ):
        raise PdbAuthenticationError("private publication requires authenticated PDBs")
    work_path = Path(work_dir).absolute()
    parent_descriptor = -1
    transaction_descriptor = -1
    complex_descriptor = -1
    native_descriptor = -1
    transaction_leaf = f"{_TRANSACTION_PREFIX}{os.getpid()}-{secrets.token_hex(16)}"
    try:
        parent_descriptor = os.open(
            work_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        parent_metadata = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise PdbAuthenticationError("binder work directory is not a directory")
        _assert_retained_work_path(work_path, parent_descriptor)
        os.mkdir(transaction_leaf, 0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        transaction_descriptor = os.open(
            transaction_leaf,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        os.fchmod(transaction_descriptor, 0o700)
        transaction_metadata = os.fstat(transaction_descriptor)
        if (
            not stat.S_ISDIR(transaction_metadata.st_mode)
            or transaction_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(transaction_metadata.st_mode) != 0o700
        ):
            raise PdbAuthenticationError("binder input transaction is not private")
        _create_snapshot(transaction_descriptor, "complex.pdb", complex_pdb.content)
        _create_snapshot(
            transaction_descriptor, "native_target.pdb", native_target.content
        )
        os.fsync(transaction_descriptor)
        _rename_directory_noreplace(
            parent_descriptor, transaction_leaf, _FINAL_DIRECTORY_NAME
        )
        os.fsync(parent_descriptor)
        _assert_published_directory(parent_descriptor, transaction_descriptor)
        complex_identity, complex_descriptor = _reauthenticate_snapshot(
            transaction_descriptor,
            name="complex.pdb",
            expected=complex_pdb.content,
        )
        native_identity, native_descriptor = _reauthenticate_snapshot(
            transaction_descriptor,
            name="native_target.pdb",
            expected=native_target.content,
        )
        _assert_published_directory(parent_descriptor, transaction_descriptor)
        _assert_retained_work_path(work_path, parent_descriptor)
        private_inputs = PrivateBinderInputs(
            complex_identity,
            native_identity,
            transaction_descriptor,
            complex_descriptor,
            native_descriptor,
        )
        transaction_descriptor = -1
        complex_descriptor = -1
        native_descriptor = -1
        return private_inputs
    except PdbAuthenticationError:
        raise
    except (OSError, ValueError) as error:
        raise PdbAuthenticationError(
            "private binder input publication failed"
        ) from error
    finally:
        for descriptor in (
            native_descriptor,
            complex_descriptor,
            transaction_descriptor,
            parent_descriptor,
        ):
            if descriptor >= 0:
                os.close(descriptor)

__all__ = (
    "AuthenticatedPdb",
    "PdbAtom",
    "PdbAuthenticationError",
    "PrivateBinderInputs",
    "publish_private_binder_inputs",
    "read_authenticated_pdb",
)
