
from __future__ import annotations

import ctypes
import errno
import grp
import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.evaluators.contracts import EvidenceContract
from dive.benchmark.evaluators.worker_errors import (
    WorkerFailureReason,
    WorkerProtocolError,
    protocol_error,
)
from dive.signed_value.roots import EMERGENT_BULK_ROOT, EMERGENT_EVIDENCE_ROOT

_RENAME_NOREPLACE = 1
_STAGING_DIRECTORY_PREFIX = ".dive-outputs-txn-"

@dataclass(frozen=True, slots=True)
class DisposableEvidencePolicy:

    terminal_root: Path
    bulk_root: Path
    expected_gid: int | None = None
    require_group_writable_setgid: bool = False
    before_final_open: Callable[[Path, bool], None] | None = None
    before_launch: Callable[[], None] | None = None

class _OutputRequest(Protocol):
    request_path: str
    result_path: str
    stdout_path: str
    stderr_path: str
    bulk_directory: str

__all__ = (
    "DisposableEvidencePolicy",
    "close_descriptors",
    "create_new_fd",
    "ensure_directory",
    "ensure_file_parent",
    "hash_identity",
    "hash_open_descriptor",
    "open_authenticated_file",
    "open_cwd",
    "open_parent_fd",
    "open_read_fd",
    "preflight_output_paths",
    "read_descriptor_bytes",
    "reserve_outputs_atomically",
    "resolve_evidence_roots",
    "rewind_descriptor",
    "write_create_new",
)

def resolve_evidence_roots(
    evaluator: object | None, policy: DisposableEvidencePolicy | None
) -> tuple[Path, Path, int | None, bool]:

    if policy is not None:
        return (
            policy.terminal_root,
            policy.bulk_root,
            policy.expected_gid,
            policy.require_group_writable_setgid,
        )
    if evaluator is not None:
        evidence = getattr(evaluator, "evidence", None)
        if (
            not isinstance(evidence, EvidenceContract)
            or evidence.create_new is not True
        ):
            raise protocol_error(
                WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                "authenticated evidence roots are required",
            )
        if (
            Path(evidence.terminal_root) != EMERGENT_EVIDENCE_ROOT / "benchmark_ready"
            or Path(evidence.bulk_root) != EMERGENT_BULK_ROOT / "benchmark_ready"
        ):
            raise protocol_error(
                WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                "production roots differ from Task 1 contract",
            )
        return (
            Path(evidence.terminal_root),
            Path(evidence.bulk_root),
            grp.getgrnam("romerolab").gr_gid,
            True,
        )
    return (
        EMERGENT_EVIDENCE_ROOT / "benchmark_ready",
        EMERGENT_BULK_ROOT / "benchmark_ready",
        grp.getgrnam("romerolab").gr_gid,
        True,
    )

def _relative_parts(path: Path, root: Path) -> tuple[str, ...]:
    if not path.is_absolute() or not root.is_absolute():
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "path or root is not absolute",
        )
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "path escapes configured roots",
        ) from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "invalid evidence relative path",
        )
    return relative.parts

def _validate_directory_fd(
    descriptor: int, expected_gid: int | None, group_policy: bool
) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "non-directory ancestry",
        )
    if group_policy and (
        metadata.st_gid != expected_gid
        or not metadata.st_mode & stat.S_ISGID
        or not metadata.st_mode & stat.S_IWGRP
    ):
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "evidence ancestry ownership or mode is invalid",
        )

def open_parent_fd(
    path: Path,
    root: Path,
    *,
    write: bool,
    expected_gid: int | None,
    group_policy: bool,
) -> tuple[int, str]:

    parts = _relative_parts(path, root)
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "cannot open evidence root",
        ) from error
    try:
        _validate_directory_fd(descriptor, expected_gid, group_policy)
        for part in parts[:-1]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not write:
                    raise
                try:
                    os.mkdir(part, 0o2775 if group_policy else 0o755, dir_fd=descriptor)
                    if group_policy:
                        os.chmod(part, 0o2775, dir_fd=descriptor, follow_symlinks=False)
                except FileExistsError:
                    pass
                except OSError as error:
                    raise protocol_error(
                        WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                        "cannot create evidence ancestry",
                    ) from error
                try:
                    child = os.open(part, flags, dir_fd=descriptor)
                except OSError as error:
                    raise protocol_error(
                        WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                        "unsafe evidence ancestry",
                    ) from error
            except OSError as error:
                raise protocol_error(
                    WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                    "unsafe evidence ancestry",
                ) from error
            try:
                _validate_directory_fd(child, expected_gid, group_policy)
            except Exception:
                os.close(child)
                raise
            previous = descriptor
            descriptor = child
            os.close(previous)
        return descriptor, parts[-1]
    except Exception:
        os.close(descriptor)
        raise

def open_read_fd(
    path: Path,
    root: Path,
    *,
    expected_gid: int | None,
    group_policy: bool,
    policy: DisposableEvidencePolicy | None,
) -> int:

    try:
        parent, leaf = open_parent_fd(
            path,
            root,
            write=False,
            expected_gid=expected_gid,
            group_policy=group_policy,
        )
    except FileNotFoundError as error:
        raise protocol_error(
            WorkerFailureReason.RESULT_MISSING,
            "output is missing",
        ) from error
    descriptor: int | None = None
    transferred = False
    try:
        if policy is not None and policy.before_final_open is not None:
            policy.before_final_open(path, False)
        descriptor = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise protocol_error(
                WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
                "output is not a regular file",
            )
        transferred = True
        return descriptor
    except WorkerProtocolError:
        raise
    except OSError as error:
        if error.errno == errno.ENOENT:
            raise protocol_error(
                WorkerFailureReason.RESULT_MISSING,
                "output is missing",
            ) from error
        raise protocol_error(
            WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
            "unsafe or missing output",
        ) from error
    finally:
        if descriptor is not None and not transferred:
            try:
                os.close(descriptor)
            except OSError:
                pass
        os.close(parent)

def write_create_new(
    path: Path,
    root: Path,
    payload: bytes,
    *,
    expected_gid: int | None,
    group_policy: bool,
    policy: DisposableEvidencePolicy | None,
) -> Path:

    parent, leaf = open_parent_fd(
        path,
        root,
        write=True,
        expected_gid=expected_gid,
        group_policy=group_policy,
    )
    try:
        if policy is not None and policy.before_final_open is not None:
            policy.before_final_open(path, True)
        descriptor = os.open(
            leaf,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o664,
            dir_fd=parent,
        )
    except FileExistsError:
        raise
    except OSError as error:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "unsafe create-new destination",
        ) from error
    finally:
        os.close(parent)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("create-new write made no forward progress")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as error:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "create-new write failed",
        ) from error
    finally:
        os.close(descriptor)
    return path

def create_new_fd(
    path: Path,
    root: Path,
    expected_gid: int | None,
    group_policy: bool,
    policy: DisposableEvidencePolicy | None,
) -> int:

    parent, leaf = open_parent_fd(
        path,
        root,
        write=True,
        expected_gid=expected_gid,
        group_policy=group_policy,
    )
    descriptor: int | None = None
    transferred = False
    try:
        if policy is not None and policy.before_final_open is not None:
            policy.before_final_open(path, True)
        descriptor = os.open(
            leaf,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o664,
            dir_fd=parent,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise protocol_error(
                WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                "create-new destination is not a regular file",
            )
        transferred = True
        return descriptor
    except FileExistsError:
        raise
    except WorkerProtocolError:
        raise
    except OSError as error:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "unsafe create-new destination",
        ) from error
    finally:
        if not transferred:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        try:
            os.close(parent)
        except OSError:
            pass

def hash_open_descriptor(descriptor: int) -> tuple[str, int]:

    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("descriptor is not regular")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1 << 20):
            digest.update(block)
        return digest.hexdigest(), os.fstat(descriptor).st_size
    except OSError as error:
        raise protocol_error(
            WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
            "cannot hash retained descriptor",
        ) from error

def read_descriptor_bytes(descriptor: int, *, rewind: bool = True) -> bytes:

    if rewind:
        os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(iter(lambda: os.read(descriptor, 1 << 20), b""))

def rewind_descriptor(descriptor: int) -> None:

    os.lseek(descriptor, 0, os.SEEK_SET)

def ensure_file_parent(
    path: Path,
    root: Path,
    *,
    expected_gid: int | None,
    group_policy: bool,
) -> None:

    descriptor, _ = open_parent_fd(
        path,
        root,
        write=True,
        expected_gid=expected_gid,
        group_policy=group_policy,
    )
    os.close(descriptor)

def ensure_directory(
    path: Path,
    root: Path,
    *,
    expected_gid: int | None,
    group_policy: bool,
) -> None:

    descriptor, _ = open_parent_fd(
        path / "placeholder",
        root,
        write=True,
        expected_gid=expected_gid,
        group_policy=group_policy,
    )
    os.close(descriptor)

def _dedicated_output_directory(paths: tuple[Path, ...], request_path: Path) -> Path:

    if not paths or len(set(paths)) != len(paths):
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "declared outputs must be distinct",
        )
    parents = {path.parent for path in paths}
    expected = request_path.parent / "outputs"
    if parents != {expected}:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "worker outputs must share the dedicated run output directory",
        )
    return expected

def preflight_output_paths(
    request: _OutputRequest,
    *,
    evaluator: object | None = None,
    policy: DisposableEvidencePolicy | None = None,
    request_path: Path | None = None,
) -> None:

    terminal_root, bulk_root, expected_gid, group_policy = resolve_evidence_roots(
        evaluator, policy
    )
    declared_request_path = Path(request.request_path)
    ensure_file_parent(
        declared_request_path,
        terminal_root,
        expected_gid=expected_gid,
        group_policy=group_policy,
    )
    if request_path is not None:
        output_directory = _dedicated_output_directory(
            (
                Path(request.stdout_path),
                Path(request.stderr_path),
                Path(request.result_path),
            ),
            declared_request_path,
        )
        parent, leaf = open_parent_fd(
            output_directory,
            terminal_root,
            write=False,
            expected_gid=expected_gid,
            group_policy=group_policy,
        )
        try:
            try:
                os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise protocol_error(
                    WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                    "dedicated output directory already exists",
                )
        finally:
            os.close(parent)
    else:
        for path in (
            Path(request.result_path),
            Path(request.stdout_path),
            Path(request.stderr_path),
        ):
            ensure_file_parent(
                path,
                terminal_root,
                expected_gid=expected_gid,
                group_policy=group_policy,
            )
    ensure_directory(
        Path(request.bulk_directory),
        bulk_root,
        expected_gid=expected_gid,
        group_policy=group_policy,
    )
    if request_path is not None and request_path != declared_request_path:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "request path differs from request",
        )

def hash_identity(
    identity: ArtifactIdentity,
    root: Path,
    *,
    expected_gid: int | None,
    group_policy: bool,
    policy: DisposableEvidencePolicy | None,
) -> None:

    path = Path(identity.path)
    try:
        descriptor = open_read_fd(
            path,
            root,
            expected_gid=expected_gid,
            group_policy=group_policy,
            policy=policy,
        )
    except WorkerProtocolError as error:
        raise protocol_error(
            WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
            f"missing output: {path}",
        ) from error
    try:
        digest, observed_size = hash_open_descriptor(descriptor)
    except WorkerProtocolError as error:
        raise protocol_error(
            WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
            f"cannot hash output: {path}",
        ) from error
    finally:
        os.close(descriptor)
    if digest != identity.sha256 or observed_size != identity.size_bytes:
        raise protocol_error(
            WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
            f"output identity drift: {path}",
        )

def open_authenticated_file(path: Path, identity: ArtifactIdentity, label: str) -> int:

    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            f"cannot open authenticated {label}",
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise protocol_error(
                WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
                f"authenticated {label} is not a regular file",
            )
        try:
            digest, size = hash_open_descriptor(descriptor)
        except WorkerProtocolError as error:
            raise protocol_error(
                WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
                f"cannot hash authenticated {label}",
            ) from error
        if digest != identity.sha256 or size != identity.size_bytes:
            raise protocol_error(
                WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
                f"authenticated {label} identity drifted",
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise

def open_cwd(path: str) -> int:

    descriptor: int | None = None
    transferred = False
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise protocol_error(
                WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
                "authenticated working directory is not a directory",
            )
        transferred = True
        return descriptor
    except WorkerProtocolError:
        raise
    except OSError as error:
        raise protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "cannot open authenticated working directory",
        ) from error
    finally:
        if descriptor is not None and not transferred:
            try:
                os.close(descriptor)
            except OSError:
                pass

def _stage_unnamed_output(parent_descriptor: int) -> int:

    temporary_flag = getattr(os, "O_TMPFILE", 0)
    if not temporary_flag:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "unnamed output staging is unavailable",
        )
    descriptor: int | None = None
    transferred = False
    try:
        descriptor = os.open(
            ".",
            os.O_RDWR | temporary_flag | getattr(os, "O_CLOEXEC", 0),
            0o664,
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 0:
            raise protocol_error(
                WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                "worker output staging inode is not unnamed and regular",
            )
        transferred = True
        return descriptor
    except WorkerProtocolError:
        raise
    except OSError as error:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "cannot stage unnamed worker output",
        ) from error
    finally:
        if descriptor is not None and not transferred:
            try:
                os.close(descriptor)
            except OSError:
                pass

def _staging_directory_leaf() -> str:

    return f"{_STAGING_DIRECTORY_PREFIX}{os.getpid()}-{os.urandom(16).hex()}"

def _single_directory_leaf(leaf: str, *, label: str) -> bytes:

    if not leaf or leaf in {".", ".."} or "/" in leaf or "\x00" in leaf:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            f"invalid {label} directory entry",
        )
    return os.fsencode(leaf)

def _rename_directory_noreplace(
    parent_descriptor: int, source_leaf: str, destination_leaf: str
) -> None:

    source_name = _single_directory_leaf(source_leaf, label="staging")
    destination_name = _single_directory_leaf(destination_leaf, label="output")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as error:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "atomic no-replace directory publication is unavailable",
        ) from error
    function = getattr(libc, "renameat2", None)
    ctypes.set_errno(0)
    if function is not None:
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source_name,
            parent_descriptor,
            destination_name,
            _RENAME_NOREPLACE,
        )
    else:
        syscall = getattr(libc, "syscall", None)
        number = {"x86_64": 316, "aarch64": 276}.get(os.uname().machine)
        if syscall is None or number is None:
            raise protocol_error(
                WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                "atomic no-replace directory publication is unavailable",
            )
        result = syscall(
            ctypes.c_long(number),
            ctypes.c_int(parent_descriptor),
            source_name,
            ctypes.c_int(parent_descriptor),
            destination_name,
            ctypes.c_uint(_RENAME_NOREPLACE),
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "atomic no-replace directory publication is unavailable",
        )
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number))
    if error_number == 0:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "atomic no-replace directory publication is unavailable",
        )
    raise OSError(error_number, os.strerror(error_number))

def _rollback_staging_namespace(
    directory_descriptor: int | None,
    parent_descriptor: int,
    staging_leaf: str,
    linked_leaves: list[str],
) -> None:

    cleanup_error: OSError | None = None
    if directory_descriptor is not None:
        for leaf in reversed(linked_leaves):
            try:
                os.unlink(leaf, dir_fd=directory_descriptor)
            except OSError as error:
                cleanup_error = cleanup_error or error
    try:
        os.rmdir(staging_leaf, dir_fd=parent_descriptor)
    except OSError as error:
        cleanup_error = cleanup_error or error
    if cleanup_error is not None:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "staging output rollback failed",
        ) from cleanup_error

def reserve_outputs_atomically(
    paths: tuple[Path, ...],
    root: Path,
    *,
    request_path: Path,
    expected_gid: int | None,
    group_policy: bool,
    policy: DisposableEvidencePolicy,
) -> list[int]:

    output_directory = _dedicated_output_directory(paths, request_path)
    try:
        parent_descriptor, directory_leaf = open_parent_fd(
            output_directory,
            root,
            write=False,
            expected_gid=expected_gid,
            group_policy=group_policy,
        )
    except (FileNotFoundError, WorkerProtocolError) as error:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "output directory parent is unavailable for create-new claim",
        ) from error
    try:
        parent_metadata = os.fstat(parent_descriptor)
    except OSError as error:
        try:
            os.close(parent_descriptor)
        except OSError:
            pass
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "cannot validate the output directory parent",
        ) from error
    if parent_metadata.st_uid != os.geteuid():
        os.close(parent_descriptor)
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "output directory parent is not exclusively owned",
        )
    parent_is_shared = bool(parent_metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
    if parent_is_shared and not parent_metadata.st_mode & stat.S_ISVTX:
        try:
            os.fchmod(
                parent_descriptor,
                stat.S_IMODE(parent_metadata.st_mode) | stat.S_ISVTX,
            )
            parent_metadata = os.fstat(parent_descriptor)
        except OSError as error:
            os.close(parent_descriptor)
            raise protocol_error(
                WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                "cannot make the output directory parent exclusive",
            ) from error
        if not parent_metadata.st_mode & stat.S_ISVTX:
            os.close(parent_descriptor)
            raise protocol_error(
                WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                "output directory parent is not exclusively owned",
            )

    staged: list[int] = []
    linked_leaves: list[str] = []
    directory_descriptor: int | None = None
    staging_leaf = ""
    created_staging = False
    published = False
    transferred = False
    try:
        staged.extend(_stage_unnamed_output(parent_descriptor) for _ in paths)
        try:
            os.stat(
                directory_leaf,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise protocol_error(
                WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                "dedicated output directory already exists",
            )
        staging_leaf = _staging_directory_leaf()
        if staging_leaf == directory_leaf:
            raise protocol_error(
                WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                "staging directory collided with the published output name",
            )
        os.mkdir(staging_leaf, 0o700, dir_fd=parent_descriptor)
        created_staging = True
        directory_descriptor = os.open(
            staging_leaf,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        os.fchmod(directory_descriptor, 0o700)
        directory_metadata = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != os.geteuid()
            or directory_metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
        ):
            raise protocol_error(
                WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                "staging output directory is not private and controller-owned",
            )

        for path, descriptor in zip(paths, staged, strict=True):
            os.link(
                f"/proc/self/fd/{descriptor}",
                path.name,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=True,
            )
            linked_leaves.append(path.name)
            expected = os.fstat(descriptor)
            observed = os.stat(
                path.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(observed.st_mode)
                or (observed.st_dev, observed.st_ino)
                != (expected.st_dev, expected.st_ino)
                or expected.st_nlink != 1
            ):
                raise protocol_error(
                    WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                    "staged output differs from its unnamed inode",
                )
        if policy.before_final_open is not None:
            policy.before_final_open(output_directory, True)
        _rename_directory_noreplace(parent_descriptor, staging_leaf, directory_leaf)
        published = True
        os.fchmod(directory_descriptor, 0o2775 if group_policy else 0o755)
        _validate_directory_fd(directory_descriptor, expected_gid, group_policy)
        transferred = True
        return staged
    except FileExistsError:
        raise
    except WorkerProtocolError:
        raise
    except OSError as error:
        raise protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "cannot publish dedicated worker output directory",
        ) from error
    finally:
        rollback_error: WorkerProtocolError | None = None
        if not transferred:
            if created_staging and not published:
                try:
                    _rollback_staging_namespace(
                        directory_descriptor,
                        parent_descriptor,
                        staging_leaf,
                        linked_leaves,
                    )
                except OSError as error:
                    rollback_error = protocol_error(
                        WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                        f"staging output rollback failed: {error.errno}",
                    )
                except WorkerProtocolError as error:
                    rollback_error = error
            close_descriptors(staged)
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass
        try:
            os.close(parent_descriptor)
        except OSError:
            pass
        if rollback_error is not None:
            raise rollback_error

def close_descriptors(descriptors: list[int]) -> None:

    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except OSError:
            pass
