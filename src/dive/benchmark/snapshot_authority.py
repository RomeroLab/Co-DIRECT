
from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import os
import resource
import stat
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dive.benchmark.contracts import ArtifactIdentity

PRODUCTION_SNAPSHOT_COUNT = 38_400

SNAPSHOT_AUTHORITY_FD_BUDGET = 6
SNAPSHOT_FD_SAFE_MARGIN = 64

IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_UNMOUNT = 0x00002000
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
IN_ONLYDIR = 0x01000000
IN_DONT_FOLLOW = 0x02000000
IN_EXCL_UNLINK = 0x04000000

_REJECTED_EVENT_MASK = (
    IN_MODIFY
    | IN_ATTRIB
    | IN_CLOSE_WRITE
    | IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_CREATE
    | IN_DELETE
    | IN_DELETE_SELF
    | IN_MOVE_SELF
    | IN_UNMOUNT
    | IN_Q_OVERFLOW
    | IN_IGNORED
)
_WATCH_MASK = _REJECTED_EVENT_MASK | IN_ONLYDIR | IN_DONT_FOLLOW | IN_EXCL_UNLINK
_EVENT_HEADER = struct.Struct("iIII")

class SnapshotAuthorityError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class _DirectoryState:
    role: str
    path: Path
    descriptor: int
    watch_descriptor: int
    initial_stat: os.stat_result

@dataclass(frozen=True, slots=True)
class _ManifestState:
    role: str
    path: Path
    descriptor: int
    initial_stat: os.stat_result
    raw: bytes
    identity: ArtifactIdentity

@dataclass(frozen=True, slots=True)
class _SnapshotState:
    role: str
    name: str
    path: Path
    initial_stat: os.stat_result
    expected: ArtifactIdentity

@dataclass(frozen=True, slots=True)
class _InotifyEvent:
    watch_descriptor: int
    mask: int
    cookie: int
    name: str

class MonitoredSnapshotAuthority:

    def __init__(
        self,
        monitor_descriptor: int,
        directories: Mapping[str, _DirectoryState],
        manifests: Mapping[str, _ManifestState],
        expected_count: int | None,
    ) -> None:
        self._monitor_descriptor = monitor_descriptor
        self._directories = dict(directories)
        self._manifests = dict(manifests)
        self._expected_count = expected_count
        self._snapshots: dict[Path, _SnapshotState] | None = None
        self._closed = False

    @classmethod
    def begin(
        cls,
        *,
        directories: Mapping[str, Path],
        manifests: Mapping[str, Path],
        expected_count: int | None,
    ) -> MonitoredSnapshotAuthority:

        if (
            set(directories) != {"query", "target"}
            or set(manifests) != {"query", "target"}
            or (
                expected_count is not None
                and (
                    type(expected_count) is not int
                    or expected_count <= 0
                    or expected_count > PRODUCTION_SNAPSHOT_COUNT
                )
            )
        ):
            raise SnapshotAuthorityError(
                "snapshot monitored-tree universe is not exact or bounded"
            )
        preflight_monitored_tree_authority()
        monitor_descriptor = -1
        retained_directories: dict[str, _DirectoryState] = {}
        retained_manifests: dict[str, _ManifestState] = {}
        try:
            monitor_descriptor = _inotify_init_fd()
            seen_directories: set[tuple[int, int]] = set()
            for role in ("query", "target"):
                path = Path(directories[role])
                descriptor = _open_directory(path)
                try:
                    initial = os.fstat(descriptor)
                    pathname = os.stat(path, follow_symlinks=False)
                    inode = (initial.st_dev, initial.st_ino)
                    if (
                        _stat_key(initial) != _stat_key(pathname)
                        or not _secure_directory(initial)
                        or inode in seen_directories
                    ):
                        raise SnapshotAuthorityError(
                            f"snapshot {role} directory is not secure and stable: {path}"
                        )
                    try:
                        watch_descriptor = _inotify_add_watch(
                            monitor_descriptor, path, _WATCH_MASK
                        )
                    except OSError as error:
                        raise SnapshotAuthorityError(
                            f"Linux inotify watch is unavailable for {path}: {error}"
                        ) from error
                    after_watch = os.fstat(descriptor)
                    watched_path = os.stat(path, follow_symlinks=False)
                    if not (
                        _stat_key(initial)
                        == _stat_key(after_watch)
                        == _stat_key(watched_path)
                    ):
                        raise SnapshotAuthorityError(
                            f"snapshot {role} directory changed while watched"
                        )
                except BaseException:
                    _close_descriptor(descriptor)
                    raise
                seen_directories.add(inode)
                retained_directories[role] = _DirectoryState(
                    role, path, descriptor, watch_descriptor, initial
                )

            for role in ("query", "target"):
                path = Path(manifests[role])
                descriptor = _open_regular(path, "snapshot manifest")
                try:
                    initial, raw, identity = _stable_read_descriptor(
                        descriptor, path, require_read_only=False
                    )
                except BaseException:
                    _close_descriptor(descriptor)
                    raise
                retained_manifests[role] = _ManifestState(
                    role, path, descriptor, initial, raw, identity
                )
            authority = cls(
                monitor_descriptor,
                retained_directories,
                retained_manifests,
                expected_count,
            )
            authority._raise_on_events("monitored-tree initialization")
            return authority
        except BaseException:
            for item in retained_manifests.values():
                _close_descriptor(item.descriptor)
            for item in retained_directories.values():
                _close_descriptor(item.descriptor)
            _close_descriptor(monitor_descriptor)
            raise

    @property
    def count(self) -> int:
        return 0 if self._snapshots is None else len(self._snapshots)

    def manifest_bytes(
        self, role: str, expected: ArtifactIdentity | None = None
    ) -> bytes:
        self._require_open()
        try:
            manifest = self._manifests[role]
        except KeyError as error:
            raise SnapshotAuthorityError(
                f"unknown retained snapshot manifest role: {role}"
            ) from error
        if expected is not None and manifest.identity != expected:
            raise SnapshotAuthorityError(
                f"retained {role} snapshot manifest identity changed"
            )
        return manifest.raw

    def authenticate(self, declarations: Mapping[Path, ArtifactIdentity]) -> None:

        self._require_open()
        if self._snapshots is not None:
            raise SnapshotAuthorityError("snapshot tree was already authenticated")
        if (
            not declarations
            or len(declarations) > PRODUCTION_SNAPSHOT_COUNT
            or (
                self._expected_count is not None
                and len(declarations) != self._expected_count
            )
            or len(set(declarations)) != len(declarations)
        ):
            raise SnapshotAuthorityError(
                "snapshot monitored-tree declaration universe is not exact"
            )
        snapshots: dict[Path, _SnapshotState] = {}
        expected_names: dict[str, set[str]] = {"query": set(), "target": set()}
        seen_inodes: set[tuple[int, int]] = set()
        try:
            for path in sorted(declarations, key=str):
                expected = declarations[path]
                role = self._role_for_path(path)
                name = path.name
                if (
                    path.parent != self._directories[role].path
                    or name in expected_names[role]
                    or not name
                    or name in {".", ".."}
                ):
                    raise SnapshotAuthorityError(
                        f"snapshot filename declaration is invalid: {path}"
                    )
                expected_names[role].add(name)
                initial, observed = _stable_read_child(
                    self._directories[role], name, path
                )
                inode = (initial.st_dev, initial.st_ino)
                if inode in seen_inodes or observed != expected:
                    raise SnapshotAuthorityError(
                        f"snapshot identity or unique inode changed: {path}"
                    )
                seen_inodes.add(inode)
                snapshots[path] = _SnapshotState(role, name, path, initial, expected)
            self._require_exact_filenames(expected_names)
            self._validate_directory_paths()
            self._raise_on_events("initial snapshot traversal")
            self._snapshots = snapshots
        except BaseException:
            self.close()
            raise

    def finish(self) -> None:

        self._require_open()
        if self._snapshots is None:
            self.close()
            raise SnapshotAuthorityError("snapshot tree was not authenticated")
        errors: list[str] = []
        expected_names: dict[str, set[str]] = {"query": set(), "target": set()}
        try:
            for path, snapshot in sorted(
                self._snapshots.items(), key=lambda item: str(item[0])
            ):
                expected_names[snapshot.role].add(snapshot.name)
                try:
                    final, observed = _stable_read_child(
                        self._directories[snapshot.role], snapshot.name, path
                    )
                    if (
                        _stat_key(final) != _stat_key(snapshot.initial_stat)
                        or observed != snapshot.expected
                    ):
                        errors.append(f"snapshot changed: {path}")
                except Exception as error:
                    errors.append(f"snapshot changed: {path}: {error}")
            try:
                self._require_exact_filenames(expected_names)
                self._validate_directory_paths()
            except Exception as error:
                errors.append(str(error))
            for manifest in self._manifests.values():
                try:
                    final, raw, identity = _stable_read_descriptor(
                        manifest.descriptor,
                        manifest.path,
                        require_read_only=False,
                    )
                    if (
                        _stat_key(final) != _stat_key(manifest.initial_stat)
                        or raw != manifest.raw
                        or identity != manifest.identity
                    ):
                        errors.append(
                            f"retained {manifest.role} snapshot manifest changed"
                        )
                except Exception as error:
                    errors.append(
                        f"retained {manifest.role} snapshot manifest changed: {error}"
                    )
            try:

                self._raise_on_events("final snapshot traversal")
            except SnapshotAuthorityError as error:
                errors.append(str(error))
        finally:
            self.close()
        if errors:
            raise SnapshotAuthorityError("; ".join(errors[:4]))

    def close(self) -> None:
        if self._closed:
            return
        for item in self._manifests.values():
            _close_descriptor(item.descriptor)
        for item in self._directories.values():
            _close_descriptor(item.descriptor)
        _close_descriptor(self._monitor_descriptor)
        self._closed = True

    def _role_for_path(self, path: Path) -> str:
        matches = [
            role
            for role, directory in self._directories.items()
            if path.parent == directory.path
        ]
        if len(matches) != 1:
            raise SnapshotAuthorityError(
                f"snapshot path is outside the two monitored flat trees: {path}"
            )
        return matches[0]

    def _require_exact_filenames(self, expected: Mapping[str, set[str]]) -> None:
        for role, directory in self._directories.items():
            try:
                observed = os.listdir(directory.descriptor)
            except OSError as error:
                raise SnapshotAuthorityError(
                    f"cannot list monitored {role} snapshot directory"
                ) from error
            if len(observed) != len(set(observed)) or set(observed) != expected[role]:
                raise SnapshotAuthorityError(
                    f"monitored {role} snapshot filename universe changed"
                )

    def _validate_directory_paths(self) -> None:
        for role, directory in self._directories.items():
            try:
                descriptor_stat = os.fstat(directory.descriptor)
                pathname_stat = os.stat(directory.path, follow_symlinks=False)
            except OSError as error:
                raise SnapshotAuthorityError(
                    f"monitored {role} snapshot directory changed"
                ) from error
            if not (
                _stat_key(directory.initial_stat)
                == _stat_key(descriptor_stat)
                == _stat_key(pathname_stat)
                and _secure_directory(pathname_stat)
            ):
                raise SnapshotAuthorityError(
                    f"monitored {role} snapshot directory changed"
                )

    def _raise_on_events(self, stage: str) -> None:
        events = _decode_inotify_events(_read_inotify_bytes(self._monitor_descriptor))
        rejected = [event for event in events if event.mask & _REJECTED_EVENT_MASK]
        if not rejected:
            return
        if any(event.mask & IN_Q_OVERFLOW for event in rejected):
            raise SnapshotAuthorityError(
                f"inotify queue overflow during {stage}; snapshot authority lost"
            )
        preview = ", ".join(
            f"wd={event.watch_descriptor} mask=0x{event.mask:x} name={event.name!r}"
            for event in rejected[:4]
        )
        raise SnapshotAuthorityError(
            f"inotify rejected snapshot event during {stage}: {preview}"
        )

    def _require_open(self) -> None:
        if self._closed:
            raise SnapshotAuthorityError("snapshot monitored-tree authority is closed")

def preflight_monitored_tree_authority() -> None:

    if not sys.platform.startswith("linux"):
        raise SnapshotAuthorityError(
            "Linux inotify is unavailable for snapshot authentication"
        )
    try:
        current_open = len(os.listdir("/proc/self/fd"))
    except OSError as error:
        raise SnapshotAuthorityError(
            "cannot count descriptors for snapshot monitor preflight"
        ) from error
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required_limit = (
        current_open + SNAPSHOT_AUTHORITY_FD_BUDGET + SNAPSHOT_FD_SAFE_MARGIN
    )
    if soft != resource.RLIM_INFINITY and soft < required_limit:
        raise SnapshotAuthorityError(
            "snapshot monitor RLIMIT_NOFILE is insufficient: "
            f"soft={soft}, current={current_open}, "
            f"authority={SNAPSHOT_AUTHORITY_FD_BUDGET}, "
            f"margin={SNAPSHOT_FD_SAFE_MARGIN}, required={required_limit}"
        )
    try:
        descriptor = _inotify_init_fd()
    except OSError as error:
        raise SnapshotAuthorityError(
            f"Linux inotify is unavailable for snapshot authentication: {error}"
        ) from error
    _close_descriptor(descriptor)

def _inotify_init_fd() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "inotify_init1", None)
    if function is None:
        raise OSError(errno.ENOSYS, "inotify_init1 is unavailable")
    function.argtypes = [ctypes.c_int]
    function.restype = ctypes.c_int
    descriptor = function(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
    status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    if not descriptor_flags & fcntl.FD_CLOEXEC or not status_flags & os.O_NONBLOCK:
        _close_descriptor(descriptor)
        raise OSError(errno.EINVAL, "inotify descriptor flags are not secure")
    return descriptor

def _inotify_add_watch(descriptor: int, path: Path, mask: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "inotify_add_watch", None)
    if function is None:
        raise OSError(errno.ENOSYS, "inotify_add_watch is unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    function.restype = ctypes.c_int
    watch_descriptor = function(descriptor, os.fsencode(path), mask)
    if watch_descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return watch_descriptor

def _read_inotify_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        try:
            chunk = os.read(descriptor, 1 << 20)
        except BlockingIOError:
            break
        except InterruptedError:
            continue
        except OSError as error:
            raise SnapshotAuthorityError("cannot read inotify event queue") from error
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)

def _decode_inotify_events(raw: bytes) -> tuple[_InotifyEvent, ...]:
    events: list[_InotifyEvent] = []
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < _EVENT_HEADER.size:
            raise SnapshotAuthorityError("malformed inotify event queue")
        watch_descriptor, mask, cookie, name_length = _EVENT_HEADER.unpack_from(
            raw, offset
        )
        offset += _EVENT_HEADER.size
        if name_length > len(raw) - offset:
            raise SnapshotAuthorityError("malformed inotify event name")
        encoded_name = raw[offset : offset + name_length]
        offset += name_length
        try:
            name = os.fsdecode(encoded_name.rstrip(b"\0"))
        except UnicodeError as error:
            raise SnapshotAuthorityError("malformed inotify event filename") from error
        events.append(_InotifyEvent(watch_descriptor, mask, cookie, name))
    return tuple(events)

def _open_directory(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        return os.open(path, flags)
    except OSError as error:
        raise SnapshotAuthorityError(
            f"cannot open monitored snapshot directory: {path}"
        ) from error

def _open_regular(path: Path, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError as error:
        raise SnapshotAuthorityError(f"cannot open retained {label}: {path}") from error

def _stable_read_child(
    directory: _DirectoryState, name: str, path: Path
) -> tuple[os.stat_result, ArtifactIdentity]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory.descriptor)
    except OSError as error:
        raise SnapshotAuthorityError(
            f"cannot open monitored snapshot: {path}"
        ) from error
    try:
        initial, _raw, observed = _stable_read_descriptor(
            descriptor,
            path,
            dir_fd=directory.descriptor,
            name=name,
            require_read_only=True,
        )
        return initial, observed
    finally:
        _close_descriptor(descriptor)

def _stable_read_descriptor(
    descriptor: int,
    path: Path,
    *,
    dir_fd: int | None = None,
    name: str | None = None,
    require_read_only: bool,
) -> tuple[os.stat_result, bytes, ArtifactIdentity]:
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & (0o222 if require_read_only else 0o022)
        ):
            raise SnapshotAuthorityError(
                f"retained snapshot file is not secure and unique: {path}"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if dir_fd is None:
            pathname = os.stat(path, follow_symlinks=False)
        else:
            if name is None:
                raise SnapshotAuthorityError("snapshot child name is unavailable")
            pathname = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            absolute_pathname = os.stat(path, follow_symlinks=False)
            if _stat_key(pathname) != _stat_key(absolute_pathname):
                raise SnapshotAuthorityError(
                    f"snapshot pathname escaped its retained directory: {path}"
                )
    except OSError as error:
        raise SnapshotAuthorityError(f"retained snapshot changed: {path}") from error
    if not (_stat_key(before) == _stat_key(after) == _stat_key(pathname)):
        raise SnapshotAuthorityError(f"retained snapshot changed while read: {path}")
    raw = b"".join(chunks)
    identity = ArtifactIdentity(
        str(path.resolve()), hashlib.sha256(raw).hexdigest(), len(raw)
    )
    return after, raw, identity

def _secure_directory(value: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(value.st_mode)
        and value.st_nlink >= 2
        and stat.S_IMODE(value.st_mode) & 0o022 == 0
    )

def _stat_key(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )

def _close_descriptor(descriptor: int) -> None:
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass
