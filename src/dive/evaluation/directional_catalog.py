
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Mapping
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys

CATALOG_SCHEMA_VERSION = 1
PROJECT_PREFIXES = ("dive", "proteinfoundation")
_ROOT_PREFIXES = MappingProxyType(
    {"dive": "src/dive", "upstream": "src/proteinfoundation"}
)
_PINNED_UPSTREAM_COMMIT = "32b71ae1d9a8767414eae3921cf35969430db82f"
_HEADER_KEYS = frozenset(
    {"schema_version", "dive_commit", "upstream_commit", "identity_sha256", "modules"}
)
_MODULE_RECORD_KEYS = frozenset(
    {"name", "root_class", "relative_path", "size_bytes", "sha256"}
)

class CatalogError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class CatalogModule:
    root_class: str
    relative_path: str
    size_bytes: int
    sha256: str
    source_bytes: bytes

@dataclass(frozen=True, slots=True)
class SourceCatalog:
    schema_version: int
    dive_commit: str
    upstream_commit: str
    modules: Mapping[str, CatalogModule]
    identity_sha256: str

def _logical_name(root_class: str, relative_path: str) -> str:
    prefix = _ROOT_PREFIXES[root_class]
    if not relative_path.startswith(prefix + "/") or not relative_path.endswith(".py"):
        raise CatalogError(f"catalog member outside its root prefix: {relative_path}")
    parts = relative_path[len("src/") :].removesuffix(".py").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CatalogError(f"catalog member path is unsafe: {relative_path}")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        raise CatalogError(f"catalog member has no logical name: {relative_path}")
    return ".".join(parts)

def _exact_commit(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or set(value) - set("0123456789abcdef")
    ):
        raise CatalogError(f"{name} is not an exact 40-hex commit")
    return value

def build_source_catalog(
    *,
    dive_sources: Mapping[str, bytes],
    upstream_sources: Mapping[str, bytes],
    dive_commit: str,
    upstream_commit: str,
) -> SourceCatalog:

    dive_commit = _exact_commit("dive_commit", dive_commit)
    upstream_commit = _exact_commit("upstream_commit", upstream_commit)
    if upstream_commit != _PINNED_UPSTREAM_COMMIT:
        raise CatalogError("upstream commit differs from the emergent subsystem pin")
    modules: dict[str, CatalogModule] = {}
    seen_paths: set[str] = set()
    for root_class, sources in (("dive", dive_sources), ("upstream", upstream_sources)):
        for relative_path, raw in sources.items():
            if type(relative_path) is not str or type(raw) is not bytes:
                raise CatalogError("catalog member is not an exact path/bytes pair")
            if relative_path.startswith("/") or "\\" in relative_path:
                raise CatalogError(f"catalog member path is unsafe: {relative_path}")
            if relative_path in seen_paths:
                raise CatalogError(f"duplicate catalog path: {relative_path}")
            seen_paths.add(relative_path)
            name = _logical_name(root_class, relative_path)
            if not raw and not relative_path.endswith("/__init__.py"):
                raise CatalogError(f"ordinary catalog module is empty: {relative_path}")
            if name in modules:
                raise CatalogError(f"duplicate catalog logical name: {name}")
            modules[name] = CatalogModule(
                root_class=root_class,
                relative_path=relative_path,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                source_bytes=raw,
            )
    if not modules:
        raise CatalogError("catalog is empty")
    for name in modules:
        parent, separator, _ = name.rpartition(".")
        if separator and parent not in modules:
            raise CatalogError(f"catalog package {parent!r} is missing its initializer")
    frozen = MappingProxyType(dict(sorted(modules.items())))
    return SourceCatalog(
        schema_version=CATALOG_SCHEMA_VERSION,
        dive_commit=dive_commit,
        upstream_commit=upstream_commit,
        modules=frozen,
        identity_sha256=_catalog_identity(
            CATALOG_SCHEMA_VERSION, dive_commit, upstream_commit, frozen
        ),
    )

def _catalog_identity(
    schema_version: int,
    dive_commit: str,
    upstream_commit: str,
    modules: Mapping[str, CatalogModule],
) -> str:
    digest = hashlib.sha256()
    header = json.dumps(
        {
            "schema_version": schema_version,
            "dive_commit": dive_commit,
            "upstream_commit": upstream_commit,
            "module_count": len(modules),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big") + header)
    for name in sorted(modules):
        entry = modules[name]
        row = json.dumps(
            {
                "name": name,
                "root_class": entry.root_class,
                "relative_path": entry.relative_path,
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(len(row).to_bytes(8, "big") + row)
        digest.update(len(entry.source_bytes).to_bytes(8, "big") + entry.source_bytes)
    return digest.hexdigest()

def serialize_catalog(catalog: SourceCatalog) -> bytes:

    header = json.dumps(
        {
            "schema_version": catalog.schema_version,
            "dive_commit": catalog.dive_commit,
            "upstream_commit": catalog.upstream_commit,
            "identity_sha256": catalog.identity_sha256,
            "modules": [
                {
                    "name": name,
                    "root_class": entry.root_class,
                    "relative_path": entry.relative_path,
                    "size_bytes": entry.size_bytes,
                    "sha256": entry.sha256,
                }
                for name, entry in sorted(catalog.modules.items())
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    body = b"".join(
        catalog.modules[name].source_bytes for name in sorted(catalog.modules)
    )
    return len(header).to_bytes(8, "big") + header + body

def deserialize_catalog(raw: bytes) -> SourceCatalog:

    if type(raw) is not bytes or len(raw) < 8:
        raise CatalogError("catalog payload is not exact bytes")
    header_length = int.from_bytes(raw[:8], "big")
    header_end = 8 + header_length
    if header_end > len(raw):
        raise CatalogError("catalog payload header is truncated")
    try:
        header = json.loads(raw[8:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CatalogError("catalog payload header is invalid") from error
    if type(header) is not dict or set(header) != _HEADER_KEYS:
        raise CatalogError("catalog payload header keys are not exact")
    if type(header["schema_version"]) is not int:
        raise CatalogError("catalog payload schema_version is not an exact int")
    if type(header["dive_commit"]) is not str:
        raise CatalogError("catalog payload dive_commit is not an exact string")
    if type(header["upstream_commit"]) is not str:
        raise CatalogError("catalog payload upstream_commit is not an exact string")
    if type(header["identity_sha256"]) is not str:
        raise CatalogError("catalog payload identity_sha256 is not an exact string")
    if type(header["modules"]) is not list:
        raise CatalogError("catalog payload modules is not an exact list")
    offset = header_end
    dive_sources: dict[str, bytes] = {}
    upstream_sources: dict[str, bytes] = {}
    for record in header["modules"]:
        if type(record) is not dict or set(record) != _MODULE_RECORD_KEYS:
            raise CatalogError("catalog payload module record keys are not exact")
        if type(record["name"]) is not str:
            raise CatalogError("catalog payload module record name is not a string")
        if record["root_class"] not in ("dive", "upstream"):
            raise CatalogError("catalog payload module record root_class is invalid")
        if type(record["relative_path"]) is not str:
            raise CatalogError(
                "catalog payload module record relative_path is not a string"
            )
        if type(record["sha256"]) is not str:
            raise CatalogError("catalog payload module record sha256 is not a string")
        size = record["size_bytes"]
        if type(size) is not int or size < 0 or offset + size > len(raw):
            raise CatalogError("catalog payload body is truncated")
        body = raw[offset : offset + size]
        offset += size
        if hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise CatalogError("catalog payload body digest differs")
        target = dive_sources if record["root_class"] == "dive" else upstream_sources
        target[record["relative_path"]] = body
    if offset != len(raw):
        raise CatalogError("catalog payload has trailing bytes")
    catalog = build_source_catalog(
        dive_sources=dive_sources,
        upstream_sources=upstream_sources,
        dive_commit=header["dive_commit"],
        upstream_commit=header["upstream_commit"],
    )
    if catalog.identity_sha256 != header["identity_sha256"]:
        raise CatalogError("catalog payload identity differs from its content")
    return catalog

class CatalogFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):

    def __init__(self, catalog: SourceCatalog) -> None:
        self._catalog = catalog
        self._loaded: dict[str, ModuleType] = {}
        self._installed = False

    @property
    def catalog(self) -> SourceCatalog:
        return self._catalog

    @property
    def loaded_modules(self) -> Mapping[str, ModuleType]:
        return MappingProxyType(dict(self._loaded))

    def install(self) -> None:
        if self._installed:
            raise CatalogError("catalog realm is already installed")
        assert_no_preloaded_project_modules()
        sys.meta_path.insert(0, self)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        sys.meta_path.remove(self)
        for name in list(self._loaded):
            sys.modules.pop(name, None)
        self._loaded.clear()
        self._installed = False

    def find_spec(self, fullname, path=None, target=None):
        if not _is_project_module(fullname):
            return None
        entry = self._catalog.modules.get(fullname)
        if entry is None:
            raise CatalogError(
                f"project module {fullname!r} is absent from the authenticated catalog"
            )
        is_package = entry.relative_path.endswith("/__init__.py")
        spec = importlib.machinery.ModuleSpec(
            fullname,
            self,
            origin=f"<catalog:{entry.relative_path}>",
            is_package=is_package,
        )
        if is_package:
            spec.submodule_search_locations = []
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        entry = self._catalog.modules[module.__name__]
        if hashlib.sha256(entry.source_bytes).hexdigest() != entry.sha256:
            raise CatalogError(f"catalog bytes for {module.__name__!r} changed")
        module.__file__ = f"<catalog:{entry.relative_path}>"
        code = compile(entry.source_bytes, module.__file__, "exec", dont_inherit=True)
        exec(code, module.__dict__)
        self._loaded[module.__name__] = module

def _is_project_module(fullname: str) -> bool:
    root, _, _ = fullname.partition(".")
    return root in PROJECT_PREFIXES

def assert_no_preloaded_project_modules() -> None:

    stale = sorted(name for name in sys.modules if _is_project_module(name))
    if stale:
        raise CatalogError(f"worker inherited preloaded project modules: {stale}")

def install_catalog_realm(catalog: SourceCatalog) -> CatalogFinder:
    finder = CatalogFinder(catalog)
    finder.install()
    return finder
