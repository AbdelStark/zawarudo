"""Trusted model-package format + safe loader for WMCP world-model checkpoints.

A *model package* is a self-describing directory the runtime can load deterministically and
safely. It is the only trusted source of weights — the service NEVER loads weights or model code
referenced by an inference request (PRD non-goal #6; ``model-packaging`` skill).

Package layout (see ``LEWM_INTEGRATION_GUIDE.md`` §2)::

    manifest.json                  # conforms to schemas/model-manifest.schema.json (+ a `weights` block)
    config.json                    # upstream model config (published HF config)
    weights.safetensors            # preferred; or weights.pt (pickled, gated) or weights.json (synthetic)
    preprocessing.json             # image preprocessing the model was trained with
    action_space.json              # action dim / layout / bounds
    normalizers/action_scaler.json # action normalizer parameters
    sources.lock.json              # pinned upstream commits + HF revision
    checksums.txt                  # sha256 of every other file (sha256sum format) — verified on load

The loader is intentionally split so the pure-data parts (checksums, manifest, shapes) need no heavy
dependencies, while the actual weight materialisation imports ``torch``/``safetensors`` lazily.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# --- package file-name constants -------------------------------------------------------------

MANIFEST_FILE = "manifest.json"
CONFIG_FILE = "config.json"
PREPROCESSING_FILE = "preprocessing.json"
ACTION_SPACE_FILE = "action_space.json"
ACTION_SCALER_FILE = "normalizers/action_scaler.json"
SOURCES_LOCK_FILE = "sources.lock.json"
CHECKSUMS_FILE = "checksums.txt"

# Top-level keys the manifest schema marks ``required`` (schemas/model-manifest.schema.json).
MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "model_id",
    "model_family",
    "model_type",
    "source_repository",
    "artifact_uri",
    "framework",
    "runtime",
    "supported_operations",
    "inputs",
    "outputs",
    "limits",
)

# Weight container formats we understand.
SAFETENSORS = "safetensors"
PT = "pt"
JSON_WEIGHTS = "json"  # portable, stdlib-only — for tests / synthetic packages, never real weights.


# --- errors ----------------------------------------------------------------------------------


class PackageError(Exception):
    """Base error for anything wrong with a model package."""


class ChecksumError(PackageError):
    """A file's checksum did not match (or a file is missing/extra). Loading must fail closed."""


class ManifestError(PackageError):
    """The manifest is missing required fields or is otherwise malformed."""


class ShapeMismatchError(PackageError):
    """Materialised weight shapes do not match the manifest's declared shapes."""


# --- checksums (pure stdlib) -----------------------------------------------------------------


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_package_files(package_dir: Path, *, exclude: set[str]) -> list[Path]:
    files = [
        p
        for p in sorted(package_dir.rglob("*"))
        if p.is_file() and p.relative_to(package_dir).as_posix() not in exclude
    ]
    return files


def compute_checksums(package_dir: str | Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    """Map each package file (posix relative path) to its sha256, excluding ``checksums.txt``."""
    package_dir = Path(package_dir)
    exclude = {CHECKSUMS_FILE} | (exclude or set())
    out: dict[str, str] = {}
    for path in _iter_package_files(package_dir, exclude=exclude):
        rel = path.relative_to(package_dir).as_posix()
        out[rel] = sha256_file(path)
    return out


def write_checksums(package_dir: str | Path, *, exclude: set[str] | None = None) -> Path:
    """Write ``checksums.txt`` in ``sha256sum`` format (``<hex>  <relpath>`` per line)."""
    package_dir = Path(package_dir)
    sums = compute_checksums(package_dir, exclude=exclude)
    lines = [f"{digest}  {rel}\n" for rel, digest in sorted(sums.items())]
    target = package_dir / CHECKSUMS_FILE
    target.write_text("".join(lines), encoding="utf-8")
    return target


def read_checksums(package_dir: str | Path) -> dict[str, str]:
    package_dir = Path(package_dir)
    target = package_dir / CHECKSUMS_FILE
    if not target.exists():
        raise ChecksumError(f"missing {CHECKSUMS_FILE} in {package_dir}")
    out: dict[str, str] = {}
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # sha256sum format: "<hex>  <path>" (two spaces) — but tolerate single-space too.
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ChecksumError(f"malformed checksum line: {raw!r}")
        digest, rel = parts
        out[rel.strip()] = digest.lower()
    return out


def verify_checksums(package_dir: str | Path, *, exclude: set[str] | None = None) -> None:
    """Recompute every file's sha256 and compare to ``checksums.txt``. Fail closed on any drift.

    Raises :class:`ChecksumError` if a file is missing, extra, or has a mismatched digest.
    """
    package_dir = Path(package_dir)
    declared = read_checksums(package_dir)
    actual = compute_checksums(package_dir, exclude=exclude)

    declared_keys = set(declared)
    actual_keys = set(actual)
    missing = declared_keys - actual_keys
    extra = actual_keys - declared_keys
    if missing:
        raise ChecksumError(f"files listed in {CHECKSUMS_FILE} but absent: {sorted(missing)}")
    if extra:
        raise ChecksumError(f"files present but not in {CHECKSUMS_FILE}: {sorted(extra)}")
    mismatched = [rel for rel in declared_keys if declared[rel] != actual[rel]]
    if mismatched:
        raise ChecksumError(f"checksum mismatch for: {sorted(mismatched)}")


# --- manifest (pure stdlib) ------------------------------------------------------------------


def load_json(package_dir: str | Path, name: str) -> Any:
    path = Path(package_dir) / name
    if not path.exists():
        raise PackageError(f"missing package file: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(package_dir: str | Path) -> dict[str, Any]:
    manifest = load_json(package_dir, MANIFEST_FILE)
    if not isinstance(manifest, dict):
        raise ManifestError("manifest.json must be a JSON object")
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a manifest against the required schema fields.

    Uses ``jsonschema`` if importable (full validation), otherwise a structural required-keys check.
    """
    missing = [k for k in MANIFEST_REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ManifestError(f"manifest missing required keys: {missing}")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or "backend" not in runtime:
        raise ManifestError("manifest.runtime must be an object containing 'backend'")
    if not isinstance(manifest.get("supported_operations"), Sequence):
        raise ManifestError("manifest.supported_operations must be an array")

    try:
        jsonschema = importlib.import_module("jsonschema")  # type: Any
    except ImportError:
        return  # structural check above is sufficient when jsonschema is unavailable.
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "model-manifest.schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=dict(manifest), schema=schema)


# --- weight shape inspection (no torch needed for safetensors/json) --------------------------


def _safetensors_header(path: str | Path) -> dict[str, Any]:
    """Parse a safetensors header (JSON prefix) without importing the safetensors library."""
    with open(path, "rb") as fh:
        size_bytes = fh.read(8)
        if len(size_bytes) != 8:
            raise PackageError(f"{path} is too small to be a safetensors file")
        (header_len,) = struct.unpack("<Q", size_bytes)
        header_json = fh.read(header_len)
    header = json.loads(header_json)
    header.pop("__metadata__", None)
    return header


def read_state_shapes(package_dir: str | Path, manifest: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return ``{tensor_name: shape}`` for the package weights, materialising as little as possible."""
    weights = manifest.get("weights")
    if not isinstance(weights, Mapping):
        raise ManifestError("manifest.weights must declare the weight file and format")
    fmt = weights.get("format")
    rel = weights.get("file")
    if not isinstance(rel, str):
        raise ManifestError("manifest.weights.file must be a string path")
    path = Path(package_dir) / rel
    if not path.exists():
        raise PackageError(f"weights file not found: {rel}")

    if fmt == SAFETENSORS:
        header = _safetensors_header(path)
        return {name: tuple(int(x) for x in meta["shape"]) for name, meta in header.items()}
    if fmt == JSON_WEIGHTS:
        blob = json.loads(path.read_text(encoding="utf-8"))
        tensors = blob.get("tensors", {})
        return {name: tuple(int(x) for x in meta["shape"]) for name, meta in tensors.items()}
    if fmt == PT:
        torch = importlib.import_module("torch")  # type: Any
        state = torch.load(path, map_location="cpu", weights_only=True)
        return {name: tuple(int(x) for x in tensor.shape) for name, tensor in state.items()}
    raise ManifestError(f"unknown weights format: {fmt!r}")


def verify_shapes(actual: Mapping[str, Sequence[int]], manifest: Mapping[str, Any]) -> None:
    """Compare materialised tensor shapes against ``manifest.weights.tensors``. Fail closed."""
    weights = manifest.get("weights")
    if not isinstance(weights, Mapping):
        raise ManifestError("manifest.weights missing")
    declared = weights.get("tensors")
    if not isinstance(declared, Mapping):
        raise ManifestError("manifest.weights.tensors must declare expected shapes")

    declared_shapes = {name: tuple(int(x) for x in meta["shape"]) for name, meta in declared.items()}
    actual_shapes = {name: tuple(int(x) for x in shape) for name, shape in actual.items()}

    missing = set(declared_shapes) - set(actual_shapes)
    extra = set(actual_shapes) - set(declared_shapes)
    if missing:
        raise ShapeMismatchError(f"weights missing tensors declared in manifest: {sorted(missing)}")
    if extra:
        raise ShapeMismatchError(f"weights have tensors not declared in manifest: {sorted(extra)}")
    bad = {n: (declared_shapes[n], actual_shapes[n]) for n in declared_shapes if declared_shapes[n] != actual_shapes[n]}
    if bad:
        raise ShapeMismatchError(f"shape mismatch (declared vs actual): {bad}")


# --- action scaler -----------------------------------------------------------------------------


@dataclass
class ActionScaler:
    """Normalize/denormalize action tensors.

    ``transform`` maps raw actions -> model space; ``inverse_transform`` reverses it. Implemented for
    array-like inputs (numpy / torch tensors broadcast naturally). ``bounds`` (low/high) are carried
    for validation and round-tripping but not enforced by ``transform``.

    Supported ``kind`` values: ``identity`` (no-op), ``standardize`` ((x-mean)/std),
    ``minmax`` (x -> (x-min)/(max-min)).
    """

    kind: str = "identity"
    mean: list[float] | None = None
    std: list[float] | None = None
    minimum: list[float] | None = None
    maximum: list[float] | None = None
    bounds: dict[str, list[float]] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionScaler":
        return cls(
            kind=str(data.get("kind", data.get("type", "identity"))),
            mean=data.get("mean"),
            std=data.get("std"),
            minimum=data.get("min"),
            maximum=data.get("max"),
            bounds=data.get("bounds"),
        )

    @classmethod
    def load(cls, package_dir: str | Path) -> "ActionScaler":
        return cls.from_dict(load_json(package_dir, ACTION_SCALER_FILE))

    def transform(self, x: Any) -> Any:
        if self.kind == "identity":
            return x
        if self.kind == "standardize":
            mean, std = self._as_arraylike(x, self.mean), self._as_arraylike(x, self.std)
            return (x - mean) / std
        if self.kind == "minmax":
            lo, hi = self._as_arraylike(x, self.minimum), self._as_arraylike(x, self.maximum)
            return (x - lo) / (hi - lo)
        raise PackageError(f"unknown action scaler kind: {self.kind!r}")

    def inverse_transform(self, x: Any) -> Any:
        if self.kind == "identity":
            return x
        if self.kind == "standardize":
            mean, std = self._as_arraylike(x, self.mean), self._as_arraylike(x, self.std)
            return x * std + mean
        if self.kind == "minmax":
            lo, hi = self._as_arraylike(x, self.minimum), self._as_arraylike(x, self.maximum)
            return x * (hi - lo) + lo
        raise PackageError(f"unknown action scaler kind: {self.kind!r}")

    @staticmethod
    def _as_arraylike(reference: Any, values: list[float] | None) -> Any:
        """Coerce ``values`` to match ``reference``'s array type (torch/numpy) when possible."""
        if values is None:
            raise PackageError("action scaler parameter missing for kind requiring it")
        # python scalar: broadcast a single-element parameter list to a scalar
        if isinstance(reference, (int, float)):
            if len(values) == 1:
                return values[0]
            raise PackageError("cannot broadcast multi-element scaler parameter onto a scalar action")
        # torch tensor
        if hasattr(reference, "new_tensor"):
            return reference.new_tensor(values)
        # numpy array
        if type(reference).__module__ == "numpy":
            numpy = importlib.import_module("numpy")  # type: Any
            return numpy.asarray(values, dtype=reference.dtype)
        return values


# --- model freezing (lazy torch) --------------------------------------------------------------


def freeze_module(model: Any) -> Any:
    """Put a torch module in eval mode and disable all gradients. Returns the same module."""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def assert_frozen(model: Any) -> None:
    """Raise if the module is training or any parameter still requires grad."""
    if getattr(model, "training", False):
        raise PackageError("model is in training mode; expected eval()")
    leaking = [name for name, p in model.named_parameters() if p.requires_grad]
    if leaking:
        raise PackageError(f"parameters still require grad: {leaking[:8]}")


# --- top-level loader -------------------------------------------------------------------------


@dataclass
class LoadedPackage:
    """Everything the runtime needs after a verified load."""

    package_dir: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    preprocessing: dict[str, Any]
    action_space: dict[str, Any]
    action_scaler: ActionScaler
    device: str = "cpu"
    weights_format: str | None = None
    model: Any = None
    state_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return str(self.manifest["model_id"])

    @property
    def revision(self) -> str:
        return str(self.manifest.get("artifact_revision", self.manifest.get("source_revision", "unknown")))


def load_package(
    package_dir: str | Path,
    *,
    device: str = "cpu",
    build_model: Any | None = None,
    verify: bool = True,
    allow_pickle: bool = False,
) -> LoadedPackage:
    """Load and verify a model package, returning a :class:`LoadedPackage`.

    Steps: verify checksums (fail closed) -> load+validate manifest -> read weight shapes and verify
    them against the manifest. If ``build_model`` is supplied it is called with the model ``config``,
    its ``state_dict`` is loaded from the package, and the module is moved to ``device``, set to eval,
    frozen, and asserted grad-free.

    ``allow_pickle`` must be ``True`` to load a ``.pt`` (pickled) weights file; the safe default refuses
    pickled weights in favour of safetensors (RFC-0005 open issue #2).
    """
    package_dir = Path(package_dir)
    if verify:
        verify_checksums(package_dir)

    manifest = load_manifest(package_dir)
    validate_manifest(manifest)
    config = load_json(package_dir, CONFIG_FILE) if (package_dir / CONFIG_FILE).exists() else {}
    preprocessing = (
        load_json(package_dir, PREPROCESSING_FILE) if (package_dir / PREPROCESSING_FILE).exists() else {}
    )
    action_space = (
        load_json(package_dir, ACTION_SPACE_FILE) if (package_dir / ACTION_SPACE_FILE).exists() else {}
    )
    action_scaler = (
        ActionScaler.load(package_dir)
        if (package_dir / ACTION_SCALER_FILE).exists()
        else ActionScaler()
    )

    weights = manifest.get("weights", {})
    weights_format = weights.get("format") if isinstance(weights, Mapping) else None

    state_shapes = read_state_shapes(package_dir, manifest)
    verify_shapes(state_shapes, manifest)

    model: Any = None
    if build_model is not None:
        if weights_format == PT and not allow_pickle:
            raise PackageError(
                "refusing to load pickled .pt weights without allow_pickle=True; "
                "convert to safetensors (RFC-0005 open issue #2)"
            )
        state = _load_state_dict(package_dir, manifest, allow_pickle=allow_pickle)
        model = build_model(config)
        model.load_state_dict(state)
        model = freeze_module(model.to(device))
        assert_frozen(model)

    return LoadedPackage(
        package_dir=package_dir,
        manifest=manifest,
        config=config,
        preprocessing=preprocessing,
        action_space=action_space,
        action_scaler=action_scaler,
        device=device,
        weights_format=weights_format,
        model=model,
        state_shapes=state_shapes,
    )


def _load_state_dict(package_dir: str | Path, manifest: Mapping[str, Any], *, allow_pickle: bool) -> Any:
    """Materialise the package weights into a torch ``state_dict`` (lazy torch import)."""
    weights = manifest["weights"]
    fmt = weights["format"]
    path = Path(package_dir) / weights["file"]
    torch = importlib.import_module("torch")  # type: Any

    if fmt == SAFETENSORS:
        safetensors_torch = importlib.import_module("safetensors.torch")  # type: Any
        return safetensors_torch.load_file(str(path))
    if fmt == PT:
        if not allow_pickle:
            raise PackageError("pickled .pt load requires allow_pickle=True")
        return torch.load(path, map_location="cpu", weights_only=True)
    if fmt == JSON_WEIGHTS:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return {name: torch.tensor(meta["data"]) for name, meta in blob.get("tensors", {}).items()}
    raise ManifestError(f"unknown weights format: {fmt!r}")
