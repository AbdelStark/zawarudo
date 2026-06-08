"""Tests for the trusted model-package format + safe loader (issue #2)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from wmcp_jepa_service import model_package as mp
from wmcp_jepa_service import packaging


# --- checksums ---------------------------------------------------------------------------------


def test_checksums_roundtrip_and_verify(tmp_path: Path) -> None:
    pkg = packaging.build_synthetic_package(tmp_path / "pkg")
    # build_synthetic_package writes checksums.txt; a fresh verify must pass.
    mp.verify_checksums(pkg)
    sums = mp.read_checksums(pkg)
    assert mp.MANIFEST_FILE in sums
    assert mp.CHECKSUMS_FILE not in sums  # checksums.txt never checksums itself


def test_verify_checksums_fail_closed_on_tampered_file(tmp_path: Path) -> None:
    pkg = packaging.build_synthetic_package(tmp_path / "pkg")
    manifest_path = pkg / mp.MANIFEST_FILE
    data = json.loads(manifest_path.read_text())
    data["model_id"] = "tampered"
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(mp.ChecksumError, match="checksum mismatch"):
        mp.verify_checksums(pkg)


def test_verify_checksums_fail_closed_on_missing_file(tmp_path: Path) -> None:
    pkg = packaging.build_synthetic_package(tmp_path / "pkg")
    (pkg / mp.CONFIG_FILE).unlink()
    with pytest.raises(mp.ChecksumError, match="absent"):
        mp.verify_checksums(pkg)


def test_verify_checksums_fail_closed_on_extra_file(tmp_path: Path) -> None:
    pkg = packaging.build_synthetic_package(tmp_path / "pkg")
    (pkg / "stowaway.bin").write_bytes(b"\x00\x01")
    with pytest.raises(mp.ChecksumError, match="not in"):
        mp.verify_checksums(pkg)


# --- manifest ----------------------------------------------------------------------------------


def test_generated_manifest_conforms_to_schema(tmp_path: Path) -> None:
    pkg = packaging.build_synthetic_package(tmp_path / "pkg")
    manifest = mp.load_manifest(pkg)
    mp.validate_manifest(manifest)  # raises on failure
    for key in mp.MANIFEST_REQUIRED_KEYS:
        assert key in manifest
    assert manifest["latent_space"]["dimension"] == 192
    assert manifest["action_space"]["action_dim"] == 10
    assert manifest["runtime"]["backend"] == "lewm"


def test_validate_manifest_rejects_missing_required_key() -> None:
    manifest = packaging.pusht_manifest(weights_file="w.json", weights_format="json", tensors={})
    del manifest["limits"]
    with pytest.raises(mp.ManifestError, match="missing required keys"):
        mp.validate_manifest(manifest)


def test_validate_manifest_requires_runtime_backend() -> None:
    manifest = packaging.pusht_manifest(weights_file="w.json", weights_format="json", tensors={})
    manifest["runtime"] = {"device": "cpu"}
    with pytest.raises(mp.ManifestError, match="backend"):
        mp.validate_manifest(manifest)


# --- shape verification ------------------------------------------------------------------------


def test_read_state_shapes_json_format(tmp_path: Path) -> None:
    pkg = packaging.build_synthetic_package(tmp_path / "pkg")
    manifest = mp.load_manifest(pkg)
    shapes = mp.read_state_shapes(pkg, manifest)
    assert shapes["encoder.weight"] == (4, 8)
    assert shapes["encoder.bias"] == (4,)


def test_verify_shapes_match(tmp_path: Path) -> None:
    pkg = packaging.build_synthetic_package(tmp_path / "pkg")
    manifest = mp.load_manifest(pkg)
    mp.verify_shapes(mp.read_state_shapes(pkg, manifest), manifest)  # raises on mismatch


def test_verify_shapes_mismatch_raises() -> None:
    manifest = packaging.pusht_manifest(
        weights_file="w.json", weights_format="json", tensors={"a": {"shape": [2, 2], "dtype": "float32"}}
    )
    with pytest.raises(mp.ShapeMismatchError):
        mp.verify_shapes({"a": (2, 3)}, manifest)


def test_verify_shapes_extra_tensor_raises() -> None:
    manifest = packaging.pusht_manifest(
        weights_file="w.json", weights_format="json", tensors={"a": {"shape": [2], "dtype": "float32"}}
    )
    with pytest.raises(mp.ShapeMismatchError, match="not declared"):
        mp.verify_shapes({"a": (2,), "b": (1,)}, manifest)


def test_safetensors_header_shape_parse_without_lib(tmp_path: Path) -> None:
    # Hand-craft a minimal safetensors file (8-byte header len + JSON header) and parse shapes.
    header = {"w": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 48]}, "__metadata__": {"k": "v"}}
    blob = json.dumps(header).encode("utf-8")
    path = tmp_path / "weights.safetensors"
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * 48)
    parsed = mp._safetensors_header(path)
    assert "__metadata__" not in parsed
    assert tuple(parsed["w"]["shape"]) == (3, 4)


# --- action scaler -----------------------------------------------------------------------------


def test_action_scaler_identity_roundtrip(tmp_path: Path) -> None:
    pkg = packaging.build_synthetic_package(tmp_path / "pkg")
    scaler = mp.ActionScaler.load(pkg)
    assert scaler.kind == "identity"
    actions = [0.1, -0.2, 0.3]
    assert scaler.transform(actions) == actions
    assert scaler.inverse_transform(actions) == actions
    assert scaler.bounds is not None and len(scaler.bounds["low"]) == 10


def test_action_scaler_standardize_roundtrip_floats() -> None:
    scaler = mp.ActionScaler(kind="standardize", mean=[1.0], std=[2.0])
    # scalar arithmetic path (python float) round-trips
    x = 5.0
    assert scaler.inverse_transform(scaler.transform(x)) == pytest.approx(x)


# --- loader checksum gate ----------------------------------------------------------------------


def test_load_package_aborts_on_checksum_mismatch(tmp_path: Path) -> None:
    pkg = packaging.build_synthetic_package(tmp_path / "pkg")
    (pkg / mp.CONFIG_FILE).write_text('{"hacked": true}')
    with pytest.raises(mp.ChecksumError):
        mp.load_package(pkg)


def test_load_package_metadata_without_model(tmp_path: Path) -> None:
    pkg = packaging.build_synthetic_package(tmp_path / "pkg")
    loaded = mp.load_package(pkg)  # no build_model -> metadata + shape verification only
    assert loaded.model is None
    assert loaded.model_id == "lewm-pusht"
    assert loaded.weights_format == mp.JSON_WEIGHTS
    assert "encoder.weight" in loaded.state_shapes


# --- torch-gated: freeze / eval / no-grad ------------------------------------------------------


def test_load_package_freezes_and_evals_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(8, 4)

        def forward(self, x):  # pragma: no cover - not executed in this test
            return self.encoder(x)

    ref = Tiny()
    named = {name: param.detach() for name, param in ref.state_dict().items()}
    pkg = packaging.build_package(tmp_path / "pkg", named_weights=named, config={"latent_dim": 192})

    loaded = mp.load_package(pkg, build_model=lambda cfg: Tiny())
    assert loaded.model is not None
    assert loaded.model.training is False
    assert all(not p.requires_grad for p in loaded.model.parameters())
    mp.assert_frozen(loaded.model)
