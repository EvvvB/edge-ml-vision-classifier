from __future__ import annotations

import hashlib

from app.inference.model import MODEL_HASH_HEX_CHARS, model_identity


def test_model_identity_hashes_weights_and_reads_manifest(tmp_path) -> None:
    weights = tmp_path / "toy.pt"
    weights.write_bytes(b"weights-bytes")
    (tmp_path / "toy.manifest.json").write_text(
        '{"model_version": "toy-v1"}', encoding="utf-8"
    )

    identity = model_identity(weights)

    expected_hash = hashlib.sha256(b"weights-bytes").hexdigest()[
        :MODEL_HASH_HEX_CHARS
    ]
    assert identity == {
        "hash": expected_hash,
        "manifest": {"model_version": "toy-v1"},
    }


def test_model_identity_survives_missing_files(tmp_path) -> None:
    assert model_identity(tmp_path / "missing.pt") == {
        "hash": None,
        "manifest": None,
    }


def test_manifest_must_be_a_json_object(tmp_path) -> None:
    weights = tmp_path / "toy.pt"
    weights.write_bytes(b"weights-bytes")
    (tmp_path / "toy.manifest.json").write_text("[1, 2]", encoding="utf-8")

    assert model_identity(weights)["manifest"] is None
