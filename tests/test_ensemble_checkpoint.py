from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from research_agent.models.ensemble_checkpoint import (
    load_ensemble_checkpoint,
    write_ensemble_checkpoint,
)


def _numpy_member(name: str = "base") -> dict[str, Any]:
    return {
        "name": name,
        "kind": "numpy",
        "groups": [],
        "loss": "pointwise",
        "embedding_dim": 2,
        "feature_dim": 3,
        "primary": 0.61,
        "epochs_run": 7,
        "best_epoch": 4,
        "state": {
            "V": np.arange(6, dtype=np.float32).reshape(3, 2),
            "W": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "b": np.asarray(0.25, dtype=np.float32),
        },
        "encoder_manifest": {
            "version": 1,
            "fields": ["user_id", "video_id"],
            "unknown_index": 0,
        },
        "encoder_arrays": {
            "field_offsets": np.array([0, 10, 20], dtype=np.int64),
            "field_names": np.array(["user_id", "video_id"], dtype="U8"),
        },
    }


def _torch_member(name: str = "listwise") -> dict[str, Any]:
    return {
        "name": name,
        "kind": "torch",
        "groups": ["watch", "history"],
        "loss": "listwise",
        "embedding_dim": 2,
        "feature_dim": 4,
        "primary": 0.62,
        "epochs_run": 9,
        "best_epoch": 6,
        "state": {
            "embedding.weight": np.arange(8, dtype=np.float32).reshape(4, 2),
            "linear.weight": np.arange(4, dtype=np.float64).reshape(4, 1),
            "bias": np.asarray(-0.5, dtype=np.float32),
        },
        "encoder_manifest": {"version": 1, "null_policy": "reserved_bucket"},
        "encoder_arrays": {
            "vocabulary": np.array([2, 7, 11], dtype=np.uint32),
            "present": np.array([True, False, True], dtype=np.bool_),
        },
    }


class EnsembleCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "ensemble.npz"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, members: list[dict[str, Any]] | None = None) -> Path:
        selected = members or [_numpy_member(), _torch_member()]
        return write_ensemble_checkpoint(
            self.path,
            seed=17,
            config={"variant": "six_way", "patience": 4},
            trained_members=[member["name"] for member in selected],
            active_members=selected,
            weights=[0.4, 0.6] if len(selected) == 2 else [1.0],
            validation_score_sha256="a" * 64,
            validation_primary=0.619,
        )

    def _archive(self) -> dict[str, np.ndarray]:
        with np.load(self.path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]).copy() for key in archive.files}

    def _replace_archive(self, arrays: dict[str, np.ndarray]) -> None:
        np.savez_compressed(self.path, **arrays)

    def test_round_trip_preserves_mixed_states_and_encoder_arrays(self) -> None:
        self._write()
        checkpoint = load_ensemble_checkpoint(self.path)

        self.assertEqual(checkpoint.manifest["active_members"], ["base", "listwise"])
        np.testing.assert_array_equal(
            checkpoint.states[0]["V"], _numpy_member()["state"]["V"]
        )
        self.assertEqual(checkpoint.states[0]["V"].dtype, np.dtype(np.float32))
        self.assertEqual(checkpoint.states[0]["W"].dtype, np.dtype(np.float64))
        self.assertEqual(
            checkpoint.states[1]["linear.weight"].dtype, np.dtype(np.float64)
        )
        np.testing.assert_array_equal(
            checkpoint.encoders[0]["field_names"],
            _numpy_member()["encoder_arrays"]["field_names"],
        )
        np.testing.assert_array_equal(
            checkpoint.encoders[1]["present"],
            _torch_member()["encoder_arrays"]["present"],
        )
        encoder_descriptor = checkpoint.manifest["members"][0]["encoder"]
        self.assertEqual(encoder_descriptor["version"], 1)
        self.assertEqual(
            set(encoder_descriptor["encoder_arrays"]),
            {"field_offsets", "field_names"},
        )
        with np.load(self.path, allow_pickle=False) as archive:
            self.assertFalse(any(archive[key].dtype.hasobject for key in archive.files))

    def test_changed_array_content_fails_digest_validation(self) -> None:
        self._write()
        arrays = self._archive()
        arrays["member_0_state_0"][0, 0] += 1.0
        self._replace_archive(arrays)

        with self.assertRaisesRegex(ValueError, "digest does not match"):
            load_ensemble_checkpoint(self.path)

    def test_missing_referenced_array_fails_closed(self) -> None:
        self._write()
        arrays = self._archive()
        del arrays["member_1_encoder_0"]
        self._replace_archive(arrays)

        with self.assertRaisesRegex(ValueError, "array set mismatch"):
            load_ensemble_checkpoint(self.path)

    def test_unreferenced_array_fails_closed(self) -> None:
        self._write()
        arrays = self._archive()
        arrays["surprise"] = np.array([1], dtype=np.int64)
        self._replace_archive(arrays)

        with self.assertRaisesRegex(ValueError, "unreferenced: surprise"):
            load_ensemble_checkpoint(self.path)

    def test_duplicate_member_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            self._write([_numpy_member("same"), _numpy_member("same")])

    def test_active_members_must_preserve_trained_order(self) -> None:
        members = [_numpy_member("first"), _torch_member("second")]
        with self.assertRaisesRegex(ValueError, "preserve trained member order"):
            write_ensemble_checkpoint(
                self.path,
                seed=1,
                config={},
                trained_members=["second", "first"],
                active_members=members,
                weights=[0.5, 0.5],
                validation_score_sha256="b" * 64,
                validation_primary=0.6,
            )

    def test_object_state_and_encoder_arrays_are_rejected(self) -> None:
        state_member = _numpy_member()
        state_member["state"]["V"] = np.array([object()], dtype=object)
        encoder_member = _numpy_member()
        encoder_member["encoder_arrays"]["field_names"] = np.array(
            [{"unsafe": True}], dtype=object
        )
        for member in (state_member, encoder_member):
            with self.subTest(member=member):
                with self.assertRaisesRegex(ValueError, "object or structured dtype"):
                    self._write([member])

    def test_state_shape_is_validated_before_write(self) -> None:
        member = _numpy_member()
        member["state"]["V"] = np.zeros((2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "has shape"):
            self._write([member])

    def test_weight_dimension_and_values_are_validated_before_write(self) -> None:
        member = _numpy_member()
        for weights in ([[1.0]], [0.0], [float("nan")]):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    write_ensemble_checkpoint(
                        self.path,
                        seed=1,
                        config={},
                        trained_members=["base"],
                        active_members=[member],
                        weights=weights,
                        validation_score_sha256="c" * 64,
                        validation_primary=0.6,
                    )

    def test_manifest_member_order_mutation_is_rejected(self) -> None:
        self._write()
        arrays = self._archive()
        manifest = json.loads(str(arrays["manifest_json"].item()))
        manifest["active_members"] = ["listwise", "base"]
        arrays["manifest_json"] = np.asarray(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        )
        self._replace_archive(arrays)

        with self.assertRaises(ValueError):
            load_ensemble_checkpoint(self.path)

    def test_array_dtype_mutation_is_rejected(self) -> None:
        self._write()
        arrays = self._archive()
        arrays["member_0_encoder_0"] = arrays["member_0_encoder_0"].astype(np.int32)
        self._replace_archive(arrays)

        with self.assertRaisesRegex(ValueError, "dtype does not match"):
            load_ensemble_checkpoint(self.path)


if __name__ == "__main__":
    unittest.main()
