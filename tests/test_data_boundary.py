import gzip
import json
import tempfile
import unittest
from pathlib import Path

from data import load
from research_agent.data_boundary import (
    load_research_splits,
    load_staged_splits,
    stage_research_splits,
)


class DataBoundaryTests(unittest.TestCase):
    @staticmethod
    def write_dataset(root: Path) -> None:
        (root / "video_features_basic_pure.csv").write_text(
            "video_id,author_id\nv1,a1\n",
            encoding="utf-8",
        )
        header = "date,user_id,video_id,tab,duration_ms,long_view\n"
        (root / "log_standard_4_08_to_4_21_pure.csv").write_text(
            header + "20220408,u1,v1,1,1000,1\n",
            encoding="utf-8",
        )
        (root / "log_standard_4_22_to_5_08_pure.csv").write_text(
            header
            + "20220422,u1,v1,1,1000,0\n"
            + "20220429,SHOULD_NOT_LOAD,v1,1,1000,1\n",
            encoding="utf-8",
        )

    def test_research_loader_does_not_materialize_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_dataset(root)

            research = load_research_splits(root, ("train", "valid"))
            complete = load(root)

            self.assertEqual(set(research), {"train", "valid"})
            self.assertNotIn("SHOULD_NOT_LOAD", [row[1] for rows in research.values() for row in rows])
            self.assertIn("SHOULD_NOT_LOAD", [row[1] for row in complete["test"]])

    def test_staged_input_rejects_test_dated_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poisoned.json.gz"
            payload = {
                "schema_version": 2,
                "source_data_sha256": "a" * 64,
                "staging_code_sha256": "b" * 64,
                "splits": {
                    "train": [[20220408, "u", "v", "a", "1", 1000.0, 1, 9, 0.0, "m", "t", 0]],
                    "valid": [[20220429, "TEST_ROW", "v", "a", "1", 1000.0, 1, 9, 0.0, "m", "t", 0]],
                },
            }
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)

            with self.assertRaisesRegex(ValueError, "date boundary"):
                load_staged_splits(
                    path,
                    source_data_sha256="a" * 64,
                    staging_code_sha256="b" * 64,
                )

    def test_invalid_cached_stage_is_regenerated_from_canonical_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_dataset(root)
            destination = root / "cache" / "research.json.gz"
            destination.parent.mkdir()
            destination.write_bytes(b"not a valid gzip payload")

            stage_research_splits(
                root,
                destination,
                train_split="train",
                validation_split="valid",
                source_data_sha256="a" * 64,
                staging_code_sha256="b" * 64,
            )
            staged = load_staged_splits(
                destination,
                source_data_sha256="a" * 64,
                staging_code_sha256="b" * 64,
            )

            self.assertEqual(set(staged), {"train", "valid"})
            self.assertEqual([row[1] for row in staged["valid"]], ["u1"])


if __name__ == "__main__":
    unittest.main()
