from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.contracts import BenchmarkContract
from research_agent.manifest import ensure_run_manifest
from research_agent.store import ArtifactStore


def write_dataset(root: Path, *, label: int = 1) -> None:
    (root / "video_features_basic_pure.csv").write_text(
        "video_id,author_id\nv1,a1\n",
        encoding="utf-8",
    )
    header = "date,user_id,video_id,tab,duration_ms,long_view\n"
    (root / "log_standard_4_08_to_4_21_pure.csv").write_text(
        header + f"20220408,u1,v1,1,1000,{label}\n",
        encoding="utf-8",
    )
    (root / "log_standard_4_22_to_5_08_pure.csv").write_text(
        header + "20220422,u1,v1,1,1000,0\n",
        encoding="utf-8",
    )


class ManifestTests(unittest.TestCase):
    def test_same_path_dataset_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            write_dataset(root, label=1)
            store = ArtifactStore(Path(directory) / "runs")
            contract = BenchmarkContract(data_dir=root)
            ensure_run_manifest(store, contract, create=True)

            write_dataset(root, label=0)

            with self.assertRaisesRegex(RuntimeError, "immutable run inputs changed"):
                ensure_run_manifest(store, contract, create=False)

    def test_changed_data_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first.mkdir()
            second.mkdir()
            write_dataset(first)
            write_dataset(second)
            store = ArtifactStore(Path(directory) / "runs")
            ensure_run_manifest(
                store,
                BenchmarkContract(data_dir=first),
                create=True,
            )

            with self.assertRaisesRegex(RuntimeError, "immutable run inputs changed"):
                ensure_run_manifest(
                    store,
                    BenchmarkContract(data_dir=second),
                    create=False,
                )

    def test_missing_manifest_fails_closed_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            write_dataset(root)
            with self.assertRaisesRegex(RuntimeError, "manifest is missing"):
                ensure_run_manifest(
                    ArtifactStore(Path(directory) / "runs"),
                    BenchmarkContract(data_dir=root),
                    create=False,
                )


if __name__ == "__main__":
    unittest.main()
