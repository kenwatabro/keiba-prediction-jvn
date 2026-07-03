import tempfile
import unittest
from pathlib import Path


from src.project_paths import existing_or_default


class ProjectPathTests(unittest.TestCase):
    def test_existing_or_default_uses_new_path_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_path = root / "data" / "datasets" / "train_data.csv"
            legacy_path = root / "data" / "processed" / "train_data.csv"
            default_path.parent.mkdir(parents=True)
            legacy_path.parent.mkdir(parents=True)
            default_path.write_text("new\n", encoding="utf-8")
            legacy_path.write_text("legacy\n", encoding="utf-8")

            self.assertEqual(existing_or_default(default_path, legacy_path), default_path)

    def test_existing_or_default_falls_back_to_legacy_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_path = root / "data" / "datasets" / "train_data.csv"
            legacy_path = root / "data" / "processed" / "train_data.csv"
            legacy_path.parent.mkdir(parents=True)
            legacy_path.write_text("legacy\n", encoding="utf-8")

            self.assertEqual(existing_or_default(default_path, legacy_path), legacy_path)

    def test_existing_or_default_returns_new_path_when_neither_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_path = root / "data" / "datasets" / "train_data.csv"
            legacy_path = root / "data" / "processed" / "train_data.csv"

            self.assertEqual(existing_or_default(default_path, legacy_path), default_path)


if __name__ == "__main__":
    unittest.main()
