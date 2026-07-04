from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PackageModelArtifacts:
    model_name: str = "model.txt"
    features_name: str = "features.json"

    def model_path(self, package_dir: Path) -> Path:
        return package_dir / self.model_name

    def features_path(self, package_dir: Path) -> Path:
        return package_dir / self.features_name

    def model_metadata_path(self, package_dir: Path) -> Path:
        return self.model_path(package_dir).with_suffix(".features.json")

    def required_names(self) -> tuple[str, str]:
        return self.model_name, self.features_name


PACKAGE_MODEL_ARTIFACTS = PackageModelArtifacts()
