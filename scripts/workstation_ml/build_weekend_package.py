import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "src" / "model"
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
DATA_LOADER_DIR = PROJECT_ROOT / "src" / "data_loader"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(PREPROCESSING_DIR))
sys.path.insert(0, str(DATA_LOADER_DIR))

from make_dataset import DEFAULT_RAW_DIR, make_dataset, make_prediction_dataset  # noqa: E402
from trainer import (  # noqa: E402
    AVAILABILITY_CONTRACT_CHOICES,
    build_explanation_metadata,
    build_feature_metadata_path,
    train_model,
)
from audit_raw_coverage import build_coverage_report  # noqa: E402
from model.model_registry import PACKAGE_MODEL_ARTIFACTS  # noqa: E402
from project_paths import PACKAGES_DIR, default_train_data_path  # noqa: E402


DEFAULT_TRAIN_DATA = default_train_data_path()
DEFAULT_PACKAGE_ROOT = PACKAGES_DIR
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_TYPE = "weekend_package"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_metadata() -> dict[str, object]:
    status = run_git(["status", "--short"])
    return {
        "branch": run_git(["branch", "--show-current"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def load_feature_columns(features_path: Path) -> list[str]:
    metadata = json.loads(features_path.read_text(encoding="utf-8"))
    columns = metadata.get("feature_columns")
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        raise ValueError(f"feature_columns must be a list of strings in {features_path}")
    return columns


def load_feature_metadata(features_path: Path) -> dict[str, object]:
    metadata = json.loads(features_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Feature metadata must be a JSON object in {features_path}")
    return metadata


def summarize_explanation_metadata(features_path: Path) -> dict[str, object]:
    metadata = load_feature_metadata(features_path)
    explanation = metadata.get("explanation")
    if not isinstance(explanation, dict):
        return {"present": False}

    display_names = explanation.get("feature_display_names")
    feature_groups = explanation.get("feature_groups")
    return {
        "present": True,
        "method": explanation.get("method"),
        "score_space": explanation.get("score_space"),
        "default_top_k": explanation.get("default_top_k"),
        "display_name_count": len(display_names) if isinstance(display_names, dict) else 0,
        "group_count": len(feature_groups) if isinstance(feature_groups, dict) else 0,
    }


def ensure_explanation_metadata(features_path: Path) -> None:
    metadata = load_feature_metadata(features_path)
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(isinstance(column, str) for column in feature_columns):
        raise ValueError(f"feature_columns must be a list of strings in {features_path}")
    if isinstance(metadata.get("explanation"), dict):
        return
    metadata["explanation"] = build_explanation_metadata(feature_columns)
    features_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def summarize_csv_dates(csv_path: Path) -> dict[str, object]:
    if not csv_path.exists():
        return {"exists": False}
    df = pd.read_csv(csv_path, usecols=lambda column: column in {"RaceDate", "RaceKey"}, low_memory=False)
    summary: dict[str, object] = {"exists": True, "rows": int(len(df))}
    if "RaceKey" in df.columns:
        summary["race_count"] = int(df["RaceKey"].nunique())
    if "RaceDate" in df.columns:
        race_dates = pd.to_datetime(df["RaceDate"], errors="coerce").dropna()
        if not race_dates.empty:
            summary["start_date"] = race_dates.min().strftime("%Y-%m-%d")
            summary["end_date"] = race_dates.max().strftime("%Y-%m-%d")
            summary["dates"] = sorted(race_dates.dt.strftime("%Y-%m-%d").unique().tolist())
    return summary


def validate_prediction_base(prediction_path: Path, features_path: Path) -> dict[str, object]:
    feature_columns = load_feature_columns(features_path)
    prediction_df = pd.read_csv(prediction_path, low_memory=False)
    required_columns = ["RaceDate", "RaceKey", "JyoCD", "RaceNum", "HassoTime", "Umaban"]
    missing_required = [column for column in required_columns if column not in prediction_df.columns]
    if missing_required:
        raise ValueError(
            "prediction_base_weekend.csv is missing required race-day columns: "
            + ", ".join(missing_required)
        )
    umaban = pd.to_numeric(prediction_df["Umaban"], errors="coerce").fillna(0)
    missing_number_mask = umaban.le(0)
    if "Wakuban" in prediction_df.columns:
        wakuban = pd.to_numeric(prediction_df["Wakuban"], errors="coerce").fillna(0)
        missing_number_mask = missing_number_mask | wakuban.le(0)
    if missing_number_mask.any():
        missing_rows = prediction_df.loc[missing_number_mask].copy()
        race_dates = pd.to_datetime(missing_rows["RaceDate"], errors="coerce").dt.strftime("%Y-%m-%d")
        sample_racekeys = sorted(missing_rows["RaceKey"].dropna().astype(str).unique().tolist())[:10]
        counts_by_date = {
            str(date): int(count)
            for date, count in race_dates.value_counts(dropna=False).sort_index().items()
        }
        raise ValueError(
            "prediction_base_weekend.csv contains rows with missing Wakuban/Umaban. "
            "Refetch finalized race-card data before packaging. "
            f"rows={int(missing_number_mask.sum())}, races={int(missing_rows['RaceKey'].nunique())}, "
            f"counts_by_date={counts_by_date}, sample_racekeys={sample_racekeys}"
        )
    missing = [column for column in feature_columns if column not in prediction_df.columns]
    if missing:
        raise ValueError(
            "prediction_base_weekend.csv is missing model feature columns: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )
    return {
        "rows": int(len(prediction_df)),
        "race_count": int(prediction_df["RaceKey"].nunique()),
        "feature_count": len(feature_columns),
        "columns": list(prediction_df.columns),
    }


def write_racekeys(prediction_path: Path, racekeys_path: Path) -> list[str]:
    prediction_df = pd.read_csv(prediction_path, usecols=["RaceKey"], dtype={"RaceKey": "string"})
    racekeys = sorted(prediction_df["RaceKey"].dropna().astype(str).unique().tolist())
    racekeys_path.write_text("\n".join(racekeys) + ("\n" if racekeys else ""), encoding="utf-8")
    return racekeys


def copy_model_artifacts(model_source: Path, features_source: Path, package_dir: Path) -> tuple[Path, Path]:
    model_path = PACKAGE_MODEL_ARTIFACTS.model_path(package_dir)
    features_path = PACKAGE_MODEL_ARTIFACTS.features_path(package_dir)
    model_metadata_path = PACKAGE_MODEL_ARTIFACTS.model_metadata_path(package_dir)
    if model_source.resolve() != model_path.resolve():
        shutil.copy2(model_source, model_path)
    if features_source.resolve() != features_path.resolve():
        shutil.copy2(features_source, features_path)
    if features_source.resolve() != model_metadata_path.resolve():
        shutil.copy2(features_source, model_metadata_path)
    ensure_explanation_metadata(features_path)
    ensure_explanation_metadata(model_metadata_path)
    return model_path, features_path


def build_or_copy_model(args: argparse.Namespace, package_dir: Path) -> tuple[Path, Path]:
    if args.model_source:
        model_source = Path(args.model_source)
        features_source = Path(args.features_source) if args.features_source else build_feature_metadata_path(model_source)
        if not model_source.exists():
            raise FileNotFoundError(f"Model source not found: {model_source}")
        if not features_source.exists():
            raise FileNotFoundError(f"Feature metadata source not found: {features_source}")
        return copy_model_artifacts(model_source, features_source, package_dir)

    train_data = Path(args.train_data)
    if args.build_train_data:
        make_dataset(
            raw_dir=args.raw_dir,
            output_dir=train_data.parent,
            output_filename=train_data.name,
            include_o1=args.include_o1,
            include_wh=args.include_wh,
            include_hc=args.include_hc,
            include_wc=args.include_wc,
        )
    if not train_data.exists():
        raise FileNotFoundError(f"Training data not found: {train_data}")

    model_path = PACKAGE_MODEL_ARTIFACTS.model_path(package_dir)
    train_model(
        data_path=train_data,
        model_path=model_path,
        target_col=args.target,
        objective_name=args.objective,
        drop_raw_ids=args.drop_raw_ids,
        include_market_features=args.include_market_features,
        availability_contract=args.availability_contract,
    )
    features_path = build_feature_metadata_path(model_path)
    package_features_path = PACKAGE_MODEL_ARTIFACTS.features_path(package_dir)
    shutil.copy2(features_path, package_features_path)
    return model_path, package_features_path


def filter_prediction_dates(source_path: Path, prediction_dates: list[str], output_path: Path | None = None) -> None:
    output = output_path or source_path
    if not prediction_dates:
        if output.resolve() != source_path.resolve():
            shutil.copy2(source_path, output)
        return
    df = pd.read_csv(source_path, low_memory=False)
    if "RaceDate" not in df.columns:
        raise ValueError("prediction_base_weekend.csv must contain RaceDate when --prediction-date is used.")
    dates = {pd.Timestamp(value).strftime("%Y-%m-%d") for value in prediction_dates}
    race_dates = pd.to_datetime(df["RaceDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    filtered = df.loc[race_dates.isin(dates)].copy()
    if filtered.empty:
        raise ValueError(f"No prediction rows matched --prediction-date values: {sorted(dates)}")
    matched_dates = set(pd.to_datetime(filtered["RaceDate"], errors="coerce").dt.strftime("%Y-%m-%d").dropna())
    missing_dates = sorted(dates - matched_dates)
    if missing_dates:
        raise ValueError(f"prediction_base_weekend.csv is missing requested dates: {missing_dates}")
    filtered.to_csv(output, index=False)


def try_filter_prediction_dates(source_path: Path, prediction_dates: list[str], output_path: Path) -> bool:
    try:
        filter_prediction_dates(source_path, prediction_dates, output_path=output_path)
    except ValueError as exc:
        print(f"Reusable prediction base skipped: {exc}")
        return False
    return True


def build_or_copy_prediction_base(args: argparse.Namespace, package_dir: Path) -> Path:
    prediction_path = package_dir / "prediction_base_weekend.csv"
    if args.prediction_base_source:
        source = Path(args.prediction_base_source)
        if not source.exists():
            raise FileNotFoundError(f"Prediction base source not found: {source}")
        if source.resolve() == prediction_path.resolve():
            raise ValueError(
                "--prediction-base-source must not point to the package output prediction_base_weekend.csv; "
                "use --prediction-base-cache for reusable unfiltered data."
            )
        shutil.copy2(source, prediction_path)
        filter_prediction_dates(prediction_path, args.prediction_date)
        return prediction_path

    prediction_base_cache = Path(args.prediction_base_cache) if args.prediction_base_cache else None
    if prediction_base_cache and prediction_base_cache.exists():
        if try_filter_prediction_dates(prediction_base_cache, args.prediction_date, prediction_path):
            return prediction_path
        print(f"Rebuilding prediction base cache: {prediction_base_cache}")

    build_output_path = prediction_base_cache or prediction_path
    make_prediction_dataset(
        raw_dir=args.raw_dir,
        output_dir=build_output_path.parent,
        output_filename=build_output_path.name,
        prediction_date=None,
        include_o1=args.include_o1,
        include_wh=args.include_wh,
        include_hc=args.include_hc,
        include_wc=args.include_wc,
    )
    if not build_output_path.exists():
        raise FileNotFoundError(f"Prediction base was not created: {build_output_path}")
    filter_prediction_dates(build_output_path, args.prediction_date, output_path=prediction_path)
    return prediction_path


def write_manifest(
    args: argparse.Namespace,
    package_dir: Path,
    tarball_path: Path,
    model_path: Path,
    features_path: Path,
    prediction_path: Path,
    racekeys_path: Path,
    validation_summary: dict[str, object],
    racekeys: list[str],
) -> Path:
    contents = sorted(path.name for path in package_dir.iterdir() if path.is_file() and path.name != "manifest.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "package_id": package_dir.name,
        "package_date": args.package_date,
        "created_at": utc_now_iso(),
        "git": git_metadata(),
        "command": " ".join(sys.argv),
        "artifacts": {
            "model": model_path.name,
            "features": features_path.name,
            "model_features_metadata": build_feature_metadata_path(model_path).name,
            "prediction_base": prediction_path.name,
            "racekeys": racekeys_path.name,
            "tarball": str(tarball_path),
            "contents": contents,
        },
        "training": {
            "mode": "copied" if args.model_source else "trained",
            "data_path": str(Path(args.train_data)),
            "data_summary": summarize_csv_dates(Path(args.train_data)),
            "target": args.target,
            "objective": args.objective,
            "drop_raw_ids": bool(args.drop_raw_ids),
            "include_market_features": bool(args.include_market_features),
            "availability_contract": args.availability_contract,
        },
        "prediction_base": {
            "source": str(Path(args.prediction_base_source)) if args.prediction_base_source else "built_from_raw",
            "requested_dates": args.prediction_date,
            "summary": summarize_csv_dates(prediction_path),
            "validation": validation_summary,
        },
        "raw_coverage": build_coverage_report(Path(args.raw_dir), ["RACE", "SLOP", "WOOD"], None, None),
        "train_coverage": summarize_csv_dates(Path(args.train_data)),
        "prediction_coverage": summarize_csv_dates(prediction_path),
        "raw": {
            "raw_dir": str(Path(args.raw_dir)),
            "include_o1": bool(args.include_o1),
            "include_wh": bool(args.include_wh),
            "include_hc": bool(args.include_hc),
            "include_wc": bool(args.include_wc),
        },
        "racekeys": {
            "count": len(racekeys),
            "path": racekeys_path.name,
        },
        "race_day_rules": {
            "retrain_on_race_day": False,
            "feature_contract": features_path.name,
            "record_data_as_of_timestamp": True,
        },
        "explanation": summarize_explanation_metadata(features_path),
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def create_tarball(package_dir: Path, tarball_path: Path) -> None:
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "w:gz") as tar:
        for path in sorted(package_dir.iterdir()):
            if path.is_file():
                tar.add(path, arcname=f"{package_dir.name}/{path.name}")


def build_weekend_package(args: argparse.Namespace) -> tuple[Path, Path]:
    package_root = Path(args.package_root)
    package_dir = package_root / f"weekend_{args.package_date}"
    package_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = build_or_copy_prediction_base(args, package_dir)
    model_path, features_path = build_or_copy_model(args, package_dir)
    validation_summary = validate_prediction_base(prediction_path, features_path)
    racekeys_path = package_dir / "racekeys_weekend.txt"
    racekeys = write_racekeys(prediction_path, racekeys_path)
    if not racekeys:
        raise ValueError("racekeys_weekend.txt would be empty.")

    tarball_path = package_root / f"weekend_package_{args.package_date}.tar.gz"
    write_manifest(
        args,
        package_dir,
        tarball_path,
        model_path,
        features_path,
        prediction_path,
        racekeys_path,
        validation_summary,
        racekeys,
    )
    create_tarball(package_dir, tarball_path)
    return package_dir, tarball_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Friday weekend prediction package for race-day Ubuntu use.")
    parser.add_argument("--package-date", default=datetime.now().strftime("%Y%m%d"), help="Package date, usually Friday, as YYYYMMDD.")
    parser.add_argument("--package-root", default=str(DEFAULT_PACKAGE_ROOT), help="Directory where package folders and tarballs are written.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Raw JV-Link text directory.")
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Training CSV path.")
    parser.add_argument("--build-train-data", action="store_true", help="Rebuild --train-data from raw files before training.")
    parser.add_argument("--prediction-base-source", default=None, help="Existing prediction base CSV to copy instead of building from raw files.")
    parser.add_argument("--prediction-base-cache", default=None, help="Reusable unfiltered prediction base CSV. Built from raw files if missing.")
    parser.add_argument("--model-source", default=None, help="Existing LightGBM model to copy instead of training.")
    parser.add_argument("--features-source", default=None, help="Existing features JSON to copy with --model-source.")
    parser.add_argument("--prediction-date", action="append", default=[], help="Restrict weekend prediction rows to a race date. Repeat for Sat/Sun.")
    parser.add_argument("--target", default="TargetWin", choices=["TargetTop3", "TargetWin"], help="Model target when training.")
    parser.add_argument("--objective", default="binary", choices=["binary", "lambdarank"], help="LightGBM objective when training.")
    parser.add_argument("--drop-raw-ids", action="store_true", help="Exclude raw owner/jockey/trainer ID columns when training.")
    parser.add_argument("--include-market-features", action="store_true", help="Include market columns in the trained model.")
    parser.add_argument(
        "--availability-contract",
        choices=AVAILABILITY_CONTRACT_CHOICES,
        default=None,
        help="Restrict trained model features to a deployment-time availability contract.",
    )
    parser.add_argument("--include-o1", action="store_true", help="Include O1 odds snapshots when building datasets.")
    parser.add_argument("--include-wh", action="store_true", help="Include WH body-weight features when building datasets.")
    parser.add_argument("--include-hc", action="store_true", help="Include HC hanro workout features when building datasets.")
    parser.add_argument("--include-wc", action="store_true", help="Include WC wood-chip workout features when building datasets.")
    return parser.parse_args()


def main() -> None:
    package_dir, tarball_path = build_weekend_package(parse_args())
    print(f"Weekend package directory: {package_dir}")
    print(f"Weekend package tarball: {tarball_path}")


if __name__ == "__main__":
    main()
