#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PACKAGE_DATE="$(date +%Y%m%d)"
PACKAGE_ROOT="${PROJECT_ROOT}/data/packages"
RAW_DIR="${PROJECT_ROOT}/data/raw"
TRAIN_DATA="${PROJECT_ROOT}/data/datasets/train_data.csv"
if [[ ! -f "${TRAIN_DATA}" && -f "${PROJECT_ROOT}/data/processed/train_data.csv" ]]; then
  TRAIN_DATA="${PROJECT_ROOT}/data/processed/train_data.csv"
fi
REMOTE="k@192.168.0.100"
REMOTE_DIR="~/projects/keiba-prediction-jvn/data/packages"
BUILD_TRAIN_DATA=0
DROP_RAW_IDS=1
INCLUDE_HC=0
INCLUDE_WC=0
INCLUDE_WH=0
INCLUDE_O1=0
INCLUDE_MARKET_FEATURES=0
PREDICTION_BASE_SOURCE=""
PREDICTION_BASE_CACHE=""
REBUILD_PREDICTION_BASE=0
RETRAIN_MODEL=0
PREDICTION_DATES=()
ALLOW_STALE_TRAIN_DATA=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Build the Friday weekend package and send the tarball to the mini PC.

Usage:
  scripts/build_and_send_weekend_package.sh \
    --package-date 20260501 \
    --prediction-date 2026-05-02 \
    --prediction-date 2026-05-03

Defaults:
  --remote k@192.168.0.100
  --remote-dir ~/projects/keiba-prediction-jvn/data/packages
  --package-root ./data/packages
  reuse ./data/datasets/train_data.csv, falling back to ./data/processed/train_data.csv
  --drop-raw-ids

Options:
  --package-date YYYYMMDD
  --prediction-date YYYY-MM-DD       Repeat for Saturday/Sunday.
  --remote USER@HOST
  --remote-dir PATH
  --python PATH
  --raw-dir PATH
  --train-data PATH
  --package-root PATH
  --prediction-base-source PATH      Reuse an existing prediction_base_weekend.csv.
  --prediction-base-cache PATH       Reusable unfiltered prediction base cache.
  --rebuild-prediction-base          Rebuild prediction_base_weekend.csv from raw files.
  --retrain-model                    Retrain even if package model artifacts exist.
  --build-train-data                 Rebuild train_data from raw files before training.
  --allow-stale-train-data           Reuse stale train data intentionally.
  --keep-raw-ids
  --include-o1
  --include-wh
  --include-hc
  --include-wc
  --include-market-features
  --dry-run                         Print commands without running them.
  -h, --help
USAGE
}

quote_cmd() {
  printf "%q " "$@"
  printf "\n"
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    quote_cmd "$@"
  else
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package-date)
      PACKAGE_DATE="$2"
      shift 2
      ;;
    --prediction-date)
      PREDICTION_DATES+=("$2")
      shift 2
      ;;
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --raw-dir)
      RAW_DIR="$2"
      shift 2
      ;;
    --train-data)
      TRAIN_DATA="$2"
      shift 2
      ;;
    --package-root)
      PACKAGE_ROOT="$2"
      shift 2
      ;;
    --prediction-base-source)
      PREDICTION_BASE_SOURCE="$2"
      shift 2
      ;;
    --prediction-base-cache)
      PREDICTION_BASE_CACHE="$2"
      shift 2
      ;;
    --rebuild-prediction-base)
      REBUILD_PREDICTION_BASE=1
      shift
      ;;
    --retrain-model)
      RETRAIN_MODEL=1
      shift
      ;;
    --build-train-data)
      BUILD_TRAIN_DATA=1
      shift
      ;;
    --allow-stale-train-data)
      ALLOW_STALE_TRAIN_DATA=1
      shift
      ;;
    --keep-raw-ids)
      DROP_RAW_IDS=0
      shift
      ;;
    --include-o1)
      INCLUDE_O1=1
      shift
      ;;
    --include-wh)
      INCLUDE_WH=1
      shift
      ;;
    --include-hc)
      INCLUDE_HC=1
      shift
      ;;
    --include-wc)
      INCLUDE_WC=1
      shift
      ;;
    --include-market-features)
      INCLUDE_MARKET_FEATURES=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "${PACKAGE_DATE}" =~ ^[0-9]{8}$ ]]; then
  echo "--package-date must be YYYYMMDD: ${PACKAGE_DATE}" >&2
  exit 2
fi

if [[ "${#PREDICTION_DATES[@]}" -eq 0 ]]; then
  echo "At least one --prediction-date is required." >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" && ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" && "${BUILD_TRAIN_DATA}" != "1" && ! -f "${TRAIN_DATA}" ]]; then
  echo "Training data not found: ${TRAIN_DATA}" >&2
  echo "Run with --build-train-data to rebuild it from raw files." >&2
  exit 1
fi

FRESHNESS_ARGS=(
  "${PYTHON}"
  "${PROJECT_ROOT}/scripts/workstation_ml/check_train_data_freshness.py"
  --raw-dir "${RAW_DIR}"
  --train-data "${TRAIN_DATA}"
)
for prediction_date in "${PREDICTION_DATES[@]}"; do
  FRESHNESS_ARGS+=(--prediction-date "${prediction_date}")
done
if [[ "${ALLOW_STALE_TRAIN_DATA}" == "1" ]]; then
  FRESHNESS_ARGS+=(--allow-stale-train-data)
fi

if [[ "${BUILD_TRAIN_DATA}" != "1" ]]; then
  echo "Checking training data freshness..."
  run_cmd "${FRESHNESS_ARGS[@]}"
fi

BUILD_ARGS=(
  "${PYTHON}"
  "${PROJECT_ROOT}/scripts/workstation_ml/build_weekend_package.py"
  --package-date "${PACKAGE_DATE}"
  --package-root "${PACKAGE_ROOT}"
  --raw-dir "${RAW_DIR}"
  --train-data "${TRAIN_DATA}"
)

TARBALL="${PACKAGE_ROOT}/weekend_package_${PACKAGE_DATE}.tar.gz"
PACKAGE_DIR="${PACKAGE_ROOT}/weekend_${PACKAGE_DATE}"
MODEL_SOURCE="${PACKAGE_DIR}/model.txt"
FEATURES_SOURCE="${PACKAGE_DIR}/model.features.json"
DEFAULT_PREDICTION_BASE_CACHE="${PACKAGE_ROOT}/prediction_base_weekend_${PACKAGE_DATE}_full.csv"

if [[ -z "${PREDICTION_BASE_CACHE}" ]]; then
  PREDICTION_BASE_CACHE="${DEFAULT_PREDICTION_BASE_CACHE}"
fi

if [[ "${RETRAIN_MODEL}" != "1" && -f "${MODEL_SOURCE}" && -f "${FEATURES_SOURCE}" ]]; then
  BUILD_ARGS+=(--model-source "${MODEL_SOURCE}" --features-source "${FEATURES_SOURCE}")
  echo "Reusing model artifacts: ${MODEL_SOURCE}"
fi

if [[ -n "${PREDICTION_BASE_SOURCE}" ]]; then
  BUILD_ARGS+=(--prediction-base-source "${PREDICTION_BASE_SOURCE}")
  echo "Reusing prediction base: ${PREDICTION_BASE_SOURCE}"
else
  BUILD_ARGS+=(--prediction-base-cache "${PREDICTION_BASE_CACHE}")
  if [[ "${REBUILD_PREDICTION_BASE}" == "1" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "Would rebuild prediction base cache: ${PREDICTION_BASE_CACHE}"
    else
      rm -f "${PREDICTION_BASE_CACHE}"
    fi
  elif [[ -f "${PREDICTION_BASE_CACHE}" ]]; then
    echo "Reusing prediction base cache: ${PREDICTION_BASE_CACHE}"
  else
    echo "No reusable prediction base cache found at ${PREDICTION_BASE_CACHE}."
    echo "build_weekend_package.py will build it from raw files this time."
  fi
fi

if [[ "${BUILD_TRAIN_DATA}" == "1" ]]; then
  BUILD_ARGS+=(--build-train-data)
fi
if [[ "${DROP_RAW_IDS}" == "1" ]]; then
  BUILD_ARGS+=(--drop-raw-ids)
fi
if [[ "${INCLUDE_HC}" == "1" ]]; then
  BUILD_ARGS+=(--include-hc)
fi
if [[ "${INCLUDE_WC}" == "1" ]]; then
  BUILD_ARGS+=(--include-wc)
fi
if [[ "${INCLUDE_WH}" == "1" ]]; then
  BUILD_ARGS+=(--include-wh)
fi
if [[ "${INCLUDE_O1}" == "1" ]]; then
  BUILD_ARGS+=(--include-o1)
fi
if [[ "${INCLUDE_MARKET_FEATURES}" == "1" ]]; then
  BUILD_ARGS+=(--include-market-features)
fi
for prediction_date in "${PREDICTION_DATES[@]}"; do
  BUILD_ARGS+=(--prediction-date "${prediction_date}")
done

echo "Building weekend package ${PACKAGE_DATE}..."
run_cmd "${BUILD_ARGS[@]}"

if [[ "${DRY_RUN}" != "1" && ! -f "${TARBALL}" ]]; then
  echo "Expected tarball was not created: ${TARBALL}" >&2
  exit 1
fi

echo "Package directory: ${PACKAGE_DIR}"
echo "Tarball: ${TARBALL}"
echo "Sending tarball to ${REMOTE}:${REMOTE_DIR}/"
run_cmd ssh "${REMOTE}" "mkdir -p ${REMOTE_DIR}"
run_cmd scp "${TARBALL}" "${REMOTE}:${REMOTE_DIR}/"

cat <<EOF
Done.

Mini PC unpack command:
  tar -xzf ${REMOTE_DIR}/$(basename "${TARBALL}") -C ${REMOTE_DIR}/

Expected mini PC package dir:
  ${REMOTE_DIR}/weekend_${PACKAGE_DATE}
EOF
