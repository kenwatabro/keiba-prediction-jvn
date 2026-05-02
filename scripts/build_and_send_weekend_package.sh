#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PACKAGE_DATE="$(date +%Y%m%d)"
PACKAGE_ROOT="${PROJECT_ROOT}/data/packages"
RAW_DIR="${PROJECT_ROOT}/data/raw"
TRAIN_DATA="${PROJECT_ROOT}/data/processed/train_data.csv"
REMOTE="k@192.168.0.100"
REMOTE_DIR="~/projects/keiba-prediction-jvn/data/packages"
BUILD_TRAIN_DATA=0
DROP_RAW_IDS=1
INCLUDE_HC=1
INCLUDE_WC=1
INCLUDE_WH=0
INCLUDE_O1=0
INCLUDE_MARKET_FEATURES=0
PREDICTION_DATES=()
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
  reuse ./data/processed/train_data.csv
  --drop-raw-ids
  --include-hc
  --include-wc

Options:
  --package-date YYYYMMDD
  --prediction-date YYYY-MM-DD       Repeat for Saturday/Sunday.
  --remote USER@HOST
  --remote-dir PATH
  --python PATH
  --raw-dir PATH
  --train-data PATH
  --package-root PATH
  --build-train-data                 Rebuild train_data from raw files before training.
  --keep-raw-ids
  --include-o1
  --include-wh
  --no-include-hc
  --no-include-wc
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
    --build-train-data)
      BUILD_TRAIN_DATA=1
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
    --no-include-hc)
      INCLUDE_HC=0
      shift
      ;;
    --no-include-wc)
      INCLUDE_WC=0
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

BUILD_ARGS=(
  "${PYTHON}"
  "${PROJECT_ROOT}/scripts/build_weekend_package.py"
  --package-date "${PACKAGE_DATE}"
  --package-root "${PACKAGE_ROOT}"
  --raw-dir "${RAW_DIR}"
  --train-data "${TRAIN_DATA}"
)

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

TARBALL="${PACKAGE_ROOT}/weekend_package_${PACKAGE_DATE}.tar.gz"
PACKAGE_DIR="${PACKAGE_ROOT}/weekend_${PACKAGE_DATE}"

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
