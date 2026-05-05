#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARBALL="${1:-}"

if [[ -z "${TARBALL}" ]]; then
  TARBALL="$(find "${ROOT_DIR}" "${ROOT_DIR}/data/packages" -maxdepth 1 -type f -name 'weekend_package_*.tar.gz' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
fi

if [[ -z "${TARBALL}" || ! -f "${TARBALL}" ]]; then
  echo "Usage: $0 path/to/weekend_package_YYYYMMDD.tar.gz" >&2
  exit 2
fi

mkdir -p "${ROOT_DIR}/data/packages"
tar -xzf "${TARBALL}" -C "${ROOT_DIR}/data/packages"

package_listing="$(tar -tzf "${TARBALL}")"
package_dir="${package_listing%%/*}"
if [[ -z "${package_dir}" ]]; then
  echo "Could not determine package directory from ${TARBALL}" >&2
  exit 1
fi

mkdir -p \
  "${ROOT_DIR}/data/raceday" \
  "${ROOT_DIR}/data/env" \
  "${ROOT_DIR}/data/locks"

env_file="${ROOT_DIR}/data/env/race-day.env"
if [[ ! -f "${env_file}" ]]; then
  cat > "${env_file}" <<EOF
DISCORD_WEBHOOK_URL=
JRA_VAN_PACKAGE_DIR=${ROOT_DIR}/data/packages/${package_dir}
JRA_VAN_OUTPUT_ROOT=${ROOT_DIR}/data/raceday
EOF
else
  tmp_file="${env_file}.tmp"
  awk -v package_dir="${ROOT_DIR}/data/packages/${package_dir}" -v output_root="${ROOT_DIR}/data/raceday" '
    BEGIN { seen_package = 0; seen_output = 0 }
    /^JRA_VAN_PACKAGE_DIR=/ { print "JRA_VAN_PACKAGE_DIR=" package_dir; seen_package = 1; next }
    /^JRA_VAN_OUTPUT_ROOT=/ { print "JRA_VAN_OUTPUT_ROOT=" output_root; seen_output = 1; next }
    { print }
    END {
      if (!seen_package) print "JRA_VAN_PACKAGE_DIR=" package_dir
      if (!seen_output) print "JRA_VAN_OUTPUT_ROOT=" output_root
    }
  ' "${env_file}" > "${tmp_file}"
  mv "${tmp_file}" "${env_file}"
fi

echo "Unpacked ${TARBALL}"
echo "Package directory: ${ROOT_DIR}/data/packages/${package_dir}"
echo "Race-day env: ${env_file}"
echo "Set DISCORD_WEBHOOK_URL in ${env_file} before enabling notifications."
