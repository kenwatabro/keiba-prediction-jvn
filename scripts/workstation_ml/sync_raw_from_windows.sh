#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <windows_raw_dir_on_/mnt/*> [wsl_target_dir]"
  echo "Example: $0 /mnt/c/work/jra-van-data/raw data/raw"
  exit 1
fi

src_dir="$1"
dest_dir="${2:-data/raw}"

if [[ ! -d "$src_dir" ]]; then
  echo "Source directory not found: $src_dir" >&2
  exit 1
fi

mkdir -p "$dest_dir"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required for raw sync." >&2
  exit 1
fi

shopt -s nullglob
files=("$src_dir"/*.txt)
manifest_files=("$src_dir"/manifests/*.json)
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .txt files found in $src_dir" >&2
  exit 1
fi

rsync -a --update "${files[@]}" "$dest_dir"/
echo "Synced ${#files[@]} raw .txt file(s) from $src_dir to $dest_dir"

if [[ ${#manifest_files[@]} -gt 0 ]]; then
  mkdir -p "$dest_dir/manifests"
  rsync -a --update "${manifest_files[@]}" "$dest_dir/manifests"/
  echo "Synced ${#manifest_files[@]} manifest file(s) from $src_dir/manifests to $dest_dir/manifests"
fi
