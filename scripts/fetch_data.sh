#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/data/raw"
REPO_URL="https://github.com/cBioPortal/datahub.git"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required. Install Git LFS, then rerun: make data" >&2
  exit 1
fi

mkdir -p "$RAW_DIR"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cohortshift-datahub.XXXXXX")"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none "$REPO_URL" "$TMP_DIR/datahub"

fetch_study() {
  local study_id="$1"
  shift
  local study_path="public/$study_id"
  local dest="$RAW_DIR/$study_id"

  git -C "$TMP_DIR/datahub" -c lfs.fetchexclude="" lfs pull -I "$study_path"
  mkdir -p "$dest"
  for file in "$@"; do
    local source="$TMP_DIR/datahub/$study_path/$file"
    if [[ ! -s "$source" ]] || head -n 1 "$source" | grep -q '^version https://git-lfs.github.com/spec/v1$'; then
      echo "Expected downloaded data file is unavailable: $source" >&2
      exit 1
    fi
    cp "$source" "$dest/$file"
  done
}

fetch_study aml_ohsu_2022 \
  data_clinical_sample.txt \
  data_clinical_patient.txt \
  data_mrna_seq_rpkm.txt \
  data_mutations.txt \
  data_gene_panel_matrix.txt \
  data_sv.txt

fetch_study laml_tcga_pan_can_atlas_2018 \
  data_clinical_sample.txt \
  data_clinical_patient.txt \
  data_mrna_seq_v2_rsem.txt \
  data_mutations.txt \
  data_gene_panel_matrix.txt \
  data_sv.txt

echo "Public cBioPortal inputs saved under $RAW_DIR"
