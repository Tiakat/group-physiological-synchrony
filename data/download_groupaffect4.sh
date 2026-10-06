#!/usr/bin/env bash
# Download the GroupAffect-4 public subset (<4 GB) into data/raw/.
# Source: https://zenodo.org/records/20809799 (CC-BY, privacy-processed)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw && cd raw
BASE="https://zenodo.org/records/20809799/files"
for f in affectai_metadata.zip affectai_physio.zip affectai_beh.zip affectai_transcripts.zip; do
  if [ ! -f "$f" ]; then
    echo "downloading $f ..."
    curl -L -o "$f" "$BASE/$f?download=1"
  else
    echo "$f already present, skipping"
  fi
done
echo "done. unzip as needed; raw zips are git-ignored."
