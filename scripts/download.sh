#!/usr/bin/env bash
# Download the e5 index and the wiki-18 corpus into SAVE_PATH (default data/wiki18), as the
# retrievers read them: e5_Flat.index and wiki_dump.jsonl (about 80 GB).
#
# Usage: bash scripts/download.sh
set -euo pipefail
SAVE_PATH="${SAVE_PATH:-data/wiki18}"
mkdir -p "${SAVE_PATH}"
python scripts/download.py --save_path "${SAVE_PATH}"
cat "${SAVE_PATH}/part_aa" "${SAVE_PATH}/part_ab" > "${SAVE_PATH}/e5_Flat.index"
gzip -dc "${SAVE_PATH}/wiki-18.jsonl.gz" > "${SAVE_PATH}/wiki_dump.jsonl"
rm -f "${SAVE_PATH}/part_aa" "${SAVE_PATH}/part_ab" "${SAVE_PATH}/wiki-18.jsonl.gz"
echo "wrote ${SAVE_PATH}/e5_Flat.index and ${SAVE_PATH}/wiki_dump.jsonl"
