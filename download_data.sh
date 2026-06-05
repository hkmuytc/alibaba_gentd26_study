#!/usr/bin/env bash
# Downloads the GenTD26 dataset (Alibaba cluster-trace-v2026-GenAI) into data/raw/
# Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI

set -euo pipefail

BASE_URL="https://github.com/alibaba/clusterdata/raw/master/cluster-trace-v2026-GenAI"
RAW_DIR="$(dirname "$0")/data/raw"

mkdir -p "$RAW_DIR"

FILES=(
    "pod_gpu_duty_cycle_anon.tar.gz"
    "pod_gpu_memory_used_bytes_anon.tar.gz"
    "pod_memory_util_anon.tar.gz"
    "qps.tar.gz"
    "data_trace_processed.tar.gz"
)

echo "Downloading GenTD26 dataset to $RAW_DIR ..."

for FILE in "${FILES[@]}"; do
    CSV="${FILE%.tar.gz}.csv"
    if [[ -f "$RAW_DIR/$CSV" ]]; then
        echo "  [skip] $CSV already exists"
        continue
    fi
    echo "  Downloading $FILE ..."
    curl -fL --progress-bar "$BASE_URL/$FILE" -o "$RAW_DIR/$FILE"
    echo "  Extracting $FILE ..."
    tar -xzf "$RAW_DIR/$FILE" -C "$RAW_DIR"
    rm "$RAW_DIR/$FILE"
done

echo "Done. Files in $RAW_DIR:"
ls -lh "$RAW_DIR"/*.csv 2>/dev/null || echo "  (no CSV files found — check extraction above)"
