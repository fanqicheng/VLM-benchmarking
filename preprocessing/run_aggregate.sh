
Copy

#!/bin/bash
# run_aggregate.sh
# Runs aggregation for all models on a single dataset
#
# Usage:
 
 
 
# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DATASET=""
TOP_K=""
 
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --top_k)   TOP_K="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
 
if [[ -z "$DATASET" || -z "$TOP_K" ]]; then
    echo "Error: --dataset and --top_k are both required"
    echo "Usage: bash run_aggregate.sh --dataset CAM16 --top_k 0.05"
    exit 1
fi
 
# ---------------------------------------------------------------------------
# Paths — edit these to match your setup
# ---------------------------------------------------------------------------
SCRIPT="path/to/aggregate_features.py"
H5_ROOT="path/to/patch_features"
OUT_ROOT="path/to/aggregate_feature"
 
# All models to run
MODELS=(
    "conch"
    "plip"
    "keep"
    "biomedclip"
    "pathgen-clip"
    "pathoclip"
    "musk"
    "mi-zero"
    "quiltnet"
)
 
# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================"
echo "  Dataset : $DATASET"
echo "  Top-k   : $TOP_K"
echo "  Models  : ${MODELS[*]}"
echo "========================================"
 
SUCCESS=()
FAILED=()
 
for MODEL in "${MODELS[@]}"; do
    H5_DIR="${H5_ROOT}/${DATASET}/${MODEL}"
    OUT_DIR="${OUT_ROOT}/${DATASET}/${MODEL}"
 
    echo ""
    echo "── Model: $MODEL ──"
 
    # Skip if h5 dir doesn't exist
    if [[ ! -d "$H5_DIR" ]]; then
        echo "   ⚠ Skipping: $H5_DIR not found"
        FAILED+=("$MODEL (missing h5 dir)")
        continue
    fi
 
    mkdir -p "$OUT_DIR"
 
    python "$SCRIPT" \
        --h5_dir  "$H5_DIR" \
        --out_dir "$OUT_DIR" \
        --top_k   "$TOP_K"
 
    if [[ $? -eq 0 ]]; then
        echo "   ✔ Done → $OUT_DIR"
        SUCCESS+=("$MODEL")
    else
        echo "   ✗ Failed: $MODEL"
        FAILED+=("$MODEL")
    fi
done
 
# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Summary for $DATASET"
echo "========================================"
echo "  ✔ Success (${#SUCCESS[@]}): ${SUCCESS[*]}"
echo "  ✗ Failed  (${#FAILED[@]}):  ${FAILED[*]}"
echo "========================================"
 
