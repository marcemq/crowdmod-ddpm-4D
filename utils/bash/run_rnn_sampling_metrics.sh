#!/bin/bash
set -euo pipefail

# # ── Dataset config and files ──────────────────────────────────────────────
configs_list=(
    "config/ATC.yml"
    "config/HERMES-T.yml"
    "config/HERMES-BO.yml"
    "config/HERMES-BN.yml"
    "config/HERMES-CR-90.yml"
)

datafiles_list=(
    "config/ATC_datafiles.yml"
    "config/HERMES-T_datafiles.yml"
    "config/HERMES-BO_datafiles.yml"
    "config/HERMES-BN_datafiles.yml"
    "config/HERMES-CR-90_datafiles.yml"
)

# ── CLI args ────────────────────────────────────────────────────────────────
DS_IDX="${1:-1}"          # dataset index or ALL (default: 1 → HERMES-T)
MODEL_CKPT="${2:-000}"    # checkpoint tag  (default: 000)

# ── Script usage ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $(basename "$0") [DS_IDX|ALL=1] [MODEL_CKPT=000]"
    echo ""
    echo "  DS_IDX     Dataset index (0=ATC, 1=HERMES-T, 2=HERMES-BO, 3=HERMES-BN,"
    echo "                            4=HERMES-CR-90), or ALL to run every dataset"
    echo "  MODEL_CKPT Checkpoint tag string (e.g. 000, 050, 120)"
    echo ""
    echo "Examples:"
    echo "  $(basename "$0")          # HERMES-T, ckpt 000"
    echo "  $(basename "$0") 0        # ATC, ckpt 000"
    echo "  $(basename "$0") 4 050    # HERMES-CR-90, ckpt 050"
    echo "  $(basename "$0") ALL      # all datasets, ckpt 000"
    echo "  $(basename "$0") ALL 050  # all datasets, ckpt 050"
    exit 0
fi

banner() { echo; echo "══════════════════════════════════════"; echo "$@"; echo "══════════════════════════════════════"; }

# ── ALL mode: one child invocation per dataset (own trap, isolated failures) ─
if [[ "${DS_IDX^^}" == "ALL" ]]; then
    failed=()
    for i in "${!configs_list[@]}"; do
        banner "ALL mode → dataset ${i}: ${configs_list[$i]}"
        "$0" "$i" "$MODEL_CKPT" || failed+=("$i:${configs_list[$i]}")
    done

    banner "ALL mode finished"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "Failed datasets:"
        printf '  - %s\n' "${failed[@]}"
        exit 1
    fi
    echo "All datasets completed successfully."
    exit 0
fi

# ── Single dataset mode ─────────────────────────────────────────────────────
# Validate DS_IDX is an integer in range
if ! [[ $DS_IDX =~ ^[0-9]+$ ]] || [[ $DS_IDX -ge ${#configs_list[@]} ]]; then
    echo "Error: DS_IDX='$DS_IDX' invalid (use 0–$((${#configs_list[@]}-1)) or ALL)"
    exit 1
fi

config="${configs_list[$DS_IDX]}"
datafiles="${datafiles_list[$DS_IDX]}"

# ── Always restore config on exit (clean or crash) ──────────────────────────
trap 'echo "[trap] Restoring $config"; git restore "$config"' EXIT

# ── Final global output dir ─────────────────────────────────────────────────
final_dir="output_rnn"
mkdir -p "$final_dir"

# Read the dataset output dir BEFORE any yq edits
out_dir="$(yq -r '.DATA_FS.OUTPUT_DIR' "$config")"

# ── Shared flags ────────────────────────────────────────────────────────────
COMMON_FLAGS=(
    --config-yml-file="$config"
    --configList-yml-file="$datafiles"
    --model-sample-to-load="$MODEL_CKPT"
)
METRICS_FLAGS=(
    --chunk-repd-past-seq=20
    --batches-to-use=20
)

# ── Helper: run samples + metrics ───────────────────────────────────────────
run_pair_convrnn() {
    python3 generate_samples.py \
        "${COMMON_FLAGS[@]}" \
        --arch="ConvRNN" \
        --plot-type="Dynamic" \
        --from-fixed-past=True

    python3 generate_metrics.py \
        "${COMMON_FLAGS[@]}" \
        "${METRICS_FLAGS[@]}" \
        --arch="ConvRNN"
}

# ── Main ────────────────────────────────────────────────────────────────────
banner "Dataset : ${config}  |  Datafiles : ${datafiles}  |  Ckpt : ${MODEL_CKPT}"

# 1. ConvRNN with GRU
banner "ConvRNN with GRU cell"
run_pair_convrnn

# 2. ConvRNN with LSTM
banner "ConvRNN with LSTM cell"
yq -i '.MODEL.CONVRNN.CELL_CLASS = "ConvLSTMCell"' "$config"
run_pair_convrnn
git restore "$config"

# 3. Move results to the global output dir (merge if already there)
if [[ -d "$out_dir" ]]; then
    dest="$final_dir/$(basename "$out_dir")"
    mkdir -p "$dest"
    cp -a "$out_dir"/. "$dest"/
    rm -rf "$out_dir"
    echo "Moved $out_dir → $dest"
else
    echo "Warning: output dir '$out_dir' not found, nothing to move."
fi

banner "Completed ${config}"