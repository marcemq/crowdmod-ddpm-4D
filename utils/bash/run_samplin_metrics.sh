#!/bin/bash
set -euo pipefail

# ── CLI args ────────────────────────────────────────────────────────────────
DS_IDX="${1:-1}"          # dataset index (default: 1 → HERMES-BO)
MODEL_CKPT="${2:-000}"    # checkpoint tag  (default: 000)

# ── Script usage ───────────────────────────────────────────────────────────-
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $(basename "$0") [DS_IDX=1] [MODEL_CKPT=000]"
    echo ""
    echo "  DS_IDX     Dataset index (0=ATC, 1=HERMES-BO, 2=HERMES-BN,"
    echo "                            3=HERMES-CR-90, 4=HERMES-CR-90-OBST)"
    echo "  MODEL_CKPT Checkpoint tag string (e.g. 000, 050, 120)"
    echo ""
    echo "Examples:"
    echo "  $(basename "$0")          # HERMES-BO, ckpt 000"
    echo "  $(basename "$0") 0        # ATC, ckpt 000"
    echo "  $(basename "$0") 3 050    # HERMES-CR-90, ckpt 050"
    exit 0
fi

# Validate DS_IDX is in range
if [[ $DS_IDX -lt 0 || $DS_IDX -ge ${#configs_list[@]} ]]; then
    echo "Error: DS_IDX=$DS_IDX out of range (0–$((${#configs_list[@]}-1)))"
    exit 1
fi

# # ── Dataset config and files ──────────────────────────────────────────────
configs_list=(
    "config/ATC_ddpm.yml"
    "config/HERMES-BO.yml"
    "config/HERMES-BN.yml"
    "config/HERMES-CR-90.yml"
    "config/HERMES-CR-90-OBST.yml"
)

datafiles_list=(
    "config/ATC_datafiles.yml"
    "config/HERMES-BO_datafiles.yml"
    "config/HERMES-BN_datafiles.yml"
    "config/HERMES-CR-90_datafiles.yml"
    "config/HERMES-CR-90-OBST_datafiles.yml"
)

config="${configs_list[$DS_IDX]}"
datafiles="${datafiles_list[$DS_IDX]}"

# ── Always restore config on exit (clean or crash) ──────────────────────────
trap 'echo "[trap] Restoring $config"; git restore "$config"' EXIT

# ── DDIM dividers to sweep ───────────────────────────────────────────────────
ddim_div=(2 4 5 10 20 50 100 200 300)

# ── Shared flags ─────────────────────────────────────────────────────────────
COMMON_FLAGS=(
    --config-yml-file="$config"
    --configList-yml-file="$datafiles"
    --model-sample-to-load="$MODEL_CKPT"
    --arch="DDPM-UNet"
)
METRICS_FLAGS=(
    --chunk-repd-past=20
    --batches-to-use=20
)

# ── Helper: run samples + metrics ────────────────────────────────────────────
run_pair() {
    python3 generate_samples.py \
        "${COMMON_FLAGS[@]}" \
        --plot-type="Dynamic" \
        --from-fixed-past=True

    python3 generate_metrics.py \
        "${COMMON_FLAGS[@]}" \
        "${METRICS_FLAGS[@]}"
}

banner() { echo; echo "══════════════════════════════════════"; echo "$@"; echo "══════════════════════════════════════"; }

# ── Main ─────────────────────────────────────────────────────────────────────
banner "Dataset : ${config}  |  Datafiles : ${datafiles}  |  Ckpt : ${MODEL_CKPT}"

# 1. Vanilla DDPM
banner "DDPM — no guidance"
run_pair

# 2. DDPM + sparsity guidance
banner "DDPM — sparsity guidance"
yq -i '.MODEL.DDPM.GUIDANCE = "Sparsity"' "$config"
run_pair

git restore $config

# 3. DDIM sweep
for div in "${ddim_dividers[@]}"; do

    banner "DDIM  |  DDIM_DIV = $div  |  guidance = sparsity"

    yq -i '.MODEL.DDPM.SAMPLER   = "DDIM"'     "$config"
    yq -i '.MODEL.DDPM.GUIDANCE  = "Sparsity"' "$config"
    yq -i '.MODEL.DDPM.DDIM_DIVIDER = env(DIV)' "$config" DIV="$div"

    run_pair
    git restore "$config"

done

banner "All runs completed."