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
LAMBDAS="${3:-0.004,0.016,0.032,0.064,0.128}"     # LAMBDA_GUIDANCE values, comma separated

# ── Script usage ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $(basename "$0") [DS_IDX|ALL=1] [MODEL_CKPT=000] [LAMBDAS=0.004,0.016,0.032,0.064,0.128]"
    echo ""
    echo "  DS_IDX     Dataset index (0=ATC, 1=HERMES-T, 2=HERMES-BO, 3=HERMES-BN,"
    echo "                            4=HERMES-CR-90), or ALL"
    echo "  MODEL_CKPT Checkpoint tag string (e.g. 000, 050, 120)"
    echo "  LAMBDAS    Comma-separated LAMBDA_GUIDANCE values to sweep"
    echo ""
    echo "Examples:"
    echo "  $(basename "$0")                    # HERMES-T, ckpt 000, default lambdas"
    echo "  $(basename "$0") 0                  # ATC"
    echo "  $(basename "$0") 4 050              # HERMES-CR-90, ckpt 050"
    echo "  $(basename "$0") ALL                # all datasets"
    echo "  $(basename "$0") 2 000 0.004,0.016  # HERMES-BO, only two lambdas"
    exit 0
fi

banner() { echo; echo "══════════════════════════════════════"; echo "$@"; echo "══════════════════════════════════════"; }

# ── ALL mode: one child invocation per dataset (own trap, isolated failures) ─
if [[ "${DS_IDX^^}" == "ALL" ]]; then
    failed=()
    for i in "${!configs_list[@]}"; do
        banner "ALL mode → dataset ${i}: ${configs_list[$i]}"
        "$0" "$i" "$MODEL_CKPT" "$LAMBDAS" || failed+=("$i:${configs_list[$i]}")
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
# Validate DS_IDX is in range
if ! [[ $DS_IDX =~ ^[0-9]+$ ]] || [[ $DS_IDX -ge ${#configs_list[@]} ]]; then
    echo "Error: DS_IDX='$DS_IDX' invalid (use 0–$((${#configs_list[@]}-1)) or ALL)"
    exit 1
fi

config="${configs_list[$DS_IDX]}"
datafiles="${datafiles_list[$DS_IDX]}"

# ── Always restore config on exit (clean or crash) ──────────────────────────
trap 'echo "[trap] Restoring $config"; git restore "$config"' EXIT

# ── Log file (one per dataset run) ──────────────────────────────────────────
mkdir -p logs
log_file="logs/run_fm_$(basename "$config" .yml)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1

# ── Sweep parameters ────────────────────────────────────────────────────────
IFS=',' read -ra lambda_list <<< "$LAMBDAS"
ddim_div=(2 4 5 10 20 50 100)

# ── Output dirs (read BEFORE any yq edits) ──────────────────────────────────
out_dir="$(yq -r '.DATA_FS.OUTPUT_DIR' "$config")"
out_dir="${out_dir%/}"
sweep_dir="${out_dir}_sweep"
mkdir -p "$sweep_dir"

# ── Shared flags ─────────────────────────────────────────────────────────────
COMMON_FLAGS=(
    --config-yml-file="$config"
    --configList-yml-file="$datafiles"
    --model-sample-to-load="$MODEL_CKPT"
    --arch="DDPM-UNet"
)
METRICS_FLAGS=(
    --chunk-repd-past-seq=20
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

# Restore the config from git, then (optionally) re-apply a LAMBDA_GUIDANCE value
reset_config() {
    git restore "$config"
    if [[ -n "${1:-}" ]]; then
        LAM="$1" yq -i '.MODEL.DDPM.LAMBDA_GUIDANCE = env(LAM)' "$config"
    fi
}

# Abort if a previous (leftover) output dir would contaminate this batch
check_clean_out() {
    if [[ -d "$out_dir" ]]; then
        echo "Error: '$out_dir' already exists. Move or remove it before running."
        exit 1
    fi
}

# Rename out_dir → <out_dir>_<tag> and move it into the sweep folder
collect() {
    local tag="$1"
    local dest="$sweep_dir/$(basename "$out_dir")_${tag}"
    if [[ ! -d "$out_dir" ]]; then
        echo "Error: expected output dir '$out_dir' not found."
        exit 1
    fi
    if [[ -e "$dest" ]]; then
        echo "Error: destination '$dest' already exists."
        exit 1
    fi
    mv "$out_dir" "$dest"
    echo "Moved $out_dir → $dest"
}

# ── Main ─────────────────────────────────────────────────────────────────────
banner "Dataset : ${config}  |  Datafiles : ${datafiles}  |  Ckpt : ${MODEL_CKPT} |  Lambdas : ${LAMBDAS}"

# 1. Vanilla DDPM
banner "DDPM — no guidance (baseline)"
check_clean_out
reset_config
run_pair
collect "none"

# 2. Lambda sweep: guided DDPM + DDIM sweep for each LAMBDA_GUIDANCE
for lam in "${lambda_list[@]}"; do
    tag="${lam#0.}"    # 0.004 → 004, 0.128 → 128

    banner "LAMBDA_GUIDANCE = ${lam}"
    check_clean_out

    # 2a. DDPM + sparsity guidance
    banner "[λ=${lam}] DDPM — sparsity guidance"
    reset_config "$lam"
    yq -i '.MODEL.DDPM.GUIDANCE = "Sparsity"' "$config"
    run_pair

    # 2b. DDIM sweep (always with sparsity guidance)
    for div in "${ddim_div[@]}"; do
        banner "[λ=${lam}] DDIM | DDIM_DIV = $div | guidance = sparsity"
        reset_config "$lam"
        yq -i '.MODEL.DDPM.SAMPLER  = "DDIM"'     "$config"
        yq -i '.MODEL.DDPM.GUIDANCE = "Sparsity"' "$config"
        DIV="$div" yq -i '.MODEL.DDPM.DDIM_DIVIDER = env(DIV)' "$config"
        run_pair
    done

    reset_config    # back to the committed config
    collect "$tag"
done

banner "Completed ${config} → results in ${sweep_dir}"