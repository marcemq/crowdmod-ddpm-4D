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
DS_IDX="${1:-1}"            # dataset index or ALL (default: 1 → HERMES-T)
MODEL_CKPT="${2:-000}"      # checkpoint tag  (default: 000)

# ── Script usage ───────────────────────────────────────────────────────────-
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $(basename "$0") [DS_IDX=1] [MODEL_CKPT=000]"
    echo ""
    echo "  DS_IDX     Dataset index (0=ATC, 1=HERMES-T, 2=HERMES-BO, 3=HERMES-BN,"
    echo "                            4=HERMES-CR-90) or ALL"
    echo "  MODEL_CKPT Checkpoint tag string (e.g. 000, 050, 120)"
    echo ""
    echo "Examples:"
    echo "  $(basename "$0")            # HERMES-T, ckpt 000"
    echo "  $(basename "$0") 0          # ATC, ckpt 000"
    echo "  $(basename "$0") 4 050      # HERMES-CR-90, ckpt 050"
    echo "  $(basename "$0") ALL      # HERMES-CR-90, ckpt 050"
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
# Validate DS_IDX is in range
if ! [[ $DS_IDX =~ ^[0-9]+$ ]] || [[ $DS_IDX -ge ${#configs_list[@]} ]]; then
    echo "Error: DS_IDX='$DS_IDX' invalid (use 0–$((${#configs_list[@]}-1)) or ALL)"
    exit 1
fi

config="${configs_list[$DS_IDX]}"
datafiles="${datafiles_list[$DS_IDX]}"

# ── Always restore config on exit (clean or crash) ──────────────────────────
trap 'echo "[trap] Restoring $config"; git restore "$config"' EXIT

# ── Log file ───────────────────────────────────────────────────
log_file="logs/run_fm_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1

# ── Integration steps to sweep ───────────────────────────────────────────────────
euler_steps=(500 250 150 100 50)
heun_steps=(250 150 75 50 25)

# ── Output dirs (read BEFORE any yq edits) ──────────────────────────────────
out_dir="$(yq -r '.DATA_FS.OUTPUT_DIR' "$config")"
out_dir="${out_dir%/}"
fm_dir="${out_dir}_fm"
mkdir -p "$fm_dir"

# ── Shared flags ─────────────────────────────────────────────────────────────
COMMON_FLAGS=(
    --config-yml-file="$config"
    --configList-yml-file="$datafiles"
    --model-sample-to-load="$MODEL_CKPT"
    --arch="FM-UNet"
)
METRICS_FLAGS=(
    --chunk-repd-past-seq=2
    --batches-to-use=2
)

# ── Helper: run samples + metrics ────────────────────────────────────────────
run_pair() {
    python3 generate_samples.py \
        "${COMMON_FLAGS[@]}" \
        --plot-type="Dynamic" \
        --from-fixed-past="True"

    python3 generate_metrics.py \
        "${COMMON_FLAGS[@]}" \
        "${METRICS_FLAGS[@]}"
}

# Restore the config from git
reset_config() {
    git restore "$config"
}

# Abort if a previous (leftover) output dir would contaminate this batch
check_clean_out() {
    if [[ -d "$out_dir" ]]; then
        echo "Error: '$out_dir' already exists. Move or remove it before running."
        exit 1
    fi
}

# Move the content of out_dir (the FM-UNet_... run folder) into
#   <out_dir>_fm/<out_dir>_<linear|conic>/
collect() {
    local w_type="${1,,}"     # Linear → linear, Conic → conic
    local target="$fm_dir/$(basename "$out_dir")_${w_type}"

    if [[ ! -d "$out_dir" ]]; then
        echo "Error: expected output dir '$out_dir' not found."
        exit 1
    fi

    mkdir -p "$target"
    local item name
    for item in "$out_dir"/*; do
        name="$(basename "$item")"
        if [[ -e "$target/$name" ]]; then
            echo "Error: '$target/$name' already exists (two runs with the same folder name?)."
            exit 1
        fi
        mv "$item" "$target/$name"
        echo "Moved $item → $target/$name"
    done
    rmdir "$out_dir"
}

# One experiment: W_TYPE (Linear|Conic), INTEGRATOR (Euler|Heun), STEPS
run_experiment() {
    local w_type="$1" intg="$2" steps="$3"

    banner "FM ${w_type} | ${intg} integrator with ${steps} steps"
    check_clean_out
    reset_config                                   # reset FIRST, then edit

    W="$w_type" INTG="$intg" \
        yq -i '.MODEL.FM.W_TYPE = env(W) | .MODEL.FM.INTEGRATOR = env(INTG)' "$config"
    STEPS="$steps" \
        yq -i ".MODEL.FM.INTEGRATOR_STEPS.${intg^^} = env(STEPS)" "$config"

    run_pair
    collect "$w_type"
}

# ── Main ─────────────────────────────────────────────────────────────────────
banner "Dataset : ${config}  |  Datafiles : ${datafiles}  |  Ckpt : ${MODEL_CKPT}"

for w_type in Linear Conic; do
    for s in "${euler_steps[@]}"; do run_experiment "$w_type" Euler "$s"; done
    for s in "${heun_steps[@]}";  do run_experiment "$w_type" Heun  "$s"; done
done

reset_config
banner "Completed ${config} → results in ${fm_dir}"