#!/bin/bash
set -euo pipefail

# ── CLI args ────────────────────────────────────────────────────────────────
DS_IDX="${1:-1}"            # dataset index (default: 1 → HERMES-T)
MODEL_CKPT="${2:-000}"      # checkpoint tag  (default: 000)

# ── Script usage ───────────────────────────────────────────────────────────-
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $(basename "$0") [DS_IDX=1] [MODEL_CKPT=000]"
    echo ""
    echo "  DS_IDX     Dataset index (0=ATC, 1=HERMES-T, 2=HERMES-BO, 3=HERMES-BN,"
    echo "                            4=HERMES-CR-90, 5=HERMES-CR-90-OBST)"
    echo "  MODEL_CKPT Checkpoint tag string (e.g. 000, 050, 120)"
    echo ""
    echo "Examples:"
    echo "  $(basename "$0")            # HERMES-T, ckpt 000"
    echo "  $(basename "$0") 0          # ATC, ckpt 000"
    echo "  $(basename "$0") 4 050      # HERMES-CR-90, ckpt 050"
    exit 0
fi

# # ── Dataset config and files ──────────────────────────────────────────────
configs_list=(
    "config/ATC_ddpm.yml"
    "config/HERMES-T.yml"
    "config/HERMES-BO.yml"
    "config/HERMES-BN.yml"
    "config/HERMES-CR-90.yml"
    "config/HERMES-CR-90-OBST.yml"
)

datafiles_list=(
    "config/ATC_datafiles.yml"
    "config/HERMES-T_datafiles.yml"
    "config/HERMES-BO_datafiles.yml"
    "config/HERMES-BN_datafiles.yml"
    "config/HERMES-CR-90_datafiles.yml"
    "config/HERMES-CR-90-OBST_datafiles.yml"
)

# Validate DS_IDX is in range
if [[ $DS_IDX -lt 0 || $DS_IDX -ge ${#configs_list[@]} ]]; then
    echo "Error: DS_IDX=$DS_IDX out of range (0–$((${#configs_list[@]}-1)))"
    exit 1
fi

config="${configs_list[$DS_IDX]}"
datafiles="${datafiles_list[$DS_IDX]}"

# ── Always restore config on exit (clean or crash) ──────────────────────────
trap 'echo "[trap] Restoring $config"; git restore "$config"' EXIT

# ── Integration steps to sweep ───────────────────────────────────────────────────
euler_steps=(1000 950 900)
heun_steps=(500 450 400)

# ── Shared flags ─────────────────────────────────────────────────────────────
COMMON_FLAGS=(
    --config-yml-file="$config"
    --configList-yml-file="$datafiles"
    --model-sample-to-load="$MODEL_CKPT"
    --arch="FM-UNet"
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
        --from-fixed-past="True"

    python3 generate_metrics.py \
        "${COMMON_FLAGS[@]}" \
        "${METRICS_FLAGS[@]}"
}

banner() { echo; echo "══════════════════════════════════════"; echo "$@"; echo "══════════════════════════════════════"; }

# ── Main ─────────────────────────────────────────────────────────────────────
banner "Dataset : ${config}  |  Datafiles : ${datafiles}  |  Ckpt : ${MODEL_CKPT}"

# 1. FM_linear + Euler
for euler_step in "${euler_steps[@]}"; do
    banner "FM_Linear | Euler integrator with $euler_step step"
    EULER_STEP="$euler_step" yq -i '.MODEL.FM.INTEGRATOR_STEPS.EULER = env(EULER_STEP)' "$config"

    run_pair
    git restore "$config"

done

# 2. FM_linear + Heun
for heun_step in "${heun_steps[@]}"; do
    banner "FM_Linear | Heun integrator with $heun_step step"
    yq -i '.MODEL.FM.INTEGRATOR = "Heun"' "$config"
    HEUN_STEP="$heun_step" yq -i '.MODEL.FM.INTEGRATOR_STEPS.HEUN = env(HEUN_STEP)' "$config"

    run_pair
    git restore "$config"

done

# 3. FM_Conic + Euler
for euler_step in "${euler_steps[@]}"; do
    banner "FM_Conic | Euler integrator with $euler_step step"
    yq -i '.MODEL.FM.W_TYPE = "Conic"' "$config"
    EULER_STEP="$euler_step" yq -i '.MODEL.FM.INTEGRATOR_STEPS.EULER = env(EULER_STEP)' "$config"

    run_pair
    git restore "$config"

done

# 4. FM_Conic + Heun
for heun_step in "${heun_steps[@]}"; do
    banner "FM_Conic | Heun integrator with $heun_step step"
    yq -i '.MODEL.FM.W_TYPE     = "Conic"' "$config"
    yq -i '.MODEL.FM.INTEGRATOR = "Heun"' "$config"
    HEUN_STEP="$heun_step" yq -i '.MODEL.FM.INTEGRATOR_STEPS.HEUN = env(HEUN_STEP)' "$config"

    run_pair
    git restore "$config"

done

banner "All runs completed."