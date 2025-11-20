#!/bin/bash

# Configs list per dataset
configs_list=(
    "config/ATC_ddpm.yml"
    "config/HERMES-BO.yml"
    "config/HERMES-CR-120.yml"
    "config/HERMES-CR-120-OBST.yml"
)

# Datafiles list per dataset
datafiles_list=(
    "config/ATC_ddpm_datafiles.yml"
    "config/HERMES-BO_datafiles.yml"
    "config/HERMES-CR-120_datafiles.yml"
    "config/HERMES-CR-120-OBST_datafiles.yml"
)

# Final global output dir
final_dir="output_ddim"
mkdir -p "$final_dir"

# Loop over indices
for i in "${!configs_list[@]}"; do

    config="${configs_list[$i]}"
    datafiles="${datafiles_list[$i]}"

    echo "======================================"
    echo "Running with:"
    echo "  Config:     $config"
    echo "  Datafile :  $datafiles"
    echo "======================================"

    yq -i '.MODEL.DDPM.SAMPLER = "DDIM"' "$config"
    yq -i '.MODEL.DDPM.GUIDANCE = "sparsity"' "$config"

    python3 generate_samples.py \
            --config-yml-file="$config" \
            --configList-yml-file="$datafiles" \
            --model-sample-to-load="000" \
            --plot-type="Dynamic" \
            --arch="DDPM-UNet"

    echo ""

    git restore $config
done

mv output_* "$final_dir"/ 2>/dev/null

echo "Moved outputs to : $final_dir"
echo ""