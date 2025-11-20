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
final_dir="outputs_ddim"
mkdir -p "$final_dir"

for config in "${configs_list[@]}"; do
    for datafiles in "${datafiles_list[@]}"; do

        echo "======================================"
        echo "Running with:"
        echo "  Config:     $config"
        echo "  Datafile :  $datafiles"
        echo "======================================"

        python3 generate_samples.py \
            --config-yml-file="$config" \
            --configList-yml-file="$datafiles" \
            --model-sample-to-load="000" \
            --plot-type="Dynamic" \
            --arch="DDPM-UNet"

        echo ""
    done
done

mv output_* "$final_dir"/ 2>/dev/null

echo "Moved outputs to : $final_dir"
echo ""