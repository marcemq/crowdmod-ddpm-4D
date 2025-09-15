import argparse
import sys
import os, re
import logging
import pickle
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.myparser import getYamlConfig
from utils.utils import get_filenames_paths

def count_samples(cfg, output_yml="samples_per_file.yml"):
    filenames = get_filenames_paths(cfg)
    samples_per_filename = []

    logging.info(f"Counting samples from {len(filenames)} files...")

    for idx, filename in enumerate(filenames):
        logging.info(f"Checking: {filename} ({idx+1}/{len(filenames)})")
        try:
            with open(filename, "rb") as file:
                seq_per_file = pickle.load(file)
                samples = seq_per_file.shape[0]
                samples_per_filename.append([os.path.basename(filename), int(samples)])
                logging.info(f"  -> {samples} samples")
        except MemoryError:
            logging.error("MemoryError: Unable to load pickle data for counting.")
        except Exception as e:
            logging.error(f"Error loading {filename}: {str(e)}")
        logging.info("-------------------------------------")

    total_samples = sum(s for _, s in samples_per_filename)
    logging.info(f"Total samples in dataset: {total_samples}")

    # Dump to YAML
    output_dict = {"DATA_LIST": samples_per_filename}
    with open(output_yml, "w") as f:
        yaml.safe_dump(output_dict, f, default_flow_style=None, sort_keys=False)

    logging.info(f"Sample counts written to {output_yml}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a diffusion model for crowd macroproperties.")
    parser.add_argument('--config-yml-file', type=str, default='config/4test/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/4test/ATC_ddpm_datafiles.yml',help='Configuration YML macroprops list for specific dataset.')
    parser.add_argument('--arch', type=str, default='DDPM-UNet', help='Architecture to be used, options: DDPM-UNet|ConvGRU')
    args = parser.parse_args()
    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    count_samples(cfg)